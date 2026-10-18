"""The RSC1 file: compressed activations plus the metadata to use them later.

Layout:   b"RSC1" | u64 header offset | u64 header length | blobs ... | JSON header
The header is written last (so the writer streams), and the two u64 fields are patched at close. A file whose
header offset is 0 was not closed and is refused.

Header (JSON):
  format, version, library, created
  model        whatever the caller passes (id, revision, n_layers, d_model, dtype, ...); see capture.model_info
  capture      site, seq_len, n_seq, d_model, n_chunks, hardware, extra
  spec         codec spec string;  calib: outlier channels per stored layer
  lossless     {"codec": "zstd", "level": L, "shuffle": bool}
  layers       one entry per stored layer, in storage order: index, layer, site, quant, keyframe, outlier_channels,
               hook names, measured fidelity (rel_l2, max_abs) over the whole file
  annotations  name -> {dtype, shape_tail}: per-sequence ([N] or [N, ...]) arrays stored chunk by chunk
  chunks       one entry per write call: seq_start, n_seq, tokens blob, attention_mask blob, per-layer stream blobs,
               annotation blobs. Blob = {offset, nbytes, raw_nbytes, crc32, dtype, shape}
  bytes        stored vs raw byte counts and the ratio
"""
import bisect
import copy
import json
import operator
import os
import platform
import struct
import time
import zlib
from typing import Optional

import numpy as np
import torch

from . import lossless as LL
from . import quant as Q
from .quant import CODEC_VERSION
from ._version import __version__
from .codec import Calib, Decoder, Encoder, calibrate
from .spec import Spec

FORMAT = b"RSC1"
VERSION = 1
HEAD = len(FORMAT) + 16


def _hardware():
    info = {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _as_tensor(a):
    return a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a))


class Writer:
    """One file per capture. `write` once per batch of sequences; each call becomes one chunk."""

    def __init__(self, path, *, model, layers, spec, calib: Optional[Calib] = None, site="resid_post",
                 seq_len=None, level=3, shuffle=True, hook_names=None, extra=None):
        """hook_names: layer -> {"hf": ..., "transformer_lens": ...} (keys may be int or str).
        model, hook_names and extra must be JSON-serializable (checked here, before the file is opened)."""
        self.path = str(path)
        self.spec = Spec.parse(spec)
        self.layers = [int(l) for l in layers]
        if not self.layers or self.layers != sorted(set(self.layers)):
            raise ValueError("layers must be a non-empty, strictly increasing list")
        self.calib = calib if self.spec.outliers else None
        if self.calib is not None:
            self.calib.check(self.spec, len(self.layers))
        self.site, self.seq_len, self.level, self.shuffle = site, seq_len, level, shuffle
        self.model = dict(model)
        self.hook_names = {str(k): v for k, v in (hook_names or {}).items()}
        self.extra = dict(extra or {})
        json.dumps({"model": self.model, "hook_names": self.hook_names, "extra": self.extra})
        self.written = None          # the header as written, after close()
        self.enc = None
        self.chunks, self.n_seq, self.d = [], 0, None
        self.ann = {}
        self.has_mask = None
        L = len(self.layers)
        self.sse = torch.zeros(L, dtype=torch.float64)
        self.ss = torch.zeros(L, dtype=torch.float64)
        self.maxabs = torch.zeros(L, dtype=torch.float64)
        self.act_raw = self.act_stored = self.tok_stored = self.ann_stored = 0
        self.t0 = time.time()
        self.f = open(self.path, "wb")
        self.f.write(FORMAT + struct.pack("<QQ", 0, 0))
        self.pos = HEAD
        self.closed = False

    # -- blobs
    def _blob(self, t):
        t = _as_tensor(t)
        comp, raw = LL.compress(t, self.level, self.shuffle)
        self.f.write(comp)
        rec = {"offset": self.pos, "nbytes": len(comp), "raw_nbytes": raw, "crc32": zlib.crc32(comp),
               "dtype": LL.dtype_name(t.dtype), "shape": list(t.shape)}
        self.pos += len(comp)
        return rec

    def _check_batch(self, acts, tokens, annotations, attention_mask):
        """Everything that can be wrong with a batch, raised before any byte is written."""
        if len(acts) != len(self.layers):
            raise ValueError(f"got {len(acts)} layers, writer has {len(self.layers)}")
        if tokens.dim() != 2 or tokens.is_floating_point():
            raise ValueError(f"tokens must be an integer [B, T] array, got {tuple(tokens.shape)} {tokens.dtype}")
        B, T = tokens.shape
        if B == 0:
            raise ValueError("empty batch")
        d = acts[0].shape[-1]
        for i, a in enumerate(acts):
            if tuple(a.shape) != (B, T, d):
                raise ValueError(f"activation shape {tuple(a.shape)} != tokens {(B, T)} x d={d}")
            if not torch.isfinite(a).all():
                raise ValueError(f"layer {self.layers[i]}: non-finite activations")
        if self.seq_len is not None and T != self.seq_len:
            raise ValueError(f"seq_len {T} != {self.seq_len}")
        if self.d is not None and d != self.d:
            raise ValueError(f"d {d} != {self.d}")
        for i in range(len(self.layers)):
            Q.check_dim(self.spec.quant_for(i), d)
        if self.spec.sink > T:
            raise ValueError(f"sink{self.spec.sink} > T={T}")
        if attention_mask is not None and tuple(attention_mask.shape) != (B, T):
            raise ValueError(f"attention_mask shape {tuple(attention_mask.shape)} != {(B, T)}")
        has_mask = attention_mask is not None
        if self.has_mask is not None and has_mask != self.has_mask:
            raise ValueError("attention_mask must be given for every chunk or for none")
        for name, t in annotations.items():
            if t.dim() == 0 or t.shape[0] != B:
                raise ValueError(f"annotation {name!r}: first dim must be the batch size {B}, got {tuple(t.shape)}")
            LL.check_dtype(t, f"annotation {name!r}")
            meta = {"dtype": LL.dtype_name(t.dtype), "shape_tail": list(t.shape[1:])}
            if self.ann.get(name, meta) != meta:
                raise ValueError(f"annotation {name!r} changed shape or dtype between chunks")
        if self.chunks and set(annotations) != set(self.ann):
            raise ValueError(f"annotations {sorted(annotations)} differ from earlier chunks {sorted(self.ann)}")
        return B, T, d

    def write(self, acts, tokens, annotations=None, attention_mask=None):
        """acts: [L, B, T, d] tensor or list of L tensors [B, T, d] in `layers` order; tokens [B, T] ints.
        annotations: name -> array whose first dim is B (per-sequence labels, e.g. doc ids, probe targets).
        Returns a copy of the chunk record."""
        if self.closed:
            raise ValueError("writer is closed")
        if isinstance(acts, torch.Tensor):
            acts = [acts[i] for i in range(acts.shape[0])]
        acts = [_as_tensor(a) for a in acts]
        tokens = _as_tensor(tokens).cpu()
        annotations = {k: _as_tensor(v).cpu() for k, v in (annotations or {}).items()}
        if attention_mask is not None:
            attention_mask = _as_tensor(attention_mask).cpu()
        B, T, d = self._check_batch(acts, tokens, annotations, attention_mask)
        if self.enc is None:
            if self.calib is None and self.spec.outliers:
                self.calib = calibrate(torch.stack([a.float().cpu() for a in acts]), self.spec, source="first_chunk")
            self.enc = Encoder(self.spec, self.calib, len(self.layers))
        self.enc.reset()
        # encode everything first; only a fully encoded chunk reaches the file and the counters
        encoded, err = [], torch.zeros(len(acts), 3, dtype=torch.float64)
        for i, x in enumerate(acts):
            streams, xhat = self.enc.encode_layer(i, x)
            xf = x.float()
            e = (xf - xhat).double()
            err[i] = torch.tensor([float(e.pow(2).sum()), float(xf.double().pow(2).sum()), float(e.abs().max())])
            encoded.append(streams)
        try:
            chunk = {"seq_start": self.n_seq, "n_seq": B, "tokens": self._blob(tokens.to(torch.int32)), "layers": {}}
            if attention_mask is not None:
                chunk["attention_mask"] = self._blob(attention_mask.to(torch.uint8))
            for i, streams in enumerate(encoded):
                chunk["layers"][str(self.layers[i])] = {k: self._blob(t) for k, t in streams.items()}
            if annotations:
                chunk["annotations"] = {name: self._blob(t) for name, t in annotations.items()}
        except Exception:
            self.abort()
            raise
        self.seq_len, self.d, self.has_mask = T, d, attention_mask is not None
        self.sse += err[:, 0]
        self.ss += err[:, 1]
        self.maxabs = torch.maximum(self.maxabs, err[:, 2])
        self.act_raw += B * T * d * 2 * len(acts)
        self.tok_stored += chunk["tokens"]["nbytes"]
        self.act_stored += sum(r["nbytes"] for recs in chunk["layers"].values() for r in recs.values())
        for name, t in annotations.items():
            self.ann.setdefault(name, {"dtype": LL.dtype_name(t.dtype), "shape_tail": list(t.shape[1:])})
            self.ann_stored += chunk["annotations"][name]["nbytes"]
        self.chunks.append(chunk)
        self.n_seq += B
        return copy.deepcopy(chunk)

    def header(self):
        """The header as it would be written now (close() writes and keeps the final one in `written`)."""
        L = len(self.layers)
        layers = []
        for i, layer in enumerate(self.layers):
            layers.append({
                "index": i, "layer": layer, "site": f"{self.site}.{layer}", "quant": self.spec.quant_for(i),
                "keyframe": self.spec.is_key(i),
                "outlier_channels": self.calib.outliers[i].tolist() if self.spec.outliers and self.calib else None,
                "hook": self.hook_names.get(str(layer)),
                "fidelity": {"rel_l2": float((self.sse[i] / self.ss[i]).sqrt()) if self.ss[i] > 0 else 0.0,
                             "max_abs": float(self.maxabs[i])}})
        total = float(self.ss.sum())
        return {
            "format": FORMAT.decode(), "version": VERSION, "codec": CODEC_VERSION, "library": f"rescomp {__version__}",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": self.model,
            "capture": {"site": self.site, "seq_len": self.seq_len, "n_seq": self.n_seq, "d_model": self.d,
                        "n_layers_stored": L, "n_chunks": len(self.chunks), "hardware": _hardware(),
                        "seconds": round(time.time() - self.t0, 2), "extra": self.extra},
            "spec": self.spec.name,
            "calib": (self.calib or Calib(None, "none")).to_dict(),
            "lossless": {"codec": "zstd", "level": self.level, "shuffle": self.shuffle},
            "layers": layers,
            "fidelity": {"rel_l2": float((self.sse.sum() / total).sqrt()) if total > 0 else 0.0,
                         "max_abs": float(self.maxabs.max())},
            "annotations": self.ann,
            "chunks": self.chunks,
            "bytes": {"activations_raw_bf16": self.act_raw, "activations_stored": self.act_stored,
                      "tokens_stored": self.tok_stored, "annotations_stored": self.ann_stored,
                      "ratio_vs_bf16": (self.act_raw / self.act_stored) if self.act_stored else None},
        }

    def close(self):
        """Write the header and finish the file. Returns the header (also kept as `written`).
        A file with no chunks is refused and left unreadable."""
        if self.closed:
            return self.written
        try:
            if not self.chunks:
                raise ValueError("nothing written: a file needs at least one chunk")
            header = self.header()
            h = json.dumps(header, separators=(",", ":")).encode()
            self.f.write(h)
            self.f.seek(len(FORMAT))
            self.f.write(struct.pack("<QQ", self.pos, len(h)))
        except Exception:
            self.abort()
            raise
        self.f.close()
        self.closed, self.written = True, header
        return header

    def abort(self):
        """Close the file without a header; readers will refuse it."""
        self.f.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()
        else:
            self.abort()


def transcode(src, dst, spec, calib_seqs=32, level=3, shuffle=True, extra=None):
    """Re-encode a file with another spec, chunk by chunk, without the model. Meant for a lossless source
    (fp32, or bf16 of a bf16 model): the new file's fidelity is measured against the source's reconstruction,
    whose own rel L2 is recorded as extra["source_rel_l2"]. Returns the new header."""
    spec = Spec.parse(spec)
    with Reader(src) as r:
        calib = None
        if spec.outliers:
            n = min(calib_seqs, r.n_seq)
            calib = calibrate(r.read_stack(seqs=slice(0, n)), spec, source=f"first {n} sequences of {os.path.basename(src)}")
        ex = dict(r.header["capture"].get("extra") or {})
        ex.update({"transcoded_from": os.path.basename(src), "source_spec": r.spec.name,
                   "source_rel_l2": r.header["fidelity"]["rel_l2"]}, **(extra or {}))
        hooks = {l["layer"]: l["hook"] for l in r.header["layers"] if l.get("hook")}
        with Writer(dst, model=r.header["model"], layers=r.layers, spec=spec, calib=calib, site=r.header["capture"]["site"],
                    seq_len=r.seq_len, level=level, shuffle=shuffle, hook_names=hooks, extra=ex) as w:
            for c in r.chunks:
                seqs = slice(c["seq_start"], c["seq_start"] + c["n_seq"])
                ann = {k: r.annotation(k, seqs) for k in r.annotations} or None
                w.write(r.read_stack(seqs=seqs), r.tokens(seqs), annotations=ann, attention_mask=r.attention_mask(seqs))
    return w.written


class Reader:
    def __init__(self, path, verify=True):
        self.path, self.verify = str(path), verify
        with open(self.path, "rb") as f:
            head = f.read(HEAD)
            if head[: len(FORMAT)] != FORMAT:
                raise ValueError(f"{self.path}: not an RSC1 file")
            off, n = struct.unpack("<QQ", head[len(FORMAT):])
            if off == 0:
                raise ValueError(f"{self.path}: file was not closed (no header)")
            f.seek(off)
            raw = f.read(n)
        if len(raw) != n:
            raise ValueError(f"{self.path}: truncated file (header)")
        self.header = json.loads(raw)
        if self.header.get("version", 0) > VERSION:
            raise ValueError(f"file version {self.header['version']} is newer than this library ({VERSION})")
        if self.header.get("codec", 1) != CODEC_VERSION:
            raise ValueError(f"{self.path}: codec version {self.header.get('codec', 1)} (this library decodes "
                             f"{CODEC_VERSION}; version 1 files were written with the pre-torchao quantizers)")
        self.spec = Spec.parse(self.header["spec"])
        self.calib = Calib.from_dict(self.header["calib"])
        self.layers = [l["layer"] for l in self.header["layers"]]
        cap = self.header["capture"]
        self.n_seq, self.seq_len, self.d = cap["n_seq"], cap["seq_len"], cap["d_model"]
        self.chunks = self.header["chunks"]
        self.starts = [c["seq_start"] for c in self.chunks]
        self.shuffle = self.header["lossless"]["shuffle"]
        self._cache = (None, None)
        self.f = open(self.path, "rb")

    @property
    def n_chunks(self):
        return len(self.chunks)

    @property
    def annotations(self):
        return list(self.header["annotations"])

    @property
    def has_attention_mask(self):
        return "attention_mask" in self.chunks[0]

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- blobs
    def _read_blob(self, rec):
        self.f.seek(rec["offset"])
        b = self.f.read(rec["nbytes"])
        if len(b) != rec["nbytes"]:
            raise ValueError("truncated file")
        if self.verify and zlib.crc32(b) != rec["crc32"]:
            raise ValueError(f"crc mismatch at offset {rec['offset']}")
        return LL.decompress(b, rec["dtype"], rec["shape"], rec["raw_nbytes"], self.shuffle)

    # -- sequences -> chunks
    def _ids(self, seqs):
        """None (all) | int | slice | iterable of ints | boolean mask -> list of sequence indices, order kept."""
        if seqs is None:
            return list(range(self.n_seq))
        if isinstance(seqs, slice):
            seqs = range(*seqs.indices(self.n_seq))
        elif isinstance(seqs, (torch.Tensor, np.ndarray)):
            if seqs.dtype == torch.bool or seqs.dtype == np.bool_:
                if len(seqs) != self.n_seq:
                    raise ValueError(f"boolean mask has {len(seqs)} entries, file has {self.n_seq} sequences")
                seqs = np.flatnonzero(np.asarray(seqs))
            seqs = seqs.tolist()
            seqs = [seqs] if isinstance(seqs, int) else seqs
        else:
            try:
                seqs = [operator.index(seqs)]
            except TypeError:
                pass
        ids = [operator.index(s) for s in seqs]
        for s in ids:
            if not 0 <= s < self.n_seq:
                raise IndexError(f"sequence {s} out of range [0, {self.n_seq})")
        return ids

    def _chunk_of(self, s):
        return bisect.bisect_right(self.starts, s) - 1

    def _gather(self, ids, fetch, empty):
        """fetch(chunk_index) -> tensor [n_seq_in_chunk, ...]; returns rows for ids in order.
        empty: (shape_tail, dtype) of the result when ids is empty."""
        if not ids:
            return torch.empty((0,) + tuple(empty[0]), dtype=empty[1])
        by_chunk = {}
        for p, s in enumerate(ids):
            by_chunk.setdefault(self._chunk_of(s), []).append(p)
        out = None
        for ci, ps in by_chunk.items():
            t = fetch(ci)
            if out is None:
                out = torch.empty((len(ids),) + tuple(t.shape[1:]), dtype=t.dtype)
            start = self.chunks[ci]["seq_start"]
            out[ps] = t[[ids[p] - start for p in ps]]
        return out

    # -- public reads
    def tokens(self, seqs=None):
        return self._gather(self._ids(seqs), lambda ci: self._read_blob(self.chunks[ci]["tokens"]).long(),
                            ([self.seq_len], torch.int64))

    def attention_mask(self, seqs=None):
        if not self.has_attention_mask:
            return None
        return self._gather(self._ids(seqs), lambda ci: self._read_blob(self.chunks[ci]["attention_mask"]),
                            ([self.seq_len], torch.uint8))

    def annotation(self, name, seqs=None):
        if name not in self.header["annotations"]:
            raise KeyError(f"no annotation {name!r}; have {self.annotations}")
        a = self.header["annotations"][name]
        return self._gather(self._ids(seqs), lambda ci: self._read_blob(self.chunks[ci]["annotations"][name]),
                            (a["shape_tail"], LL.DTYPES[a["dtype"]]))

    def streams(self, chunk, layer):
        """Raw (decompressed) streams of one layer of one chunk: name -> tensor."""
        return {k: self._read_blob(r) for k, r in self.chunks[chunk]["layers"][str(layer)].items()}

    def _decode_chunk(self, ci, upto):
        """Decoded layers of chunk ci, with index `upto` filled in (a delta layer needs the earlier layers of its
        keyframe group). The last chunk's decoded layers stay cached: up to L x n_seq_in_chunk x T x d fp32."""
        cached_ci, xs = self._cache
        if cached_ci != ci:
            xs = [None] * len(self.layers)
        if xs[upto] is not None:
            return xs
        dec = Decoder(self.spec, self.calib, len(self.layers))
        first = upto - (upto % self.spec.K) if self.spec.K else upto
        # resume from the latest decoded layer of this keyframe group, if any
        have = [i for i in range(first, upto) if xs[i] is not None]
        if have:
            first = have[-1] + 1
            dec.prev, dec.expect = xs[have[-1]], first
        for i in range(first, upto + 1):
            xs[i] = dec.decode_layer(i, self.streams(ci, self.layers[i]))
        self._cache = (ci, xs)
        return xs

    def _indices(self, layers):
        try:
            return [self.layers.index(l) for l in layers]
        except ValueError:
            raise KeyError(f"layer not stored; have {self.layers}") from None

    def read(self, layer, seqs=None, dtype=torch.float32):
        """Reconstruction of one layer: [n, T, d] in `dtype` (fp32 is exactly the encoder's reconstruction)."""
        idx, = self._indices([layer])
        return self._gather(self._ids(seqs), lambda ci: self._decode_chunk(ci, idx)[idx],
                            ([self.seq_len, self.d], torch.float32)).to(dtype)

    def read_stack(self, layers=None, seqs=None, dtype=torch.float32):
        """[L, n, T, d]: like read() for several layers, decoding each chunk once."""
        idxs = self._indices(layers or self.layers)
        fetch = lambda ci: torch.stack([self._decode_chunk(ci, i)[i] for i in idxs], 1)
        out = self._gather(self._ids(seqs), fetch, ([len(idxs), self.seq_len, self.d], torch.float32))
        return out.transpose(0, 1).contiguous().to(dtype)

    def layer_info(self, layer):
        return self.header["layers"][self._indices([layer])[0]]

    def summary(self):
        h, b, c = self.header, self.header["bytes"], self.header["capture"]
        m = h["model"]
        lines = [f"{os.path.basename(self.path)}: {h['format']} v{h['version']} ({h['library']}, {h['created']})",
                 f"model: {m.get('id')} rev {m.get('revision')} n_layers={m.get('n_layers')} d_model={m.get('d_model')} "
                 f"dtype={m.get('dtype')}",
                 f"capture: {c['n_seq']} seqs x {c['seq_len']} tokens, site {c['site']}, {c['n_chunks']} chunks, "
                 f"layers {self.layers}",
                 f"spec: {h['spec']}   lossless: zstd-{h['lossless']['level']}"
                 f"{' + byte shuffle' if h['lossless']['shuffle'] else ''}",
                 f"bytes: activations {b['activations_stored']:,} stored / {b['activations_raw_bf16']:,} raw bf16 "
                 f"= {b['ratio_vs_bf16']:.2f}x; tokens {b['tokens_stored']:,}; annotations {b['annotations_stored']:,}",
                 f"fidelity: rel L2 {h['fidelity']['rel_l2']:.3e}, max abs err {h['fidelity']['max_abs']:.3g}"]
        for l in h["layers"]:
            lines.append(f"  layer {l['layer']:3d} {l['quant']:10s} {'key' if l['keyframe'] else 'delta':5s} "
                         f"rel L2 {l['fidelity']['rel_l2']:.3e} max abs {l['fidelity']['max_abs']:.3g}"
                         + (f" outliers {l['outlier_channels']}" if l["outlier_channels"] else ""))
        if self.annotations:
            ann = ", ".join("%s %s%s" % (k, v["dtype"], v["shape_tail"]) for k, v in h["annotations"].items())
            lines.append(f"annotations: {ann}")
        return "\n".join(lines)
