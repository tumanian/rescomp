"""Storage, format version 2: tensors apart from metadata.

    <stem>.00000.rsc, <stem>.00001.rsc, ...   append-only files of compressed tensor blobs, rotated at a size limit
    <stem>.meta.parquet                        one row: model, spec, calibration, stored layers, fidelity, byte counts
    <stem>.chunks.parquet                      one row per chunk: which sequences it holds and at which positions
    <stem>.blobs.parquet                       one row per blob: chunk, layer, stream, file, byte range, crc, dtype, shape
    <stem>.tokens.parquet                      one row per sequence: token ids, attention mask, annotations as columns

A chunk is one encoded batch [L, B, T, d]: B sequences (ids `seq_ids`), each contributing T tokens starting at
its position `starts[i]`. A sequence may span several chunks (a prompt chunk, then one chunk per generated token);
`Reader2.read` concatenates them. Chunks are encoded independently, so any chunk can be decoded alone.

`Store` takes already-encoded streams (the Sink owns the encoder); `Reader2` decodes. `rescomp.open(stem)` picks
this reader or the single-file RSC1 one.
"""
import json
import os
import struct
import time
import zlib
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from . import lossless as LL
from ._version import __version__
from .codec import Calib, Decoder
from .spec import Spec

MAGIC2 = b"RSC2"
VERSION2 = 2
HEAD2 = len(MAGIC2) + 4
TABLES = ("meta", "chunks", "blobs", "tokens")


def table_path(stem, name):
    return f"{stem}.{name}.parquet"


def blob_path(stem, n):
    return f"{stem}.{n:05d}.rsc"


def is_v2(stem):
    return os.path.exists(table_path(stem, "meta"))


# ---------------------------------------------------------------- blob files

class BlobFiles:
    """Append-only tensor files with rotation. append(tensor) -> blob record; the record says which file."""

    def __init__(self, stem, rotate_bytes=4 << 30, level=3, shuffle=True, start_file=0, start_offset=None):
        self.stem, self.rotate_bytes, self.level, self.shuffle = str(stem), int(rotate_bytes), level, shuffle
        self.n = start_file
        self.f = None
        self.pos = 0
        if start_offset is not None:                     # reopen an existing last file for appending
            self.f = open(blob_path(self.stem, self.n), "r+b")
            self.f.seek(start_offset)
            self.pos = start_offset

    def _open_new(self):
        if self.f:
            self.f.close()
            self.n += 1
        self.f = open(blob_path(self.stem, self.n), "wb")
        self.f.write(MAGIC2 + struct.pack("<I", VERSION2))
        self.pos = HEAD2

    def append(self, t):
        t = t.detach() if isinstance(t, torch.Tensor) else torch.as_tensor(np.asarray(t))
        comp, raw = LL.compress(t, self.level, self.shuffle)
        if self.f is None or (self.pos > HEAD2 and self.pos + len(comp) > self.rotate_bytes):
            self._open_new()
        self.f.write(comp)
        rec = {"file": self.n, "offset": self.pos, "nbytes": len(comp), "raw_nbytes": raw, "crc32": zlib.crc32(comp),
               "dtype": LL.dtype_name(t.dtype), "shape": list(t.shape)}
        self.pos += len(comp)
        return rec

    def flush(self):
        if self.f:
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()
            self.f = None

    @property
    def files(self):
        return [blob_path(self.stem, i) for i in range(self.n + 1) if os.path.exists(blob_path(self.stem, i))]


# ---------------------------------------------------------------- annotations <-> arrow

def _to_arrow_column(values):
    """A list of per-sequence values (scalars or arrays) -> arrow array (nested lists for arrays)."""
    out = []
    for v in values:
        if v is None:
            out.append(None)
        elif isinstance(v, torch.Tensor):
            out.append(v.detach().cpu().tolist())
        elif isinstance(v, np.ndarray):
            out.append(v.tolist())
        elif isinstance(v, (np.generic,)):
            out.append(v.item())
        else:
            out.append(v)
    return pa.array(out)


def _from_arrow(value):
    if value is None:
        return None
    if isinstance(value, list):
        return torch.tensor(value)
    return value


# ---------------------------------------------------------------- writer

class Store:
    """Format-2 writer. Give it encoded chunks and per-sequence tokens; call flush() to make the files readable."""

    def __init__(self, stem, *, model, layers, spec, calib=None, site="resid_post", level=3, shuffle=True,
                 hook_names=None, extra=None, rotate_bytes=4 << 30):
        self.stem = str(stem)
        self.spec = Spec.parse(spec)
        self.layers = [int(l) for l in layers]
        if not self.layers or self.layers != sorted(set(self.layers)):
            raise ValueError("layers must be a non-empty, strictly increasing list")
        self.calib = calib if self.spec.outliers else None
        if self.calib is not None:
            self.calib.check(self.spec, len(self.layers))
        self.model, self.site, self.level, self.shuffle = dict(model), site, level, shuffle
        self.hook_names = {str(k): v for k, v in (hook_names or {}).items()}
        self.extra = dict(extra or {})
        json.dumps({"model": self.model, "hook_names": self.hook_names, "extra": self.extra})
        os.makedirs(os.path.dirname(os.path.abspath(self.stem)), exist_ok=True)
        self.blobs = BlobFiles(self.stem, rotate_bytes, level, shuffle)
        self.chunk_rows, self.blob_rows = [], []
        self.seq = {}                    # seq_id -> {"tokens": list[int] | None, "mask": list[int] | None, "ann": {}}
        self.d = None
        L = len(self.layers)
        self.sse = np.zeros(L)
        self.ss = np.zeros(L)
        self.maxabs = np.zeros(L)
        self.act_raw = self.act_stored = 0
        self.t0 = time.time()
        self.closed = False

    # -- sequences
    def new_sequence(self, tokens=None, mask=None, **annotations):
        """Register a sequence; returns its id. tokens/mask may be extended later (generation)."""
        sid = len(self.seq)
        self.seq[sid] = {"tokens": None if tokens is None else [int(t) for t in tokens],
                         "mask": None if mask is None else [int(m) for m in mask], "ann": dict(annotations)}
        return sid

    def extend_sequence(self, seq_id, tokens=None, mask=None):
        s = self.seq[seq_id]
        if tokens is not None:
            s["tokens"] = (s["tokens"] or []) + [int(t) for t in tokens]
        if mask is not None:
            s["mask"] = (s["mask"] or []) + [int(m) for m in mask]

    def annotate(self, seq_id, **annotations):
        self.seq[seq_id]["ann"].update(annotations)

    # -- chunks
    def add_chunk(self, seq_ids, starts, streams, err):
        """streams: per stored layer, name -> encoded tensor (from Encoder.encode_layer); err: per layer
        (sum of squared error, sum of squares, max abs error) for the fidelity record. Returns the chunk id."""
        if self.closed:
            raise ValueError("store is closed")
        if len(streams) != len(self.layers):
            raise ValueError(f"got {len(streams)} layers, store has {len(self.layers)}")
        T = int(streams[0]["codes"].shape[-2] if streams[0]["codes"].dim() >= 3 else 0)
        cid = len(self.chunk_rows)
        for i, st in enumerate(streams):
            for name, t in st.items():
                rec = self.blobs.append(t)
                self.blob_rows.append({"chunk": cid, "layer": self.layers[i], "stream": name, **rec})
                self.act_stored += rec["nbytes"]
        B = len(seq_ids)
        self.chunk_rows.append({"chunk": cid, "seq_ids": [int(s) for s in seq_ids], "starts": [int(s) for s in starts],
                                "n_tokens": T, "n_seq": B})
        err = np.asarray(err, dtype=np.float64).reshape(len(self.layers), 3)
        self.sse += err[:, 0]
        self.ss += err[:, 1]
        self.maxabs = np.maximum(self.maxabs, err[:, 2])
        return cid

    def note_raw(self, nbytes_bf16, d):
        """Count the raw bf16 size of what a chunk holds (the Sink knows the shapes; streams do not)."""
        self.act_raw += int(nbytes_bf16)
        self.d = d

    # -- tables
    def _meta(self):
        layers = [{"index": i, "layer": l, "site": f"{self.site}.{l}", "quant": self.spec.quant_for(i),
                   "keyframe": self.spec.is_key(i),
                   "outlier_channels": self.calib.outliers[i].tolist() if self.calib is not None else None,
                   "hook": self.hook_names.get(str(l)),
                   "fidelity": {"rel_l2": float(np.sqrt(self.sse[i] / self.ss[i])) if self.ss[i] > 0 else 0.0,
                                "max_abs": float(self.maxabs[i])}} for i, l in enumerate(self.layers)]
        total = float(self.ss.sum())
        n_tok = sum(c["n_tokens"] * c["n_seq"] for c in self.chunk_rows)
        return {"format": MAGIC2.decode(), "version": VERSION2, "library": f"rescomp {__version__}",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": json.dumps(self.model), "spec": self.spec.name,
                "calib": json.dumps((self.calib or Calib(None, "none")).to_dict()),
                "lossless": json.dumps({"codec": "zstd", "level": self.level, "shuffle": self.shuffle}),
                "site": self.site, "d_model": self.d, "layers": json.dumps(layers),
                "fidelity": json.dumps({"rel_l2": float(np.sqrt(self.sse.sum() / total)) if total > 0 else 0.0,
                                        "max_abs": float(self.maxabs.max())}),
                "extra": json.dumps(self.extra), "n_sequences": len(self.seq), "n_chunks": len(self.chunk_rows),
                "n_tokens": int(n_tok), "n_files": len(self.blobs.files),
                "bytes": json.dumps({"activations_raw_bf16": self.act_raw, "activations_stored": self.act_stored,
                                     "ratio_vs_bf16": (self.act_raw / self.act_stored) if self.act_stored else None}),
                "seconds": round(time.time() - self.t0, 2)}

    def _tokens_table(self):
        ids = sorted(self.seq)
        cols = {"seq_id": pa.array(ids, pa.int64()),
                "n_tokens": pa.array([len(self.seq[s]["tokens"]) if self.seq[s]["tokens"] is not None else
                                      self._n_tokens_from_chunks(s) for s in ids], pa.int32()),
                "tokens": pa.array([self.seq[s]["tokens"] for s in ids], pa.list_(pa.int32())),
                "mask": pa.array([self.seq[s]["mask"] for s in ids], pa.list_(pa.uint8()))}
        names = sorted({k for s in ids for k in self.seq[s]["ann"]})
        for name in names:
            cols[name] = _to_arrow_column([self.seq[s]["ann"].get(name) for s in ids])
        return pa.table(cols)

    def _n_tokens_from_chunks(self, sid):
        return sum(c["n_tokens"] for c in self.chunk_rows if sid in c["seq_ids"])

    def flush(self):
        """Write the four tables (whole, every time) so the files are readable now."""
        self.blobs.flush()
        pq.write_table(pa.table({k: [v] for k, v in self._meta().items()}), table_path(self.stem, "meta"))
        pq.write_table(pa.table({
            "chunk": pa.array([c["chunk"] for c in self.chunk_rows], pa.int64()),
            "seq_ids": pa.array([c["seq_ids"] for c in self.chunk_rows], pa.list_(pa.int64())),
            "starts": pa.array([c["starts"] for c in self.chunk_rows], pa.list_(pa.int32())),
            "n_tokens": pa.array([c["n_tokens"] for c in self.chunk_rows], pa.int32()),
            "n_seq": pa.array([c["n_seq"] for c in self.chunk_rows], pa.int32())}), table_path(self.stem, "chunks"))
        b = self.blob_rows
        pq.write_table(pa.table({
            "chunk": pa.array([r["chunk"] for r in b], pa.int64()), "layer": pa.array([r["layer"] for r in b], pa.int32()),
            "stream": pa.array([r["stream"] for r in b], pa.string()), "file": pa.array([r["file"] for r in b], pa.int32()),
            "offset": pa.array([r["offset"] for r in b], pa.int64()), "nbytes": pa.array([r["nbytes"] for r in b], pa.int64()),
            "raw_nbytes": pa.array([r["raw_nbytes"] for r in b], pa.int64()),
            "crc32": pa.array([r["crc32"] for r in b], pa.int64()), "dtype": pa.array([r["dtype"] for r in b], pa.string()),
            "shape": pa.array([r["shape"] for r in b], pa.list_(pa.int32()))}), table_path(self.stem, "blobs"))
        pq.write_table(self._tokens_table(), table_path(self.stem, "tokens"))

    def stats(self):
        m = self._meta()
        return {"sequences": m["n_sequences"], "chunks": m["n_chunks"], "tokens": m["n_tokens"],
                "bytes_raw_bf16": self.act_raw, "bytes_stored": self.act_stored,
                "ratio": (self.act_raw / self.act_stored) if self.act_stored else None,
                "rel_l2": {l: float(np.sqrt(self.sse[i] / self.ss[i])) if self.ss[i] > 0 else 0.0
                           for i, l in enumerate(self.layers)}, "files": self.blobs.files}

    def close(self):
        if self.closed:
            return
        self.flush()
        self.blobs.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------- reader

class Reader2:
    def __init__(self, stem, verify=True):
        self.stem, self.verify = str(stem), verify
        if not is_v2(self.stem):
            raise ValueError(f"{self.stem}: no {table_path(self.stem, 'meta')}")
        m = pq.read_table(table_path(self.stem, "meta")).to_pylist()[0]
        self.meta = m
        self.header = dict(m, model=json.loads(m["model"]), calib=json.loads(m["calib"]), lossless=json.loads(m["lossless"]),
                           layers=json.loads(m["layers"]), fidelity=json.loads(m["fidelity"]), extra=json.loads(m["extra"]),
                           bytes=json.loads(m["bytes"]))
        if m["version"] > VERSION2:
            raise ValueError(f"file version {m['version']} is newer than this library ({VERSION2})")
        self.spec = Spec.parse(m["spec"])
        self.calib = Calib.from_dict(self.header["calib"])
        self.layers = [l["layer"] for l in self.header["layers"]]
        self.d = m["d_model"]
        self.shuffle = self.header["lossless"]["shuffle"]
        self.chunks = pq.read_table(table_path(self.stem, "chunks")).to_pylist()
        self.blobs = defaultdict(dict)                   # (chunk, layer) -> stream -> record
        for r in pq.read_table(table_path(self.stem, "blobs")).to_pylist():
            self.blobs[(r["chunk"], r["layer"])][r["stream"]] = r
        self.tokens_table = pq.read_table(table_path(self.stem, "tokens"))
        self._seq_rows = {r["seq_id"]: r for r in self.tokens_table.to_pylist()}
        self.n_seq = len(self._seq_rows)
        self.by_seq = defaultdict(list)                  # seq_id -> [(start, chunk, row)]
        for c in self.chunks:
            for row, (sid, st) in enumerate(zip(c["seq_ids"], c["starts"])):
                self.by_seq[sid].append((st, c["chunk"], row))
        for v in self.by_seq.values():
            v.sort()
        self._files = {}
        self._cache = (None, None)

    @property
    def annotations(self):
        return [n for n in self.tokens_table.column_names if n not in ("seq_id", "n_tokens", "tokens", "mask")]

    def sequences(self):
        """The per-sequence table (pyarrow.Table): seq_id, n_tokens, tokens, mask, annotations."""
        return self.tokens_table

    def close(self):
        for f in self._files.values():
            f.close()
        self._files = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- blobs
    def _read_blob(self, rec):
        f = self._files.get(rec["file"])
        if f is None:
            f = self._files[rec["file"]] = open(blob_path(self.stem, rec["file"]), "rb")
        f.seek(rec["offset"])
        b = f.read(rec["nbytes"])
        if len(b) != rec["nbytes"]:
            raise ValueError("truncated blob file")
        if self.verify and zlib.crc32(b) != rec["crc32"]:
            raise ValueError(f"crc mismatch in file {rec['file']} at offset {rec['offset']}")
        return LL.decompress(b, rec["dtype"], rec["shape"], rec["raw_nbytes"], self.shuffle)

    def streams(self, chunk, layer):
        return {k: self._read_blob(r) for k, r in self.blobs[(chunk, layer)].items()}

    def _decode_chunk(self, ci, upto):
        cached_ci, xs = self._cache
        if cached_ci != ci:
            xs = [None] * len(self.layers)
        if xs[upto] is not None:
            return xs
        dec = Decoder(self.spec, self.calib, len(self.layers))
        first = upto - (upto % self.spec.K) if self.spec.K else upto
        have = [i for i in range(first, upto) if xs[i] is not None]
        if have:
            first = have[-1] + 1
            dec.prev, dec.expect = xs[have[-1]], first
        for i in range(first, upto + 1):
            xs[i] = dec.decode_layer(i, self.streams(ci, self.layers[i]))
        self._cache = (ci, xs)
        return xs

    # -- public reads
    def _ids(self, seqs):
        if seqs is None:
            return sorted(self._seq_rows)
        if isinstance(seqs, int):
            return [seqs]
        if isinstance(seqs, slice):
            return sorted(self._seq_rows)[seqs]
        ids = [int(s) for s in (seqs.tolist() if hasattr(seqs, "tolist") else seqs)]
        for s in ids:
            if s not in self._seq_rows:
                raise IndexError(f"no sequence {s}")
        return ids

    def read(self, layer, seqs=None, dtype=torch.float32):
        """One layer's reconstruction for each sequence, whole: a list of [T_i, d] tensors (an int `seqs` gives
        one tensor). Chunks of a sequence are concatenated along the token axis."""
        if layer not in self.layers:
            raise KeyError(f"layer {layer} not stored; have {self.layers}")
        idx = self.layers.index(layer)
        out = []
        for sid in self._ids(seqs):
            parts = [self._decode_chunk(ci, idx)[idx][row] for _, ci, row in self.by_seq[sid]]
            out.append(torch.cat(parts, 0).to(dtype) if parts else torch.empty(0, self.d, dtype=dtype))
        return out[0] if isinstance(seqs, int) else out

    def read_padded(self, layer, seqs=None, dtype=torch.float32):
        """[n, T_max, d] right-padded with zeros, plus a [n, T_max] mask of real tokens."""
        parts = self.read(layer, None if seqs is None else list(self._ids(seqs)), dtype)
        T = max((p.shape[0] for p in parts), default=0)
        out = torch.zeros(len(parts), T, self.d, dtype=dtype)
        mask = torch.zeros(len(parts), T, dtype=torch.bool)
        for i, p in enumerate(parts):
            out[i, : p.shape[0]] = p
            mask[i, : p.shape[0]] = True
        return out, mask

    def tokens(self, seqs=None):
        """Token ids per sequence (list of LongTensors, or None where no tokens were recorded)."""
        rows = [self._seq_rows[s] for s in self._ids(seqs)]
        out = [None if r["tokens"] is None else torch.tensor(r["tokens"], dtype=torch.long) for r in rows]
        return out[0] if isinstance(seqs, int) else out

    def attention_mask(self, seqs=None):
        rows = [self._seq_rows[s] for s in self._ids(seqs)]
        out = [None if r["mask"] is None else torch.tensor(r["mask"], dtype=torch.uint8) for r in rows]
        return out[0] if isinstance(seqs, int) else out

    def annotation(self, name, seqs=None):
        if name not in self.annotations:
            raise KeyError(f"no annotation {name!r}; have {self.annotations}")
        out = [_from_arrow(self._seq_rows[s][name]) for s in self._ids(seqs)]
        return out[0] if isinstance(seqs, int) else out

    def summary(self):
        h, b = self.header, self.header["bytes"]
        m = h["model"]
        ratio = f"{b['ratio_vs_bf16']:.2f}x" if b["ratio_vs_bf16"] else "-"
        lines = [f"{os.path.basename(self.stem)}: {h['format']} v{h['version']} ({h['library']}, {h['created']})",
                 f"model: {m.get('id')} rev {m.get('revision')} n_layers={m.get('n_layers')} d_model={m.get('d_model')}",
                 f"stored: {h['n_sequences']} sequences, {h['n_tokens']} tokens, {h['n_chunks']} chunks in {h['n_files']} "
                 f"blob file(s); layers {self.layers}; site {h['site']}",
                 f"spec: {h['spec']}   lossless: zstd-{h['lossless']['level']}{' + byte shuffle' if self.shuffle else ''}",
                 f"bytes: {b['activations_stored']:,} stored / {b['activations_raw_bf16']:,} raw bf16 = {ratio}",
                 f"fidelity: rel L2 {h['fidelity']['rel_l2']:.3e}, max abs err {h['fidelity']['max_abs']:.3g}"]
        for l in h["layers"]:
            lines.append(f"  layer {l['layer']:3d} {l['quant']:10s} {'key' if l['keyframe'] else 'delta':5s} "
                         f"rel L2 {l['fidelity']['rel_l2']:.3e}" + (f" outliers {l['outlier_channels']}" if l["outlier_channels"] else ""))
        if self.annotations:
            lines.append("annotations: " + ", ".join(self.annotations))
        return "\n".join(lines)
