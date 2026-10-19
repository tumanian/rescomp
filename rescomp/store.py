"""Storage, format version 3: tensors apart from metadata, on the local disk or in S3.

    <stem>.00000.rsc, <stem>.00001.rsc, ...   append-only files of compressed tensor blobs, rotated at a size limit
    <stem>.meta.parquet                        one row: model, spec, outlier channels, stored layers, fidelity, byte counts
    <stem>.chunks.parquet                      one row per chunk: which sequences it holds and at which positions
    <stem>.blobs.parquet                       one row per blob: chunk, layer, stream, file, byte range, crc, dtype, shape
    <stem>.tokens.parquet                      one row per sequence: token ids, attention mask, annotations as columns

A chunk is one encoded batch [L, B, T, d]: B sequences (ids `seq_ids`), each contributing T tokens starting at
its position `starts[i]`. A sequence may span several chunks (a prompt chunk, then one chunk per generated token);
`Reader.read` concatenates them. Chunks are encoded independently, so any chunk can be decoded alone.

Where a capture lives is a `Location` behind pyarrow's filesystem layer:

    location("runs/run1")                                  -> local files runs/run1.*
    location("s3://bucket/prefix/run1")                    -> objects prefix/run1.* in bucket
    location("run1", bucket="bucket", prefix="prefix")     -> the same
    s3={"region": ..., "endpoint_override": ..., "access_key": ..., ...}   -> pyarrow.fs.S3FileSystem options
      (credentials otherwise come from the usual AWS environment / profile / instance role)

S3 fits the layout: tensor files are immutable once closed (each flush closes the current one, so a capture
is readable mid-run), the Parquet tables are small whole-object rewrites, and blob reads are offset-plus-length
range requests.

`Store` takes already-encoded streams (the Sink owns the encoder); `Reader` decodes; `transcode` re-encodes a
stored capture with another spec without the model.
"""
import json
import os
import struct
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import torch

from . import lossless as LL
from .lossless import lossless
from ._version import __version__
from .bf16_wrapper import OutlierChannels, check_channels, choose_channels
from .codec import Decoder, Encoder
from .quant import CODEC_VERSION
from .spec import Spec

FORMAT = b"RSC3"
FORMAT_VERSION = 3
FILE_HEAD = len(FORMAT) + 4          # each .rsc file starts with the magic and the format version
S3_ROTATE_BYTES = 1 << 30            # objects of at most 1 GiB: a few multipart parts, cheap to re-upload or re-fetch
LOCAL_ROTATE_BYTES = 4 << 30


# ---------------------------------------------------------------- where a capture lives

@dataclass
class Location:
    fs: pafs.FileSystem
    path: str            # the stem's path inside fs (no scheme)
    remote: bool
    uri: str             # how to print it

    @property
    def rotate_default(self):
        return S3_ROTATE_BYTES if self.remote else LOCAL_ROTATE_BYTES

    def exists(self, path):
        return self.fs.get_file_info(path).type != pafs.FileType.NotFound

    def ensure_dir(self):
        if not self.remote:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)


def location(stem, bucket=None, prefix=None, s3=None):
    if isinstance(stem, Location):
        return stem
    stem = str(stem)
    if bucket is None and stem.startswith("s3://"):
        bucket, _, rest = stem[5:].partition("/")
        prefix, _, stem = rest.rpartition("/")
    if bucket is None:
        path = os.path.abspath(stem)
        return Location(pafs.LocalFileSystem(), path, False, path)
    if not stem or "/" in stem.strip("/"):
        raise ValueError(f"the stem on S3 must be a plain name (got {stem!r}); put directories in prefix=")
    opts = dict(s3 or {})
    if "region" not in opts and "endpoint_override" not in opts:
        try:
            opts["region"] = pafs.resolve_s3_region(bucket)
        except Exception:
            pass
    key = "/".join(p for p in [(prefix or "").strip("/"), stem.strip("/")] if p)
    return Location(pafs.S3FileSystem(**opts), f"{bucket}/{key}", True, f"s3://{bucket}/{key}")


def table_path(loc, name):
    return f"{loc.path}.{name}.parquet"


def blob_path(loc, n):
    return f"{loc.path}.{n:05d}.rsc"


def is_capture(stem, bucket=None, prefix=None, s3=None):
    """Whether a capture exists at this stem (its meta table does)."""
    loc = location(stem, bucket, prefix, s3)
    return loc.exists(table_path(loc, "meta"))


# ---------------------------------------------------------------- validators

def check_layers(layers):
    """The stored layers: a non-empty, strictly increasing list of ints."""
    layers = [int(l) for l in layers]
    if not layers or layers != sorted(set(layers)):
        raise ValueError("layers must be a non-empty, strictly increasing list")
    return layers


def check_streams(streams, n_layers):
    """One stream dict per stored layer, each with its codes."""
    if len(streams) != n_layers:
        raise ValueError(f"got {len(streams)} layers, store has {n_layers}")
    for st in streams:
        if "codes" not in st:
            raise ValueError("each layer's streams must include its codes")


# ---------------------------------------------------------------- blob files

class BlobFiles:
    """Append-only tensor files with rotation. append(tensor) -> blob record; the record says which file.
    On a remote filesystem flush() closes the current file (an object is only readable once complete) and the
    next append starts a new one."""

    def __init__(self, loc, rotate_bytes=None, codec=None):
        self.loc, self.codec = loc, codec or lossless()
        self.rotate_bytes = int(rotate_bytes or loc.rotate_default)
        self.n = -1
        self.f = None
        self.pos = 0
        self.files = []

    def _open_new(self):
        self.close()
        self.n += 1
        self.f = self.loc.fs.open_output_stream(blob_path(self.loc, self.n))
        self.f.write(FORMAT + struct.pack("<I", FORMAT_VERSION))
        self.pos = FILE_HEAD
        self.files.append(blob_path(self.loc, self.n))

    def append(self, t):
        t = t.detach() if isinstance(t, torch.Tensor) else torch.as_tensor(np.asarray(t))
        comp, raw = self.codec.compress(t)
        if self.f is None or (self.pos > FILE_HEAD and self.pos + len(comp) > self.rotate_bytes):
            self._open_new()
        self.f.write(comp)
        rec = {"file": self.n, "offset": self.pos, "nbytes": len(comp), "raw_nbytes": raw, "crc32": zlib.crc32(comp),
               "dtype": LL.dtype_name(t.dtype), "shape": list(t.shape)}
        self.pos += len(comp)
        return rec

    def flush(self):
        if self.f is None:
            return
        if self.loc.remote:
            self.close()                                 # finalize the object; the next append opens a new one
        else:
            self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None


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
    """The writer. Give it encoded chunks and per-sequence tokens; call flush() to make the files readable."""

    def __init__(self, stem, *, model, layers, spec=None, channels=None, site="resid_post", level=3, shuffle=True,
                 hook_names=None, extra=None, rotate_bytes=None, bucket=None, prefix=None, s3=None):
        """stem: a local path stem, an s3://bucket/prefix/name URI, or a name with bucket= and prefix=.
        spec may be None at construction (a regime that needs the model width) and set with set_spec()
        before the first chunk. level/shuffle configure the zstd lossless stage (rescomp.lossless);
        rotate_bytes defaults to 4 GiB locally and 1 GiB on S3."""
        self.loc = location(stem, bucket, prefix, s3)
        self.stem = self.loc.uri
        self.spec = None if spec is None else Spec.parse(spec)
        self.layers = check_layers(layers)
        self.channels = channels
        self.chunk_rows, self.blob_rows = [], []
        if self.spec is not None:
            self.set_spec(self.spec)
        self.model, self.site = dict(model), site
        self.codec = lossless(level=level, shuffle=shuffle)
        self.hook_names = {str(k): v for k, v in (hook_names or {}).items()}
        self.extra = dict(extra or {})
        self.regime = None               # the regime name this spec came from, if any
        json.dumps({"model": self.model, "hook_names": self.hook_names, "extra": self.extra})
        self.loc.ensure_dir()
        self.blobs = BlobFiles(self.loc, rotate_bytes, self.codec)
        self.seq = {}                    # seq_id -> {"tokens": list[int] | None, "mask": list[int] | None, "ann": {}}
        self.d = None
        L = len(self.layers)
        self.sse = np.zeros(L)
        self.ss = np.zeros(L)
        self.maxabs = np.zeros(L)
        self.act_raw = self.act_stored = 0
        self.n_measured = 0              # chunks whose reconstruction error was measured
        self.t0 = time.time()
        self.closed = False

    def set_spec(self, spec):
        if self.chunk_rows:
            raise ValueError("spec cannot change after the first chunk")
        self.spec = Spec.parse(spec)
        if not self.spec.outliers:
            self.channels = None
        elif self.channels is not None:
            check_channels(self.channels, self.spec, len(self.layers))

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
        if self.spec is None:
            raise ValueError("set_spec() before the first chunk")
        check_streams(streams, len(self.layers))
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
        if err is not None:
            err = np.asarray(err, dtype=np.float64).reshape(len(self.layers), 3)
            self.sse += err[:, 0]
            self.ss += err[:, 1]
            self.maxabs = np.maximum(self.maxabs, err[:, 2])
            self.n_measured += 1
        return cid

    def note_raw(self, nbytes_bf16, d):
        """Count the raw bf16 size of what a chunk holds (the Sink knows the shapes; streams do not)."""
        self.act_raw += int(nbytes_bf16)
        self.d = d

    # -- tables
    def _meta(self):
        spec = self.spec or Spec("bf16")
        layers = [{"index": i, "layer": l, "site": f"{self.site}.{l}", "quant": spec.quant_for(i),
                   "keyframe": spec.is_key(i),
                   "outlier_channels": self.channels.ids[i].tolist() if self.channels is not None else None,
                   "hook": self.hook_names.get(str(l)),
                   "fidelity": {"rel_l2": float(np.sqrt(self.sse[i] / self.ss[i])) if self.ss[i] > 0 else None,
                                "max_abs": float(self.maxabs[i]) if self.n_measured else None}} for i, l in enumerate(self.layers)]
        total = float(self.ss.sum())
        n_tok = sum(c["n_tokens"] * c["n_seq"] for c in self.chunk_rows)
        return {"format": FORMAT.decode(), "version": FORMAT_VERSION, "codec": CODEC_VERSION, "library": f"rescomp {__version__}",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": json.dumps(self.model),
                "spec": self.spec.name if self.spec else None, "regime": self.regime,
                "outlier_channels": json.dumps((self.channels or OutlierChannels(None, "none")).to_dict()),
                "lossless": json.dumps(self.codec.to_dict()),
                "site": self.site, "d_model": self.d, "layers": json.dumps(layers),
                "fidelity": json.dumps({"rel_l2": float(np.sqrt(self.sse.sum() / total)) if total > 0 else None,
                                        "max_abs": float(self.maxabs.max()) if self.n_measured else None,
                                        "measured_chunks": self.n_measured}),
                "extra": json.dumps(self.extra), "n_sequences": len(self.seq), "n_chunks": len(self.chunk_rows),
                "n_tokens": int(n_tok), "n_files": len(self.blobs.files), "location": self.loc.uri,
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
        fs = self.loc.fs
        pq.write_table(pa.table({k: [v] for k, v in self._meta().items()}), table_path(self.loc, "meta"), filesystem=fs)
        pq.write_table(pa.table({
            "chunk": pa.array([c["chunk"] for c in self.chunk_rows], pa.int64()),
            "seq_ids": pa.array([c["seq_ids"] for c in self.chunk_rows], pa.list_(pa.int64())),
            "starts": pa.array([c["starts"] for c in self.chunk_rows], pa.list_(pa.int32())),
            "n_tokens": pa.array([c["n_tokens"] for c in self.chunk_rows], pa.int32()),
            "n_seq": pa.array([c["n_seq"] for c in self.chunk_rows], pa.int32())}), table_path(self.loc, "chunks"), filesystem=fs)
        b = self.blob_rows
        pq.write_table(pa.table({
            "chunk": pa.array([r["chunk"] for r in b], pa.int64()), "layer": pa.array([r["layer"] for r in b], pa.int32()),
            "stream": pa.array([r["stream"] for r in b], pa.string()), "file": pa.array([r["file"] for r in b], pa.int32()),
            "offset": pa.array([r["offset"] for r in b], pa.int64()), "nbytes": pa.array([r["nbytes"] for r in b], pa.int64()),
            "raw_nbytes": pa.array([r["raw_nbytes"] for r in b], pa.int64()),
            "crc32": pa.array([r["crc32"] for r in b], pa.int64()), "dtype": pa.array([r["dtype"] for r in b], pa.string()),
            "shape": pa.array([r["shape"] for r in b], pa.list_(pa.int32()))}), table_path(self.loc, "blobs"), filesystem=fs)
        pq.write_table(self._tokens_table(), table_path(self.loc, "tokens"), filesystem=fs)

    def stats(self):
        m = self._meta()
        return {"sequences": m["n_sequences"], "chunks": m["n_chunks"], "tokens": m["n_tokens"],
                "bytes_raw_bf16": self.act_raw, "bytes_stored": self.act_stored,
                "ratio": (self.act_raw / self.act_stored) if self.act_stored else None,
                "rel_l2": {l: float(np.sqrt(self.sse[i] / self.ss[i])) if self.ss[i] > 0 else None
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

class Reader:
    def __init__(self, stem, verify=True, bucket=None, prefix=None, s3=None):
        self.loc = location(stem, bucket, prefix, s3)
        self.stem, self.verify = self.loc.uri, verify
        if not is_capture(self.loc):
            raise ValueError(f"{self.stem}: no {os.path.basename(table_path(self.loc, 'meta'))} (a capture is opened by its stem)")
        fs = self.loc.fs
        m = pq.read_table(table_path(self.loc, "meta"), filesystem=fs).to_pylist()[0]
        self.meta = m
        self.header = dict(m, model=json.loads(m["model"]), outlier_channels=json.loads(m["outlier_channels"]), lossless=json.loads(m["lossless"]),
                           layers=json.loads(m["layers"]), fidelity=json.loads(m["fidelity"]), extra=json.loads(m["extra"]),
                           bytes=json.loads(m["bytes"]))
        if m["version"] != FORMAT_VERSION:
            raise ValueError(f"{self.stem}: format version {m['version']}; this library reads {FORMAT_VERSION}")
        if m.get("codec", 1) != CODEC_VERSION:
            raise ValueError(f"{self.stem}: codec version {m.get('codec', 1)}; this library decodes {CODEC_VERSION}")
        self.spec = Spec.parse(m["spec"])
        self.regime = m.get("regime")
        self.channels = OutlierChannels.from_dict(self.header["outlier_channels"])
        self.layers = [l["layer"] for l in self.header["layers"]]
        self.d = m["d_model"]
        self.codec = lossless(self.header["lossless"])
        self.chunks = pq.read_table(table_path(self.loc, "chunks"), filesystem=fs).to_pylist()
        self.blobs = defaultdict(dict)                   # (chunk, layer) -> stream -> record
        for r in pq.read_table(table_path(self.loc, "blobs"), filesystem=fs).to_pylist():
            self.blobs[(r["chunk"], r["layer"])][r["stream"]] = r
        self.tokens_table = pq.read_table(table_path(self.loc, "tokens"), filesystem=fs)
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
            f = self._files[rec["file"]] = self.loc.fs.open_input_file(blob_path(self.loc, rec["file"]))
        b = bytes(f.read_at(rec["nbytes"], rec["offset"]))         # on S3: one range request
        if len(b) != rec["nbytes"]:
            raise ValueError("truncated blob file")
        if self.verify and zlib.crc32(b) != rec["crc32"]:
            raise ValueError(f"crc mismatch in file {rec['file']} at offset {rec['offset']}")
        return self.codec.decompress(b, rec["dtype"], rec["shape"], rec["raw_nbytes"])

    def streams(self, chunk, layer):
        return {k: self._read_blob(r) for k, r in self.blobs[(chunk, layer)].items()}

    def _decode_chunk(self, ci, upto):
        cached_ci, xs = self._cache
        if cached_ci != ci:
            xs = [None] * len(self.layers)
        if xs[upto] is not None:
            return xs
        dec = Decoder(self.spec, self.channels, len(self.layers))
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
                 f"spec: {h['spec']}{' (regime ' + h['regime'] + ')' if h.get('regime') else ''}   lossless: {h['lossless']}",
                 f"bytes: {b['activations_stored']:,} stored / {b['activations_raw_bf16']:,} raw bf16 = {ratio}",
                 ("fidelity: not measured (Sink(fidelity=True) measures it)" if h["fidelity"]["rel_l2"] is None else
                  f"fidelity: rel L2 {h['fidelity']['rel_l2']:.3e}, max abs err {h['fidelity']['max_abs']:.3g}")]
        for l in h["layers"]:
            f = l["fidelity"]["rel_l2"]
            lines.append(f"  layer {l['layer']:3d} {l['quant']:10s} {'key' if l['keyframe'] else 'delta':5s} "
                         + (f"rel L2 {f:.3e}" if f is not None else "rel L2 -")
                         + (f" outliers {l['outlier_channels']}" if l["outlier_channels"] else ""))
        if self.annotations:
            lines.append("annotations: " + ", ".join(self.annotations))
        return "\n".join(lines)


def _sample_channels(r, spec, n_seqs):
    """Outlier channels for `spec` from the stored activations of the chunks holding the first n_seqs sequences
    (chunks are rectangular, so whole chunks are taken)."""
    chunks, seen = [], set()
    for c in r.chunks:
        if len(seen) >= n_seqs:
            break
        chunks.append(c["chunk"])
        seen.update(c["seq_ids"])
    sample = [torch.cat([r._decode_chunk(ci, i)[i].flatten(0, 1) for ci in chunks]) for i in range(len(r.layers))]
    return choose_channels(torch.stack(sample).unsqueeze(1), spec, source=f"first {len(seen)} sequences of {os.path.basename(r.stem)}")


@torch.no_grad()
def transcode(src, dst, spec, sample_seqs=32, level=3, rotate_bytes=None, extra=None, src_loc=None, dst_loc=None):
    """Re-encode a stored capture with another spec, chunk by chunk, without the model. Meant for a lossless
    source (exact / bf16); the source spec and its measured error are recorded in the new header's extra.
    Outlier channels (+out) are chosen on the first `sample_seqs` sequences of the source.
    src_loc / dst_loc: {"bucket": ..., "prefix": ..., "s3": {...}} for either end. Returns the new Store's stats."""
    from .regimes import resolve
    with Reader(src, **(src_loc or {})) as r:
        spec = resolve(spec, r.d)
        channels = _sample_channels(r, spec, sample_seqs) if spec.outliers else None
        ex = dict(r.header["extra"])
        ex.update({"transcoded_from": r.stem, "source_spec": r.spec.name,
                   "source_rel_l2": r.header["fidelity"]["rel_l2"]}, **(extra or {}))
        hooks = {l["layer"]: l["hook"] for l in r.header["layers"] if l.get("hook")}
        with Store(dst, model=r.header["model"], layers=r.layers, spec=spec, channels=channels, site=r.header["site"],
                   level=level, hook_names=hooks, extra=ex, rotate_bytes=rotate_bytes, **(dst_loc or {})) as st:
            for row in r.tokens_table.to_pylist():
                sid = row["seq_id"]
                new = st.new_sequence(row["tokens"], row["mask"], **{k: row[k] for k in r.annotations})
                assert new == sid
            enc = Encoder(spec, channels, len(r.layers))
            for c in r.chunks:
                ci = c["chunk"]
                acts = [r._decode_chunk(ci, i)[i] for i in range(len(r.layers))]
                enc.reset()
                streams, err = [], []
                for i, x in enumerate(acts):
                    strm, xhat = enc.encode_layer(i, x)
                    e = x - xhat
                    err.append((float(e.pow(2).sum()), float(x.pow(2).sum()), float(e.abs().max())))
                    streams.append(strm)
                st.add_chunk(c["seq_ids"], c["starts"], streams, err)
                st.note_raw(acts[0].numel() * 2 * len(acts), r.d)
            return st.stats()
