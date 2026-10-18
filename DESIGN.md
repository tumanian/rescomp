# Design: the sink

Decisions taken 2026-10-18, built the same day (`rescomp/sink.py`, `rescomp/store.py`, `rescomp/instrument.py`). This replaces the "capture wraps the model" API: the production flow is
*initialize once (hooks), run the model untouched, look at what is stored*. Instrumentation (patching
reconstructions back into a model, KL, coherence) is a separate regime and must not leak into it.

## Production flow

```python
rec = rc.Sink("run1", layers=rc.hf_layers(model, [8, 9, 10, 11]),   # module objects to hook; helpers for HF/TL
                  spec="int4_g384+layerK4+out12",
                  tokens=None)                                             # None: a pre-hook captures input ids
with rec:                                   # registers forward hooks on the given modules
    model(**batch)                          # one forward call = one chunk
    model.generate(**prompt)                # steps are stitched into one chunk per generate call
rec.flush()                                 # pending chunks written; files readable from here on
rec.stats()                                 # chunks, sequences, bytes, ratio, per-layer rel L2 (from memory)
```

- **Hooks attach to plain `nn.Module` objects.** Any module whose output is the residual stream: Hugging Face
  decoder blocks, TransformerLens hook points, custom models. `rc.hf_layers` / `rc.tl_layers` only pick modules.
- **Tokens are optional.** Given explicitly (`rec.tokens(ids)` before a forward, or `tokens=` at init for a fixed
  set) or, when not given, taken by a forward pre-hook on the top-level model (`input_ids` keyword or first
  positional argument; a one-line extractor for custom models). The attention mask is taken the same way.
- **Quantize on the device, compress and write off the hot path** (decided 2026-10-18 after a first version
  that staged raw bf16 on the CPU). Quantization is a map-and-reduce over the activations and belongs on the GPU
  with the forward pass; each layer's hook runs it as the layer is produced and keeps the encoded streams
  (2-4x smaller than raw) on the device. zstd and the file writes happen at `flush()`, which first moves the
  streams to the host (forced by an on-device buffer limit, 1 GiB default, and at close). Budget: under 1% of
  forward latency; `scripts/sink_overhead.py` measures latency and peak GPU memory. Outlier calibration uses
  the first call, per layer, in each layer's hook.
- **File rotation at a size limit**, large (default 4 GB per tensor file): `run1.00000.rsc`, `run1.00001.rsc`, ...
  The index records which file each blob lives in.
- **Generation is handled on read, not on write.** With the KV cache each step yields `[B, 1, d]`. Every chunk
  carries a sequence id and a start position; the reader concatenates the chunks of a sequence. A call with more
  than one token and no cache starts a new sequence, a one-token call with a cache continues it (read off the call's
  arguments by the pre-hook). Each call is encoded on its own (small one-token blobs compress ~10-20% worse than a merged
  sequence would); the reader concatenates. Sequence length is per chunk.

## Storage layout (format version 2)

Tensors and metadata are separated:

| File | Contents |
|---|---|
| `run1.<n>.rsc` | only the compressed tensor blobs (codes, scales, outliers, sink rows), one blob per (chunk, layer, stream), concatenated; a magic and a format version at the front |
| `run1.meta.parquet` | one row: format version, library version, created, model info (JSON), spec, calibration (JSON), lossless settings, site, stored layers with hook names and measured fidelity (JSON) |
| `run1.blobs.parquet` | one row per blob: chunk, sequence id, start position, n_tokens, layer, stream, file number, offset, nbytes, raw_nbytes, crc32, dtype, shape |
| `run1.tokens.parquet` | one row per sequence: seq id, n_tokens, token ids (list<int32>), attention mask (list<uint8>, optional), and every annotation as its own column (scalars as plain columns, per-token arrays as list columns, per-layer routing as list columns) |

Why: the per-sequence table is directly queryable (pandas, DuckDB, Polars) without touching the tensors, tokens
and labels live next to each other, and the tensor file stays a dumb append-only stream that rotation can cut.
`rc.Reader` opens the set by its stem and keeps today's API (`read`, `read_stack`, `tokens`, `annotation`).
Dependency: `pyarrow`.

## Instrumentation regime (separate)

`rescomp.instrument`: a `Patcher` that hooks the same modules and replaces their outputs with reconstructions
(from a file, or live through the encoder), plus the measurements that exist today (file / per-layer / live KL,
perplexity ratio, generation coherence). Uses the same module-object convention as the sink; the
`ModelAdapter` protocol goes away once this exists.

## Bytes and devices

The codec (version 2) is pinned to torchao's numerics. Encoding the same activations on a GPU and on the CPU gives
identical bytes for the plain casts, int8, uabs, mxfp4 and int4 keyframes, and can differ in the last bit for
nvfp4 and for delta layers (reciprocal scales and accumulation order). Every file is self-consistent: decoding on
any device returns exactly the reconstruction the encoder computed on the device that wrote it.

## Status

Built as designed: defaults are 4 GB rotation and 2 GB staging buffer; the offline mode encodes on flush only;
`rc.open()` reads both the format-2 stem and the single-file RSC1 format. Not done: a background encoder thread.
