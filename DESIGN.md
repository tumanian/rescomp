# Design: the recorder (next step, not built yet)

Decisions taken 2026-10-18. This replaces the "capture wraps the model" API: the production flow is
*initialize once (hooks), run the model untouched, look at what is stored*. Instrumentation (patching
reconstructions back into a model, KL, coherence) is a separate regime and must not leak into it.

## Production flow

```python
rec = rc.Recorder("run1", layers=rc.hf_layers(model, [8, 9, 10, 11]),   # module objects to hook; helpers for HF/TL
                  spec="int4_g384+layerK4+out12", encode="cpu",            # or encode="inline"
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
- **Where to encode is the user's choice at initialization.**
  - `encode="inline"`: the closed-loop encoder runs in the hook on the model's device; lowest memory, adds
    latency to the model.
  - `encode="cpu"` (offline): hooks copy raw bf16 activations to a staging buffer and return; encoding happens
    at `flush()`, so the model runs at full speed and encoding never competes with the forward pass. Outlier
    calibration uses the first `calib_seqs` buffered sequences. A byte limit on the buffer forces a flush.
- **File rotation at a size limit**, large (default 4 GB per tensor file): `run1.00000.rsc`, `run1.00001.rsc`, ...
  The index records which file each blob lives in.
- **Generation**: with the KV cache each step yields `[B, 1, d]`; the recorder stitches steps along the sequence
  axis until the generate call returns, and stitches tokens the same way. Sequence length is per chunk.

## Storage layout (format version 2)

Tensors and metadata are separated:

| File | Contents |
|---|---|
| `run1.<n>.rsc` | only the compressed tensor blobs (codes, scales, outliers, sink rows), one blob per (chunk, layer, stream), concatenated; a magic and a format version at the front |
| `run1.meta.parquet` | one row: format version, library version, created, model info (JSON), spec, calibration (JSON), lossless settings, site, stored layers with hook names and measured fidelity (JSON) |
| `run1.blobs.parquet` | one row per blob: chunk, layer, stream, file number, offset, nbytes, raw_nbytes, crc32, dtype, shape |
| `run1.tokens.parquet` | one row per sequence: seq id, chunk, position in chunk, seq_len, token ids (list<int32>), attention mask (list<uint8>, optional), and every annotation as its own column (scalars as plain columns, per-token arrays as list columns, per-layer routing as list columns) |

Why: the per-sequence table is directly queryable (pandas, DuckDB, Polars) without touching the tensors, tokens
and labels live next to each other, and the tensor file stays a dumb append-only stream that rotation can cut.
`rc.Reader` opens the set by its stem and keeps today's API (`read`, `read_stack`, `tokens`, `annotation`).
Dependency: `pyarrow`.

## Instrumentation regime (separate)

`rescomp.instrument`: a `Patcher` that hooks the same modules and replaces their outputs with reconstructions
(from a file, or live through the encoder), plus the measurements that exist today (file / per-layer / live KL,
perplexity ratio, generation coherence). Uses the same module-object convention as the recorder; the
`ModelAdapter` protocol goes away once this exists.

## Open points

- Generation stitching in the first version, or forward-only first.
- Default rotation size (4 GB proposed) and staging-buffer limit (2 GB proposed).
- Whether a background encoder thread is worth having for the offline mode, or encode-on-flush only (proposed:
  flush only, as decided).
- Keep RSC1 readable (one-file format) or migrate: proposed to keep `Reader` able to open both.
