# Design: the sink

Decisions taken 2026-10-18, built the same day; module layout reworked 2026-10-19 (see the end). This replaces the "capture wraps the model" API: the production flow is
*initialize once (hooks), run the model untouched, look at what is stored*. Instrumentation (patching
reconstructions back into a model, KL, coherence) is a separate regime and must not leak into it.

## Production flow

```python
rec = rc.Sink("run1", layers=rc.bind(model, [8, 9, 10, 11]),         # module objects to hook; HF/TL detected
                  spec="int4_g384+layerK4+out12",
                  tokens=None)                                             # None: a pre-hook captures input ids
with rec:                                   # registers forward hooks on the given modules
    model(**batch)                          # one forward call = one chunk
    model.generate(**prompt)                # steps are stitched into one chunk per generate call
rec.flush()                                 # pending chunks written; files readable from here on
rec.stats()                                 # chunks, sequences, bytes, ratio, per-layer rel L2 (from memory)
```

- **Hooks attach to plain `nn.Module` objects.** Any module whose output is the residual stream: Hugging Face
  decoder blocks, TransformerLens hook points, custom models. `rc.bind` only picks modules: a `ModelBinding` per
  framework says which modules carry the residual stream and what the header records; `{layer: module}` for the rest.
- **Tokens are optional.** Given explicitly (`rec.tokens(ids)` before a forward, or `tokens=` at init for a fixed
  set) or, when not given, taken by a forward pre-hook on the top-level model (`input_ids` keyword or first
  positional argument; a one-line extractor for custom models). The attention mask is taken the same way.
- **Quantize on the device, compress and write off the hot path** (decided 2026-10-18 after a first version
  that staged raw bf16 on the CPU). Quantization is a map-and-reduce over the activations and belongs on the GPU
  with the forward pass; each layer's hook runs it as the layer is produced and keeps the encoded streams
  (2-4x smaller than raw) on the device. zstd and the file writes happen at `flush()`, which first moves the
  streams to the host (forced by an on-device buffer limit, 1 GiB default, and at close). Budget: under 1% of
  forward latency; `scripts/sink_overhead.py` measures latency and peak GPU memory. Outlier channels are chosen on
  the first call, per layer, in each layer's hook (`bf16_wrapper.py`).
- **File rotation at a size limit**, large (default 4 GB per tensor file): `run1.00000.rsc`, `run1.00001.rsc`, ...
  The index records which file each blob lives in.
- **Generation is handled on read, not on write.** With the KV cache each step yields `[B, 1, d]`. Every chunk
  carries a sequence id and a start position; the reader concatenates the chunks of a sequence. A call with more
  than one token and no cache starts a new sequence, a one-token call with a cache continues it (read off the call's
  arguments by the pre-hook). Each call is encoded on its own (small one-token blobs compress ~10-20% worse than a merged
  sequence would); the reader concatenates. Sequence length is per chunk.

## Storage layout (format version 3)

Tensors and metadata are separated:

| File | Contents |
|---|---|
| `run1.<n>.rsc` | only the compressed tensor blobs (codes, scales, outliers, sink rows), one blob per (chunk, layer, stream), concatenated; a magic and a format version at the front |
| `run1.meta.parquet` | one row: format version, library version, created, model info (JSON), spec, outlier channels (JSON), lossless settings, site, stored layers with hook names and measured fidelity (JSON) |
| `run1.blobs.parquet` | one row per blob: chunk, sequence id, start position, n_tokens, layer, stream, file number, offset, nbytes, raw_nbytes, crc32, dtype, shape |
| `run1.tokens.parquet` | one row per sequence: seq id, n_tokens, token ids (list<int32>), attention mask (list<uint8>, optional), and every annotation as its own column (scalars as plain columns, per-token arrays as list columns, per-layer routing as list columns) |

Why: the per-sequence table is directly queryable (pandas, DuckDB, Polars) without touching the tensors, tokens
and labels live next to each other, and the tensor file stays a dumb append-only stream that rotation can cut.
`rc.Reader` opens the set by its stem (`read`, `read_padded`, `tokens`, `attention_mask`, `annotation`).
Dependency: `pyarrow`. Format 3 (2026-10-19) is format 2 with codec 3 (torchao-pinned quantizers, `uabs` gone);
the reader names a mismatched format or codec version instead of guessing.

## Instrumentation regime (separate)

`rescomp.instrument`: a `Patcher` that hooks the same modules and replaces their outputs with reconstructions
(from a file, or live through the encoder), plus the measurements that exist today (file / per-layer / live KL,
perplexity ratio, generation coherence). Uses the same module-object convention as the sink, through the same
`CallHooks` (`hooks.py`); it runs over any `run(tokens) -> logits` callable.

## Measured overhead (A10G, 2026-10-18)

`scripts/sink_overhead.py`, median forward time with and without the sink attached, buffer never flushed during
the timed passes, peak GPU memory from the allocator. The sink does the quantizer kernels in the hook and
nothing else; zstd and the writes happen at flush (3.2 s for 13 Qwen3-8B chunks, 0.5 GB).

| Model, batch | regime | forward | with sink | with sink, compiled quantizer | peak GPU memory added |
|---|---|---|---|---|---|
| GPT-2, 16 x 256, layers 8-11 | compact | 35.7 ms | +30% | +24% | +150 MiB |
| Qwen3-8B, 4 x 512, layers 24-35 | compact | 609 ms | +5.8% | +3.5% | +650 MiB (buffered streams, capped by `buffer_bytes`) |
| Qwen3-8B, 4 x 512, layers 24-35 | faithful | 609 ms | +5.6% | +2.4% | +1.25 GiB (same) |
| Qwen3-8B, 4 x 512, layers 24-35 | exact | 609 ms | +2.2% | - | +2.5 GiB (same) |

`fidelity=True` adds about 2 points. The budget is 1%; what remains is per-layer fixed cost (Python in the
hook, the outlier gather/scatter, kernel launches, dynamo guards), about 1.8 ms per hooked layer on Qwen3-8B
and 2 ms on GPT-2, so small models that run a forward in tens of milliseconds pay the most. Next step: profile
one hooked layer with torch.profiler and fold the remaining ops into the compiled function (outlier handling,
the fp32 cast, the delta), or write the quantizer as one Triton kernel.

## Bytes and devices

The codec (version 3) is pinned to torchao's numerics. Encoding the same activations on a GPU and on the CPU gives
identical bytes for the plain casts, int8, mxfp4 and int4 keyframes, and can differ in the last bit for
nvfp4 and for delta layers (reciprocal scales and accumulation order). Every file is self-consistent: decoding on
any device returns exactly the reconstruction the encoder computed on the device that wrote it. The compiled
per-layer function (`compile=True`) gives the same bytes as eager: inductor is told to emulate the fp16 casts of
scales and codes (`emulate_precision_casts`), since by default it keeps fused intermediates in fp32.

## Status

Built as designed: 4 GiB rotation locally, 1 GiB on S3, a 1 GiB on-device buffer; S3 through pyarrow's filesystem
layer (each flush closes the current object; the reader fetches blobs with range requests). The single-file RSC1
reader is gone. Not done: a background encoder thread; the sub-1% overhead work (CUDA graphs for fixed shapes).

## Module layout (2026-10-19)

The library was reworked in six small commits that kept bytes and numbers intact (checked by the golden hashes,
the suite on x86 and GPU, `scripts/spot_check.py` on a real GPT-2 capture and a re-run of the GPT-2 study row):

- `hooks.py`: `Layers` and `CallHooks`, the forward hooks and the token pre-hook, shared by `Sink` and `Patcher`.
- `bindings.py`: `ModelBinding` (`blocks`, `root`, `hook_name`, `info`, `extract_tokens`, `moe_layers`) with
  `HuggingFaceBinding` (`hf.py`), `TransformerLensBinding` (`tl.py`), `ModuleBinding` and the `bind()` factory
  that picks one by the model's type without importing a framework that is not installed.
- `codec.py`: the per-layer tensor work is one function (`layer_values`, compilable), the checks are functions
  (`check_layer_index`, `check_order`, `check_width`), `_CodecState` is what carries over between layers.
- `bf16_wrapper.py`: outlier channels and sink rows (`OutlierChannels`, `choose_channels`, `split`, `restore`),
  with the explanation of why they are kept exactly.
- `quant.py`: `Quantizer` subclasses and `quantizer(name)`; `lossless.py`: `Lossless` subclasses and `lossless()`.
  `uabs` was dropped (not a regime).
- `store.py`: also holds `Location` / `location` (local or S3); `check_layers`, `check_streams`;
  `transcode(sample_seqs=)` picks the outlier channels on the first sequences of the source.
- `scripts/`: the CLI, the wikitext corpus helper, the study, overhead/profile and spot-check scripts.
