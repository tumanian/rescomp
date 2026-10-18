# rescomp: a small library for storing compressed residual-stream activations

*One-pager, 2026-10-17. Builds on the actcomp experiments (REPORT.md, HILLCLIMB.md, WIDE_SAE_REPORT.md in the taste_graph worktree).*

## Goal

Take residual-stream activations from a transformer, compress them, and write them to a file that carries
everything an interpretability experiment needs later: which model, which layer, which tokens, and any labels.
Reading back must be exact for lossless codecs and bit-identical to the encoder's own reconstruction for lossy ones.

## What goes in the file

One file = one capture: a model, a set of layers, N sequences of T tokens. Format `RSC1`: a 4-byte magic, the
header's offset and length, the raw blobs, then the JSON header (written last, so the writer can stream). Header fields:

| Group | Contents |
|---|---|
| `model` | HF id, revision, loader, n_layers, d_model, dtype, attention type, tokenizer id |
| `capture` | corpus, seq_len, n_seq, hook site naming (HF module path and TransformerLens name), created, hardware, library version |
| `tokens` | token ids [N, T] (zstd) and the attention mask if any, per chunk |
| `layers` | for each stored layer: index, site, codec spec, calibration (outlier channel ids), measured rel L2 and max abs error at write time |
| `chunks` | the file is chunked along the sequence axis (e.g. 64 sequences per chunk); each chunk holds one blob per (layer group, stream) with offset, raw and stored bytes, and a checksum. You can read one layer of one chunk without touching the rest |
| `annotations` | named arrays aligned to tokens ([N, T]) or sequences ([N]), for probes and labels, each zstd-coded |

## Codecs (modular registry, one spec string per file or per layer)

Every codec is `transform -> quantize -> byte-shuffle + zstd`. The lossless stage is CPU zstd so the library has no
GPU dependency; nvCOMP can be added as a backend later.

| Spec | What it is | When |
|---|---|---|
| `fp32`, `bf16`, `fp16` | plain casts, then zstd (bf16 is lossless for bf16 models, 1.3–1.5×) | the simple baseline you asked for |
| `fp8_e4m3`, `int8_tok` | 8-bit with a per-token scale | ~2.4× at <1% error |
| `int4_g{G}` (+ `int{2..8}_g{G}`) | integer groups of G values, fp16 scale per group | the hill-climb winner's base |
| `+layerK{K}` | closed-loop layer delta, keyframe every K layers | halves the error on middle layers |
| `+out{n}` | n highest-energy channels kept in bf16 | the fidelity lever (error ÷2–4) |
| `+sink{s}` | first s tokens of each sequence kept in bf16 | BOS / attention-sink rows |
| `+key=bf16` | **hybrid**: keyframe layers lossless, only deltas lossy | keeps some frames exact |
| `uabs{step}` | fixed-step uniform quantizer, zigzag codes, zstd | best at the high-fidelity end |
| `mxfp4`, `nvfp4` | 4-bit floats with block scales | tight budgets on Qwen |

Left out on purpose (rejected by the ≥10% simplicity rule last time): Hadamard/PCA rotation, low-rank correction,
top-k sparsity, token deltas, mean centring.

Recommended defaults from the prior runs: GPT-2 `int4_g384+layerK4+out12` (5.0×, ppl +2.4%); Qwen3-8B
`int4_g512+layerK4+out32` (5.2×, ppl +6%); high fidelity `uabs0.1+out8` (2.5×, ppl +0.5%).

## API

```python
import rescomp as rc
cal = rc.calibrate(x_sample, spec)                               # outlier channels from a small sample
with rc.Writer("gpt2_l8-11.rsc", model=info, layers=[8,9,10,11], spec=spec, calib=cal) as w:
    w.write(acts, tokens, annotations={"doc_id": ids})          # acts [L, B, T, d]; call per batch
r = rc.Reader("gpt2_l8-11.rsc")
r.header["model"]; r.tokens(); r.read(layer=9, seqs=slice(0, 16))  # -> [16, T, d] fp32
rc.capture(model_name, layers, n_seq, seq_len, spec, out)      # HF model -> file, in one call
```

CLI: `rescomp info FILE`, `rescomp capture`, `rescomp eval`.

## Tests (pytest, CPU only, run locally)

Bit-exact round trips of every spec through write and read; keyframe/hybrid independence; partial reads by layer and
chunk; error bounds (fp16 relative, uabs ≤ step/2, int4 ≤ scale/2); outlier calibration; metadata and annotation
round trips; corruption detection; a tiny random HF model through capture and the KL evaluator (identity gives 0).

## Evaluation on AWS (the existing g5.xlarge, A10G 24 GB)

Models: GPT-2 small, Qwen3-1.7B, Qwen3-8B (and Gemma-2-2B if time allows). Late third of layers, as before.
For each model and a short ladder of specs (bf16, fp16, int8, int4 winner, hybrid keyframes, uabs):

- **compression ratio** from file bytes
- **chain KL**: all compressed layers replaced by their reconstruction in one forward pass, mean KL vs the clean model
- **perplexity**: teacher-forced perplexity ratio on held-out wikitext
- **coherence**: generate with compression live, then paired perplexity of the generated text under the clean model,
  plus repetition rate. (The blind LLM judge was shown insensitive last time; it is not used.)

Budget: under $10 of GPU time; the box shuts down from inside when done.
