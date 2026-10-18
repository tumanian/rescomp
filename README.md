# rescomp

Compress the residual-stream activations of a running transformer and keep them, with the tokens, labels and
model metadata that interpretability work needs later. Pure PyTorch + zstd + Parquet, CPU or GPU, any framework
whose layers are `nn.Module`s. The model is not wrapped: you add a sink, run the model as you always do, and read
the files afterwards.

- `DESIGN.md`: the design decisions behind the sink and the storage layout.
- `RESULTS.md`: what each codec costs, measured on eight models (GPT-2 to Qwen3-30B-A3B, dense and MoE).
- `PLAN.md`: the original one-page plan.

## Install

```bash
pip install -e .            # torch>=2.8, numpy, zstandard, pyarrow, torchao
pip install -e '.[hf]'      # + transformers, datasets
pytest                      # ~210 tests, CPU only, under a minute
```

## Use

```python
import torch, rescomp as rc
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16).cuda().eval()
batch = tok(["The history of the Roman Empire", "To make a tomato sauce, first"], return_tensors="pt", padding=True).to("cuda")

sink = rc.Sink("runs/demo", layers=rc.hf_layers(model, [24, 28, 32, 35]),
               spec="int4_g512+layerK4+out32+key=bf16", encode="cpu")     # tokens come from a pre-hook
with sink, torch.no_grad():
    logits = model(**batch).logits                 # one forward call = one chunk
    out = model.generate(**batch, max_new_tokens=80)   # prompt + steps = one sequence per prompt
print(sink.stats())                                # chunks, sequences, bytes, ratio, per-layer rel L2

r = rc.open("runs/demo")
r.sequences()                                      # Parquet table: seq_id, n_tokens, tokens, mask, annotations
x = r.read(32, seqs=[2, 3])                        # each sequence whole, [T_i, d] fp32 (the encoder's own reconstruction)
r.tokens(2); r.header["model"]; r.header["layers"][1]["fidelity"]
```

- `layers`: `rc.hf_layers(model, [...])` for Hugging Face, `rc.tl_layers(model, [...])` for TransformerLens, or
  any `{layer: nn.Module}` whose outputs are the residual stream (then pass `rc.Layers(modules=..., root=model)`
  if you want the token pre-hook).
- `encode="cpu"` (default): hooks copy raw bf16 to a staging buffer and return; encoding runs at `sink.flush()`,
  when the buffer passes its limit, and at close. `encode="inline"`: the codec runs in the hook on the model's device.
- Tokens: taken from each call by a pre-hook on the root model (`input_ids`, `attention_mask`, and the KV-cache
  signal that tells a generation step from a new prompt); `sink.tokens(ids)` overrides, `tokens="off"` disables.
- `sink.annotate(doc_id=[...], split="train")` labels the sequences the next new-sequence call starts; labels become
  Parquet columns.
- Outlier channels (`+out`) are calibrated on the first chunk unless `calib=` is given.

### Storage (format 2)

`<stem>.00000.rsc, ...` hold only the compressed tensor blobs (append-only, rotated at `rotate_bytes`, 4 GB by
default). `<stem>.meta.parquet` has the model, spec, calibration, stored layers with hook names and measured
fidelity, and byte counts; `<stem>.chunks.parquet` which sequences each chunk holds and where; `<stem>.blobs.parquet`
one row per blob (byte range, crc, dtype, shape); `<stem>.tokens.parquet` one row per sequence (tokens, mask,
annotations). `rc.open()` also still reads the single-file RSC1 format from the earlier study.

### Lower level

`rc.encode(X, spec, calib)` / `rc.decode(streams, spec, calib)` on an `[L, B, T, d]` stack; `rc.Encoder(...)
.encode_layer(i, x)` one layer at a time (closed-loop deltas); `rc.Store` to write encoded chunks yourself;
`rc.transcode(src, dst, spec)` to re-encode an RSC1 file.

## Specs

`<quant>[+layerK<K>][+out<n>][+sink<s>][+key=<quant>]`

| Part | Meaning |
|---|---|
| `fp32`, `bf16`, `fp16` | plain casts (bf16 is lossless for a bf16 model; fp16 too, at 1.5×), then byte-shuffle + zstd |
| `fp8_e4m3`, `int8_tok` | 8-bit codes with one fp16 scale per token |
| `int4_g<G>`, `int2_g<G>`, `int8_g<G>` | torchao affine codes (symmetric, no clipping, full range) with one fp16 scale per group of G channels (G must divide d) |
| `uabs<step>` | fixed step in activation units; zigzag integer codes in byte planes, so zstd does the entropy coding |
| `mxfp4`, `nvfp4` | torchao's OCP MX / NVIDIA FP4: e2m1 codes with an e8m0 scale per 32 or an e4m3 scale per 16 channels |
| `+layerK<K>` | code stored layer i as `x_i - xhat_{i-1}` except every K-th layer (closed loop, so errors do not accumulate) |
| `+out<n>` | keep the n highest-energy channels of each layer in bf16 (calibrated; stored in the header) |
| `+sink<s>` | keep the first s tokens of every sequence in bf16 |
| `+key=<quant>` | hybrid: keyframe layers use this quantizer (`key=bf16` keeps every K-th layer exact) |

Measured on eight models (see `RESULTS.md`): `fp16` 1.5× lossless; `int8_tok+out8` 2.5× at ≤ 0.7% perplexity;
`uabs(0.1·rms)+out8` 3.0× at ≤ 0.6%; `int4_g(d/8)+layerK4+out(d/128)+key=bf16` 2.4–3.0× with a third of the
layers exact; `int4_g(d/8)+layerK4+out(d/128)` 5.0–5.2× at +4–7% (more on small models); never int2.

## Instrumentation (separate from the sink)

`rescomp.instrument` feeds reconstructions back into a running model and measures the change, with the same
module-object convention and any `run(tokens) -> logits` callable:

```python
from rescomp import instrument as I
L = rc.hf_layers(model, [24, 28, 32, 35])
run = lambda t: model(t, use_cache=False).logits
I.kl_ppl(run, tokens, I.Patcher(L, I.LiveCodec("int4_g512+layerK4+out32")))   # codec inside the forward pass
I.kl_ppl(run, tokens, I.Patcher(L, rc.open("runs/demo")), seq_ids=range(64))  # stored reconstructions
I.per_layer(run, tokens, L, I.LiveCodec("int8_tok+out8"))                       # one layer at a time
I.coherence(run, prompts, I.Patcher(L, I.LiveCodec(spec)), tokenizer=tok)       # generation, paired perplexity
```

`rescomp.evaluate` and the `ModelAdapter` protocol (`rescomp.model`, `rescomp.hf.HFAdapter`) are the earlier,
adapter-based versions of the same measurements; `scripts/run_models.py` still uses them for the study in RESULTS.md.
