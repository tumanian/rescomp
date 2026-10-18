# rescomp

Compress residual-stream activations from a transformer and store them in one self-describing file, with the
model, layer, token and label metadata that interpretability work needs later. Pure PyTorch + zstd, CPU or GPU,
and framework-agnostic: the codec and file format never see a model, and capture and evaluation talk to a small
`ModelAdapter` interface (Hugging Face, TransformerLens and plain-function adapters are included).

- `PLAN.md`: the one-page design.
- `RESULTS.md`: measurements on real models (ratio, KL, perplexity, coherence).

## Install

```bash
pip install -e .            # torch, numpy, zstandard
pip install -e '.[hf]'      # + transformers, datasets (capture from Hugging Face models)
pytest                      # ~185 tests, CPU only, ~1 min (tests/test_golden.py pins every codec's bytes)
```

## Use

```python
import rescomp as rc
from rescomp import capture as C, evaluate as E
from rescomp.hf import load_model

model, tok = load_model("gpt2")                                      # HFAdapter over any HF causal LM
tokens = C.wikitext_sequences(tok, n_seq=128, seq_len=256)           # or C.pack_sequences(tok, texts, ...)
C.capture(model, tokens, layers=[8, 9, 10, 11], spec="int4_g384+layerK4+out12", out="gpt2.rsc",
          annotations={"doc_id": doc_ids})                           # any per-sequence arrays ride along

r = rc.Reader("gpt2.rsc")
print(r.summary())
r.header["model"], r.header["layers"][1]                             # model id/revision/config, per-layer quant + fidelity
x = r.read(layer=9, seqs=slice(0, 16))                               # [16, T, d] fp32: exactly the encoder's reconstruction
r.tokens(), r.annotation("doc_id")

E.evaluate_file("gpt2.rsc", model)                                   # KL / perplexity with the last stored layer patched in
E.evaluate_file_per_layer("gpt2.rsc", model)                         # ... one layer at a time
E.evaluate_live(model, tokens, layers, spec, calib)                  # codec inside the forward pass (errors compound)
E.coherence(model, prompts, layers, spec, calib, tokenizer=tok)      # generation with the codec on, paired perplexity
rc.transcode("gpt2_bf16.rsc", "gpt2_int4.rsc", "int4_g384+layerK4+out12")   # re-encode without the model
```

Lower level: `rc.encode(X, spec, calib)` / `rc.decode(streams, spec, calib)` on an `[L, B, T, d]` stack, or
`rc.Encoder(...).encode_layer(i, x)` one layer at a time (closed-loop deltas).

### Other models and frameworks

Capture and evaluation only need a `ModelAdapter` (`rescomp/model.py`): `n_layers`, `d_model`, `device`, `info()`
for the header, `hook_names(layer)`, `moe_layers()`, and `forward(tokens, capture=(), patch=None, routers=())`
returning `(logits, {layer: activation}, routing)`. Three are included:

- `rescomp.hf.HFAdapter` for Hugging Face causal LMs (the only module that imports `transformers`), with
  multi-GPU `device_map` and MoE router capture.
- `rescomp.model.TLAdapter` for a TransformerLens `HookedTransformer` (site `blocks.<l>.hook_resid_post`).
- `rescomp.model.FunctionAdapter` for anything else: hand it one function that runs tokens to logits, collects the
  requested layers and applies `rescomp.model.apply_patch(layer, x, patch)` after each layer.

`tests/test_generic.py` runs the full pipeline on a hand-written torch model with `transformers` blocked from
importing.

Command line: `rescomp info FILE`, `rescomp capture --model gpt2 --layers 8 9 10 11 --spec ... --out FILE`,
`rescomp eval FILE [--live] [--coherence]`.

## Specs

`<quant>[+layerK<K>][+out<n>][+sink<s>][+key=<quant>]`

| Part | Meaning |
|---|---|
| `fp32`, `bf16`, `fp16` | plain casts (bf16 is lossless for a bf16 model), then byte-shuffle + zstd |
| `fp8_e4m3`, `int8_tok` | 8-bit codes with one fp16 scale per token |
| `int4_g<G>`, `int2_g<G>`, `int8_g<G>` | integer codes with one fp16 scale per group of G channels (G must divide d) |
| `uabs<step>` | fixed step in activation units; zigzag integer codes in byte planes, so zstd does the entropy coding |
| `mxfp4`, `nvfp4` | 4-bit floats with a shared block scale (32 or 16 channels) |
| `+layerK<K>` | code stored layer i as `x_i - xhat_{i-1}` except every K-th layer (closed loop, so errors do not accumulate) |
| `+out<n>` | keep the n highest-energy channels of each layer in bf16 (chosen on a calibration sample; stored in the header) |
| `+sink<s>` | keep the first s tokens of every sequence in bf16 |
| `+key=<quant>` | hybrid: keyframe layers use this quantizer (e.g. `key=bf16` keeps every K-th layer exact) |

Starting points from the earlier study: `int4_g384+layerK4+out12` for GPT-2 small, `int4_g512+layerK4+out32`
for Qwen3-8B (about 5x at +2-6% perplexity), `uabs<0.1 x rms>+out8` for high fidelity. Compare specs at matched
KL or perplexity: without `+out`, the massive channels make 8-bit codes collapse and look better than they are.

## File format (RSC1)

`"RSC1" | u64 header offset | u64 header length | blobs | JSON header`. One `write` call = one chunk of sequences;
every (chunk, layer, stream) is its own zstd blob with a CRC, so one layer of one chunk can be read alone (a delta
layer also decodes the earlier layers of its keyframe group). The header records the model (id, revision,
architecture, config summary, dtype, tokenizer), the capture (site, hook names in HF and TransformerLens naming,
seq_len, hardware, library version), the spec and calibration, per-layer measured fidelity (rel L2, max abs error),
byte counts, tokens and attention masks per chunk, and named per-sequence annotations. `rescomp info FILE` prints it.

## Evaluate on real models

`scripts/run_models.py` captures a model once (bf16), transcodes to a ladder of specs and measures each
(`scripts/report.py` renders the tables); `scripts/aws.sh` drives the GPU box. See `RESULTS.md`.
