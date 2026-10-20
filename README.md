# rescomp

Compress the residual-stream activations of a running transformer and keep them, with the tokens, labels and
model metadata that interpretability work needs later; or keep only a mean and a last-token row per sequence, which
is what most probes train on. Pure PyTorch + zstd + Parquet, CPU or GPU, any framework whose layers are
`nn.Module`s.

- `DESIGN.md`: the design decisions behind the sink and the storage layout.
- `RESULTS.md`: what each codec costs, measured on eight models (GPT-2 to Qwen3-30B-A3B, dense and MoE).
- `PLAN.md`: the original one-page plan.

## Install

```bash
pip install -e .            # torch>=2.8, numpy, zstandard, pyarrow, torchao
pip install -e '.[hf]'      # + transformers
pip install -e '.[scripts]' # + datasets, for the scripts/ study and CLI
pytest                      # ~155 tests, CPU only, under a minute
```

## Use

### Pooled rows for probes

A probe wants one vector per sequence, not one per token. `rc.TransformSink` hooks the model like the `Sink`
below but stores only a few rows per sequence: the mean over the real tokens and the last real token. Measured on
SST-2 (RESULTS.md): the rows equal a lossless full capture's pooled rows bit for bit, probes trained on either
give the same predictions, the capture is 11-12x smaller than the full one on short sentences (70-140x at 256-512
tokens), and reading it is 4x (GPT-2) to 113x (Qwen2.5-72B) faster than replaying the model to get the same
features. `model` and `tok` are
loaded as in the Sink example below.

```python
sink = rc.TransformSink("runs/sst2", layers=rc.bind(model, [14, 21], tokenizer=tok))   # mean + last rows, fp32
with sink, torch.no_grad():
    for texts, labels in loader:
        sink.annotate(label=labels)                     # one label per sentence, stored next to its rows
        model(**tok(texts, return_tensors="pt", padding=True).to("cuda"))
r = rc.open("runs/sst2")
X, y = r.pooled(21, "mean").numpy(), r.annotation("label")   # [N, d] in sequence order; or "last"
from sklearn.linear_model import LogisticRegression
probe = LogisticRegression(max_iter=1000).fit(X[:2000], y[:2000]); print(probe.score(X[2000:], y[2000:]))
```

- Each layer's hook pools the activation on the device (`[B, T, d]` -> `[B, 2, d]` fp32); the rows accumulate per
  layer until `chunk_seqs` (4096) sequences are in, then go through the codec and into the store as one chunk.
  The site becomes `resid_post.mean+last`; `r.pooled_rows` names the rows, `r.read(21)` gives the `[2, d]` rows
  of each sequence.
- The default codec is `"fp32"`: the rows are stored bit for bit (a mean is an fp32 value, so the `"exact"`
  regime, an fp16 cast, would round it to about 5e-4 relative; it still stores the last token of a bf16 model bit
  for bit and takes 1.5x less space). The lossy regimes work as well (outlier channels are chosen on the first chunk).
- A KV-cache continuation call (a generation step) is skipped: the rows and the stored tokens describe the prompt.
- Cost (A10G, `scripts/pooling_overhead.py`): Qwen3-8B with 12 hooked layers runs 0.7% slower, GPT-2 small 18%
  (a fixed 0.4-1.6 ms per hooked layer); memory is the accumulated rows only (k x d x 4 bytes per sequence per
  layer). Stored: 20 KB per sequence per layer on Qwen3-8B (13 KB with `exact`), 141x less than an `exact` full
  capture and 41x less than `compact`. DESIGN.md has the table.
- `scripts/probes.py` trains logistic-regression probes (SST-2) on the pooled rows of a lossless full capture and
  of a TransformSink capture, tests each on the other and on the lossy regimes, and times each way of getting the
  features (read the pooled capture, read the full capture, replay the model); RESULTS.md has the tables.
- `r.pooled(layer, "mean")` also works on a full capture from a `Sink`: the row is computed from the stored
  tokens with the attention mask, so a probe reads either kind of capture the same way.

### Every token: the Sink

```python
import torch, rescomp as rc
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16).cuda().eval()
batch = tok(["The history of the Roman Empire", "To make a tomato sauce, first"], return_tensors="pt", padding=True).to("cuda")

sink = rc.Sink("runs/demo", layers=rc.bind(model, [24, 28, 32, 35], tokenizer=tok),
               codec="balanced")                                          # a regime, a spec string, or rc.Codec(...)
with sink, torch.no_grad():
    logits = model(**batch).logits                 # one forward call = one chunk
    out = model.generate(**batch, max_new_tokens=80)   # prompt + steps = one sequence per prompt
print(sink.stats())                                # chunks, sequences, bytes, ratio, per-layer rel L2

r = rc.open("runs/demo")
r.sequences()                                      # Parquet table: seq_id, n_tokens, tokens, mask, annotations
x = r.read(32, seqs=[2, 3])                        # each sequence whole, [T_i, d] fp32 (the encoder's own reconstruction)
r.tokens(2); r.header["model"]; r.header["layers"][1]["fidelity"]
```

- `layers`: `rc.bind(model, [...])` recognises Hugging Face and TransformerLens models by type (`ModelBinding`
  subclasses in `hf.py` / `tl.py`; the framework is imported only when present). Any other model:
  `rc.bind({layer: module, ...}, root=model)` with the modules whose outputs are the residual stream (`root` gets
  the token pre-hook).
- Each layer's hook quantizes that layer on the model's device (cheap kernels) and keeps the encoded streams
  there; the model frees its activations as usual. `sink.flush()` moves the streams to the host, runs zstd and
  writes; it is forced when the on-device buffer passes `buffer_bytes` (1 GiB default) and at close.
  Measured on an A10G (`scripts/sink_overhead.py`): Qwen3-8B with 12 hooked layers runs 2.4-5.8% slower
  (`compile=True` gives the lower numbers) with about 150 MiB of working memory plus the buffered streams;
  GPT-2 small, whose forward is only 36 ms, 24-30% slower. DESIGN.md has the table and the path to under 1%.
- Tokens: taken from each call by a pre-hook on the root model (`input_ids`, `attention_mask`, and the KV-cache
  signal that tells a generation step from a new prompt); `sink.tokens(ids)` overrides, `tokens="off"` stores no token ids (the attention mask is still read and stored:
  pooling needs it to skip padding).
- `sink.annotate(doc_id=[...], split="train")` labels the sequences the next new-sequence call starts; labels become
  Parquet columns.
- `codec`: one object for how activations are stored, lossy or lossless: `rc.Codec("compact", level=5)`,
  `rc.Codec("int4_g512+layerK4+out32")`, `rc.Codec("bf16")`. Inside it are the quantizer, the layer deltas, the
  outlier channels kept in bf16 (chosen per layer on the first forward call unless `channels=` is given) and the
  zstd stage; which parts run on the device and which on the host is its business, not yours. `codec.lossy` says
  which kind it is; `compile=True` runs the per-layer kernels through torch.compile (same bytes).

### Storage (format 3), local or S3

A capture lives under a stem: a local path (`runs/demo`), an `s3://bucket/prefix/demo` URI, or a name with
`bucket=` and `prefix=`; `Sink`, `Store`, `rc.open`/`Reader` and `rc.transcode` all take these. S3 goes through
pyarrow's own S3 filesystem (credentials from the usual environment, profile or instance role; `s3={...}` passes
options such as `region` or `endpoint_override` for MinIO-style endpoints). On S3 each flush closes the current
tensor object, so a capture is readable mid-run, and objects are capped at 1 GiB (4 GiB locally); the reader
fetches blobs with range requests.

```python
with rc.Sink("demo", layers=..., codec="balanced", bucket="my-acts", prefix="team/qwen3-8b") as sink: ...
r = rc.open("s3://my-acts/team/qwen3-8b/demo")
```

`<stem>.00000.rsc, ...` hold only the compressed tensor blobs (append-only, rotated at `rotate_bytes`, 4 GB by
default). `<stem>.meta.parquet` has the model, spec, outlier channels, stored layers with hook names and measured
fidelity, and byte counts; `<stem>.chunks.parquet` which sequences each chunk holds and where; `<stem>.blobs.parquet`
one row per blob (byte range, crc, dtype, shape); `<stem>.tokens.parquet` one row per sequence (tokens, mask,
annotations).

### Lower level

`codec.encode(X)` / `codec.decode(streams)` on an `[L, B, T, d]` stack; `codec.encoder(n).encode_layer(i, x)`
one layer at a time on the model's device (closed-loop deltas); `codec.pack(t)` / `codec.unpack(...)` the host
half; `rc.Store(stem, model=..., layers=..., codec=...)` to write encoded chunks yourself; `rc.transcode(src, dst,
codec)` to re-encode a stored capture with another codec or regime; `reader.codec` is the one a file was written
with. `rescomp.bf16_wrapper.choose_channels(X, spec)` picks outlier channels from a sample offline.

## Regimes

Four named settings cover the useful range; pass one as `codec=`. Each is a spec template whose group and
outlier counts scale with the model width `d` (resolved when the sink sees its first batch).

| Regime | Spec | Keep in mind |
|---|---|---|
| `exact` | `fp16` | bit-exact for bf16 models |
| `faithful` | `int8_tok+out8` | every layer at single-layer KL ≈ 0.001 |
| `balanced` | `int4_g(d/8)+layerK4+out(d/128)+key=bf16` | every 4th layer stored exactly; the rest at KL ≤ 0.005 |
| `compact` | `int4_g(d/8)+layerK4+out(d/128)` | the 5× point; small models pay more |

Measured on eight models (compression ratio vs bf16 / perplexity change with the codec live in the forward pass,
late third of layers, wikitext; details and per-layer numbers in `RESULTS.md`):

| Model | `exact` | `faithful` | `balanced` | `compact` |
|---|---|---|---|---|
| GPT-2 | 1.5× / +0.0% | 2.4× / +0.1% | 3.0× / +1.9% | 4.7× / +2.9% |
| Qwen3-1.7B | 1.5× / +0.0% | 2.5× / -0.3% | 2.6× / +1.1% | 5.0× / +6.8% |
| Gemma-2-2B | 1.5× / +0.0% | 2.6× / +0.2% | 2.7× / +1.2% | 5.0× / +7.0% |
| Qwen3-8B | 1.5× / +0.0% | 2.6× / +0.1% | 3.0× / +1.8% | 5.2× / +6.8% |
| Qwen3-14B | 1.5× / +0.0% | 2.6× / +0.1% | 3.0× / +1.2% | 5.2× / +4.3% |
| Gemma-2-9B | 1.5× / +0.0% | 2.6× / +0.5% | 3.0× / +1.2% | 5.1× / +3.9% |
| OLMoE-1B-7B (MoE) | 1.5× / +0.0% | 2.5× / +0.7% | 2.4× / +0.6% | 5.0× / +4.1% |
| Qwen3-30B-A3B (MoE) | 1.5× / +0.0% | 2.7× / +0.2% | 3.0× / +1.8% | 5.1× / +5.7% |

![Four regimes on eight models](docs/regimes.png)

The table and chart are the study's numbers (codec version 1, the library's own quantizers). Codec 3 pins the
int-group, MXFP4 and NVFP4 paths to torchao; re-measured on GPT-2 it gives `compact` 4.6× / +2.8% (torchao's
nibble packing compresses about 2% less under zstd), `faithful` 2.4× / +0.4% (was +0.1%; the difference is
within the noise of a bf16 forward), `exact` and `balanced` unchanged.

## Specs

Anything beyond the four regimes is a spec string: `<quant>[+layerK<K>][+out<n>][+sink<s>][+key=<quant>]`

| Part | Meaning |
|---|---|
| `fp32`, `bf16`, `fp16` | plain casts (bf16 is lossless for a bf16 model; fp16 too, at 1.5×), then byte-shuffle + zstd |
| `fp8_e4m3`, `int8_tok` | 8-bit codes with one fp16 scale per token |
| `int4_g<G>`, `int8_g<G>` | torchao affine codes (symmetric, no clipping, full range) with one fp16 scale per group of G channels (G must divide d) |
| `mxfp4`, `nvfp4` | torchao's OCP MX / NVIDIA FP4: e2m1 codes with an e8m0 scale per 32 or an e4m3 scale per 16 channels |
| `+layerK<K>` | code stored layer i as `x_i - xhat_{i-1}` except every K-th layer (closed loop, so errors do not accumulate) |
| `+out<n>` | keep the n highest-energy channels of each layer in bf16 (chosen on the first call; stored in the header) |
| `+sink<s>` | keep the first s tokens of every sequence in bf16 |
| `+key=<quant>` | hybrid: keyframe layers use this quantizer (`key=bf16` keeps every K-th layer exact) |

Beyond the regimes, `nvfp4+layerK4+out8` gives 3.7× at +0.5–2%. The study's `uabs` (fixed-step) codec is not in
the library: it was not one of the four regimes, and its numbers stay in RESULTS.md for reference.

## Instrumentation (separate from the sink)

`rescomp.instrument` feeds reconstructions back into a running model and measures the change, with the same
module-object convention and any `run(tokens) -> logits` callable:

```python
from rescomp import instrument as I
L = rc.bind(model, [24, 28, 32, 35])
run = lambda t: model(t, use_cache=False).logits
I.kl_ppl(run, tokens, I.Patcher(L, I.LiveCodec("int4_g512+layerK4+out32")))   # codec inside the forward pass
I.kl_ppl(run, tokens, I.Patcher(L, rc.open("runs/demo")), seq_ids=range(64))  # stored reconstructions
I.per_layer(run, tokens, L, I.LiveCodec("int8_tok+out8"))                       # one layer at a time
I.coherence(run, prompts, I.Patcher(L, I.LiveCodec(spec)), tokenizer=tok)       # generation, paired perplexity
```

`scripts/run_models.py` runs the study in RESULTS.md with these pieces (capture once with the `exact` regime,
`rc.transcode` to each spec, then the measurements). `scripts/cli.py` (`info`, `capture`, `eval`),
`scripts/corpus.py` (wikitext windows) and `scripts/spot_check.py` (a real GPT-2 capture checked against the
tokenizer and forward hooks) live beside it; none of them is part of the library.

## Layout

| Module | What it holds |
|---|---|
| `sink.py` | `Sink`: hooks a running model, encodes each layer in its hook, buffers on the device, flushes to a `Store` |
| `transform_sink.py`, `transforms.py` | `TransformSink`: the `Sink` with a transform (`Mean`, `Last`, stacked) in place of the per-layer encoding; a few rows per sequence for probes |
| `hooks.py` | `Layers` (what to hook) and `CallHooks` (the forward hooks and the token pre-hook), shared by the Sink and the Patcher |
| `bindings.py`, `hf.py`, `tl.py` | `ModelBinding` and `bind()`: which modules carry the residual stream, what the header records; Hugging Face (`load_model`, `decoder_blocks`) and TransformerLens bindings |
| `codec.py` | `Codec`: how activations are stored. The device half (`encoder()`: closed-loop deltas, the compiled per-layer function) and the host half (`pack`/`unpack`); `to_dict`/`from_header` for the file header |
| `stage.py` | `Stage`, the root of the codec's parts, with the registry and `stage(kind, name)` factory |
| `quant.py` | `Quantizer` stages (casts, fp8, int8/int4 groups on torchao, MXFP4, NVFP4), `quantizer(name)` |
| `lossless.py` | `Lossless` stages (byte-shuffle + zstd), `lossless(record)` |
| `bf16_wrapper.py` | what the quantizer does not touch: outlier channels and sink rows, kept in bf16, and how they are chosen |
| `spec.py`, `regimes.py` | spec strings and the four regimes |
| `store.py` | format 3 on disk or S3: `Store`, `Reader`, `transcode`, `location` |
| `instrument.py` | the separate regime: `Patcher`, `LiveCodec`, KL / perplexity / per-layer / coherence |
