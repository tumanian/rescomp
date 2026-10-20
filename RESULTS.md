# Results: rescomp on four models

*2026-10-17. Eight models on three AWS boxes (one A10G for most, one L40S for Qwen3-14B, four A10Gs for the 30B
mixture of experts). Numbers regenerate with `scripts/run_all.sh`, `scripts/run_extras.sh`, `scripts/run_larger.sh`
and `scripts/report.py`; the raw records (`results/<model>/results.jsonl`, generations) are kept out of the repo.*

## Setup

*Codec note: these numbers were measured with codec version 1 (the library's own quantizers). Codec version 3
pins the int-group, MXFP4 and NVFP4 paths to torchao (no-clipping symmetric int grid, torchao's nibble packing)
and drops `uabs`. Re-measured on GPT-2 (2026-10-19, same data and layers): `exact` 1.51×, `faithful`
`int8_tok+out8` 2.42× at live KL 0.0217 / ppl ×1.0035, `compact` `int4_g96+layerK4+out8` 4.59× at live KL 0.047 /
ppl ×1.028 (was 4.70× / 0.050 / ×1.029), per-layer KL 0.029 keyframe / 0.037 delta, coherence ×1.06 (t 1.5).
The ratio of the int4 rows is about 2% lower under codec 3 (the packing compresses slightly less); errors are the same.*

- **Models** (bf16, Hugging Face): GPT-2 small (12 layers, d = 768), Qwen3-1.7B (28 layers, d = 2048),
  Gemma-2-2B (26 layers, d = 2304, eager attention, BOS-first windows), Qwen3-8B (36 layers, d = 4096); then, as
  additions, Qwen3-14B (40 layers, d = 5120), Gemma-2-9B (42 layers, d = 3584), OLMoE-1B-7B (16 layers, d = 2048,
  64 experts, 8 active) and Qwen3-30B-A3B (48 layers, d = 2048, 128 experts, 8 active).
- **Layers**: the late third of each model, as in the earlier study: GPT-2 8–11, Qwen3-1.7B 19–27, Gemma-2-2B
  17–25, Qwen3-8B 24–35, Qwen3-14B 28–39, Gemma-2-9B 30–41, OLMoE 11–15, Qwen3-30B-A3B 36–47. The site is the
  residual stream after each decoder block (`resid_post`).
- **Data**: wikitext-103 paragraphs, seed 0, packed into windows: GPT-2 128 × 256 tokens, the others 64 × 512.
  The first 32 windows also calibrate the outlier channels and the `uabs` step (0.1 × the rms of the non-outlier
  channels). The split is the training split, as before; this only affects absolute perplexities, and every number
  below is a ratio or difference against the clean model on the same tokens.
- **Procedure**: capture each model once losslessly (`bf16`), transcode to each spec with `rescomp.transcode`,
  then measure.

### What the columns mean

| Column | Meaning |
|---|---|
| Ratio vs bf16 | raw bf16 bytes of the stored layers ÷ stored bytes (codes + scales + outliers, after zstd) |
| rel L2 | ‖x − x̂‖ / ‖x‖ over all stored layers |
| KL (file), ppl × (file) | every stored layer replaced by its reconstruction *from the file*. Each layer's reconstruction was made from the clean activations, so the model's output only depends on the **last** stored layer's reconstruction. This is what a probe or a replay on the file sees for that layer |
| KL (live), ppl × (live) | the codec runs inside the forward pass, each delta layer coded against the already-patched previous layer: errors compound, as in generation. The honest number for "run the model on compressed activations" |
| top-1 | fraction of positions where the patched model's top token equals the clean model's (file mode) |
| Coherence ppl ×, t | 80-token continuations sampled (T = 0.8, top-k 50) with the live codec on, for 10 prompts × 3 seeds, and the same prompts and seeds without it. Each continuation is scored by the clean model; the column is exp(mean log-perplexity difference) and the paired t over the 10 prompt means. The blind LLM judge used last time was not sensitive and is not used |
| rep4 | fraction of 4-grams that already occurred in the continuation (looping): codec / clean |
| per-layer KL | one stored layer at a time replaced by its reconstruction (extras pass; in the per-model notes) |

KL is mean next-token KL(clean ‖ patched) in nats. A KL of 0.01 is small (the earlier study's bf16 noise floor on
GPT-2 was 0.017); 0.1 is visible in generated text; 1 is broken.

## Tables

Specs with `{G}`-style sizes were resolved per model: groups of d/2 channels with d/64 outlier channels (the GPT-2
winner's proportions, e.g. `int4_g384+layerK4+out12`), and groups of d/8 with d/128 outliers (the earlier Qwen3-8B
winner's, e.g. `int4_g512+layerK4+out32`). `uabs` steps are 0.1 × rms. Rows are sorted by ratio.

### gpt2  (layers 8-11, 128 seqs x 256 tokens)

| Spec | Ratio vs bf16 | rel L2 | KL (last layer) | ppl x (last layer) | KL (live) | ppl x (live) | top-1 | Coherence ppl x | t | rep4 codec/clean | Enc s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16` | 1.40 | 0.0e+00 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 3 |
| `fp16` | 1.51 | 0.0e+00 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 2 |
| `fp8_e4m3+out8` | 2.35 | 1.1e-02 | 0.0214 | 1.0006 | 0.0225 | 1.0015 | 0.916 | - | - | - | 2 |
| `int8_tok+out8` | 2.42 | 1.0e-02 | 0.0165 | 1.0001 | 0.0217 | 1.0008 | 0.933 | - | - | - | 2 |
| `uabs0.4464+out8` | 2.91 | 1.1e-02 | 0.0106 | 1.0010 | 0.0231 | 1.0021 | 0.955 | 1.033 | 0.7 | 0.024/0.031 | 4 |
| `int4_g384+layerK4+out12+key=bf16` | 3.03 | 3.3e-02 | 0.0270 | 1.0074 | 0.0308 | 1.0096 | 0.905 | 1.106 | 2.2 | 0.018/0.031 | 2 |
| `nvfp4+layerK4+out8` | 3.64 | 2.3e-02 | 0.0224 | 1.0020 | 0.0287 | 1.0107 | 0.914 | - | - | - | 3 |
| `mxfp4+layerK4+out8` | 3.87 | 3.6e-02 | 0.0251 | 1.0057 | 0.0368 | 1.0195 | 0.907 | - | - | - | 3 |
| `int4_g64+layerK4+out8` | 4.45 | 4.2e-02 | 0.0331 | 1.0114 | 0.0434 | 1.0223 | 0.892 | - | - | - | 2 |
| `int4_g384+layerK4+out12` | 5.00 | 4.0e-02 | 0.0271 | 1.0079 | 0.0423 | 1.0245 | 0.903 | 1.092 | 1.6 | 0.023/0.031 | 2 |
| `int4_g384+out12` | 5.10 | 7.3e-02 | 0.0484 | 1.0263 | 0.0867 | 1.0707 | 0.870 | - | - | - | 2 |
| `int2_g16+layerK4+out16` | 7.71 | 1.3e-01 | 0.0600 | 1.0361 | 0.2944 | 1.3286 | 0.852 | 1.495 | 9.0 | 0.010/0.031 | 2 |

### Qwen/Qwen3-1.7B  (layers 19-27, 64 seqs x 512 tokens)

| Spec | Ratio vs bf16 | rel L2 | KL (last layer) | ppl x (last layer) | KL (live) | ppl x (live) | top-1 | Coherence ppl x | t | rep4 codec/clean | Enc s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16` | 1.40 | 0.0e+00 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 21 |
| `fp16` | 1.51 | 0.0e+00 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 18 |
| `fp8_e4m3+out8` | 2.37 | 1.6e-02 | 0.0016 | 0.9998 | 0.0058 | 1.0023 | 0.980 | - | - | - | 23 |
| `int8_tok+out8` | 2.48 | 1.0e-02 | 0.0009 | 1.0004 | 0.0031 | 0.9974 | 0.986 | - | - | - | 23 |
| `int4_g1024+layerK4+out32+key=bf16` | 2.67 | 3.3e-02 | 0.0000 | 1.0000 | 0.0228 | 1.0177 | 1.000 | 0.967 | -0.7 | 0.026/0.054 | 25 |
| `uabs2.645+out8` | 2.99 | 1.7e-02 | 0.0009 | 1.0000 | 0.0077 | 1.0021 | 0.986 | 1.029 | 0.5 | 0.024/0.054 | 55 |
| `nvfp4+layerK4+out8` | 3.71 | 4.0e-02 | 0.0102 | 1.0049 | 0.0294 | 1.0176 | 0.946 | - | - | - | 34 |
| `mxfp4+layerK4+out8` | 3.94 | 5.5e-02 | 0.0252 | 1.0034 | 0.0621 | 1.0129 | 0.918 | - | - | - | 35 |
| `int4_g64+layerK4+out8` | 4.47 | 6.2e-02 | 0.0265 | 1.0156 | 0.0608 | 1.0434 | 0.911 | - | - | - | 25 |
| `int4_g1024+layerK4+out32` | 5.19 | 8.1e-02 | 0.0382 | 1.0210 | 0.1167 | 1.0808 | 0.896 | 1.159 | 2.6 | 0.042/0.054 | 25 |
| `int4_g1024+out32` | 5.56 | 1.2e-01 | 0.0382 | 1.0210 | 0.2465 | 1.2265 | 0.896 | - | - | - | 25 |
| `int2_g16+layerK4+out16` | 8.37 | 2.6e-01 | 0.4251 | 1.2479 | 1.2798 | 2.5755 | 0.692 | 8.608 | 12.3 | 0.004/0.054 | 26 |

### google/gemma-2-2b  (layers 17-25, 64 seqs x 512 tokens)

| Spec | Ratio vs bf16 | rel L2 | KL (last layer) | ppl x (last layer) | KL (live) | ppl x (live) | top-1 | Coherence ppl x | t | rep4 codec/clean | Enc s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16` | 1.40 | 0.0e+00 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 26 |
| `fp16` | 1.52 | 2.1e-14 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 22 |
| `fp8_e4m3+out8` | 2.39 | 2.0e-02 | 0.0018 | 1.0009 | 0.0045 | 1.0041 | 0.981 | - | - | - | 28 |
| `int8_tok+out8` | 2.56 | 1.4e-02 | 0.0010 | 1.0010 | 0.0026 | 1.0023 | 0.990 | - | - | - | 26 |
| `int4_g1152+layerK4+out36+key=bf16` | 2.67 | 3.9e-02 | 0.0000 | 1.0000 | 0.0136 | 1.0133 | 1.000 | 1.000 | -0.0 | 0.023/0.021 | 27 |
| `uabs0.6756+out8` | 2.98 | 2.0e-02 | 0.0008 | 1.0003 | 0.0050 | 1.0045 | 0.989 | 0.993 | -0.1 | 0.022/0.021 | 65 |
| `nvfp4+layerK4+out8` | 3.72 | 5.1e-02 | 0.0099 | 1.0091 | 0.0212 | 1.0196 | 0.954 | - | - | - | 40 |
| `mxfp4+layerK4+out8` | 3.95 | 6.7e-02 | 0.0258 | 1.0341 | 0.0487 | 1.0552 | 0.935 | - | - | - | 39 |
| `int4_g64+layerK4+out8` | 4.47 | 7.7e-02 | 0.0202 | 1.0213 | 0.0454 | 1.0487 | 0.937 | - | - | - | 26 |
| `int4_g1152+layerK4+out36` | 5.09 | 8.8e-02 | 0.0273 | 1.0266 | 0.0620 | 1.0599 | 0.930 | 0.976 | -0.6 | 0.021/0.021 | 29 |
| `int4_g1152+out36` | 5.25 | 1.2e-01 | 0.0273 | 1.0266 | 0.1248 | 1.1379 | 0.930 | - | - | - | 32 |
| `int2_g16+layerK4+out16` | 8.39 | 3.3e-01 | 0.3579 | 1.4170 | 1.1733 | 3.1870 | 0.767 | 3.119 | 12.7 | 0.011/0.021 | 30 |

### Qwen/Qwen3-8B  (layers 24-35, 64 seqs x 512 tokens)

| Spec | Ratio vs bf16 | rel L2 | KL (last layer) | ppl x (last layer) | KL (live) | ppl x (live) | top-1 | Coherence ppl x | t | rep4 codec/clean | Enc s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16` | 1.40 | 0.0e+00 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 64 |
| `fp16` | 1.51 | 6.7e-14 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 54 |
| `fp8_e4m3+out8` | 2.38 | 1.3e-02 | 0.0011 | 1.0009 | 0.0056 | 1.0033 | 0.983 | - | - | - | 77 |
| `int8_tok+out8` | 2.65 | 1.1e-02 | 0.0009 | 1.0005 | 0.0031 | 1.0011 | 0.985 | - | - | - | 54 |
| `uabs0.7303+out8` | 3.03 | 1.3e-02 | 0.0006 | 1.0005 | 0.0086 | 1.0040 | 0.990 | 0.927 | -2.3 | 0.032/0.012 | 152 |
| `int4_g2048+layerK4+out64+key=bf16` | 3.05 | 3.4e-02 | 0.0057 | 1.0034 | 0.0207 | 1.0141 | 0.961 | 0.970 | -0.6 | 0.023/0.012 | 74 |
| `nvfp4+layerK4+out8` | 3.73 | 2.7e-02 | 0.0022 | 1.0015 | 0.0205 | 1.0120 | 0.976 | - | - | - | 95 |
| `mxfp4+layerK4+out8` | 3.97 | 3.7e-02 | 0.0061 | 1.0089 | 0.0448 | 1.0456 | 0.957 | - | - | - | 91 |
| `int4_g64+layerK4+out8` | 4.52 | 4.4e-02 | 0.0077 | 1.0054 | 0.0455 | 1.0343 | 0.952 | - | - | - | 66 |
| `int4_g2048+layerK4+out64` | 5.31 | 5.7e-02 | 0.0057 | 1.0037 | 0.1028 | 1.0896 | 0.960 | 1.040 | 0.7 | 0.020/0.012 | 66 |
| `int4_g2048+out64` | 5.97 | 1.1e-01 | 0.0241 | 1.0156 | 0.3321 | 1.3266 | 0.918 | - | - | - | 75 |
| `int2_g16+layerK4+out16` | 8.53 | 1.9e-01 | 0.0741 | 1.0579 | 0.9861 | 2.1477 | 0.864 | 2.442 | 8.6 | 0.078/0.012 | 74 |

### Qwen/Qwen3-14B  (layers 28-39, 64 seqs x 512 tokens)

| Spec | Ratio vs bf16 | rel L2 | KL (last layer) | ppl x (last layer) | KL (live) | ppl x (live) | top-1 | Coherence ppl x | t | rep4 codec/clean | Enc s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16` | 1.40 | 0.0e+00 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 59 |
| `fp16` | 1.51 | 0.0e+00 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 44 |
| `int8_tok+out8` | 2.61 | 1.1e-02 | 0.0006 | 1.0003 | 0.0025 | 1.0014 | 0.988 | - | - | - | 55 |
| `int4_g640+layerK4+out40+key=bf16` | 3.03 | 3.8e-02 | 0.0063 | 1.0035 | 0.0172 | 1.0116 | 0.958 | 0.994 | -0.2 | 0.021/0.007 | 59 |
| `uabs0.9399+out8` | 3.03 | 1.5e-02 | 0.0005 | 1.0000 | 0.0077 | 1.0037 | 0.990 | 0.957 | -0.7 | 0.020/0.007 | 108 |
| `nvfp4+layerK4+out8` | 3.73 | 3.0e-02 | 0.0018 | 1.0005 | 0.0179 | 1.0049 | 0.978 | - | - | - | 74 |
| `int4_g640+layerK4+out40` | 5.15 | 5.6e-02 | 0.0064 | 1.0025 | 0.0684 | 1.0429 | 0.958 | 0.979 | -0.5 | 0.018/0.007 | 56 |
| `int4_g2560+layerK4+out80` | 5.27 | 6.0e-02 | 0.0056 | 1.0019 | 0.0837 | 1.0620 | 0.960 | - | - | - | 56 |
| `int2_g16+layerK4+out16` | 8.54 | 2.1e-01 | 0.0698 | 1.0455 | 0.8966 | 1.9941 | 0.870 | - | - | - | 59 |


## Findings

1. **The ratio of a recipe barely depends on the model.** Across GPT-2 (d = 768) to Qwen3-14B (d = 5120) the same
   spec lands within a few percent of the same ratio: bf16 1.40×, fp16 1.51×, int8 per token 2.4–2.65×, `uabs`
   3.0×, NVFP4 3.7×, int4 with 64-channel groups 4.5×, int4 with large groups 5.0–5.3×, int2 7.7–8.5×. The bytes
   are set by the code width and the scale overhead, and zstd gains about the same on every model.

2. **The quality cost of a recipe does depend on the model, and not simply on size.** The large-group int4 recipe
   (about 5.1–5.3×) costs, in live perplexity: GPT-2 +2.5%, Qwen3-1.7B +8.1%, Gemma-2-2B +6.0%, Qwen3-8B +9.0%
   (d/2 groups) and Qwen3-14B +4.3%. The 14B model is the most tolerant and the 1.7B the least. Pick the operating
   point per model with its own table, not by analogy.

3. **Simple fp16 is a free 8% over bf16 and lossless for bf16 models.** bf16 values are exactly representable in
   fp16 (the rel L2 of 1e-14 on Gemma and Qwen3-8B is fp16's loss of tiny subnormal values), and fp16's extra
   mantissa bits are zeros that zstd removes: 1.51× against 1.40×. It is the right default for "just store it".

4. **Closed-loop layer deltas are what make large groups usable.** Removing `+layerK4` from the 5× int4 recipe
   raises live perplexity from +2.5% to +7.1% on GPT-2, +8% to +23% on Qwen3-1.7B, +6% to +14% on Gemma-2-2B and
   +9% to +33% on Qwen3-8B, for 2–12% more ratio. The earlier study's recipe (int4 large groups + delta + a few
   bf16 outlier channels) reproduces here: GPT-2 5.00× at KL 0.042 and +2.5%, as before.

5. **The hybrid that keeps every fourth layer exact is a good middle point.** `+key=bf16` stores the keyframes
   losslessly and only codes the deltas with int4: 3.0× at +1.0–1.8% live perplexity on every model, with a third
   of the layers bit-exact for probing. It beats `uabs` (3.0×, +0.2–0.5%) only in having exact layers; for pure
   fidelity per byte `uabs` and int8 remain better.

6. **Operating points, by what you can afford to lose (live perplexity):**

   | Budget | Spec | Ratio | Models where it holds |
   |---|---|---|---|
   | ≤ 0.2% | `int8_tok+out8` | 2.4–2.65× | all five |
   | ≤ 0.5% | `uabs(0.1·rms)+out8` | 3.0× | all five |
   | ≤ 2% | `nvfp4+layerK4+out8` | 3.7× | all but Gemma-2-2B (+2.0%) |
   | ≤ 2% | `int4+layerK4+key=bf16` (hybrid) | 3.0× | all five |
   | ≤ 5% | `int4_g64+layerK4+out8` | 4.5× | GPT-2, Qwen3-8B, Qwen3-1.7B (+4.3%), Gemma (+4.9%) |
   | ≤ 5% | `int4_g(d/8)+layerK4+out(d/128)` | 5.1× | Qwen3-14B (+4.3%); others see the extras pass below |
   | ≤ 10% | `int4_g(d/2)+layerK4+out(d/64)` | 5.0–5.3× | all five |
   | never | `int2_g16+layerK4+out16` | 8× | breaks every model (+33% to +220%) |

7. **"Last layer" (file) KL understates what a running model sees.** With 12 stored layers and keyframes every 4,
   the file-mode KL of the 5× int4 recipe is 0.006 on Qwen3-8B while the live KL is 0.10, because the file's
   reconstruction of each layer was made from clean inputs and only the last one reaches the logits. When the last
   stored layer is a keyframe (9 stored layers: Qwen3-1.7B, Gemma-2-2B), the hybrid scores a perfect 1.000 in file
   mode. For interp use the per-layer single-replacement KL (extras pass) is the relevant number; for running the
   model on compressed activations use the live column.

8. **Coherence: only gross damage is detectable with 10 prompts × 3 seeds.** int2 is caught on every model
   (t = 9–14, generated-text perplexity ×1.5–8.6, collapsing 4-gram diversity). The 5× int4 recipe is neutral on
   GPT-2, Gemma, Qwen3-8B and Qwen3-14B and detectable on Qwen3-1.7B (t = 2.6, ×1.16). Two cautions: the clean
   model's perplexity of generated text can go *below* 1 for a lightly damaged model that repeats itself more
   (Qwen3-8B `uabs`: ×0.93 with rep4 0.032 vs 0.012), so read the repetition column with it; and the test's power
   is low, so "neutral" means "not grossly worse", as in the earlier study.

## Additions

### Extras pass: d/8 groups and per-layer KL

The second pass added the earlier Qwen3-8B winner's proportions (groups of d/8 channels, d/128 outlier channels)
and, for the main configs, the per-layer single-replacement KL: one stored layer at a time replaced by its
reconstruction, everything else clean. The two per-layer columns give the worst keyframe and the worst delta layer.

**GPT-2 (8–11)**

| Spec | Ratio | KL (live) | ppl × (live) | max per-layer KL, keyframes | max per-layer KL, delta layers | Coherence ppl × |
|---|---|---|---|---|---|---|
| `int8_tok+out8` | 2.42 | 0.0217 | 1.0008 | 0.019 | 0.019 | - |
| `uabs0.4464+out8` | 2.91 | 0.0231 | 1.0021 | 0.022 | 0.021 | 1.033 (t 0.5) |
| `int4_g384+layerK4+out12+key=bf16` | 3.03 | 0.0308 | 1.0096 | 0.000 | 0.027 | 1.106 (t 1.5) |
| `int4_g96+layerK4+out8` | 4.70 | 0.0498 | 1.0292 | 0.030 | 0.039 | 1.053 (t 1.0) |
| `int4_g384+layerK4+out12` | 5.00 | 0.0423 | 1.0245 | 0.033 | 0.027 | 1.092 (t 1.4) |

**Qwen3-1.7B (19–27)**

| Spec | Ratio | KL (live) | ppl × (live) | max per-layer KL, keyframes | max per-layer KL, delta layers | Coherence ppl × |
|---|---|---|---|---|---|---|
| `int8_tok+out8` | 2.48 | 0.0031 | 0.9974 | 0.001 | 0.001 | - |
| `uabs2.645+out8` | 2.99 | 0.0077 | 1.0021 | 0.003 | 0.002 | 1.029 (t 0.6) |
| `int4_g256+layerK4+out16+key=bf16` | 2.63 | 0.0179 | 1.0112 | 0.000 | 0.004 | 0.948 (t -0.8) |
| `int4_g256+layerK4+out16` | 4.96 | 0.0945 | 1.0680 | 0.040 | 0.004 | 1.180 (t 4.9) |
| `int4_g1024+layerK4+out32` | 5.19 | 0.1167 | 1.0808 | 0.038 | 0.005 | 1.159 (t 2.3) |

**Gemma-2-2B (17–25)**

| Spec | Ratio | KL (live) | ppl × (live) | max per-layer KL, keyframes | max per-layer KL, delta layers | Coherence ppl × |
|---|---|---|---|---|---|---|
| `int8_tok+out8` | 2.56 | 0.0026 | 1.0023 | 0.001 | 0.001 | - |
| `uabs0.6756+out8` | 2.98 | 0.0050 | 1.0045 | 0.002 | 0.002 | 0.993 (t -0.1) |
| `int4_g288+layerK4+out18+key=bf16` | 2.66 | 0.0132 | 1.0119 | 0.000 | 0.006 | 0.938 (t -0.8) |
| `int4_g288+layerK4+out18` | 4.98 | 0.0661 | 1.0704 | 0.029 | 0.005 | 0.949 (t -0.8) |
| `int4_g1152+layerK4+out36` | 5.09 | 0.0620 | 1.0599 | 0.027 | 0.004 | 0.976 (t -0.4) |

**Qwen3-8B (24–35)**

| Spec | Ratio | KL (live) | ppl × (live) | max per-layer KL, keyframes | max per-layer KL, delta layers | Coherence ppl × |
|---|---|---|---|---|---|---|
| `int8_tok+out8` | 2.65 | 0.0031 | 1.0011 | 0.001 | 0.001 | - |
| `uabs0.7303+out8` | 3.03 | 0.0086 | 1.0040 | 0.003 | 0.002 | 0.927 (t -1.8) |
| `int4_g512+layerK4+out32+key=bf16` | 3.03 | 0.0204 | 1.0184 | 0.000 | 0.008 | 0.876 (t -1.7) |
| `int4_g512+layerK4+out32` | 5.18 | 0.0828 | 1.0681 | 0.026 | 0.008 | 1.003 (t 0.1) |
| `int4_g2048+layerK4+out64` | 5.31 | 0.1028 | 1.0896 | 0.033 | 0.006 | 1.040 (t 0.7) |

**Qwen3-14B (28–39)**

| Spec | Ratio | KL (live) | ppl × (live) | max per-layer KL, keyframes | max per-layer KL, delta layers | Coherence ppl × |
|---|---|---|---|---|---|---|
| `int8_tok+out8` | 2.61 | 0.0025 | 1.0014 | 0.001 | 0.001 | - |
| `uabs0.9399+out8` | 3.03 | 0.0077 | 1.0037 | 0.002 | 0.002 | 0.957 (t -0.7) |
| `int4_g640+layerK4+out40+key=bf16` | 3.03 | 0.0172 | 1.0116 | 0.000 | 0.006 | 0.994 (t -0.2) |
| `int4_g640+layerK4+out40` | 5.15 | 0.0684 | 1.0429 | 0.021 | 0.006 | 0.979 (t -0.5) |
| `int4_g2560+layerK4+out80` | 5.27 | 0.0837 | 1.0620 | - | - | - |

9. **The earlier Qwen3-8B recipe reproduces.** `int4_g512+layerK4+out32` gives 5.18× at live KL 0.083 and
   perplexity +6.8% (the earlier study: 5.23×, KL 0.077, +6.1% with a different sample), and the generation test
   cannot tell it from the clean model (t = 0.1). Between the two proportions, d/8 groups cost about 2% of ratio
   for 1–3 points of perplexity on the larger models and are the safer default; on Qwen3-1.7B neither is safe
   (+6.8% and +8.1%, both detectable in generation).

10. **With layer deltas, almost all the single-layer damage sits on the keyframes.** For Qwen3-8B's 5.2×
    recipe the keyframe layers (24, 28, 32) have per-layer KL 0.013–0.026 and every delta layer 0.002–0.003; the
    same pattern holds on every model (Gemma-2-2B keyframes 0.010–0.027, deltas 0.002–0.005; Qwen3-14B keyframes
    0.011–0.031, deltas 0.002–0.004). A delta layer is coded against the previous reconstruction with a small-range
    quantizer, so its own error is small; the keyframe carries the full-range int4 error. The last stored layer
    (a keyframe in the 9-layer runs, or the final delta) is also where the live chain's compounded error shows.

11. **For probing single layers, the hybrid is the recommendation.** `+key=bf16` makes every keyframe exact and
    leaves every delta layer at per-layer KL ≤ 0.005 (≤ 0.008 for the last layer of Qwen3-8B), at 2.6–3.0× on
    every model, with live perplexity +1.0–1.8%. `uabs` at the same ratio has even per-layer KL (0.001–0.003) but no
    exact layers; `int8_tok` is 2.4–2.65× at per-layer KL ≈ 0.001 everywhere. The choice is between "a third of the
    layers bit-exact" (hybrid) and "all layers equally almost-exact" (`uabs`, int8) at the same size.

### Larger and mixture-of-experts models

These runs are additions in their own result directories; nothing above was re-run. Qwen3-30B-A3B (48 layers,
d = 2048, 128 experts with 8 active per token, 61 GB in bf16) ran on a 4 × A10G box with the model spread over the
four cards (`--device-map auto`); the library's layer-delta encoder follows each layer to its own device. The
capture also stored, for every stored layer, each token's 8 chosen experts and their router weights as per-token
annotations (`experts.<l>` and `expert_weights.<l>`, 12 of each), so the file carries the routing for expert-level
work; the .rsc files stayed on the (now terminated) box and only the measurements were pulled.

### Qwen/Qwen3-30B-A3B  (layers 36-47, 64 seqs x 512 tokens)

| Spec | Ratio vs bf16 | rel L2 | KL (last layer) | ppl x (last layer) | KL (live) | ppl x (live) | top-1 | Coherence ppl x | t | rep4 codec/clean | Enc s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16` | 1.39 | 0.0e+00 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 23 |
| `fp16` | 1.50 | 0.0e+00 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 16 |
| `int8_tok+out8` | 2.74 | 1.2e-02 | 0.0010 | 1.0008 | 0.0040 | 1.0020 | 0.985 | - | - | - | 20 |
| `int4_g256+layerK4+out16+key=bf16` | 3.01 | 5.0e-02 | 0.0204 | 1.0131 | 0.0291 | 1.0180 | 0.928 | 1.007 | 0.1 | 0.022/0.029 | 23 |
| `uabs0.09634+out8` | 3.02 | 1.4e-02 | 0.0006 | 1.0002 | 0.0078 | 1.0057 | 0.989 | 0.937 | -1.3 | 0.025/0.029 | 26 |
| `nvfp4+layerK4+out8` | 3.71 | 3.2e-02 | 0.0044 | 1.0022 | 0.0205 | 1.0066 | 0.967 | - | - | - | 21 |
| `int4_g256+layerK4+out16` | 5.13 | 6.9e-02 | 0.0204 | 1.0119 | 0.0849 | 1.0574 | 0.930 | 1.081 | 1.4 | 0.038/0.029 | 22 |
| `int4_g1024+layerK4+out32` | 5.31 | 6.9e-02 | 0.0145 | 1.0088 | 0.0951 | 1.0696 | 0.941 | - | - | - | 22 |
| `int2_g16+layerK4+out16` | 8.34 | 2.0e-01 | 0.1116 | 1.0816 | 0.9698 | 2.1335 | 0.844 | - | - | - | 22 |


Per-layer KL on Qwen3-30B-A3B (layers 36–47; keyframes 36, 40, 44):

| Spec | keyframes | delta layers | last layer (47) |
|---|---|---|---|
| `int8_tok+out8` | 0.001 | 0.001 | 0.001 |
| `uabs0.096+out8` | 0.002–0.003 | 0.001–0.002 | 0.001 |
| `int4_g256+layerK4+out16` | 0.016–0.023 | 0.002 | 0.020 |
| `int4_g256+layerK4+out16+key=bf16` | 0 | 0.002 | 0.020 |

Gemma-2-9B (42 layers, d = 3584, 18 GB; stored layers 30–41 on the 24 GB box, batch 2):

### google/gemma-2-9b  (layers 30-41, 64 seqs x 512 tokens)

| Spec | Ratio vs bf16 | rel L2 | KL (last layer) | ppl x (last layer) | KL (live) | ppl x (live) | top-1 | Coherence ppl x | t | rep4 codec/clean | Enc s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16` | 1.40 | 0.0e+00 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 58 |
| `fp16` | 1.51 | 3.9e-14 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 42 |
| `int8_tok+out8` | 2.64 | 1.4e-02 | 0.0005 | 1.0003 | 0.0033 | 1.0051 | 0.995 | - | - | - | 44 |
| `uabs0.665+out8` | 2.95 | 1.9e-02 | 0.0006 | 1.0002 | 0.0034 | 1.0030 | 0.994 | 0.997 | -0.1 | 0.008/0.019 | 72 |
| `int4_g448+layerK4+out28+key=bf16` | 3.03 | 4.2e-02 | 0.0031 | 1.0045 | 0.0104 | 1.0122 | 0.979 | 1.004 | 0.1 | 0.018/0.019 | 50 |
| `nvfp4+layerK4+out8` | 3.71 | 3.7e-02 | 0.0013 | 1.0011 | 0.0102 | 1.0112 | 0.988 | - | - | - | 62 |
| `int4_g448+layerK4+out28` | 5.10 | 7.1e-02 | 0.0031 | 1.0037 | 0.0342 | 1.0387 | 0.980 | 1.037 | 0.6 | 0.023/0.019 | 46 |
| `int4_g1792+layerK4+out56` | 5.20 | 7.4e-02 | 0.0035 | 1.0037 | 0.0382 | 1.0391 | 0.979 | - | - | - | 47 |
| `int2_g16+layerK4+out16` | 8.44 | 2.6e-01 | 0.0419 | 1.0534 | 0.6348 | 1.9252 | 0.930 | - | - | - | 49 |


Per-layer KL on Gemma-2-9B: `int8_tok` and `uabs` 0.001 on every layer; the 5.1× int4 recipe 0.007–0.010 on the
keyframes (30, 34, 38) and 0.001–0.004 elsewhere; the hybrid 0 on keyframes and ≤ 0.004 elsewhere.

14. **Gemma-2-9B is the most tolerant model measured.** The 5.1× recipe costs +3.9% live perplexity (Gemma-2-2B:
    +6.0% at the same proportions), NVFP4 at 3.7× costs +1.1%, and the hybrid at 3.0× +1.2%; the generation test is
    neutral for all three. Within the two Gemma and three dense Qwen models the larger one is the more tolerant
    each time (2B → 9B: +6.0% → +3.9%; 1.7B → 8B → 14B: +8.1% → +9.0%/+6.8% → +4.3%), with Qwen3-8B the exception at
    d/2 groups.

OLMoE-1B-7B (16 layers, all MoE, 64 experts with 8 active; stored layers 11–15, routing stored for each):

### allenai/OLMoE-1B-7B-0924  (layers 11-15, 64 seqs x 512 tokens)

| Spec | Ratio vs bf16 | rel L2 | KL (last layer) | ppl x (last layer) | KL (live) | ppl x (live) | top-1 | Coherence ppl x | t | rep4 codec/clean | Enc s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16` | 1.40 | 0.0e+00 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 12 |
| `fp16` | 1.52 | 0.0e+00 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.000 | - | - | - | 10 |
| `int4_g256+layerK4+out16+key=bf16` | 2.43 | 4.1e-02 | 0.0000 | 1.0000 | 0.0078 | 1.0058 | 1.000 | 1.086 | 3.2 | 0.012/0.029 | 14 |
| `int8_tok+out8` | 2.55 | 1.7e-02 | 0.0006 | 1.0003 | 0.0102 | 1.0071 | 0.985 | - | - | - | 11 |
| `uabs0.01227+out8` | 2.99 | 2.2e-02 | 0.0004 | 1.0004 | 0.0037 | 1.0028 | 0.990 | 1.037 | 0.8 | 0.008/0.029 | 26 |
| `nvfp4+layerK4+out8` | 3.69 | 6.2e-02 | 0.0070 | 1.0065 | 0.0143 | 1.0141 | 0.947 | - | - | - | 16 |
| `int4_g256+layerK4+out16` | 4.95 | 1.2e-01 | 0.0251 | 1.0215 | 0.0476 | 1.0410 | 0.902 | 1.151 | 2.1 | 0.011/0.029 | 13 |
| `int4_g1024+layerK4+out32` | 5.13 | 1.2e-01 | 0.0261 | 1.0242 | 0.0529 | 1.0489 | 0.901 | - | - | - | 12 |
| `int2_g16+layerK4+out16` | 8.34 | 3.9e-01 | 0.2918 | 1.3252 | 0.6284 | 1.8592 | 0.709 | - | - | - | 13 |


Per-layer KL on OLMoE: the 5× recipe 0.015 and 0.025 on its two keyframes (11, 15) and 0.002–0.004 between; the
hybrid 0 on both keyframes and ≤ 0.004 between; `int8_tok` and `uabs` ≤ 0.002 everywhere.

15. **The small MoE is as compressible as its dense peers and slightly less tolerant than the big one.** OLMoE's
    5× int4 recipe costs +4.1% live (Qwen3-30B-A3B +5.7%, Qwen3-1.7B +6.8% at the same proportions), NVFP4 at
    3.7× +1.4%, int8 +0.7%. With only 5 stored layers the keyframes are the first and last stored layer, so the
    hybrid keeps 2 of 5 layers exact at 2.4× and every other layer at KL ≤ 0.004. The generation test flags the
    hybrid (t = 3.2, ×1.09) but not the harsher 5× recipe (t = 2.1): with 10 prompts that is noise, and a reminder
    that the test only rules out gross damage.

12. **The mixture-of-experts residual compresses like a dense one.** Qwen3-30B-A3B lands on the same ratios
    (int8 2.74×, `uabs` 3.0×, NVFP4 3.7×, int4 d/8 groups 5.13×, int2 8.3×) and in the same quality band as the
    dense 8B–14B models: +0.2% for int8, +0.6% for `uabs` and NVFP4, +5.7% live for the 5.1× recipe (generation
    test neutral, t = 1.4, with more 4-gram repetition: 0.038 vs 0.029), int2 broken (×2.1). Its late residual
    stream is small in scale (rms ≈ 1 against 9–27 for the dense models), which only changes the `uabs` step.

13. **One difference: its last stored layer is the fragile one.** Layer 47's single-replacement KL is 0.020 for
    the int4 recipe *and* for the hybrid, against 0.002 for the other delta layers: the step from layer 46 to 47
    (the final block) is large relative to the signal, so the delta no longer helps there. If the last layer of
    a model is to be probed, store it as a keyframe (choose the stored range so it falls on a multiple of K, or
    use `+key=bf16` with a range ending on a keyframe).

## Probes on pooled rows (2026-10-19)

`scripts/probes.py`, on the A10G box: SST-2 sentences (2000 train, the 872 validation sentences as test) through
a bf16 model with a lossless full `Sink` (`exact`, an fp16 cast: bit-exact for bf16 activations) and a
`TransformSink` (fp32, `mean+last`) attached to the same forward, then once more per lossy regime. From every
capture `Reader.pooled(layer, row)` gives one vector per sentence: the full captures compute it from the stored
tokens with the attention mask, the TransformSink stored it. A logistic-regression probe (L-BFGS, standardized
features, small L2) per layer and row is trained on the lossless features and on the TransformSink's, and tested on
every capture's features. Three stored layers per model, so layer 6 (GPT-2) / 14 (Qwen3-1.7B) is the keyframe of
both lossy regimes and the other two are int4 delta layers in both, which is why their errors agree. (A first run
on the CPU in fp32 with an fp32 lossless codec gave the same picture within a point.)

| Model | capture | spec | stored | note |
|---|---|---|---|---|
| GPT-2, layers 6 9 11 | full exact | `fp16` | 311 MB | the lossless reference |
| | TransformSink | `fp32` | 29 MB | 11x smaller: two rows per sentence instead of every token |
| | full balanced | `int4_g96+layerK4+out8+key=bf16` | 153 MB | ratios of the full captures are inflated by the padded positions |
| | full compact | `int4_g96+layerK4+out8` | 66 MB | |
| Qwen3-1.7B, layers 14 21 27 | full exact | `fp16` | 921 MB | |
| | TransformSink | `fp32` | 75 MB | 12x smaller |
| | full balanced | `int4_g256+layerK4+out16+key=bf16` | 495 MB | |
| | full compact | `int4_g256+layerK4+out16` | 250 MB | |

Feature error against the lossless full capture (rel L2 over the test set) and probe accuracy (train source → test
source; in brackets, agreement with the same probe's predictions on the lossless features):

| Model | layer | row | TransformSink err | balanced err | compact err | lossless → lossless | lossless → TransformSink | lossless → balanced | lossless → compact | TransformSink → lossless |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-2 | 6 | mean | 0 | 0 | 9.3e-03 | 79.9% | 79.9% (100%) | 79.9% (100%) | 80.4% (96.6%) | 79.9% |
| | 6 | last | 0 | 0 | 1.2e-01 | 81.4% | 81.4% (100%) | 81.4% (100%) | 79.6% (92.7%) | 81.4% |
| | 9 | mean | 0 | 5.7e-03 | 5.8e-03 | 79.2% | 79.2% (100%) | 78.6% (98.4%) | 79.1% (98.5%) | 79.2% |
| | 9 | last | 0 | 4.0e-02 | 4.0e-02 | 81.3% | 81.3% (100%) | 81.5% (96.6%) | 80.7% (97.1%) | 81.3% |
| | 11 | mean | 0 | 3.3e-02 | 3.3e-02 | 78.4% | 78.4% (100%) | 75.9% (88.8%) | 77.2% (88.2%) | 78.4% |
| | 11 | last | 0 | 7.4e-02 | 7.4e-02 | 81.9% | 81.9% (100%) | 74.5% (81.0%) | 72.9% (80.3%) | 81.9% |
| Qwen3-1.7B | 14 | mean | 0 | 0 | 9.2e-03 | 88.3% | 88.3% (100%) | 88.3% (100%) | 88.1% (97.2%) | 88.3% |
| | 14 | last | 0 | 0 | 1.7e-01 | 87.3% | 87.3% (100%) | 87.3% (100%) | 86.7% (95.3%) | 87.3% |
| | 21 | mean | 6e-12 | 1.7e-02 | 1.7e-02 | 83.7% | 83.7% (100%) | 84.1% (99.0%) | 83.9% (98.2%) | 83.7% |
| | 21 | last | 0 | 1.1e-01 | 1.1e-01 | 87.3% | 87.3% (100%) | 86.2% (95.3%) | 86.1% (96.1%) | 87.3% |
| | 27 | mean | 0 | 3.3e-02 | 3.3e-02 | 82.7% | 82.7% (100%) | 83.5% (97.8%) | 82.7% (97.7%) | 82.7% |
| | 27 | last | 0 | 1.3e-01 | 1.3e-01 | 85.8% | 85.8% (100%) | 85.7% (92.8%) | 86.4% (92.3%) | 85.8% |

- **The TransformSink's rows are the lossless capture's pooled rows, bit for bit** on both models (error 0 on
  every layer and row but one mean at 6e-12, the GPU's summation order). A probe trained on either capture gives
  the same predictions on the other, so a probe study can use the 11-12x smaller pooled capture (70-140x smaller
  at the 256-512 token sequences of the overhead table) and the full capture interchangeably.
- **Lossy full captures work for probes on keyframe and middle layers; the last layer of GPT-2 does not.** On
  the keyframe layer `balanced` is exact (bf16 keyframes) and `compact` costs at most 2 points (93-97% agreement).
  On the int4 delta layers of Qwen3-1.7B the probes keep their accuracy within 1 point (92-99% agreement) even at
  rel L2 0.1-0.13 on the last-token rows. GPT-2's last layer is the exception: rel L2 7e-2 on the last-token
  rows costs the probe trained on lossless features 7-9 points (81.9% → 74.5 / 72.9%, 80% agreement), the mean
  probe 1-2 points. Mean rows average the error out; last-token rows carry it whole, and GPT-2's final residual
  stream is the hardest to quantize (its per-layer KL is the largest in the tables above). A probe on a compressed
  capture of a last layer should be trained on that capture or use a keyframe layer.
- Absolute accuracies (78-82% on GPT-2 small, 83-88% on Qwen3-1.7B) are what a linear probe with 2000 training
  sentences gives on SST-2; they are not the point here.

Time to get a probe's features for all 2872 sentences, three layers by two rows (A10G; the script times each
source the same way: open the capture, call `pooled()` for every layer and row; replay runs the model again with
forward hooks and pools on the GPU, nothing stored). Qwen3-8B, layers 18, 27, 35, was run the same way on the same
box (2026-10-20), after no larger GPU box could be had in us-west-2 overnight:

| Model | read the TransformSink capture | read the lossless full capture | replay the model | capture once (TransformSink alone) | train one probe |
|---|---|---|---|---|---|
| GPT-2 | 0.36 s | 6.1 s (17x) | 1.4 s (4x) | - | 0.46 s |
| Qwen3-1.7B | 0.65 s | 13.4 s (21x) | 10.5 s (16x) | - | 1.1 s |
| Qwen3-8B | 1.1 s | 25.4 s (23x) | 36.8 s (33x) | 38.7 s | 4.9 s |

- **The pooled capture is the fastest source, and its lead grows with the model: 4x faster than replaying GPT-2,
  16x for Qwen3-1.7B, 33x for Qwen3-8B.** Replay costs a forward pass of the model; reading the pooled rows costs
  only their bytes (proportional to d), so the gap keeps widening on larger models.
- **The capture pays for itself on the first reuse.** Capturing with a TransformSink costs one forward pass with
  the hooks on (38.7 s on Qwen3-8B, about one replay); every later probe session, a new layer choice or a new
  label set on the same sentences then takes about 1 s instead of 37 s, and needs no GPU or model at all.
- Reading the full capture is slower than replaying a small model: it decodes every token of every sequence with
  the CPU's zstd to pool two rows. A full capture pays off when the per-token activations are needed, not for
  sequence-level probes.
- Qwen3-8B probes behave like the other two: the TransformSink's rows equal the lossless pooled rows (rel L2 0,
  one mean at 2e-12) with 100% prediction agreement; the lossy captures cost at most 0.9 points (layer 35: 91.4%
  -> 90.5% balanced, 90.7% compact), at 97-100% agreement. Full table: `results/probes/Qwen-Qwen3-8B.md`.
- Replayed features equal the lossless capture's (rel L2 0 on GPT-2, 1.5e-12 on Qwen3-1.7B, 6e-11 on Qwen3-8B).

## Summary

- **A recipe's ratio is fixed; its cost is per model.** Across eight models from 124M to 30B parameters, dense
  and mixture-of-experts, every spec lands at the same ratio within a few percent (fp16 1.5×, int8 2.4–2.7×,
  `uabs` 3.0×, hybrid 2.4–3.0×, NVFP4 3.7×, int4 large groups 5.0–5.3×, int2 8.3–8.5×), and what it costs in live
  perplexity at 5× ranges from +2.5% (GPT-2) and +3.9% (Gemma-2-9B) to +8–9% (Qwen3-1.7B, Qwen3-8B at d/2 groups).
- **Safe defaults.** `fp16` to store losslessly at 1.5×; `int8_tok+out8` at 2.5× for ≤ 0.7% on every model;
  `uabs(0.1·rms)+out8` at 3.0× for ≤ 0.6%; `int4_g(d/8)+layerK4+out(d/128)+key=bf16` at 2.4–3.0× when a third of
  the layers must be exact; `int4_g(d/8)+layerK4+out(d/128)` at 5.0–5.2× when +4–7% perplexity is acceptable
  (+8% on the 1.7B); never int2.
- **For interp work read the per-layer column.** With layer deltas, single-layer damage sits on keyframes (and on
  a last layer whose step from its predecessor is large); the hybrid removes it. The "file" KL of a multi-layer
  file only measures its last layer.
- **Cost.** About 7 GPU-hours: the A10G box 4.4 h ($5.3), the L40S box 0.5 h ($1.1) and the 4 × A10G box 1.1 h
  ($9.2), roughly $16 in all; the two extra boxes are terminated, the main box is stopped (its 250 GB disk, about
  $20/month, keeps the model caches and each model's lossless base capture; the per-spec files were deleted to
  make room).

### Not done, and what a follow-up could do

- The routing annotations (`experts.<l>`, `expert_weights.<l>`) were written for OLMoE and Qwen3-30B-A3B but not
  analysed: per-expert activation statistics, routing entropy, and whether compressing the residual changes which
  experts the next layer picks (re-run the router on the reconstruction and count flips) are the natural next
  experiment, and the file already holds what it needs.
- Perplexity uses the wikitext training split (ratios only; the earlier study did the same); the first 32
  sequences also calibrate the outlier channels.
- The coherence test has 10 prompts × 3 seeds and only detects gross damage.
- No GPU-side entropy coder (nvCOMP) in this library: zstd on the CPU, 20–150 s per model per spec here.
