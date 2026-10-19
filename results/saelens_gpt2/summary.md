# rescomp x SAELens downstream check: gpt2 (768d, 12 layers)

*2026-10-19 08:57. Hook `blocks.7.hook_resid_post`; 20,000,087 training tokens (128-token contexts, BOS at 0, never trained on); TopK SAE d_sae = 16 x d_model, k = 32, lr 0.0003, batch 4096; seeds [0, 1, 2]. Every number below is measured on true float32 activations of 500,126 held-out tokens (BOS excluded). Cells are mean (range) over seeds.*

| arm | spec | layers | target | B/tok/layer | ratio vs fp32 | rel L2 (target) | FVE | MSE | L0 | dead frac | CE clean | CE spliced | CE zero-abl | delta CE | CE recovered | KL(clean, spliced) | cross-seed match | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | fp32 live | - | - | 4 x d | 1.00 | 0 | 0.8232 (0.0006) | 1036.082 (3.307) | 32.00 (0.00) | 0.0337 (0.0024) | 3.5773 (0.0000) | 3.7557 (0.0019) | 11.135 (0.000) | 0.1784 (0.0019) | 0.9764 (0.0003) | 0.1983 (0.0022) | 0.5500 (noise floor) | - |
| faithful | `int8_tok+out8` | [7] | keyframe | 656.6 | 4.68 | 9.02e-03 | 0.8233 (0.0001) | 1035.504 (0.620) | 32.00 (0.00) | 0.0312 (0.0032) | 3.5773 (0.0000) | 3.7549 (0.0013) | 11.135 (0.000) | 0.1776 (0.0013) | 0.9765 (0.0002) | 0.1966 (0.0016) | 0.5508 | PASS |
| balanced | `int4_g96+layerK4+out8+key=bf16` | [4, 5, 6, 7] | delta | 543.6 | 5.65 | 2.88e-02 | 0.8231 (0.0003) | 1036.703 (1.994) | 32.00 (0.00) | 0.0306 (0.0035) | 3.5773 (0.0000) | 3.7562 (0.0034) | 11.135 (0.000) | 0.1789 (0.0034) | 0.9763 (0.0005) | 0.1987 (0.0036) | 0.5486 | PASS |
| compact | `int4_g96+layerK4+out8` | [4, 5, 6, 7] | delta | 349.7 | 8.79 | 2.88e-02 | 0.8233 (0.0003) | 1035.093 (1.759) | 32.00 (0.00) | 0.0320 (0.0037) | 3.5773 (0.0000) | 3.7549 (0.0045) | 11.135 (0.000) | 0.1776 (0.0045) | 0.9765 (0.0006) | 0.1979 (0.0051) | 0.5498 | PASS |

Ratios and bytes per token cover the whole captured layer stack (keyframes included for `balanced` and `compact`),
not the target layer alone. rel L2 is ||x - xhat|| / ||x|| at the target layer over the first 512 training sequences, BOS excluded (the sink's own figure in the capture header counts the BOS position too, whose norm is far larger than any other token's, so it reads lower: faithful 3.99e-03, balanced 1.01e-02, compact 1.01e-02).

## Pass rule (fixed before running)

- |mean delta CE(arm) - mean delta CE(baseline)| <= max(baseline seed range, 0.005 nats)
- |mean FVE(arm) - mean FVE(baseline)| <= max(baseline seed range, 0.002)
- cross-seed feature match(arm) >= cross-seed match(baseline) - 0.01

Cross-seed feature match: for each pair of distinct seeds (s, s'), the mean over arm-seed-s decoder rows of the max cosine against baseline-seed-s' decoder rows, averaged over the pairs; the baseline row is the noise floor (baseline s vs baseline s').

| arm | delta CE gap (tolerance) | FVE gap (tolerance) | feature match (floor) | verdict |
|---|---|---|---|---|
| faithful | 0.0008 (0.0050) | 0.0001 (0.0020) | 0.5508 (0.5400) | PASS |
| balanced | 0.0005 (0.0050) | 0.0001 (0.0020) | 0.5486 (0.5400) | PASS |
| compact | 0.0008 (0.0050) | 0.0002 (0.0020) | 0.5498 (0.5400) | PASS |

## Per seed

| arm | seed | FVE | MSE | L0 | dead frac | CE clean | CE spliced | CE zero-abl | delta CE | CE recovered | KL(clean, spliced) | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 0.8230 | 1037.139 | 32.00 | 0.0342 | 3.5773 | 3.7562 | 11.135 | 0.1789 | 0.9763 | 0.1983 | 564 |
| baseline | 1 | 0.8235 | 1033.900 | 32.00 | 0.0322 | 3.5773 | 3.7545 | 11.135 | 0.1772 | 0.9766 | 0.1972 | 560 |
| baseline | 2 | 0.8230 | 1037.207 | 32.00 | 0.0346 | 3.5773 | 3.7564 | 11.135 | 0.1791 | 0.9763 | 0.1994 | 560 |
| faithful | 0 | 0.8233 | 1035.114 | 32.00 | 0.0309 | 3.5773 | 3.7556 | 11.135 | 0.1783 | 0.9764 | 0.1974 | 908 |
| faithful | 1 | 0.8232 | 1035.664 | 32.00 | 0.0297 | 3.5773 | 3.7543 | 11.135 | 0.1770 | 0.9766 | 0.1958 | 442 |
| faithful | 2 | 0.8232 | 1035.734 | 32.00 | 0.0329 | 3.5773 | 3.7549 | 11.135 | 0.1776 | 0.9765 | 0.1965 | 443 |
| balanced | 0 | 0.8229 | 1037.844 | 32.00 | 0.0288 | 3.5773 | 3.7581 | 11.135 | 0.1807 | 0.9761 | 0.2006 | 842 |
| balanced | 1 | 0.8231 | 1036.415 | 32.00 | 0.0323 | 3.5773 | 3.7546 | 11.135 | 0.1773 | 0.9765 | 0.1970 | 836 |
| balanced | 2 | 0.8232 | 1035.850 | 32.00 | 0.0307 | 3.5773 | 3.7560 | 11.135 | 0.1787 | 0.9764 | 0.1985 | 848 |
| compact | 0 | 0.8235 | 1034.188 | 32.00 | 0.0308 | 3.5773 | 3.7574 | 11.135 | 0.1801 | 0.9762 | 0.2006 | 728 |
| compact | 1 | 0.8232 | 1035.947 | 32.00 | 0.0343 | 3.5773 | 3.7530 | 11.135 | 0.1757 | 0.9768 | 0.1956 | 739 |
| compact | 2 | 0.8233 | 1035.144 | 32.00 | 0.0307 | 3.5773 | 3.7544 | 11.135 | 0.1771 | 0.9766 | 0.1976 | 727 |
