#!/usr/bin/env bash
# The full evaluation on the GPU box (run there: ~/venv/bin/python, cwd ~/rescomp). Late third of each model.
# Roughly: GPT-2 10 min, Qwen3-1.7B 30 min, Qwen3-8B 60-80 min, Gemma-2-2B 30 min on an A10G.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venv/bin/python}
COH='int4_g{G}+layerK4+out{O} int4_g{G}+layerK4+out{O}+key=bf16 uabs@0.1+out8 int2_g16+layerK4+out16'

$PY scripts/run_models.py --model gpt2 --layers 8 9 10 11 --n-seq 128 --seq-len 256 --batch 16 \
    --coherence-specs $COH --out results/gpt2
$PY scripts/run_models.py --model Qwen/Qwen3-1.7B --layers 19 20 21 22 23 24 25 26 27 --n-seq 64 --seq-len 512 --batch 8 \
    --coherence-specs $COH --out results/qwen3-1.7b
$PY scripts/run_models.py --model google/gemma-2-2b --layers 17 18 19 20 21 22 23 24 25 --n-seq 64 --seq-len 512 --batch 8 \
    --attn eager --bos --coherence-specs $COH --out results/gemma-2-2b
$PY scripts/run_models.py --model Qwen/Qwen3-8B --layers 24 25 26 27 28 29 30 31 32 33 34 35 --n-seq 64 --seq-len 512 --batch 4 \
    --coherence-specs $COH --out results/qwen3-8b
$PY scripts/report.py results/gpt2 results/qwen3-1.7b results/gemma-2-2b results/qwen3-8b > results/RESULTS_TABLES.md
echo ALL DONE
