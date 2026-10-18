#!/usr/bin/env bash
# Second pass on the box after run_all.sh: the d/8-group specs (the Qwen3-8B winner's proportions from the earlier
# study), coherence on them, and per-layer single-replacement KL on the main configs. Already-measured specs are
# skipped (run_models.py reads results.jsonl), so this only adds rows.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venv/bin/python}
SPECS='compact balanced int4_g{G}+layerK4+out{O} int4_g{G}+layerK4+out{O}+key=bf16 uabs@0.1+out8 int8_tok+out8 bf16'
COH='compact balanced'
PER='compact balanced int4_g{G}+layerK4+out{O} int4_g{G}+layerK4+out{O}+key=bf16 uabs@0.1+out8 int8_tok+out8'

run() { $PY scripts/run_models.py "$@" --specs $SPECS --coherence-specs $COH --per-layer-specs $PER; }
run --model gpt2 --layers 8 9 10 11 --n-seq 128 --seq-len 256 --batch 16 --out results/gpt2
run --model Qwen/Qwen3-1.7B --layers 19 20 21 22 23 24 25 26 27 --n-seq 64 --seq-len 512 --batch 8 --out results/qwen3-1.7b
run --model google/gemma-2-2b --layers 17 18 19 20 21 22 23 24 25 --n-seq 64 --seq-len 512 --batch 8 --attn eager --bos --out results/gemma-2-2b
run --model Qwen/Qwen3-8B --layers 24 25 26 27 28 29 30 31 32 33 34 35 --n-seq 64 --seq-len 512 --batch 4 --out results/qwen3-8b
$PY scripts/report.py results/gpt2 results/qwen3-1.7b results/gemma-2-2b results/qwen3-8b > results/RESULTS_TABLES.md
echo EXTRAS DONE
