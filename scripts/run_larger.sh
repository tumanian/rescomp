#!/usr/bin/env bash
# Stage 2: larger dense models and a mixture-of-experts model, as additional result directories.
#   scripts/run_larger.sh a10g    # on the 24 GB box: Gemma-2-9B (18 GB) and OLMoE-1B-7B (14 GB, 64 experts, 8 active)
#   scripts/run_larger.sh l40s    # on a 48 GB box: Qwen3-14B (28 GB)
#   scripts/run_larger.sh moe4    # on a 4-GPU box (device_map=auto): Qwen3-30B-A3B (60 GB, 128 experts, 8 active)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venv/bin/python}
SPECS='bf16 fp16 int8_tok+out8 uabs@0.1+out8 compact balanced int4_g{G}+layerK4+out{O} nvfp4+layerK4+out8 int2_g16+layerK4+out16'
COH='compact balanced uabs@0.1+out8'
PER='compact balanced uabs@0.1+out8 int8_tok+out8'
run() { $PY scripts/run_models.py "$@" --specs $SPECS --coherence-specs $COH --per-layer-specs $PER; }
case ${1:-a10g} in
  a10g)
    # Gemma-2-9B: 42 layers, d = 3584; late third 28-41 is 14 layers, keep 12 (30-41)
    run --model google/gemma-2-9b --layers 30 31 32 33 34 35 36 37 38 39 40 41 --n-seq 64 --seq-len 512 --batch 2 \
        --attn eager --bos --out results/gemma-2-9b
    # OLMoE-1B-7B: 16 layers, d = 2048, every layer is MoE; late third 11-15; routing stored for the stored layers
    run --model allenai/OLMoE-1B-7B-0924 --layers 11 12 13 14 15 --n-seq 64 --seq-len 512 --batch 4 \
        --routers stored --out results/olmoe-1b-7b ;;
  l40s)
    # Qwen3-14B: 40 layers, d = 5120; late third 27-39, keep 12 (28-39)
    run --model Qwen/Qwen3-14B --layers 28 29 30 31 32 33 34 35 36 37 38 39 --n-seq 64 --seq-len 512 --batch 2 \
        --out results/qwen3-14b ;;
  moe4)
    # Qwen3-30B-A3B: 48 layers, d = 2048, all MoE; late third 32-47, keep 12 (36-47)
    run --model Qwen/Qwen3-30B-A3B --layers 36 37 38 39 40 41 42 43 44 45 46 47 --n-seq 64 --seq-len 512 --batch 2 \
        --device-map auto --routers stored --out results/qwen3-30b-a3b ;;
esac
echo "LARGER DONE $1"
