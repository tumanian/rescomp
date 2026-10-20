#!/usr/bin/env bash
# The probe study (scripts/probes.py) on a model that needs several GPUs, on a fresh `BOX=big` box.
#   scripts/run_probes_large.sh MODEL LAYER LAYER LAYER [-- extra probes.py args]
#   e.g. scripts/run_probes_large.sh Qwen/Qwen3-235B-A22B 47 70 93
# Puts the Hugging Face model cache on the instance's local NVMe disk (fast, and as large as the model needs),
# downloads the model there, then runs probes.py with device_map=auto. The SST-2 dataset must already be in
# ~/.cache/huggingface/datasets (copied from a machine that has it); everything runs offline after the download.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$1; shift
LAYERS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do LAYERS+=("$1"); shift; done; [ "${1:-}" = "--" ] && shift
PY=${PY:-$HOME/venv/bin/python}
NVME=/opt/dlami/nvme
if ! mountpoint -q "$NVME"; then
  NVME=/mnt/nvme
  if ! mountpoint -q "$NVME"; then
    DEV=$(lsblk -dpno NAME,MODEL | awk '/Instance Storage/ {print $1; exit}')
    [ -n "$DEV" ] || { echo "no instance-store NVMe disk"; exit 1; }
    sudo mkfs.ext4 -q -F "$DEV" && sudo mkdir -p "$NVME" && sudo mount "$DEV" "$NVME"
  fi
fi
sudo mkdir -p "$NVME/hf" "$NVME/caps" && sudo chown -R "$USER" "$NVME/hf" "$NVME/caps"
export HF_HUB_CACHE=$NVME/hf HF_XET_HIGH_PERFORMANCE=1
echo "$(date +%T) cache on $NVME ($(df -h "$NVME" | tail -1 | awk '{print $4}') free); downloading $MODEL"
t0=$(date +%s)
$PY -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*'], max_workers=32)"
echo "$(date +%T) downloaded $(du -sh "$NVME/hf" | cut -f1) in $(( $(date +%s) - t0 )) s"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 $PY scripts/probes.py --model "$MODEL" --layers "${LAYERS[@]}" --device-map auto \
  --train 2000 --test 872 --batch 32 --out "$NVME/caps" "$@"
echo "LARGE PROBES DONE $MODEL"
