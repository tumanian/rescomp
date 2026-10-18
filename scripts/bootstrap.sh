#!/bin/bash
# EC2 user-data for a fresh rescomp GPU box (AWS Deep Learning Base AMI, Ubuntu 22.04): venv with CUDA torch,
# transformers, datasets, zstandard. Runs once as root on first boot; ~3 min.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -q && apt-get install -yq python3-venv python3-pip rsync
sudo -u ubuntu bash -c "
  cd ~ && python3 -m venv venv
  ~/venv/bin/pip install -q --upgrade pip
  ~/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128
  ~/venv/bin/pip install -q 'transformers>=4.56' datasets zstandard numpy pytest accelerate
"
touch /home/ubuntu/.bootstrap-done
