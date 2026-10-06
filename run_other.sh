#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data1/code/dlx/Psi0_zbs"
cd "$PROJECT_ROOT"

source .venv/bin/activate

export PYTHONPATH="$PROJECT_ROOT"
export HF_HOME="/data1/code/dlx/psi_home/cache/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

exec torchrun \
  --standalone \
  --nproc_per_node=1 \
  train.py \
  --config config/configs/other_config.json \
  "$@"
