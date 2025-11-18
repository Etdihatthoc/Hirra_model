#!/usr/bin/env bash

# Simple launch script for Stage 2 training.
# Usage:
#   ./train.sh [path/to/config_stage2.yaml]
# If config path is omitted, default file in this directory is used.
conda init
conda activate stage1env
#unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES=0
# mkdir -p /mnt/disk1/aiotlab/sondinh/Model_dice/Hirra_model/training/stage2/wandb_tmp/wandb
# chmod -R 777 /mnt/disk1/aiotlab/sondinh/Model_dice/Hirra_model/training/stage2/wandb_tmp
# export WANDB_DIR=/mnt/disk1/aiotlab/sondinh/Model_dice/Hirra_model/training/stage2/wandb_tmp
# export WANDB_CACHE_DIR=$WANDB_DIR/cache
# export WANDB_CONFIG_DIR=$WANDB_DIR/config
# export TMPDIR=$WANDB_DIR


set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/config_stage2.yaml"
CONFIG_PATH="${1:-$DEFAULT_CONFIG}"

echo "[Stage2] Using config: ${CONFIG_PATH}"
python -u "${SCRIPT_DIR}/train.py" --config "${CONFIG_PATH}"
