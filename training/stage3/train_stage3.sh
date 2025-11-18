#!/usr/bin/env bash

# Activate conda environment
conda init
conda activate stage1env

# Set CUDA device (adjust if needed)
export CUDA_VISIBLE_DEVICES=0

# CUDA memory allocation configuration
# Helps with memory fragmentation for models with variable ROI counts
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Exit on error, undefined variable, or pipe failure
set -euo pipefail

# Get script directory and config path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/config_stage3.yaml"
CONFIG_PATH="${1:-$DEFAULT_CONFIG}"

echo "============================================================"
echo "Stage 3 Training: Local (RoI) Feature Alignment"
echo "============================================================"
echo "[Stage3] Script directory: ${SCRIPT_DIR}"
echo "[Stage3] Using config: ${CONFIG_PATH}"
echo "[Stage3] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "[Stage3] PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "============================================================"
echo ""

# Run training
python -u "${SCRIPT_DIR}/train.py" --config "${CONFIG_PATH}"

echo ""
echo "============================================================"
echo "[Stage3] Training completed!"
echo "============================================================"
