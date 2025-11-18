#!/bin/bash

# Activate conda environment
conda init
conda activate stage1env

# Set CUDA device (adjust if needed)
export CUDA_VISIBLE_DEVICES=2

# CUDA memory allocation configuration
# Helps with memory fragmentation for models with variable ROI counts
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Optional: Set OMP threads for CPU parallelism
export OMP_NUM_THREADS=8

# Run training
python train.py --config config_stage4.yaml

# Optional: If you want to override the device in config
# python train.py --config config_stage4.yaml --device cuda:0
