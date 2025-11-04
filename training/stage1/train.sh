#!/bin/bash

# Stage 1 Training Script
# Simple single-GPU training

echo "========================================="
echo "Stage 1: CLIP-style Pretraining"
echo "========================================="

# Set GPU device
conda activate RG_DICE

export CUDA_VISIBLE_DEVICES=0

# Run training
python train.py --config config.yaml > run.txt 2>&1

echo "========================================="
echo "Training finished!"
echo "========================================="
