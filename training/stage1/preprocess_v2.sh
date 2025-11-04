#!/bin/bash
# Script để preprocess data với GPU + multi-threading
# Giữ nguyên cấu trúc thư mục, chỉ thay processed_npy → processed_480_npy

cd /mnt/disk1/SonDinh/SonDinh/DICE_model/training/stage1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RG_DICE

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "GPU + Multi-threading Preprocessing"
echo "=========================================="
echo "Input:  /media/gpus/New Volume/ViMed-PET/raw"
echo "Output: /mnt/disk1/SonDinh/SonDinh/DICE_model/training/stage1"
echo "Transform: processed_npy → processed_480_npy"
echo "=========================================="
echo ""

# Run preprocessing
python preprocess_data_v2.py \
    --json_path "/media/gpus/New Volume/ViMed-PET/raw/label/PETCT_parts_train_val_test.json" \
    --input_root "/media/gpus/New Volume/ViMed-PET/raw" \
    --output_root "/mnt/disk1/SonDinh/SonDinh/DICE_model/training/stage1" \
    --spatial_size 480 \
    --target_depth 201 \
    --num_workers 16 \
    --device cuda

echo ""
echo "=========================================="
echo "✅ Hoàn tất!"
echo "=========================================="
echo ""
echo "Cấu trúc output:"
echo "  /mnt/disk1/SonDinh/SonDinh/DICE_model/training/stage1/"
echo "  └── processed_480_npy/"
echo "      ├── PETCT_2017/"
echo "      ├── PETCT_2018/"
echo "      └── ..."
echo ""
echo "Kiểm tra một vài files:"
find /mnt/disk1/SonDinh/SonDinh/DICE_model/training/stage1/processed_480_npy -name "*.npy" | head -5
