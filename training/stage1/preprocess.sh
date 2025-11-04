#!/bin/bash
# Script để preprocess data với GPU acceleration
# NHANH HƠN 10-20x so với CPU!

cd /mnt/disk1/SonDinh/SonDinh/DICE_model/training/stage1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RG_DICE

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Starting GPU-accelerated preprocessing..."
echo "=========================================="

# Run preprocessing với GPU batch processing
python preprocess_data.py \
    --root_dir "/media/gpus/New Volume/ViMed-PET/raw" \
    --output_dir "/media/gpus/New Volume/ViMed-PET/raw/preprocessed_480" \
    --spatial_size 480 \
    --target_depth 201 \
    --batch_size 4 \
    --device cuda \
    --split both

echo ""
echo "=========================================="
echo "✅ Preprocessing completed!"
echo "=========================================="
echo ""
echo "Cấu trúc thư mục:"
echo "  /media/gpus/New Volume/ViMed-PET/raw/"
echo "  ├── processed_npy/       (raw data cũ)"
echo "  └── preprocessed_480/    (✨ data mới, resize sẵn!)"
echo "      ├── ct/"
echo "      └── pet/"
echo ""
echo "Bước tiếp theo:"
echo "1. Update train.py dòng 24:"
echo "   Thay: from dataset_preprocessed import create_dataloaders"
echo ""
echo "2. Update config.yaml:"
echo "   data:"
echo "     root_dir: '/media/gpus/New Volume/ViMed-PET/raw/preprocessed_480'"
echo ""
echo "3. Chạy training:"
echo "   bash train.sh"
