"""
Dataset for CLIP-style Vision-Text Pretraining

Handles:
- Loading CT/PET .npy files with variable depths
- Text extraction from Vietnamese reports
- Spatial resizing to 480x480
- Variable depth padding in collate function
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Mapping từ body part code sang Vietnamese keys trong "Mô tả hình ảnh"
BODY_PART_MAPPING = {
    'abdomen_pelvis': 'Ổ bụng - khung chậu',
    'chest': 'Lồng ngực',
    'head_neck': 'Đầu - cổ'
}


class ViMedPETCLIPDataset(Dataset):
    """
    Dataset cho CLIP-style pretraining với CT/PET và Vietnamese text.

    Returns dict với keys:
        - 'ct': [D, 480, 480] tensor
        - 'pet': [D, 480, 480] tensor
        - 'text': str (Vietnamese description)
        - 'id': int
    """

    def __init__(self, json_path, root_dir, config):
        """
        Args:
            json_path: Path to JSON manifest (train/val)
            root_dir: Root directory of dataset
            config: Data config dict
        """
        with open(json_path) as f:
            self.samples = json.load(f)

        self.root_dir = root_dir
        self.config = config

        print(f"[Dataset] Loaded {len(self.samples)} samples from {json_path}")

    def __len__(self):
        return len(self.samples)

    def _extract_body_part(self, path):
        """Extract body part code from file path."""
        for part in BODY_PART_MAPPING.keys():
            if part in path:
                return part
        return None

    def _extract_text(self, report_path, body_part):
        """
        Extract text matching body part từ report JSON.

        Args:
            report_path: Path to report JSON
            body_part: Body part code (abdomen_pelvis/chest/head_neck)

        Returns:
            str: Vietnamese text description
        """
        full_path = f"{self.root_dir}/{report_path}"

        try:
            with open(full_path, encoding='utf-8') as f:
                report = json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to load report {full_path}: {e}")
            return ""

        # Lấy từ "Mô tả hình ảnh"
        mo_ta = report.get('Mô tả hình ảnh', {})

        # Map body part code sang Vietnamese key
        vn_key = BODY_PART_MAPPING.get(body_part, '')

        # Lấy text
        text = mo_ta.get(vn_key, '')
        if isinstance(text, dict):
            # Một số báo cáo có thể lồng thêm key con – nối tất cả value lại
            text = " ".join(str(v) for v in text.values())

        return text.strip()

    def _resize_volume(self, volume, target_spatial, target_depth=201):
        """
        Resize volume to fixed spatial and depth dimensions.

        Args:
            volume: (D, H, W) numpy array
            target_spatial: Target spatial size (e.g., 480)
            target_depth: Target depth (default 201)

        Returns:
            (target_depth, target_spatial, target_spatial) tensor
        """
        D, H, W = volume.shape

        # Convert to tensor [1, 1, D, H, W]
        volume = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)

        # Resize both depth and spatial dimensions to fixed sizes
        volume = F.interpolate(
            volume,
            size=(target_depth, target_spatial, target_spatial),
            mode='trilinear',
            align_corners=False
        )

        # Remove batch and channel dims: [target_depth, target_spatial, target_spatial]
        return volume.squeeze(0).squeeze(0)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load CT/PET numpy arrays
        ct_path = f"{self.root_dir}/{sample['ct_img_path']}"
        pet_path = f"{self.root_dir}/{sample['pet_img_path']}"

        ct = np.load(ct_path)      # (D, 512, 512)
        pet = np.load(pet_path)    # (D, 256, 256)

        # Normalize by dividing by 32767
        ct = ct / self.config['normalize_by']
        pet = pet / self.config['normalize_by']

        # Resize to fixed 201 depth and 480x480 spatial
        target_size = self.config['spatial_size']
        target_depth = 201  # Fixed depth
        ct = self._resize_volume(ct, target_size, target_depth)    # (201, 480, 480)
        pet = self._resize_volume(pet, target_size, target_depth)  # (201, 480, 480)

        # Extract text matching body part
        body_part = self._extract_body_part(sample['ct_img_path'])
        text = self._extract_text(sample['report_path'], body_part)

        return {
            'ct': ct,           # [D, 480, 480]
            'pet': pet,         # [D, 480, 480]
            'text': text,
            'id': sample['id']
        }


def collate_fn(batch):
    """
    Custom collate function.
    All volumes are already fixed to 201 depth.
    KHÔNG CẦN padding vì (201-1) % 10 = 0 (đúng với temporal_patch_size=10)

    Args:
        batch: List of dicts from __getitem__

    Returns:
        dict với:
        - 'ct': [B, 201, 480, 480]
        - 'pet': [B, 201, 480, 480]
        - 'text': List of strings
        - 'id': List of ints
    """
    ct_batch = [item['ct'] for item in batch]   # List of [201, 480, 480]
    pet_batch = [item['pet'] for item in batch]  # List of [201, 480, 480]

    return {
        'ct': torch.stack(ct_batch),           # [B, 201, 480, 480]
        'pet': torch.stack(pet_batch),         # [B, 201, 480, 480]
        'text': [item['text'] for item in batch],
        'id': [item['id'] for item in batch]
    }


def create_dataloaders(config):
    """
    Factory function to create train and validation dataloaders.

    Args:
        config: Full config dict

    Returns:
        train_loader, val_loader
    """
    # Training dataset
    train_ds = ViMedPETCLIPDataset(
        json_path=f"{config['data']['root_dir']}/{config['data']['train_json']}",
        root_dir=config['data']['root_dir'],
        config=config['data']
    )
    print(f"Created training dataset with {len(train_ds)} samples.")

    # Validation dataset
    val_ds = ViMedPETCLIPDataset(
        json_path=f"{config['data']['root_dir']}/{config['data']['val_json']}",
        root_dir=config['data']['root_dir'],
        config=config['data']
    )
    print(f"Created validation dataset with {len(val_ds)} samples.")
    # Training dataloader với optimization cho CPU bottleneck
    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,                    # Pin memory for faster GPU transfer
        persistent_workers=True,            # Giữ workers sống giữa các epochs
        prefetch_factor=4,                  # Prefetch 4 batches per worker
        drop_last=True                      # Drop last incomplete batch
    )
    print(f"Created training dataloader with {len(train_loader)} batches.")
    # Validation dataloader
    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    print(f"Created validation dataloader with {len(val_loader)} batches.")
    print(f"[DataLoader] Train batches: {len(train_loader)}")
    print(f"[DataLoader] Val batches: {len(val_loader)}")

    return train_loader, val_loader


# Test
if __name__ == '__main__':
    import yaml

    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    train_loader, val_loader = create_dataloaders(config)

    # Test one batch
    batch = next(iter(train_loader))
    print(f"CT shape: {batch['ct'].shape}")
    print(f"PET shape: {batch['pet'].shape}")
    print(f"Text samples 1: {batch['text'][0][:100]}...")
    print(f"IDs: {batch['id']}")
    
    batch2 = next(iter(val_loader))
    print(f"CT shape: {batch2['ct'].shape}")
    print(f"PET shape: {batch2['pet'].shape}")
    print(f"Text samples 1: {batch2['text'][0][:100]}...")
    print(f"IDs: {batch2['id']}")
