"""
Model Builder for Stage 4: LoRA Finetuning
Loads Stage 3 checkpoint and applies LoRA to language decoder.
"""

import sys
from pathlib import Path

# Add repo root to path
CURRENT_DIR = Path(__file__).resolve()
REPO_ROOT = CURRENT_DIR.parents[3]  # /mnt/disk1/aiotlab/sondinh/Model_dice
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import torch
from typing import Dict, Any, Tuple

from hirra_model.hirra import HiRRA
from .freeze import apply_stage4_freeze_policy, count_parameters


def load_stage3_checkpoint(model: HiRRA, checkpoint_path: str) -> Dict[str, Any]:
    """
    Load Stage 3 checkpoint into the model.

    Args:
        model: HiRRA model instance
        checkpoint_path: Path to Stage 3 checkpoint (.pt file)

    Returns:
        Dictionary with checkpoint information
    """
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage 3 checkpoint not found: {ckpt_path}")

    print(f"[ModelBuilder] Loading Stage 3 checkpoint from: {ckpt_path}")

    # Load checkpoint
    payload = torch.load(ckpt_path, map_location="cpu")

    # Extract model state dict
    if "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    print(f"[ModelBuilder] Checkpoint contains {len(state_dict)} keys")

    # Load into model (strict=False to allow architecture differences)
    incompatible = model.load_state_dict(state_dict, strict=False)

    # Prepare info
    info = {
        "checkpoint": str(ckpt_path),
        "epoch": payload.get("epoch", "unknown"),
        "global_step": payload.get("global_step", "unknown"),
        "loaded_keys": len(state_dict),
        "missing_keys": len(incompatible.missing_keys) if hasattr(incompatible, 'missing_keys') else 0,
        "unexpected_keys": len(incompatible.unexpected_keys) if hasattr(incompatible, 'unexpected_keys') else 0,
    }

    # Print missing/unexpected keys if any
    if incompatible.missing_keys:
        print(f"[ModelBuilder] Missing keys: {len(incompatible.missing_keys)}")
        if len(incompatible.missing_keys) <= 10:
            for key in incompatible.missing_keys:
                print(f"  - {key}")

    if incompatible.unexpected_keys:
        print(f"[ModelBuilder] Unexpected keys: {len(incompatible.unexpected_keys)}")
        if len(incompatible.unexpected_keys) <= 10:
            for key in incompatible.unexpected_keys:
                print(f"  - {key}")

    print(f"[ModelBuilder] Successfully loaded checkpoint from epoch {info['epoch']}")

    return info


def build_stage4_model(config: Dict[str, Any]) -> Tuple[HiRRA, Dict[str, Any]]:
    """
    Build HiRRA model for Stage 4 training (LoRA finetuning).

    Steps:
    1. Load Stage 3 checkpoint to get architecture config
    2. Initialize HiRRA model with checkpoint config
    3. Load Stage 3 checkpoint weights
    4. Apply Stage 4 freeze policy (freeze all + LoRA)
    5. Count parameters

    Args:
        config: Full configuration dictionary

    Returns:
        Tuple of (model, metadata_dict)
    """
    model_cfg = config["model"]

    # Step 1: Load Stage 3 checkpoint to get architecture config
    stage3_checkpoint = model_cfg.get("stage3_checkpoint")
    if not stage3_checkpoint:
        raise ValueError("stage3_checkpoint must be specified in config")

    ckpt_path = Path(stage3_checkpoint).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage 3 checkpoint not found: {ckpt_path}")

    print("[ModelBuilder] Loading Stage 3 checkpoint to extract architecture config...")
    payload = torch.load(ckpt_path, map_location="cpu")

    # Extract architecture config from checkpoint
    if "config" in payload:
        ckpt_model_cfg = payload["config"]["model"]
        print("[ModelBuilder] Using model config from checkpoint for architecture")

        # Extract key architecture parameters
        ctvit_config_raw = ckpt_model_cfg.get("ctvit", {})

        # Filter ctvit_config to only include valid CTViT parameters
        # (remove MultimodalEncoder params like num_fusion_layers)
        valid_ctvit_keys = {
            'dim', 'codebook_size', 'image_size', 'patch_size', 'temporal_patch_size',
            'spatial_depth', 'temporal_depth', 'dim_head', 'heads', 'mlp_dim',
            'channels', 'use_vgg_and_gan', 'vgg_pretrained', 'discriminator_config'
        }
        ctvit_config = {k: v for k, v in ctvit_config_raw.items() if k in valid_ctvit_keys}

        feature_extractor_config = ckpt_model_cfg.get("feature_extractor_config", "improved")
        language_decoder_name = ckpt_model_cfg.get("language_decoder_name", "Qwen/Qwen2.5-3B-Instruct")
        original_spatial_dims = tuple(ckpt_model_cfg.get("original_spatial_dims", [201, 480, 480]))
        num_queries = ckpt_model_cfg.get("num_queries", 32)
        use_graph_reasoning = ckpt_model_cfg.get("use_graph_reasoning", True)
        graph_config = ckpt_model_cfg.get("graph_config", "basic")

        print(f"[ModelBuilder] CTViT config: {ctvit_config}")
        print(f"[ModelBuilder] Feature extractor: {feature_extractor_config}")
        print(f"[ModelBuilder] LLM: {language_decoder_name}")
    else:
        # Fallback: use config from config_stage4.yaml
        print("[ModelBuilder] Warning: No config in checkpoint, using config_stage4.yaml")
        ctvit_config = model_cfg.get("ctvit", {})
        feature_extractor_config = model_cfg.get("feature_extractor_config", "improved")
        language_decoder_name = model_cfg.get("language_decoder_name", "Qwen/Qwen2.5-3B-Instruct")
        original_spatial_dims = tuple(model_cfg.get("original_spatial_dims", [201, 480, 480]))
        num_queries = model_cfg.get("num_queries", 32)
        use_graph_reasoning = model_cfg.get("use_graph_reasoning", True)
        graph_config = model_cfg.get("graph_config", "basic")

    # Step 2: Initialize HiRRA model with checkpoint architecture
    print("[ModelBuilder] Building HiRRA model for Stage 4...")
    model = HiRRA(
        ctvit_config=ctvit_config,
        feature_extractor_config=feature_extractor_config,
        language_decoder_name=language_decoder_name,
        original_spatial_dims=original_spatial_dims,
        num_queries=num_queries,
        use_graph_reasoning=use_graph_reasoning,
        graph_config=graph_config,
        language_decoder_4bit=False,  # No quantization for LoRA training
        language_decoder_kwargs={}
    )

    # Step 3: Load Stage 3 checkpoint weights
    checkpoint_info = load_stage3_checkpoint(model, stage3_checkpoint)

    # Step 4: Apply Stage 4 freeze policy (freeze all + apply LoRA)
    print("[ModelBuilder] Applying Stage 4 freeze policy (freeze all + LoRA)...")
    apply_stage4_freeze_policy(model, config)

    # Step 5: Count parameters
    param_stats = count_parameters(model)
    print(f"[ModelBuilder] Total parameters: {param_stats['total']:,}")
    print(f"[ModelBuilder] Trainable parameters: {param_stats['trainable']:,} ({param_stats['trainable_pct']:.2f}%)")

    # Return model and metadata
    metadata = {
        "checkpoint": checkpoint_info,
        "param_stats": param_stats,
        "architecture": {
            "ctvit_config": ctvit_config,
            "feature_extractor_config": feature_extractor_config,
            "language_decoder_name": language_decoder_name,
            "num_queries": num_queries,
            "use_graph_reasoning": use_graph_reasoning,
        }
    }

    return model, metadata
