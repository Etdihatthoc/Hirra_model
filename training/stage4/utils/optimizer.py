"""
Optimizer builder for Stage 4: LoRA Finetuning
Creates optimizer for LoRA parameters (+ optionally embed_tokens and lm_head).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any


def _collect_params(module: nn.Module) -> List[torch.nn.Parameter]:
    """
    Collect trainable parameters from a module.

    Args:
        module: PyTorch module

    Returns:
        List of trainable parameters
    """
    if module is None:
        return []

    return [p for p in module.parameters() if p.requires_grad]


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Build optimizer for Stage 4 training (LoRA adapters).

    Creates parameter groups:
    1. LoRA adapters (automatically marked as trainable by PEFT)
    2. Optional: embed_tokens (if config["model"]["train_embed_tokens"] = True)
    3. Optional: lm_head (if config["model"]["train_lm_head"] = True)

    Args:
        model: HiRRA model with LoRA applied
        config: Full configuration dictionary

    Returns:
        Optimizer instance

    Raises:
        RuntimeError: If no trainable parameters found
    """
    opt_cfg = config["optimizer"]
    model_cfg = config["model"]

    param_groups = []
    seen_params = set()  # Track parameter IDs to avoid duplicates

    # ========== DEBUG: Print all trainable parameters ==========
    print("\n[Optimizer] DEBUG: All trainable parameters in language_decoder:")
    for name, param in model.language_decoder.model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}: {param.numel():,} params")
    print()

    # ========== COLLECT LORA PARAMETERS ==========
    # PEFT automatically marks LoRA adapters as trainable
    # Collect ALL trainable params from language_decoder that have "lora" in name
    lora_params = []
    lora_param_names = []
    for name, param in model.language_decoder.model.named_parameters():
        if param.requires_grad and "lora" in name.lower():
            param_id = id(param)
            if param_id not in seen_params:
                lora_params.append(param)
                lora_param_names.append(name)
                seen_params.add(param_id)

    if lora_params:
        lr_lora = opt_cfg.get("lr_lora", 1e-4)
        param_groups.append({
            "params": lora_params,
            "lr": lr_lora,
            "name": "lora_adapters"
        })
        print(f"[Optimizer] LoRA adapters: {len(lora_params)} parameter tensors, lr={lr_lora:.2e}")
        print(f"[Optimizer] LoRA param names: {lora_param_names[:5]}...")  # Show first 5

    # ========== COLLECT EMBED_TOKENS (OPTIONAL) ==========
    if model_cfg.get("train_embed_tokens", False):
        embed_layer = model.language_decoder.model.get_input_embeddings()
        if embed_layer is not None and embed_layer.weight.requires_grad:
            param_id = id(embed_layer.weight)
            if param_id not in seen_params:
                lr_embed = opt_cfg.get("lr_embed", 5e-5)
                param_groups.append({
                    "params": [embed_layer.weight],
                    "lr": lr_embed,
                    "name": "embed_tokens"
                })
                seen_params.add(param_id)
                print(f"[Optimizer] embed_tokens: {embed_layer.weight.numel():,} parameters, lr={lr_embed:.2e}")
            else:
                print(f"[Optimizer] Warning: embed_tokens already in another group, skipping")

    # ========== COLLECT LM_HEAD (OPTIONAL) ==========
    if model_cfg.get("train_lm_head", False):
        lm_head = model.language_decoder.model.lm_head
        if lm_head is not None and lm_head.weight.requires_grad:
            param_id = id(lm_head.weight)
            if param_id not in seen_params:
                lr_lm_head = opt_cfg.get("lr_lm_head", 5e-5)
                param_groups.append({
                    "params": [lm_head.weight],
                    "lr": lr_lm_head,
                    "name": "lm_head"
                })
                seen_params.add(param_id)
                print(f"[Optimizer] lm_head: {lm_head.weight.numel():,} parameters, lr={lr_lm_head:.2e}")
            else:
                print(f"[Optimizer] Warning: lm_head already in another group, skipping")

    # ========== SANITY CHECK ==========
    if not param_groups:
        raise RuntimeError(
            "No trainable parameters found. Check freeze policy. "
            "Make sure LoRA was applied correctly and some parameters are trainable."
        )

    total_trainable_params = sum(
        sum(p.numel() for p in group["params"])
        for group in param_groups
    )
    print(f"[Optimizer] Total trainable parameters: {total_trainable_params:,}")
    print(f"[Optimizer] Number of parameter groups: {len(param_groups)}")

    # ========== CREATE OPTIMIZER ==========
    optimizer_type = opt_cfg.get("type", "AdamW")

    # Common optimizer kwargs
    optimizer_kwargs = {
        "weight_decay": opt_cfg.get("weight_decay", 0.01),
        "betas": tuple(opt_cfg.get("betas", [0.9, 0.999])),
        "eps": opt_cfg.get("eps", 1e-8),
    }

    if optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
        print(f"[Optimizer] Using AdamW optimizer")

    elif optimizer_type == "AdamW8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(param_groups, **optimizer_kwargs)
            print(f"[Optimizer] Using AdamW8bit optimizer (8-bit quantized)")
        except ImportError:
            print("[Optimizer] Warning: bitsandbytes not found, falling back to AdamW")
            optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    print(f"[Optimizer] Weight decay: {optimizer_kwargs['weight_decay']}")
    print(f"[Optimizer] Betas: {optimizer_kwargs['betas']}")
    print(f"[Optimizer] Epsilon: {optimizer_kwargs['eps']}")

    return optimizer
