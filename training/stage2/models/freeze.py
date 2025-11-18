from typing import Dict, Any


def set_requires_grad(module, requires_grad: bool):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_module(module):
    set_requires_grad(module, False)


def apply_freeze_policies(model, config: Dict[str, Any]):
    model_cfg = config["model"]

    if model_cfg.get("freeze_vision_encoder", True):
        freeze_module(model.vision_encoder)

    extractor = getattr(model, "feature_extractor", None)
    projector = getattr(model, "visual_projector", None)

    if not model_cfg.get("train_fpn", False) and extractor is not None:
        freeze_module(getattr(extractor, "fpn", None))
        freeze_module(getattr(extractor, "fpn_aggregator", None))

    if model_cfg.get("freeze_roi_modules", True) and extractor is not None:
        freeze_module(getattr(extractor, "roi_extractor", None))
        freeze_module(getattr(extractor, "graph_builder", None))
        freeze_module(getattr(extractor, "graph_reasoner", None))
        freeze_module(getattr(extractor, "graph_summary_proj", None))
        if projector is not None and hasattr(projector, "roi_projector"):
            freeze_module(projector.roi_projector)

    if model_cfg.get("freeze_language_decoder", True):
        freeze_module(model.language_decoder)
        model.language_decoder.model.eval()

    # Q-Former + global projector phải train.
    if extractor is not None:
        set_requires_grad(getattr(extractor, "global_extractor", None), True)
        if model_cfg.get("train_fpn", False):
            set_requires_grad(getattr(extractor, "fpn", None), True)
            set_requires_grad(getattr(extractor, "fpn_aggregator", None), True)

    if projector is not None and hasattr(projector, "global_projector"):
        set_requires_grad(projector.global_projector, True)


def count_parameters(model) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": 100.0 * trainable / total if total > 0 else 0.0,
    }
