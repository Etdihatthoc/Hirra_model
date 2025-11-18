import math
from typing import Dict


def summarize_validation(loss_value: float) -> Dict[str, float]:
    if loss_value is None or math.isnan(loss_value):
        return {"loss": float("inf"), "perplexity": float("inf")}

    loss_clamped = min(50.0, max(1e-5, loss_value))
    return {
        "loss": loss_value,
        "perplexity": math.exp(loss_clamped)
    }
