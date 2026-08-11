"""
Shared metrics utilities for trainer and evaluator.

Provides a single consistent implementation of F1 computation
to avoid divergence between training metrics and evaluation metrics.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
from typing import Dict, List, Optional, Tuple


def compute_epoch_metrics(
    shape_targets: np.ndarray,
    shape_preds: np.ndarray,
    color_targets: np.ndarray,
    color_preds: np.ndarray,
    shape_class_names: Optional[List[str]] = None,
    color_class_names: Optional[List[str]] = None,
    zero_division: int = 0,
) -> Dict:
    """Compute all metrics for one epoch (used by both trainer and evaluator).

    Args:
        shape_targets: Ground truth shape labels, shape (N,).
        shape_preds: Predicted shape labels, shape (N,).
        color_targets: Ground truth color labels, shape (N, C).
        color_preds: Predicted color labels (binary), shape (N, C).
        shape_class_names: Optional list of shape class names for per-class metrics.
        color_class_names: Optional list of color class names for per-class metrics.
        zero_division: Value to use when a class has no predictions.

    Returns:
        Dictionary with shape_macro_f1, color_macro_f1, overall_macro_f1,
        and optionally per-class metrics.
    """
    shape_macro_f1 = f1_score(
        shape_targets, shape_preds,
        average="macro", zero_division=zero_division,
    )
    color_macro_f1 = f1_score(
        color_targets, color_preds,
        average="macro", zero_division=zero_division,
    )
    overall_macro_f1 = (shape_macro_f1 + color_macro_f1) / 2.0

    result = {
        "shape_macro_f1": float(shape_macro_f1),
        "color_macro_f1": float(color_macro_f1),
        "overall_macro_f1": float(overall_macro_f1),
    }

    # Per-class metrics if class names provided
    if shape_class_names:
        prec, rec, f1, sup = precision_recall_fscore_support(
            shape_targets, shape_preds,
            labels=list(range(len(shape_class_names))),
            zero_division=zero_division,
        )
        result["shape_per_class"] = {
            name: {"precision": float(prec[i]), "recall": float(rec[i]),
                   "f1": float(f1[i]), "support": int(sup[i])}
            for i, name in enumerate(shape_class_names)
        }

    if color_class_names:
        prec, rec, f1, sup = precision_recall_fscore_support(
            color_targets, color_preds,
            average=None, zero_division=zero_division,
        )
        result["color_per_class"] = {
            name: {"precision": float(prec[i]), "recall": float(rec[i]),
                   "f1": float(f1[i]), "support": int(sup[i])}
            for i, name in enumerate(color_class_names)
        }

    return result


def compute_confidence(logits, task: str = "shape"):
    """Compute confidence scores from logits.

    Args:
        logits: Raw model output, shape (B, C).
        task: "shape" (softmax) or "color" (sigmoid).

    Returns:
        Confidence array, shape (B,) for shape, (B, C) for color.
    """
    import torch

    if task == "shape":
        probs = torch.softmax(logits, dim=-1)
        conf, _ = probs.max(dim=-1)
        return conf.cpu().numpy()
    else:
        probs = torch.sigmoid(logits)
        return probs.cpu().numpy()