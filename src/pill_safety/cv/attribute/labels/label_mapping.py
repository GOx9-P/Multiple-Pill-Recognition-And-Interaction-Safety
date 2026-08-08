"""
Label mapping utilities for pill attribute recognition.

Handles building shape/color class name mappings and saving/loading
label mapping files. This is the **Single Source of Truth** for
class names and indices.

Key design decisions:
    - NO rare class removal. All classes are kept (augmentation handles imbalance).
    - Shape mapping uses REAL names (e.g. "CAPSULE"), never fake names like "shape_class_0".
    - Format: {"shape": ["CAPSULE", "ROUND", ...], "color": ["color_BLUE", ...]}
    - Head-tune CREATES the mapping; Last-blocks LOADS it unchanged.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def build_label_mapping(
    train_dataset,
    num_shape_classes: int,
    num_color_classes: int,
) -> Tuple[Dict, List[str], List[str]]:
    """Build label mapping from a fitted training dataset.

    Determines human-readable class names for shape and color labels.
    All classes are kept — no rare class removal.

    Args:
        train_dataset: A fitted ``RxImageDataset`` instance (training split).
        num_shape_classes: Total number of shape classes.
        num_color_classes: Total number of color classes.

    Returns:
        Tuple of (label_mapping dict, shape_class_names list, color_class_names list).
    """
    # --- Shape names ---
    shape_encoder_dict = getattr(train_dataset, "shape_encoder_dict", None)
    shape_encoder = getattr(train_dataset, "shape_encoder", None)

    if shape_encoder_dict is not None:
        shape_class_names = [
            shape_encoder_dict.get(i, f"UNKNOWN_SHAPE_{i}")
            for i in range(num_shape_classes)
        ]
    elif shape_encoder is not None and hasattr(shape_encoder, "classes_"):
        shape_class_names = list(shape_encoder.classes_)
    elif hasattr(train_dataset, "df") and "shape" in train_dataset.df.columns:
        shape_class_names = sorted(
            train_dataset.df["shape"].dropna().unique().tolist()
        )
    else:
        # Last resort — should ideally never happen
        shape_class_names = [
            f"shape_class_{i}" for i in range(num_shape_classes)
        ]

    # --- Color names ---
    color_class_names = list(train_dataset.color_cols)

    # --- Build mapping (list-based, deterministic order) ---
    label_mapping = {
        "shape": shape_class_names,
        "color": color_class_names,
    }

    return label_mapping, shape_class_names, color_class_names


def get_shape_distribution(
    train_dataset, shape_class_names: List[str]
) -> Dict[str, int]:
    """Compute per-class sample counts for shape labels."""
    from collections import Counter
    shape_dist = dict(Counter(train_dataset.shape_labels.tolist()))
    return {
        shape_class_names[int(k)]: v for k, v in shape_dist.items()
    }


def get_color_distribution(
    train_dataset, color_class_names: List[str]
) -> Dict[str, int]:
    """Compute per-class sample counts for color labels."""
    color_sums = train_dataset.color_labels.sum(axis=0)
    return {
        name: int(color_sums[i])
        for i, name in enumerate(color_class_names)
    }


def save_label_mapping(label_mapping: Dict, path: Path) -> str:
    """Save label mapping to a JSON file and return its SHA-256 hash.

    Args:
        label_mapping: Dictionary with ``"shape"`` and ``"color"`` keys (list values).
        path: Output file path.

    Returns:
        SHA-256 hex digest of the serialized mapping.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(label_mapping, sort_keys=True, ensure_ascii=False)
    mapping_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    output = {
        **label_mapping,
        "mapping_hash": mapping_hash,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return mapping_hash


def load_label_mapping(path: Path) -> Tuple[Dict, int, int, str]:
    """Load label mapping from a JSON file.

    Args:
        path: Path to the label_mapping.json file.

    Returns:
        Tuple of (label_mapping dict, num_shape_classes, num_color_classes, mapping_hash).

    Raises:
        FileNotFoundError: If the mapping file does not exist.
        ValueError: If the mapping file is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FATAL: Label mapping file not found: {path}\n"
            "Last-blocks MUST use the label mapping created by head-tune."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "shape" not in data or "color" not in data:
        raise ValueError(
            f"Malformed label mapping at {path}. "
            "Expected keys: 'shape', 'color'."
        )

    label_mapping = {
        "shape": data["shape"],
        "color": data["color"],
    }

    num_shape_classes = len(data["shape"])
    num_color_classes = len(data["color"])
    mapping_hash = data.get("mapping_hash", "")

    return label_mapping, num_shape_classes, num_color_classes, mapping_hash
