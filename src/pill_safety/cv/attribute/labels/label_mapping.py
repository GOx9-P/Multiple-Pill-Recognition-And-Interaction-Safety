"""
Label mapping utilities for pill attribute recognition.

Handles building shape/color class name mappings, removing rare color classes,
and saving/loading label mapping files.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def build_label_mapping(
    train_dataset,
    num_shape_classes: int,
    num_color_classes: int,
) -> Tuple[Dict, List[str], List[str]]:
    """Build label mapping from a fitted training dataset.

    Determines human-readable class names for shape and color labels
    by inspecting the dataset's encoder or DataFrame columns.

    Args:
        train_dataset: A fitted ``RxImageDataset`` instance (training split).
        num_shape_classes: Total number of shape classes.
        num_color_classes: Total number of color classes.

    Returns:
        Tuple of (label_mapping dict, shape_class_names list, color_class_names list).
    """
    shape_encoder_dict = getattr(train_dataset, "shape_encoder_dict", None)
    shape_encoder = getattr(train_dataset, "shape_encoder", None)

    if shape_encoder_dict is not None:
        shape_class_names = [
            shape_encoder_dict.get(i, f"shape_class_{i}")
            for i in range(num_shape_classes)
        ]
    elif shape_encoder is not None and hasattr(shape_encoder, "classes_"):
        shape_class_names = list(shape_encoder.classes_)
    elif "shape" in train_dataset.df.columns:
        shape_class_names = sorted(
            train_dataset.df["shape"].dropna().unique().tolist()
        )
    else:
        shape_class_names = [
            f"shape_class_{i}" for i in range(num_shape_classes)
        ]

    color_class_names = list(train_dataset.color_cols)

    label_mapping = {
        "shape": {int(i): name for i, name in enumerate(shape_class_names)},
        "color": color_class_names,
    }

    return label_mapping, shape_class_names, color_class_names


def get_shape_distribution(
    train_dataset, shape_class_names: List[str]
) -> Dict[str, int]:
    """Compute per-class sample counts for shape labels.

    Args:
        train_dataset: A fitted ``RxImageDataset`` instance.
        shape_class_names: Ordered list of shape class names.

    Returns:
        Dictionary mapping shape name to sample count.
    """
    shape_dist = dict(Counter(train_dataset.shape_labels.tolist()))
    return {
        shape_class_names[int(k)]: v for k, v in shape_dist.items()
    }


def remove_rare_color_classes(
    datasets: list,
    rare_columns: Optional[List[str]] = None,
    min_samples: int = 3,
) -> int:
    """Remove color classes with too few training samples from all datasets.

    Modifies datasets **in-place** by removing the specified columns from
    ``color_cols`` and ``color_labels``.

    Args:
        datasets: List of ``RxImageDataset`` instances to modify.
        rare_columns: Explicit list of color column names to remove.
            If ``None``, auto-detects columns with fewer than ``min_samples``
            in the first dataset.
        min_samples: Threshold below which a color class is considered rare.
            Only used when ``rare_columns`` is ``None``.

    Returns:
        Updated number of color classes.
    """
    if not datasets:
        return 0

    train_ds = datasets[0]

    # Auto-detect rare columns if not explicitly given
    if rare_columns is None:
        rare_columns = []
        for i, col in enumerate(train_ds.color_cols):
            if train_ds.color_labels[:, i].sum() < min_samples:
                rare_columns.append(col)

    if not rare_columns:
        return len(train_ds.color_cols)

    for ds in datasets:
        for col_name in rare_columns:
            if col_name in ds.color_cols:
                idx = ds.color_cols.index(col_name)
                ds.color_cols = [
                    c for c in ds.color_cols if c != col_name
                ]
                ds.color_labels = np.delete(ds.color_labels, idx, axis=1)

    removed_str = ", ".join(rare_columns)
    print(
        f"  [INFO] Removed rare color classes: {removed_str}. "
        f"Remaining: {len(datasets[0].color_cols)}"
    )
    return len(datasets[0].color_cols)


def save_label_mapping(
    label_mapping: Dict, path: Path
) -> None:
    """Save label mapping to a JSON file.

    Args:
        label_mapping: Dictionary with ``"shape"`` and ``"color"`` keys.
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
