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
from numbers import Real
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

    # Head/last-block artifacts trong project đã từng dùng hai format. Inference
    # phải giữ thứ tự index đúng lúc train, thay vì tự sort nhãn theo tên.
    if "shape" in data and "color" in data:
        shape_names = data["shape"]
        color_names = data["color"]
    elif "shape_classification" in data and "color_multilabel" in data:
        shape_to_index = data["shape_classification"]
        color_block = data["color_multilabel"]
        color_names = color_block.get("labels")
        color_to_index = color_block.get("mapping")

        if not isinstance(shape_to_index, dict):
            raise ValueError(
                f"Malformed shape_classification mapping at {path}."
            )
        if not isinstance(color_names, list) or not isinstance(color_to_index, dict):
            raise ValueError(
                f"Malformed color_multilabel mapping at {path}."
            )

        def names_by_index(mapping: Dict, field_name: str) -> List[str]:
            indices = list(mapping.values())
            if (
                not all(isinstance(index, int) for index in indices)
                or sorted(indices) != list(range(len(indices)))
            ):
                raise ValueError(
                    f"{field_name} indices at {path} must be consecutive from 0."
                )
            return [
                name
                for name, _ in sorted(mapping.items(), key=lambda item: item[1])
            ]

        shape_names = names_by_index(shape_to_index, "shape_classification")
        indexed_color_names = names_by_index(color_to_index, "color_multilabel")
        if color_names != indexed_color_names:
            raise ValueError(
                f"color_multilabel labels and mapping disagree at {path}."
            )
    else:
        raise ValueError(
            f"Malformed label mapping at {path}. Expected either "
            "{'shape', 'color'} or {'shape_classification', 'color_multilabel'}."
        )

    if (
        not isinstance(shape_names, list)
        or not isinstance(color_names, list)
        or not all(isinstance(name, str) and name for name in shape_names + color_names)
    ):
        raise ValueError(f"Label names at {path} must be non-empty strings.")

    label_mapping = {"shape": shape_names, "color": color_names}
    num_shape_classes = len(shape_names)
    num_color_classes = len(color_names)
    mapping_hash = data.get("mapping_hash", "")

    return label_mapping, num_shape_classes, num_color_classes, mapping_hash


def load_color_threshold_values(path: Path, color_names: List[str]) -> List[float]:
    """Nạp threshold color theo đúng thứ tự index trong label mapping.

    Hỗ trợ cả artifact cũ dạng list và artifact run mới dạng
    ``{\"thresholds\": {\"WHITE\": 0.7, ...}}``. Hàm này không phụ thuộc
    PyTorch để có thể kiểm tra contract artifact trước khi khởi tạo model.
    """

    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, dict) and isinstance(raw.get("thresholds"), list):
        values = raw["thresholds"]
    elif isinstance(raw, dict) and isinstance(raw.get("thresholds"), dict):
        threshold_by_name = raw["thresholds"]
        missing_names = [name for name in color_names if name not in threshold_by_name]
        if missing_names:
            raise ValueError(
                "Color threshold mapping is missing labels from label_mapping.json: "
                + ", ".join(missing_names)
            )
        values = [threshold_by_name[name] for name in color_names]
    elif isinstance(raw, dict) and all(name in raw for name in color_names):
        values = [raw[name] for name in color_names]
    else:
        raise ValueError(
            "optimal_thresholds.json must be a list, contain a 'thresholds' "
            "list or map, or map every label in label_mapping.json to a threshold."
        )

    if len(values) != len(color_names):
        raise ValueError(
            "Color threshold count does not match label mapping: "
            f"{len(values)} != {len(color_names)}."
        )
    if not all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
        raise ValueError("Color thresholds must be numeric values.")

    normalized_values = [float(value) for value in values]
    if any(value < 0.0 or value > 1.0 for value in normalized_values):
        raise ValueError("Color thresholds must be in [0, 1].")
    return normalized_values
