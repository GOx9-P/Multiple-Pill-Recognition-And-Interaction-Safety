"""Chuyen logits shape/color thanh attribute JSON theo label mapping hien tai."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class AttributeFormatter:
    """Giu thu tu label mapping de threshold va output khong bi lech class."""

    def __init__(self, label_mapping_path: str | Path):
        """Nap mapping moi shape_classification/color_multilabel hoac mapping cu."""
        path = Path(label_mapping_path)
        if not path.is_file():
            raise FileNotFoundError(f"Label mapping not found: {path}")
        mapping = json.loads(path.read_text(encoding="utf-8"))
        if "shape_classification" in mapping:
            self.shape_labels = [label for label, _ in sorted(mapping["shape_classification"].items(), key=lambda item: item[1])]
            self.color_labels = list(mapping["color_multilabel"]["labels"])
        else:
            self.shape_labels = [mapping["shape"][str(index)] for index in range(len(mapping["shape"]))]
            self.color_labels = list(mapping["color"])

    def format_output(self, shape_logits: np.ndarray, color_logits: np.ndarray, color_threshold: float | np.ndarray = 0.5) -> dict:
        """Tra shape top-1 va tat ca mau vuot qua threshold scalar hoac per-color."""
        shape_logits = np.asarray(shape_logits, dtype=np.float64)
        shape_probabilities = np.exp(shape_logits - shape_logits.max())
        shape_probabilities /= shape_probabilities.sum()
        shape_index = int(shape_probabilities.argmax())
        color_probabilities = 1.0 / (1.0 + np.exp(-np.asarray(color_logits, dtype=np.float64)))
        thresholds = np.full(len(self.color_labels), float(color_threshold)) if np.isscalar(color_threshold) else np.asarray(color_threshold, dtype=np.float64)
        if thresholds.shape != color_probabilities.shape:
            raise ValueError("Color thresholds must have the same length as color logits.")
        colors = [
            {"label": label, "confidence": float(probability)}
            for label, probability, threshold in zip(self.color_labels, color_probabilities, thresholds)
            if probability >= threshold
        ]
        colors.sort(key=lambda item: item["confidence"], reverse=True)
        return {
            "shape": {"label": self.shape_labels[shape_index], "confidence": float(shape_probabilities[shape_index])},
            "color": {"labels": colors},
        }
