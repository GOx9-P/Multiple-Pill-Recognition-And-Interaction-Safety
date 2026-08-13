from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pill_safety.cv.ocr.config import OCRConfig
from pill_safety.cv.ocr.engines import OCREngine
from pill_safety.cv.ocr.postprocessing.ordering import (
    item_center,
    sequence_confidence,
    smart_order_items,
)
from pill_safety.cv.ocr.preprocessing.image_ops import (
    PreparedImage,
    map_polygon_to_original,
)


def line_orientation(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, str]:
    angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
    if angle <= 30.0 or angle >= 150.0:
        orientation = "horizontal"
    elif 60.0 <= angle <= 120.0:
        orientation = "vertical"
    else:
        orientation = "oblique"
    return angle, orientation


def map_scoreline_to_original(
    scoreline: dict[str, Any],
    padded_shape: tuple[int, ...],
    rotation_degrees: int,
    prepared_image: PreparedImage,
) -> dict[str, Any]:
    """Dua hai dau mut scoreline tu variant xoay ve he toa do crop goc."""

    mapped = dict(scoreline)
    line = mapped.get("line_xyxy")
    if not mapped.get("visible") or line is None:
        return mapped
    points = map_polygon_to_original(
        [[float(line[0]), float(line[1])], [float(line[2]), float(line[3])]],
        padded_shape=padded_shape,
        rotation_degrees=rotation_degrees,
        pad_px=prepared_image.pad_px,
        original_height=prepared_image.original_height,
        original_width=prepared_image.original_width,
    )
    if points is None or len(points) != 2:
        raise ValueError(
            "Cannot map scoreline endpoints to original crop coordinates."
        )
    x1, y1 = points[0]
    x2, y2 = points[1]
    angle, orientation = line_orientation(x1, y1, x2, y2)
    mapped["line_xyxy"] = [x1, y1, x2, y2]
    mapped["angle_degrees"] = round(angle, 2)
    mapped["orientation"] = orientation
    return mapped


def point_to_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    dx, dy = x2 - x1, y2 - y1
    denominator = dx * dx + dy * dy
    if denominator <= 1e-6:
        return float(np.hypot(px - x1, py - y1))
    parameter = np.clip(
        ((px - x1) * dx + (py - y1) * dy) / denominator, 0.0, 1.0
    )
    return float(
        np.hypot(
            px - (x1 + parameter * dx),
            py - (y1 + parameter * dy),
        )
    )


def line_foreground_coverage(
    line_xyxy: list[float], foreground_mask: np.ndarray
) -> float:
    """Tinh ti le mau cua doan line nam trong foreground cua vien thuoc."""

    x1, y1, x2, y2 = [float(value) for value in line_xyxy]
    sample_count = max(2, int(np.hypot(x2 - x1, y2 - y1)) + 1)
    xs = np.rint(np.linspace(x1, x2, sample_count)).astype(np.int32)
    ys = np.rint(np.linspace(y1, y2, sample_count)).astype(np.int32)
    height, width = foreground_mask.shape[:2]
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if not np.any(valid):
        return 0.0
    return float(np.mean(foreground_mask[ys[valid], xs[valid]] > 0))


def detect_scoreline_for_split(
    image: np.ndarray,
    config: OCRConfig,
    foreground_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Tim scoreline bang Hough va loai line nam tren padding hoac nen crop."""

    height, width = image.shape[:2]
    if foreground_mask is not None and foreground_mask.shape[:2] != (height, width):
        raise ValueError("Foreground mask must match the OCR variant size.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 120)
    min_dimension = min(height, width)
    if config.scoreline_use_center_roi:
        roi = np.zeros_like(edges)
        cv2.ellipse(
            roi,
            (width // 2, height // 2),
            (max(1, int(width * 0.46)), max(1, int(height * 0.46))),
            0,
            0,
            360,
            255,
            -1,
        )
        edges = cv2.bitwise_and(edges, roi)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        max(18, int(min_dimension * 0.16)),
        minLineLength=max(24, int(min_dimension * 0.35)),
        maxLineGap=max(6, int(min_dimension * 0.08)),
    )
    best = None
    if lines is not None:
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = [float(value) for value in line]
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < max(24, int(min_dimension * 0.35)):
                continue
            center_distance = point_to_segment_distance(
                width / 2.0, height / 2.0, x1, y1, x2, y2
            )
            if (
                center_distance
                > config.scoreline_center_max_distance_ratio * min_dimension
            ):
                continue
            foreground_coverage = (
                line_foreground_coverage([x1, y1, x2, y2], foreground_mask)
                if foreground_mask is not None
                else 1.0
            )
            if foreground_coverage < config.scoreline_min_foreground_coverage:
                continue
            length_score = float(
                np.clip(length / max(0.80 * min_dimension, 1.0), 0.0, 1.0)
            )
            center_score = float(
                np.clip(
                    1.0
                    - center_distance
                    / max(
                        config.scoreline_center_max_distance_ratio * min_dimension,
                        1.0,
                    ),
                    0.0,
                    1.0,
                )
            )
            score = float(np.clip(0.60 * length_score + 0.40 * center_score, 0.0, 1.0))
            angle, orientation = line_orientation(x1, y1, x2, y2)
            if best is None or score > best["confidence"]:
                best = {
                    "visible": True,
                    "confidence": round(score, 4),
                    "line_xyxy": [x1, y1, x2, y2],
                    "angle_degrees": round(angle, 2),
                    "orientation": orientation,
                    "foreground_coverage": round(foreground_coverage, 4),
                }
    if best and best["confidence"] >= config.min_scoreline_detection_confidence:
        return best
    return {
        "visible": False,
        "confidence": 0.0,
        "line_xyxy": None,
        "angle_degrees": None,
        "orientation": "unknown",
    }


def create_line_mask(
    image_shape: tuple[int, ...],
    line_xyxy: list[float],
    keep_positive: bool = True,
) -> np.ndarray:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [float(value) for value in line_xyxy]
    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    cross = (x2 - x1) * (yy - y1) - (y2 - y1) * (xx - x1)
    return cross >= 0 if keep_positive else cross < 0


def apply_line_mask(
    image: np.ndarray,
    line_xyxy: list[float],
    keep_positive: bool = True,
    margin_px: int = 0,
) -> np.ndarray:
    height, width = image.shape[:2]
    fill = [
        int(value)
        for value in np.median(image.reshape(-1, image.shape[2]), axis=0)
    ]
    mask = create_line_mask((height, width), line_xyxy, keep_positive)
    if margin_px > 0:
        x1, y1, x2, y2 = [float(value) for value in line_xyxy]
        length = np.hypot(x2 - x1, y2 - y1)
        if length > 0:
            normal_x = -(y2 - y1) / length
            normal_y = (x2 - x1) / length
            xx, yy = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32),
            )
            distance = abs(normal_x * (xx - x1) + normal_y * (yy - y1))
            mask = mask & (distance > margin_px)
    result = image.copy()
    for channel in range(image.shape[2]):
        result[:, :, channel][~mask] = fill[channel]
    return result


def run_scoreline_side_split(
    variant: np.ndarray,
    rotation_degrees: int,
    step_id: str,
    split_dir: Path,
    json_dir: Path,
    scoreline: dict[str, Any],
    engine: OCREngine,
    config: OCRConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    line = scoreline.get("line_xyxy")
    if line is None:
        return [], {"confidence": 0.0, "reliable": False}

    margin_px = int(min(variant.shape[:2]) * config.split_margin_ratio)
    side_results = {}
    for side_name, keep_positive in [("side_a", True), ("side_b", False)]:
        masked = apply_line_mask(
            variant, line, keep_positive=keep_positive, margin_px=margin_px
        )
        crop_path = split_dir / f"{step_id}_{side_name}.jpg"
        cv2.imwrite(str(crop_path), masked)
        items = engine.predict(crop_path, json_dir, f"{step_id}_{side_name}")
        items = [item for item in items if item.get("text")]
        for item in items:
            item["center_x"], item["center_y"] = item_center(item)
        side_results[side_name] = items

    centroids = {
        name: (
            (
                np.mean([item.get("center_x") for item in items]),
                np.mean([item.get("center_y") for item in items]),
            )
            if items
            else (1e9, 1e9)
        )
        for name, items in side_results.items()
    }
    x1, y1, x2, y2 = [float(value) for value in line]
    length = max(float(np.hypot(x2 - x1, y2 - y1)), 1e-6)
    normal_x = -(y2 - y1) / length
    normal_y = (x2 - x1) / length
    if (abs(normal_x) >= abs(normal_y) and normal_x < 0) or (
        abs(normal_x) < abs(normal_y) and normal_y < 0
    ):
        normal_x, normal_y = -normal_x, -normal_y
    midpoint = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    side_projection = {
        name: normal_x * (centroids[name][0] - midpoint[0])
        + normal_y * (centroids[name][1] - midpoint[1])
        for name in side_results
    }
    ordered_sides = sorted(side_results, key=lambda name: side_projection[name])
    split_items = []
    for side_name in ordered_sides:
        split_items.extend(
            smart_order_items(side_results.get(side_name, []), rotation_degrees)
        )
    side_confidence = {
        name: sequence_confidence(items) for name, items in side_results.items()
    }
    split_info = {
        "confidence": scoreline.get("confidence", 0.0),
        "reliable": all(len(side_results[name]) > 0 for name in side_results)
        and min(side_confidence.values()) >= config.min_side_confidence,
        "orientation": scoreline.get("orientation", "unknown"),
        "ordered_sides": ordered_sides,
    }
    return split_items, split_info
