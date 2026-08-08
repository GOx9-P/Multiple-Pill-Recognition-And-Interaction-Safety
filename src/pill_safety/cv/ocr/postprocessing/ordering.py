from __future__ import annotations

from typing import Any

import numpy as np

from pill_safety.cv.ocr.config import OCRConfig


def item_center(item: dict[str, Any]) -> tuple[float, float]:
    polygon = item.get("polygon")
    if polygon:
        points = np.asarray(polygon, dtype=np.float32)
        if points.ndim == 2 and points.shape[1] >= 2:
            return float(points[:, 0].mean()), float(points[:, 1].mean())
    return float(item.get("center_x", 0)), float(item.get("center_y", 0))


def median_text_height(items: list[dict[str, Any]]) -> float:
    heights = []
    for item in items:
        polygon = item.get("polygon")
        if polygon:
            points = np.asarray(polygon, dtype=np.float32)
            if points.ndim == 2 and points.shape[0] >= 2:
                height = float(points[:, 1].max() - points[:, 1].min())
                if height > 0:
                    heights.append(height)
    return float(np.median(heights)) if heights else 20.0


def reading_order(
    items: list[dict[str, Any]],
    rotation_degrees: int = 0,
    line_group_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    # Coordinates already belong to the rotated image. No 180-degree reversal.
    del rotation_degrees
    if not items:
        return items
    centers = [item_center(item) for item in items]
    threshold = median_text_height(items) * line_group_ratio
    indexed = sorted(zip(centers, items), key=lambda pair: pair[0][1])
    lines = [[indexed[0]]]
    for index in range(1, len(indexed)):
        if abs(indexed[index][0][1] - lines[-1][-1][0][1]) <= threshold:
            lines[-1].append(indexed[index])
        else:
            lines.append([indexed[index]])
    result = []
    for line in lines:
        line.sort(key=lambda pair: pair[0][0])
        result.extend(item for _, item in line)
    return result


def smart_order_items(
    items: list[dict[str, Any]], rotation_degrees: int = 0
) -> list[dict[str, Any]]:
    if len(items) <= 1:
        return list(items)
    return reading_order(items, rotation_degrees)


def candidate_text(items: list[dict[str, Any]]) -> str:
    return " ".join(
        str(item.get("text", "")).strip()
        for item in items
        if str(item.get("text", "")).strip()
    )


def sequence_confidence(items: list[dict[str, Any]]) -> float:
    scores = [float(item.get("confidence", 0.0)) for item in items]
    if not scores:
        return 0.0
    return float(np.clip(0.5 * np.mean(scores) + 0.5 * np.min(scores), 0.0, 1.0))


def circular_order_candidates(
    items: list[dict[str, Any]], image_shape: tuple[int, ...], config: OCRConfig
) -> list[dict[str, Any]]:
    if (
        not config.enable_circular_text_order
        or len(items) < config.min_circular_boxes
    ):
        return []
    height, width = image_shape[:2]
    center_x, center_y = width / 2.0, height / 2.0
    centers = np.asarray([item_center(item) for item in items], dtype=np.float32)
    radii = np.hypot(centers[:, 0] - center_x, centers[:, 1] - center_y)
    mean_radius = float(np.mean(radii))
    if mean_radius < 0.12 * min(height, width):
        return []
    radial_cv = float(np.std(radii) / max(mean_radius, 1e-6))
    angles = np.arctan2(centers[:, 1] - center_y, centers[:, 0] - center_x)
    sorted_angles = np.sort((angles + 2 * np.pi) % (2 * np.pi))
    gaps = np.diff(np.r_[sorted_angles, sorted_angles[0] + 2 * np.pi])
    angular_coverage = float(2 * np.pi - np.max(gaps))
    if radial_cv > 0.45 or angular_coverage < np.deg2rad(70):
        return []

    clockwise = [items[index] for index in np.argsort(angles)]
    sequences = []
    seen = set()
    directions = [
        ("circular_cw", clockwise),
        ("circular_ccw", list(reversed(clockwise))),
    ]
    for direction, ordered in directions:
        for shift in range(len(ordered)):
            shifted = ordered[shift:] + ordered[:shift]
            text = candidate_text(shifted)
            key = (direction, text)
            if text and key not in seen:
                seen.add(key)
                sequences.append(
                    {"ordering": direction, "text": text, "items": shifted}
                )
    return sequences


def build_order_candidates(
    items: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    rotation_degrees: int,
    config: OCRConfig,
) -> list[dict[str, Any]]:
    linear = smart_order_items(items, rotation_degrees)
    candidates = [
        {"ordering": "linear", "text": candidate_text(linear), "items": linear}
    ]
    candidates.extend(circular_order_candidates(items, image_shape, config))
    return [candidate for candidate in candidates if candidate["text"]]
