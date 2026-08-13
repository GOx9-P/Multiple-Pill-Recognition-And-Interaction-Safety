from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pill_safety.cv.ocr.config import OCRConfig


def _polygon_mask(
    polygon: list[list[float]] | None,
    image_shape: tuple[int, ...],
) -> np.ndarray | None:
    """Raster hoa polygon PaddleOCR de so sanh truc tiep voi mask vien."""

    if not polygon:
        return None
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return None
    height, width = image_shape[:2]
    rounded = np.rint(points[:, :2]).astype(np.int32)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
    region_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(region_mask, [rounded], 1)
    return region_mask


def filter_text_regions(
    items: list[dict[str, Any]],
    foreground_mask: np.ndarray | None,
    config: OCRConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bo text region nam ngoai vien hoac co dien tich qua lon bat thuong."""

    if foreground_mask is None:
        return list(items), []

    foreground = foreground_mask > 0
    foreground_area = int(foreground.sum())
    if foreground_area == 0:
        return [], [dict(item, rejection_reason="empty_foreground_mask") for item in items]

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for item in items:
        region_mask = _polygon_mask(item.get("polygon"), foreground.shape)
        if region_mask is None:
            rejected.append(dict(item, rejection_reason="invalid_text_polygon"))
            continue
        ys, xs = np.nonzero(region_mask)
        region_area = int(region_mask.sum())
        if region_area == 0 or len(xs) == 0:
            rejected.append(dict(item, rejection_reason="empty_text_region"))
            continue

        foreground_coverage = float(foreground[region_mask > 0].mean())
        region_area_ratio = region_area / foreground_area
        if foreground_coverage < config.min_text_region_foreground_coverage:
            rejected.append(dict(item, rejection_reason="outside_pill_mask"))
            continue
        if region_area_ratio > config.max_text_region_area_ratio:
            rejected.append(dict(item, rejection_reason="text_region_too_large"))
            continue
        accepted.append(item)

    return accepted, rejected
