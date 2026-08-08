from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def draw_items(
    image: np.ndarray,
    items: list[dict[str, Any]],
    scoreline: dict[str, Any] | None = None,
) -> np.ndarray:
    canvas = image.copy()
    for index, item in enumerate(items, start=1):
        label = f"{index}: {item['text']} ({item['confidence']:.2f})"
        anchor = (10, 28 + index * 24)
        polygon = item.get("polygon")
        if polygon:
            points = np.asarray(polygon, dtype=np.int32)
            if points.ndim == 2 and points.shape[0] >= 2:
                cv2.polylines(canvas, [points], True, (0, 180, 255), 2)
                x_value = points[:, 0].min()
                y_value = points[:, 1].min()
                anchor = (int(x_value), max(18, int(y_value) - 8))
        cv2.putText(
            canvas,
            label,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if scoreline and scoreline.get("visible") and scoreline.get("line_xyxy"):
        x1, y1, x2, y2 = [
            int(round(value)) for value in scoreline["line_xyxy"]
        ]
        cv2.line(canvas, (x1, y1), (x2, y2), (255, 80, 0), 2, cv2.LINE_AA)
    return canvas
