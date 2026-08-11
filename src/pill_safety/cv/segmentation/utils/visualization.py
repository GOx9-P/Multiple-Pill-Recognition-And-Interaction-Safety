"""Vẽ ảnh minh họa kết quả segmentation để debug và audit."""

from __future__ import annotations

import cv2
import numpy as np

from ..postprocessing.instances import ProcessedInstance


def draw_segmentation_overlay(
    image_bgr: np.ndarray,
    instances: list[tuple[str, ProcessedInstance]],
) -> np.ndarray:
    """Vẽ mask, bbox, confidence và instance ID ổn định lên ảnh gốc."""

    overlay = image_bgr.copy()
    colors = (
        (0, 190, 255),
        (80, 200, 80),
        (255, 120, 80),
        (180, 90, 255),
        (255, 190, 70),
    )
    mask_layer = np.zeros_like(image_bgr)

    for index, (instance_id, instance) in enumerate(instances):
        color = colors[index % len(colors)]
        mask_layer[instance.mask > 0] = color
        x1, y1, x2, y2 = instance.bbox_xyxy
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label_y = max(y1 - 8, 18)
        cv2.putText(
            overlay,
            f"{instance_id} {instance.confidence:.2f}",
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if instances:
        overlay = cv2.addWeighted(overlay, 1.0, mask_layer, 0.35, 0.0)
    return overlay
