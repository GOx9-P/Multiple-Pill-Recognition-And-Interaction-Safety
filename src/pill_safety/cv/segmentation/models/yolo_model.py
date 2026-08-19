"""Wrapper YOLOv11-Seg dùng chung cho quá trình train và inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class RawSegmentationPrediction:
    """Kết quả thô từ model đã được đưa về hệ tọa độ ảnh gốc."""

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    mask: np.ndarray


class SegmentationModel:
    def __init__(self, base_weights: str):
        """Lưu đường dẫn weight và trì hoãn việc nạp model đến khi cần."""

        self.base_weights = base_weights
        self._model: Any | None = None

    def build(self):
        """Khởi tạo một lần đối tượng Ultralytics dùng cho train hoặc inference."""

        if self._model is None:
            # Import trễ giúp schema và unit test không phụ thuộc runtime nặng.
            from ultralytics import YOLO

            self._model = YOLO(self.base_weights)
        return self._model

    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        image_size: int,
        confidence_threshold: float,
        iou_threshold: float,
        mask_threshold: float,
        device: str | int | None,
    ) -> list[RawSegmentationPrediction]:
        """Chạy YOLO và trả bbox, confidence, mask theo tọa độ ảnh đầu vào."""

        model = self.build()
        results = model.predict(
            source=image_bgr,
            imgsz=image_size,
            conf=confidence_threshold,
            iou=iou_threshold,
            device=device,
            retina_masks=True,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or result.masks is None:
            return []

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        masks = result.masks.data.detach().cpu().numpy()
        if not (len(boxes) == len(confidences) == len(masks)):
            raise RuntimeError("YOLO returned inconsistent box, score and mask counts.")
        image_height, image_width = image_bgr.shape[:2]

        predictions: list[RawSegmentationPrediction] = []
        for bbox, confidence, mask in zip(boxes, confidences, masks):
            if mask.shape != (image_height, image_width):
                mask = cv2.resize(
                    mask,
                    (image_width, image_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            binary_mask = (mask >= mask_threshold).astype(np.uint8)
            predictions.append(
                RawSegmentationPrediction(
                    bbox_xyxy=tuple(float(value) for value in bbox),
                    confidence=float(confidence),
                    mask=binary_mask,
                )
            )
        return predictions
