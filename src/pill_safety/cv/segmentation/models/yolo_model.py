from __future__ import annotations

from ultralytics import YOLO


class SegmentationModel:
    def __init__(self, base_weights: str):
        self.base_weights = base_weights

    def build(self):
        return YOLO(self.base_weights)
