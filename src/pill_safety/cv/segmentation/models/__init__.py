"""Cung cấp lớp bọc model và kiểu dự đoán phân đoạn đã chuẩn hóa."""

from .yolo_model import RawSegmentationPrediction, SegmentationModel

__all__ = ["RawSegmentationPrediction", "SegmentationModel"]
