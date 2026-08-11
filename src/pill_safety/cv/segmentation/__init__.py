"""Cung cấp API công khai cho luồng phân đoạn viên thuốc."""

from .config import SegmentationConfig
from .predictors import SegmentationArtifacts, SegmentationPredictor

__all__ = [
    "SegmentationArtifacts",
    "SegmentationConfig",
    "SegmentationPredictor",
]
