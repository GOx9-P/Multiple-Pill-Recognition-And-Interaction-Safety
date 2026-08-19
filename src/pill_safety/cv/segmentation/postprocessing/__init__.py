from .instances import (
    ProcessedInstance,
    process_prediction,
    sort_instances_reading_order,
)
from .schema_mapper import build_segmentation_instance, build_segmentation_output

__all__ = [
    "ProcessedInstance",
    "build_segmentation_instance",
    "build_segmentation_output",
    "process_prediction",
    "sort_instances_reading_order",
]
"""Cung cấp API hậu xử lý và ánh xạ schema cho kết quả phân đoạn."""
