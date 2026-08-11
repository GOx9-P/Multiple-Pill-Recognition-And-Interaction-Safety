"""Cấu hình runtime dành riêng cho inference segmentation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SegmentationConfig:
    """Tập hợp cấu hình inference và quality gate của Module 1."""

    weights_path: Path = Path(
        "models/segmentation_yolov11_full_finetune/"
        "yolov11m_seg_mediseg_full_finetune_v1.pt"
    )
    device: str = "auto"
    image_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.60
    mask_threshold: float = 0.50

    bbox_padding_ratio: float = 0.08
    crop_size: int = 640
    crop_background_value: int = 127
    align_long_axis: bool = True
    min_alignment_aspect_ratio: float = 1.15

    min_mask_area_ratio: float = 0.0005
    max_mask_area_ratio: float = 0.90
    min_component_area_ratio: float = 0.05
    merged_solidity_threshold: float = 0.82
    non_pill_confidence_threshold: float = 0.45

    blur_variance_reference: float = 120.0
    blur_warning_score: float = 0.35
    unusable_blur_score: float = 0.85
    glare_value_threshold: int = 245
    glare_saturation_threshold: int = 45
    glare_ratio_threshold: float = 0.02
    dark_value_threshold: int = 35
    bright_value_threshold: int = 245
    lighting_ratio_threshold: float = 0.35
    unusable_lighting_ratio: float = 0.75

    output_dir: Path = Path("outputs")
    save_overlay: bool = True

    def __post_init__(self) -> None:
        """Kiểm tra cấu hình ngay khi khởi tạo để phát hiện giá trị không hợp lệ."""

        unit_interval_fields = (
            "confidence_threshold",
            "iou_threshold",
            "mask_threshold",
            "bbox_padding_ratio",
            "min_mask_area_ratio",
            "max_mask_area_ratio",
            "min_component_area_ratio",
            "merged_solidity_threshold",
            "non_pill_confidence_threshold",
            "blur_warning_score",
            "unusable_blur_score",
            "glare_ratio_threshold",
            "lighting_ratio_threshold",
            "unusable_lighting_ratio",
        )
        for field_name in unit_interval_fields:
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1], got {value}.")
        if self.image_size <= 0 or self.crop_size <= 0:
            raise ValueError("image_size and crop_size must be positive.")
        if self.blur_variance_reference <= 0:
            raise ValueError("blur_variance_reference must be positive.")
        if self.min_alignment_aspect_ratio < 1.0:
            raise ValueError("min_alignment_aspect_ratio must be at least 1.0.")
        if not 0 <= self.crop_background_value <= 255:
            raise ValueError("crop_background_value must be in [0, 255].")
        pixel_threshold_fields = (
            "glare_value_threshold",
            "glare_saturation_threshold",
            "dark_value_threshold",
            "bright_value_threshold",
        )
        for field_name in pixel_threshold_fields:
            value = int(getattr(self, field_name))
            if not 0 <= value <= 255:
                raise ValueError(f"{field_name} must be in [0, 255], got {value}.")
        if self.min_mask_area_ratio >= self.max_mask_area_ratio:
            raise ValueError("min_mask_area_ratio must be below max_mask_area_ratio.")
        if self.blur_warning_score >= self.unusable_blur_score:
            raise ValueError("blur warning threshold must be below unusable threshold.")

    def with_output_dir(self, output_dir: str | Path) -> "SegmentationConfig":
        """Trả về config mới với thư mục output được thay thế."""

        return replace(self, output_dir=Path(output_dir))

    def with_weights_path(self, weights_path: str | Path) -> "SegmentationConfig":
        """Trả về config mới với đường dẫn checkpoint được thay thế."""

        return replace(self, weights_path=Path(weights_path))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SegmentationConfig":
        """Đọc YAML theo từng nhóm và giữ giá trị mặc định khi trường bị thiếu."""

        with Path(path).open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}

        defaults = cls()
        model = raw.get("model", {})
        post = raw.get("postprocessing", {})
        quality = raw.get("quality", {})
        artifacts = raw.get("artifacts", {})

        values: dict[str, Any] = {
            "weights_path": Path(
                model.get("weights_path", defaults.weights_path)
            ),
            "device": model.get("device", defaults.device),
            "image_size": model.get("image_size", defaults.image_size),
            "confidence_threshold": model.get(
                "confidence_threshold", defaults.confidence_threshold
            ),
            "iou_threshold": model.get(
                "iou_threshold", defaults.iou_threshold
            ),
            "mask_threshold": model.get(
                "mask_threshold", defaults.mask_threshold
            ),
            "bbox_padding_ratio": post.get(
                "bbox_padding_ratio", defaults.bbox_padding_ratio
            ),
            "crop_size": post.get("crop_size", defaults.crop_size),
            "crop_background_value": post.get(
                "crop_background_value", defaults.crop_background_value
            ),
            "align_long_axis": post.get(
                "align_long_axis", defaults.align_long_axis
            ),
            "min_alignment_aspect_ratio": post.get(
                "min_alignment_aspect_ratio",
                defaults.min_alignment_aspect_ratio,
            ),
            "min_mask_area_ratio": quality.get(
                "min_mask_area_ratio", defaults.min_mask_area_ratio
            ),
            "max_mask_area_ratio": quality.get(
                "max_mask_area_ratio", defaults.max_mask_area_ratio
            ),
            "min_component_area_ratio": quality.get(
                "min_component_area_ratio",
                defaults.min_component_area_ratio,
            ),
            "merged_solidity_threshold": quality.get(
                "merged_solidity_threshold",
                defaults.merged_solidity_threshold,
            ),
            "non_pill_confidence_threshold": quality.get(
                "non_pill_confidence_threshold",
                defaults.non_pill_confidence_threshold,
            ),
            "blur_variance_reference": quality.get(
                "blur_variance_reference", defaults.blur_variance_reference
            ),
            "blur_warning_score": quality.get(
                "blur_warning_score", defaults.blur_warning_score
            ),
            "unusable_blur_score": quality.get(
                "unusable_blur_score", defaults.unusable_blur_score
            ),
            "glare_value_threshold": quality.get(
                "glare_value_threshold", defaults.glare_value_threshold
            ),
            "glare_saturation_threshold": quality.get(
                "glare_saturation_threshold",
                defaults.glare_saturation_threshold,
            ),
            "glare_ratio_threshold": quality.get(
                "glare_ratio_threshold", defaults.glare_ratio_threshold
            ),
            "dark_value_threshold": quality.get(
                "dark_value_threshold", defaults.dark_value_threshold
            ),
            "bright_value_threshold": quality.get(
                "bright_value_threshold", defaults.bright_value_threshold
            ),
            "lighting_ratio_threshold": quality.get(
                "lighting_ratio_threshold",
                defaults.lighting_ratio_threshold,
            ),
            "unusable_lighting_ratio": quality.get(
                "unusable_lighting_ratio",
                defaults.unusable_lighting_ratio,
            ),
            "output_dir": Path(
                artifacts.get("output_dir", defaults.output_dir)
            ),
            "save_overlay": artifacts.get(
                "save_overlay", defaults.save_overlay
            ),
        }
        return cls(**values)
