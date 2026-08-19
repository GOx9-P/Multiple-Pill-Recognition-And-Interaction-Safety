"""Đánh giá blur, glare và ánh sáng dùng chung cho toàn bộ CV pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from ....schemas import ImageQuality


class ImageQualitySettings(Protocol):
    """Mô tả tối thiểu các cấu hình mà image-quality gate cần sử dụng."""

    blur_variance_reference: float
    blur_warning_score: float
    unusable_blur_score: float
    glare_value_threshold: int
    glare_saturation_threshold: int
    glare_ratio_threshold: float
    dark_value_threshold: int
    bright_value_threshold: int
    lighting_ratio_threshold: float
    unusable_lighting_ratio: float


@dataclass(frozen=True)
class ImageQualityAssessment:
    """Gói kết quả schema và các cảnh báo dùng lại cho từng instance."""

    output: ImageQuality
    quality_flags: tuple[str, ...]


def assess_image_quality(
    image_bgr: np.ndarray,
    config: ImageQualitySettings,
) -> ImageQualityAssessment:
    """Ước lượng blur, glare và ánh sáng mà không suy luận danh tính thuốc.

    ``blur_score`` biểu diễn mức độ lỗi: 0 là đủ nét so với mốc cấu hình, còn
    1 là mờ nghiêm trọng. Glare và ánh sáng được tính bằng tỷ lệ pixel trên ảnh
    để threshold có thể quan sát và tinh chỉnh trên tập validation.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reference = max(float(config.blur_variance_reference), 1e-6)
    blur_score = float(np.clip(1.0 - laplacian_variance / reference, 0.0, 1.0))

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    glare_mask = (
        (value >= config.glare_value_threshold)
        & (saturation <= config.glare_saturation_threshold)
    )
    glare_ratio = float(np.mean(glare_mask))
    glare_detected = glare_ratio >= config.glare_ratio_threshold

    dark_ratio = float(np.mean(value <= config.dark_value_threshold))
    bright_ratio = float(np.mean(value >= config.bright_value_threshold))
    lighting_ratio = max(dark_ratio, bright_ratio)
    lighting_warning = lighting_ratio >= config.lighting_ratio_threshold

    flags: list[str] = []
    if blur_score >= config.blur_warning_score:
        flags.append(
            "blur_warning"
            if blur_score < config.unusable_blur_score
            else "severe_blur"
        )
    if glare_detected:
        flags.append("minor_glare")
    if lighting_warning:
        flags.append("lighting_warning")

    if (
        blur_score >= config.unusable_blur_score
        or lighting_ratio >= config.unusable_lighting_ratio
    ):
        status = "unusable"
    elif flags:
        status = "usable_with_warning"
    else:
        status = "usable"

    return ImageQualityAssessment(
        output=ImageQuality(
            status=status,
            blur_score=round(blur_score, 4),
            glare_detected=glare_detected,
            lighting_warning=lighting_warning,
        ),
        quality_flags=tuple(flags),
    )
