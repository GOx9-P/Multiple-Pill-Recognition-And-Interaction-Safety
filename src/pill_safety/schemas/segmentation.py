"""Schema Pydantic cho input và output của Module 1 segmentation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    """Từ chối các trường không được định nghĩa tại ranh giới giữa module."""

    model_config = ConfigDict(extra="forbid")


class SegmentationInferenceRequest(StrictSchema):
    """Biểu diễn input của Module 1 theo đúng tài liệu ``docs/schema.md``."""

    request_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    image_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)


class ImageQuality(StrictSchema):
    """Biểu diễn chất lượng tổng thể của ảnh đầu vào."""

    status: str = Field(min_length=1)
    blur_score: float = Field(ge=0.0, le=1.0)
    glare_detected: bool
    lighting_warning: bool


class SegmentationEvidence(StrictSchema):
    """Chứa bằng chứng và cờ an toàn của một instance segmentation."""

    confidence: float = Field(ge=0.0, le=1.0)
    occlusion_estimate: float = Field(ge=0.0, le=1.0)
    possible_merged_instance: bool
    possible_non_pill: bool


class SegmentationInstance(StrictSchema):
    """Biểu diễn một viên thuốc đã được phát hiện và tách crop/mask."""

    instance_id: str = Field(min_length=1)
    instance_token: str = Field(min_length=1)
    bbox_xyxy: list[int] = Field(min_length=4, max_length=4)
    mask_path: str = Field(min_length=1)
    color_crop_path: str = Field(min_length=1)
    shape_crop_path: str = Field(min_length=1)
    ocr_crop_path: str = Field(min_length=1)
    crop_path: str = Field(min_length=1)
    segmentation: SegmentationEvidence
    quality_flags: list[str]


class SegmentationInferenceOutput(StrictSchema):
    """Biểu diễn output Module 1 được các module CV tiếp theo sử dụng."""

    request_id: str
    session_id: str
    image_id: str
    image_quality: ImageQuality
    instances: list[SegmentationInstance]
