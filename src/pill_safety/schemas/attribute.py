"""Schema Pydantic cho input và output của Module 2 attribute recognition."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    """Từ chối các trường ngoài contract tại ranh giới giữa các module."""

    model_config = ConfigDict(extra="forbid")


class AttributeInferenceRequest(StrictSchema):
    """Biểu diễn crop và mask của một viên thuốc do Module 1 cung cấp."""

    request_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    image_id: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)
    instance_token: str = Field(min_length=1)
    crop_path: str = Field(min_length=1)
    mask_path: str = Field(min_length=1)


class AttributeAlternative(StrictSchema):
    """Biểu diễn một nhãn thay thế cùng độ tin cậy của nó."""

    label: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class ShapeResult(StrictSchema):
    """Biểu diễn kết quả shape từ shape head của ResNet18."""

    label: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[AttributeAlternative]


class ColorResult(StrictSchema):
    """Biểu diễn kết quả multi-label color từ color head của ResNet18."""

    primary: str = Field(min_length=1)
    secondary: str | None
    distribution: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    lighting_warning: bool


class UnavailableLabelResult(StrictSchema):
    """Đánh dấu thuộc tính dạng nhãn chưa được model hiện tại dự đoán."""

    label: Literal["unknown"] = "unknown"
    confidence: None = None
    source: Literal["not_predicted_by_attribute"] = "not_predicted_by_attribute"


class UnavailableScorelineResult(UnavailableLabelResult):
    """Giữ chỗ scoreline; quyết định thật phải được cập nhật từ Module 3 OCR."""

    visible: None = None


class UnavailableVisibilityResult(StrictSchema):
    """Đánh dấu thuộc tính visibility chưa có head hoặc thuật toán phụ trách."""

    visible: None = None
    confidence: None = None
    source: Literal["not_predicted_by_attribute"] = "not_predicted_by_attribute"


class AttributeInferenceOutput(StrictSchema):
    """Biểu diễn output Module 2 với chỉ shape và color là dự đoán thật."""

    request_id: str
    session_id: str
    image_id: str
    instance_id: str
    instance_token: str
    shape: ShapeResult
    color: ColorResult
    dosage_form: UnavailableLabelResult
    scoreline: UnavailableScorelineResult
    logo_or_symbol: UnavailableVisibilityResult
    damage_or_occlusion: UnavailableVisibilityResult
