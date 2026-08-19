from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OCRInferenceRequest(StrictSchema):
    request_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    image_id: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)
    instance_token: str = Field(min_length=1)
    crop_path: str = Field(min_length=1)
    mask_path: str = Field(min_length=1)


class ImprintVisibility(StrictSchema):
    visible: bool
    confidence: float = Field(ge=0.0, le=1.0)


class TextRegion(StrictSchema):
    region_id: str
    polygon: list[list[float]]
    detection_confidence: float = Field(ge=0.0, le=1.0)


class OCRObservation(StrictSchema):
    region_id: str
    rotation_degrees: int
    preprocessing: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class NormalizedCandidate(StrictSchema):
    text: str
    score: float = Field(ge=0.0, le=1.0)
    source: str
    evidence: list[str]


class ImprintResult(StrictSchema):
    visible: bool
    raw: str
    confidence: float = Field(ge=0.0, le=1.0)
    text_regions: list[TextRegion]
    ocr_observations: list[OCRObservation]
    normalized_candidates: list[NormalizedCandidate]


class ScorelineResult(StrictSchema):
    """Biểu diễn quyết định scoreline cuối cùng do Module 3 OCR quản lý."""

    visible: bool
    confidence: float = Field(ge=0.0, le=1.0)
    angle_degrees: float | None = Field(default=None, ge=0.0, le=180.0)
    orientation: Literal["horizontal", "vertical", "oblique", "unknown"]
    line_xyxy: list[float] | None = Field(
        default=None, min_length=4, max_length=4
    )
    support_count: int = Field(ge=0)
    rotation_degrees: int | None
    preprocessing: str | None
    source: Literal["ocr_hough_consensus"]


class OCRInferenceOutput(StrictSchema):
    request_id: str
    session_id: str
    image_id: str
    instance_id: str
    instance_token: str
    scoreline: ScorelineResult
    imprint_visibility: ImprintVisibility
    imprint: ImprintResult
