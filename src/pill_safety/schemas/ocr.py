from __future__ import annotations

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


class OCRInferenceOutput(StrictSchema):
    request_id: str
    session_id: str
    image_id: str
    instance_id: str
    instance_token: str
    imprint_visibility: ImprintVisibility
    imprint: ImprintResult
