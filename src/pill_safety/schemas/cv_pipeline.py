"""Schema Pydantic cho input va output fusion cua Module 4 CV Pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .attribute import (
    AttributeInferenceOutput,
    ColorResult,
    ShapeResult,
    UnavailableLabelResult,
    UnavailableVisibilityResult,
)
from .ocr import (
    ImprintVisibility,
    NormalizedCandidate,
    OCRInferenceOutput,
    OCRObservation,
    ScorelineResult,
)
from .segmentation import (
    ImageQuality,
    SegmentationEvidence,
    SegmentationInferenceOutput,
)


class StrictSchema(BaseModel):
    """Tu choi field ngoai contract cua Module 4."""

    model_config = ConfigDict(extra="forbid")


class CVPipelineInput(StrictSchema):
    """Gom cac output da co cua Module 1, Module 2 va Module 3 de fusion."""

    segmentation_output: SegmentationInferenceOutput
    attribute_outputs: list[AttributeInferenceOutput]
    ocr_outputs: list[OCRInferenceOutput]


class PipelineImprintResult(StrictSchema):
    """Phan imprint can cho Retrieval/RAG, khong dua text-region debug qua Module 4."""

    visible: bool
    raw: str
    confidence: float = Field(ge=0.0, le=1.0)
    ocr_observations: list[OCRObservation]
    normalized_candidates: list[NormalizedCandidate]


class CVPill(StrictSchema):
    """Metadata thi giac da fusion cua mot instance vien thuoc."""

    instance_id: str = Field(min_length=1)
    instance_token: str = Field(min_length=1)
    side_hint: Literal["unknown"] = "unknown"
    cv_status: Literal[
        "features_ready",
        "partial_features",
        "insufficient_visual_evidence",
        "unknown_object",
    ]
    bbox_xyxy: list[int] = Field(min_length=4, max_length=4)
    mask_path: str = Field(min_length=1)
    crop_path: str = Field(min_length=1)
    segmentation: SegmentationEvidence
    shape: ShapeResult
    color: ColorResult
    dosage_form: UnavailableLabelResult
    scoreline: ScorelineResult
    logo_or_symbol: UnavailableVisibilityResult
    damage_or_occlusion: UnavailableVisibilityResult
    imprint_visibility: ImprintVisibility
    imprint: PipelineImprintResult
    quality_flags: list[str]


class CVPipelineOutput(StrictSchema):
    """CV output v1 la contract truyen truc tiep sang Retrieval/RAG."""

    schema_version: Literal["cv_output_v1"] = "cv_output_v1"
    request_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    image_id: str = Field(min_length=1)
    image_quality: ImageQuality
    pills: list[CVPill]
