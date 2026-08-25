from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ----------------- CV Output Sub-models -----------------
class CvImageQuality(BaseModel):
    status: Optional[str] = None
    blur_score: Optional[float] = None
    glare_detected: Optional[bool] = None
    lighting_warning: Optional[bool] = None


class CvPillSegmentation(BaseModel):
    confidence: Optional[float] = None
    occlusion_estimate: Optional[float] = None
    possible_merged_instance: Optional[bool] = None
    possible_non_pill: Optional[bool] = None


class CvPillShapeAlternative(BaseModel):
    label: Optional[str] = None
    confidence: Optional[float] = None


class CvPillShape(BaseModel):
    label: Optional[str] = None
    confidence: Optional[float] = None
    alternatives: Optional[list[CvPillShapeAlternative]] = None


class CvPillColor(BaseModel):
    primary: Optional[str] = None
    secondary: Optional[str] = None
    distribution: Optional[dict[str, float]] = None
    confidence: Optional[float] = None
    lighting_warning: Optional[bool] = None


class CvPillDosageForm(BaseModel):
    label: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None


class CvPillScoreline(BaseModel):
    label: Optional[str] = None
    visible: Optional[bool] = None
    confidence: Optional[float] = None
    angle_degrees: Optional[float] = None
    orientation: Optional[str] = None
    line_xyxy: Optional[list[float]] = None
    support_count: Optional[int] = None
    rotation_degrees: Optional[int] = None
    preprocessing: Optional[str] = None
    source: Optional[str] = None


class CvPillLogo(BaseModel):
    visible: Optional[bool] = None
    confidence: Optional[float] = None
    source: Optional[str] = None


class CvPillImprintVisibility(BaseModel):
    visible: Optional[bool] = None
    confidence: Optional[float] = None


class CvPillImprintCandidate(BaseModel):
    text: str
    score: Optional[float] = None
    source: Optional[str] = None
    evidence: Optional[list[str]] = None


class CvPillOcrObservation(BaseModel):
    region_id: Optional[str] = None
    rotation_degrees: Optional[int] = None
    preprocessing: Optional[str] = None
    text: Optional[str] = None
    confidence: Optional[float] = None


class CvPillImprint(BaseModel):
    visible: Optional[bool] = None
    raw: Optional[str] = None
    confidence: Optional[float] = None
    ocr_observations: Optional[list[CvPillOcrObservation]] = None
    normalized_candidates: Optional[list[CvPillImprintCandidate]] = None


class CvPill(BaseModel):
    instance_id: str
    instance_token: Optional[str] = None
    side_hint: Optional[str] = "unknown"
    cv_status: Optional[str] = "features_ready"
    bbox_xyxy: Optional[list[float]] = None
    mask_path: Optional[str] = None
    crop_path: Optional[str] = None
    segmentation: Optional[CvPillSegmentation] = None
    shape: Optional[CvPillShape] = None
    color: Optional[CvPillColor] = None
    dosage_form: Optional[CvPillDosageForm] = None
    scoreline: Optional[CvPillScoreline] = None
    logo_or_symbol: Optional[CvPillLogo] = None
    damage_or_occlusion: Optional[CvPillLogo] = None
    imprint_visibility: Optional[CvPillImprintVisibility] = None
    imprint: Optional[CvPillImprint] = None
    quality_flags: Optional[list[str]] = None


class CvOutput(BaseModel):
    schema_version: Optional[str] = "cv_output_v1"
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    image_id: Optional[str] = None
    image_quality: Optional[CvImageQuality] = None
    pills: list[CvPill] = Field(default_factory=list)

    @field_validator("pills")
    @classmethod
    def limit_pills_count(cls, v: list[CvPill]) -> list[CvPill]:
        if len(v) > 15:
            raise ValueError("Số lượng viên thuốc trong ảnh vượt quá giới hạn an toàn là 15 viên.")
        return v


# ----------------- RAG Identification Input -----------------
class RagIdentifyRequest(BaseModel):
    schema_version: Optional[str] = "rag_request_v1"
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    market: str = "US"
    known_drug_names: list[str] = Field(default_factory=list)
    cv_output: CvOutput


# ----------------- RAG Identification Output Models -----------------
class CandidateGenerationDiagnostics(BaseModel):
    strategy: str
    queried_imprints: list[str] = Field(default_factory=list)
    num_records_before_dedup: int = 0
    num_records_after_dedup: int = 0


class CandidateEvidence(BaseModel):
    best_imprint_candidate: Optional[str] = None
    imprint_match_score: float = 0.0
    shape_score: Optional[float] = None
    color_score: Optional[float] = None
    dosage_form_score: Optional[float] = None
    scoreline_score: Optional[float] = None
    logo_score: Optional[float] = None
    market_score: float = 1.0
    top1_top2_margin: Optional[float] = None
    hard_reject: bool = False
    hard_reject_reasons: list[str] = Field(default_factory=list)
    fields: Optional[dict[str, Any]] = None


class TopCandidate(BaseModel):
    rank: int
    appearance_id: Any
    product_id: str
    drug_id: Optional[int] = None
    product_code: Optional[str] = None
    product_name: str
    final_score: float
    evidence: CandidateEvidence


class AcceptedProduct(BaseModel):
    product_id: Optional[str] = None
    drug_id: Optional[int] = None
    product_code: Optional[str] = None
    product_name: str


class PillIdentificationResult(BaseModel):
    instance_id: str
    instance_token: Optional[str] = None
    identification_status: str
    required_action: Optional[str] = None
    candidate_generation: CandidateGenerationDiagnostics
    ranking_method: Optional[str] = "idf_weighted_evidence_v1"
    idf_statistics_version: Optional[str] = None
    top_candidates: list[TopCandidate] = Field(default_factory=list)
    accepted_product: Optional[AcceptedProduct] = None
    scope_warning: Optional[str] = None
    decision_reasons: list[str] = Field(default_factory=list)


class RagIdentificationOutput(BaseModel):
    schema_version: str = "rag_identification_v1"
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    pill_results: list[PillIdentificationResult] = Field(default_factory=list)


# ----------------- DDI Input -----------------
class IdentifiedProductInput(BaseModel):
    instance_id: str
    product_id: str


class DdiRequest(BaseModel):
    schema_version: Optional[str] = "ddi_request_v0"
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    identified_products: list[IdentifiedProductInput] = Field(default_factory=list)

    @field_validator("identified_products")
    @classmethod
    def limit_products_count(cls, v: list[IdentifiedProductInput]) -> list[IdentifiedProductInput]:
        if len(v) > 15:
            raise ValueError("Số lượng thuốc đưa vào kiểm tra tương tác vượt quá giới hạn an toàn là 15 viên.")
        return v


# ----------------- Context Builder Input -----------------
class ContextBuilderInput(BaseModel):
    schema_version: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    cv_output: Optional[dict[str, Any]] = None
    rag_identification: Optional[dict[str, Any]] = None
    ddi_output: Optional[dict[str, Any]] = None


# ----------------- Report & Manual Override Models -----------------
class RagReportRequest(BaseModel):
    schema_version: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    cv_output: CvOutput
    rag_identification: Optional[dict[str, Any]] = None
    ddi_output: Optional[dict[str, Any]] = None


class RagReportResponse(BaseModel):
    schema_version: str = "llm_report_v0"
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    overall_severity: str
    provider_used: str
    formatted_report_text: str
    structured_context: dict[str, Any]


class ManualIdentifyRequest(BaseModel):
    session_id: Optional[str] = None
    instance_id: str
    manual_drug_name: Optional[str] = None
    product_id: Optional[str] = None
