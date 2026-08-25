from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class ImprintCandidate:
    text: str
    raw_text: str
    score: float
    source: str | None = None
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LabelEvidence:
    label: str | None
    confidence: float
    alternatives: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ColorEvidence:
    primary: str | None
    secondary: str | None
    distribution: dict[str, float]
    confidence: float
    lighting_warning: bool


@dataclass(frozen=True)
class ScorelineEvidence:
    label: str | None
    visible: bool | None
    confidence: float


@dataclass(frozen=True)
class LogoEvidence:
    visible: bool | None
    confidence: float


@dataclass(frozen=True)
class SegmentationEvidence:
    confidence: float
    occlusion_estimate: float | None
    possible_merged_instance: bool
    possible_non_pill: bool


@dataclass(frozen=True)
class ImageQualityEvidence:
    status: str | None
    blur_score: float | None
    glare_detected: bool
    lighting_warning: bool


@dataclass(frozen=True)
class RecognitionInput:
    instance_id: str
    instance_token: str | None
    market: str | None
    cv_status: str | None
    segmentation: SegmentationEvidence
    image_quality: ImageQualityEvidence
    imprint_visible: bool
    imprint_visibility_confidence: float
    imprint_confidence: float
    imprint_candidates: list[ImprintCandidate]
    shape: LabelEvidence | None
    color: ColorEvidence | None
    dosage_form: LabelEvidence | None
    scoreline: ScorelineEvidence | None
    logo_or_symbol: LogoEvidence | None
    quality_flags: list[str]
    max_status: str | None = None


@dataclass(frozen=True)
class CandidateRecord:
    appearance_id: int
    drug_id: int
    product_code: str | None
    product_name: str
    imprint_normalized: str | None
    shape: str | None
    primary_color: str | None
    secondary_color: str | None
    color_pattern: str | None
    score_line: bool
    logo_or_symbol: bool
    size_mm: Decimal | None
    dosage_form: str | None
    market: str | None
    source_name: str | None = None
    source_reference: str | None = None
    imprint_raw: str | None = None
    imprint_side_a: str | None = None
    imprint_side_b: str | None = None


@dataclass(frozen=True)
class RetrievalDiagnostics:
    strategy: str
    queried_imprints: list[str]
    num_records_before_dedup: int
    num_records_after_dedup: int


@dataclass(frozen=True)
class FieldScore:
    field: str
    cv_value: Any
    db_value: Any
    match_score: float
    idf_weight: float
    confidence: float
    quality_multiplier: float
    evidence_score: float
    max_score: float
    explanation: str


@dataclass(frozen=True)
class CandidateScore:
    candidate: CandidateRecord
    final_score: float
    field_scores: dict[str, FieldScore]
    best_imprint_candidate: str | None
    imprint_match_score: float
    hard_reject: bool
    hard_reject_reasons: list[str]
