"""Strongly-typed ViewModels for the Healthcare Streamlit UI.

These models decouple the UI rendering components from raw backend JSON/Pydantic schemas,
providing a stable presenter layer for Clinical AI decision support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateViewModel:
    """An individual candidate matching from the RAG pharmaceutical database."""

    rank: int
    product_name: str
    final_score: float
    imprint_score: float | None = None
    shape_score: float | None = None
    color_score: float | None = None
    brand_name: str | None = None
    generic_name: str | None = None
    rxcui: str | None = None
    ndc: str | None = None
    evidence_notes: list[str] = field(default_factory=list)


@dataclass
class PillViewModel:
    """Presentation model for an individual detected pill instance."""

    instance_id: str
    status: str  # "accepted" | "ambiguous" | "unresolved" | "rejected" | "detected"
    shape: str
    shape_confidence: float | None
    color_primary: str
    color_secondary: str | None
    color_confidence: float | None
    imprint_raw: str
    imprint_confidence: float | None
    imprint_candidates: list[str] = field(default_factory=list)
    scoreline_visible: bool | None = None
    scoreline_confidence: float | None = None
    bbox_xyxy: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    mask_path: str | None = None
    crop_path: str | None = None

    # Identified Product details
    drug_name: str | None = None
    brand_name: str | None = None
    generic_name: str | None = None
    strength: str | None = None
    rxcui: str | None = None
    ndc: str | None = None
    active_ingredients: list[dict[str, Any]] = field(default_factory=list)
    match_confidence: float | None = None

    # XAI and candidates
    top_candidates: list[CandidateViewModel] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    required_action: str | None = None
    scope_warning: str | None = None
    is_manual_override: bool = False


@dataclass
class ImageQualityViewModel:
    """Summary of CV input image quality and lighting checks."""

    status: str = "good"  # "good" | "usable_with_warning" | "poor" | "warning"
    blur_score: float = 0.0
    glare_detected: bool = False
    lighting_warning: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class InteractionPairViewModel:
    """Presentation model for a pairwise drug-drug interaction alert."""

    drug_a_name: str
    drug_b_name: str
    severity: str  # "critical" | "major" | "moderate" | "minor" | "safe"
    message: str
    mechanism: str = ""
    clinical_risk: str = ""
    management: str = ""
    source: str = "NLM / NIH DDI Standard"


@dataclass
class DuplicateIngredientViewModel:
    """Presentation model for duplicate active ingredient overdose hazards."""

    ingredient_name: str
    source_instances: list[str] = field(default_factory=list)
    severity: str = "major"
    warning: str = "Trùng lặp hoạt chất giữa các viên thuốc."
    total_strength_note: str | None = None


@dataclass
class SafetyReportViewModel:
    """Unified clinical safety evaluation combining DDI, duplicates, and LLM summary."""

    request_id: str
    session_id: str
    overall_severity: str  # "critical" | "major" | "moderate" | "safe" | "unresolved"
    identified_drugs: list[dict[str, Any]] = field(default_factory=list)
    interactions: list[InteractionPairViewModel] = field(default_factory=list)
    duplicate_warnings: list[DuplicateIngredientViewModel] = field(default_factory=list)
    formatted_report_text: str = ""
    provider_used: str = "Expert Clinical Rules"
    scope_warnings: list[str] = field(default_factory=list)
    timestamp: str = ""
