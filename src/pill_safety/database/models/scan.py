from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Index, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pill_safety.database.base import Base

if TYPE_CHECKING:
    from pill_safety.database.models.drug import DrugProduct
    from pill_safety.database.models.interaction import DrugInteraction


class PatientProfile(Base):
    __tablename__ = "patient_profiles"
    __table_args__ = (
        CheckConstraint("age IS NULL OR age >= 0", name="ck_patient_profiles_age_non_negative"),
    )

    patient_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    full_name: Mapped[str | None] = mapped_column(String(255))
    age: Mapped[int | None]
    sex: Mapped[str | None] = mapped_column(String(50))
    medical_history: Mapped[list[Any] | dict[str, Any] | None] = mapped_column(JSONB)
    allergies: Mapped[list[Any] | dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    scan_sessions: Mapped[list[ScanSession]] = relationship(back_populates="patient")


class ScanSession(Base):
    __tablename__ = "scan_sessions"
    __table_args__ = (Index("ix_scan_sessions_patient_id", "patient_id"),)

    scan_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    patient_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("patient_profiles.patient_id", ondelete="SET NULL"),
    )
    image_uri: Mapped[str | None] = mapped_column(String(500))
    request_id: Mapped[str | None] = mapped_column(String(100), index=True)
    session_key: Mapped[str | None] = mapped_column(String(100), index=True)
    image_id: Mapped[str | None] = mapped_column(String(100))
    image_quality: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    scope_warnings: Mapped[list[Any] | dict[str, Any] | None] = mapped_column(JSONB)
    llm_context: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    llm_report_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    overall_severity: Mapped[str | None] = mapped_column(String(20))
    llm_report: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    patient: Mapped[PatientProfile | None] = relationship(back_populates="scan_sessions")
    scan_items: Mapped[list[ScanItem]] = relationship(
        back_populates="scan",
        cascade="all, delete-orphan",
    )
    interaction_results: Mapped[list[ScanInteractionResult]] = relationship(
        back_populates="scan",
        cascade="all, delete-orphan",
    )


class ScanItem(Base):
    __tablename__ = "scan_items"
    __table_args__ = (
        CheckConstraint(
            "ocr_confidence IS NULL OR (ocr_confidence >= 0 AND ocr_confidence <= 1)",
            name="ck_scan_items_ocr_confidence_range",
        ),
        CheckConstraint(
            "match_probability IS NULL OR (match_probability >= 0 AND match_probability <= 1)",
            name="ck_scan_items_match_probability_range",
        ),
        CheckConstraint(
            "identification_status IN ('IDENTIFIED', 'AMBIGUOUS', 'UNKNOWN', 'INSUFFICIENT_EVIDENCE')",
            name="ck_scan_items_identification_status",
        ),
        Index("ix_scan_items_scan_id", "scan_id"),
        Index("ix_scan_items_matched_drug_id", "matched_drug_id"),
    )

    scan_item_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    scan_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("scan_sessions.scan_id", ondelete="CASCADE"),
        nullable=False,
    )
    matched_drug_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("drug_products.drug_id", ondelete="SET NULL"),
    )
    instance_id: Mapped[str | None] = mapped_column(String(100))
    instance_token: Mapped[str | None] = mapped_column(String(100))
    crop_path: Mapped[str | None] = mapped_column(String(500))
    mask_path: Mapped[str | None] = mapped_column(String(500))
    detected_imprint: Mapped[str | None] = mapped_column(String(100))
    detected_shape: Mapped[str | None] = mapped_column(String(100))
    detected_color: Mapped[str | None] = mapped_column(String(100))
    dosage_form: Mapped[str | None] = mapped_column(String(100))
    cv_status: Mapped[str | None] = mapped_column(String(50))
    cv_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    candidate_generation: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    top_candidates: Mapped[list[Any] | dict[str, Any] | None] = mapped_column(JSONB)
    ranking_evidence: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    required_action: Mapped[str | None] = mapped_column(String(100))
    scope_warning: Mapped[str | None] = mapped_column(String(255))
    top2_margin: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    ocr_confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    match_probability: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    identification_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="AMBIGUOUS",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    scan: Mapped[ScanSession] = relationship(back_populates="scan_items")
    matched_drug: Mapped[DrugProduct | None] = relationship(back_populates="scan_items")


class ScanInteractionResult(Base):
    __tablename__ = "scan_interaction_results"
    __table_args__ = (
        Index("ix_scan_interaction_results_scan_id", "scan_id"),
        Index("ix_scan_interaction_results_interaction_id", "interaction_id"),
    )

    result_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    scan_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("scan_sessions.scan_id", ondelete="CASCADE"),
        nullable=False,
    )
    interaction_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("drug_interactions.interaction_id", ondelete="CASCADE"),
        nullable=True,
    )
    ingredient_a_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("ingredients.ingredient_id", ondelete="CASCADE"),
        nullable=False,
    )
    ingredient_b_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("ingredients.ingredient_id", ondelete="CASCADE"),
        nullable=False,
    )
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    warning_message: Mapped[str] = mapped_column(Text, nullable=False)
    result_type: Mapped[str | None] = mapped_column(String(50))
    source_instance_ids: Mapped[list[Any] | dict[str, Any] | None] = mapped_column(JSONB)
    ingredient_a_name: Mapped[str | None] = mapped_column(String(255))
    ingredient_b_name: Mapped[str | None] = mapped_column(String(255))
    clinical_risk: Mapped[str | None] = mapped_column(Text)
    mechanism: Mapped[str | None] = mapped_column(Text)
    management: Mapped[str | None] = mapped_column(Text)
    source: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    evidence_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    scan: Mapped[ScanSession] = relationship(back_populates="interaction_results")
    interaction: Mapped[DrugInteraction] = relationship(back_populates="scan_results")
