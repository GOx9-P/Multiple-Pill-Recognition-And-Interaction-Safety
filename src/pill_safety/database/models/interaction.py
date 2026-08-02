from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Index, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pill_safety.database.base import Base

if TYPE_CHECKING:
    from pill_safety.database.models.ingredient import Ingredient
    from pill_safety.database.models.scan import ScanInteractionResult


class DrugInteraction(Base):
    __tablename__ = "drug_interactions"
    __table_args__ = (
        CheckConstraint(
            "severity IN ('minor', 'moderate', 'major', 'contraindicated')",
            name="ck_drug_interactions_severity",
        ),
        CheckConstraint(
            "ingredient_a_id <> ingredient_b_id",
            name="ck_drug_interactions_distinct_ingredients",
        ),
        CheckConstraint(
            "ingredient_a_id < ingredient_b_id",
            name="ck_drug_interactions_normalized_pair",
        ),
        UniqueConstraint(
            "ingredient_a_id",
            "ingredient_b_id",
            name="uq_drug_interactions_ingredient_pair",
        ),
        Index("ix_drug_interactions_ingredient_a_id", "ingredient_a_id"),
        Index("ix_drug_interactions_ingredient_b_id", "ingredient_b_id"),
    )

    interaction_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
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
    description: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(255))
    clinical_risk: Mapped[str | None] = mapped_column(Text)
    mechanism: Mapped[str | None] = mapped_column(Text)
    management: Mapped[str | None] = mapped_column(Text)
    alternative: Mapped[str | None] = mapped_column(Text)
    source_name: Mapped[str | None] = mapped_column(String(100))
    source_reference: Mapped[str | None] = mapped_column(String(500))
    source_level: Mapped[str | None] = mapped_column(String(50))
    evidence_text: Mapped[str | None] = mapped_column(Text)
    last_reviewed: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    ingredient_a: Mapped[Ingredient] = relationship(foreign_keys=[ingredient_a_id])
    ingredient_b: Mapped[Ingredient] = relationship(foreign_keys=[ingredient_b_id])
    scan_results: Mapped[list[ScanInteractionResult]] = relationship(back_populates="interaction")
