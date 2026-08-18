from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Index, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pill_safety.database.base import Base

if TYPE_CHECKING:
    from pill_safety.database.models.ingredient import ProductIngredient
    from pill_safety.database.models.scan import ScanItem


class DrugProduct(Base):
    __tablename__ = "drug_products"

    drug_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    product_code: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    manufacturer: Mapped[str | None] = mapped_column(String(255))
    dosage_form: Mapped[str | None] = mapped_column(String(100), index=True)
    product_image: Mapped[str | None] = mapped_column(String(500))
    generic_name: Mapped[str | None] = mapped_column(String(255))
    route: Mapped[str | None] = mapped_column(String(100))
    market: Mapped[str] = mapped_column(String(20), nullable=False, server_default="US")
    product_rxcui: Mapped[str | None] = mapped_column(String(50), index=True)
    source_name: Mapped[str | None] = mapped_column(String(100))
    source_reference: Mapped[str | None] = mapped_column(String(500))
    source_set_id: Mapped[str | None] = mapped_column(String(100), index=True)
    spl_version: Mapped[str | None] = mapped_column(String(50))
    published_date: Mapped[str | None] = mapped_column(String(50))
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    appearances: Mapped[list[DrugAppearance]] = relationship(
        back_populates="drug",
        cascade="all, delete-orphan",
    )
    product_ingredients: Mapped[list[ProductIngredient]] = relationship(
        back_populates="drug",
        cascade="all, delete-orphan",
    )
    scan_items: Mapped[list[ScanItem]] = relationship(back_populates="matched_drug")


class DrugAppearance(Base):
    __tablename__ = "drug_appearances"
    __table_args__ = (
        UniqueConstraint(
            "drug_id",
            "imprint",
            "shape",
            "color",
            name="uq_drug_appearances_visual_signature",
        ),
        Index("ix_drug_appearances_drug_id", "drug_id"),
        Index("ix_drug_appearances_imprint", "imprint"),
        Index("ix_drug_appearances_shape", "shape"),
        Index("ix_drug_appearances_color", "color"),
    )

    appearance_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("drug_products.drug_id", ondelete="CASCADE"),
        nullable=False,
    )
    imprint: Mapped[str] = mapped_column(String(100), nullable=False, server_default="")
    imprint_raw: Mapped[str | None] = mapped_column(String(255))
    imprint_normalized: Mapped[str | None] = mapped_column(String(255), index=True)
    imprint_side_a: Mapped[str | None] = mapped_column(String(100))
    imprint_side_b: Mapped[str | None] = mapped_column(String(100))
    shape: Mapped[str] = mapped_column(String(100), nullable=False, server_default="")
    color: Mapped[str] = mapped_column(String(100), nullable=False, server_default="")
    primary_color: Mapped[str | None] = mapped_column(String(100))
    secondary_color: Mapped[str | None] = mapped_column(String(100))
    color_pattern: Mapped[str | None] = mapped_column(String(100))
    size_mm: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    score_line: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    logo_or_symbol: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    coating: Mapped[str | None] = mapped_column(String(100))
    source_name: Mapped[str | None] = mapped_column(String(100))
    source_reference: Mapped[str | None] = mapped_column(String(500))

    drug: Mapped[DrugProduct] = relationship(back_populates="appearances")
