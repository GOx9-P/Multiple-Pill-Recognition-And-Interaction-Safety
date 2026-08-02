from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from decimal import Decimal

from sqlalchemy import BigInteger, DateTime, ForeignKey, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pill_safety.database.base import Base

if TYPE_CHECKING:
    from pill_safety.database.models.drug import DrugProduct


class Ingredient(Base):
    __tablename__ = "ingredients"

    ingredient_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    normalized_name: Mapped[str | None] = mapped_column(String(255), index=True)
    rxcui: Mapped[str | None] = mapped_column(String(50), unique=True, index=True)
    unii: Mapped[str | None] = mapped_column(String(50), index=True)
    source_name: Mapped[str | None] = mapped_column(String(100))
    source_reference: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    product_ingredients: Mapped[list[ProductIngredient]] = relationship(
        back_populates="ingredient",
        cascade="all, delete-orphan",
    )


class ProductIngredient(Base):
    __tablename__ = "product_ingredients"

    drug_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("drug_products.drug_id", ondelete="CASCADE"),
        primary_key=True,
    )
    ingredient_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("ingredients.ingredient_id", ondelete="CASCADE"),
        primary_key=True,
    )
    strength: Mapped[str | None] = mapped_column(String(100))
    strength_value: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    strength_unit: Mapped[str | None] = mapped_column(String(50))
    numerator_text: Mapped[str | None] = mapped_column(String(100))
    source_ingredient_name: Mapped[str | None] = mapped_column(String(255))

    drug: Mapped[DrugProduct] = relationship(back_populates="product_ingredients")
    ingredient: Mapped[Ingredient] = relationship(back_populates="product_ingredients")
