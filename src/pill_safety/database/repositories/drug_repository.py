from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, selectinload

from pill_safety.database.models import DrugAppearance, DrugProduct, ProductIngredient


class DrugRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_all(self) -> list[DrugProduct]:
        statement = select(DrugProduct).order_by(DrugProduct.name)
        return list(self.db.scalars(statement).all())

    def get_by_id(self, drug_id: int) -> DrugProduct | None:
        statement = (
            select(DrugProduct)
            .options(
                selectinload(DrugProduct.appearances),
                selectinload(DrugProduct.product_ingredients).selectinload(ProductIngredient.ingredient),
            )
            .where(DrugProduct.drug_id == drug_id)
        )
        return self.db.scalars(statement).first()

    def get_by_product_code(self, product_code: str) -> DrugProduct | None:
        statement = select(DrugProduct).where(DrugProduct.product_code == product_code)
        return self.db.scalars(statement).first()

    def find_by_visual_features(
        self,
        *,
        imprint: str | None = None,
        shape: str | None = None,
        color: str | None = None,
        dosage_form: str | None = None,
    ) -> list[DrugProduct]:
        statement = select(DrugProduct).join(DrugAppearance).options(selectinload(DrugProduct.appearances))

        if imprint:
            normalized_imprint = "".join(ch for ch in imprint.upper() if ch.isalnum())
            statement = statement.where(
                or_(
                    DrugAppearance.imprint.ilike(f"%{imprint}%"),
                    DrugAppearance.imprint_normalized.ilike(f"%{normalized_imprint}%"),
                )
            )
        if shape:
            statement = statement.where(DrugAppearance.shape.ilike(shape))
        if color:
            statement = statement.where(DrugAppearance.color.ilike(color))
        if dosage_form:
            statement = statement.where(DrugProduct.dosage_form.ilike(dosage_form))

        statement = statement.distinct().order_by(DrugProduct.name)
        return list(self.db.scalars(statement).all())
