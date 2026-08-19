from __future__ import annotations

from sqlalchemy.orm import Session

from pill_safety.database.models import DrugProduct
from pill_safety.database.repositories.drug_repository import DrugRepository


class DrugService:
    def __init__(self, db: Session) -> None:
        self.repository = DrugRepository(db)

    def list_drugs(self) -> list[dict]:
        return [self._summary(drug) for drug in self.repository.get_all()]

    def get_drug(self, drug_id: int) -> dict | None:
        drug = self.repository.get_by_id(drug_id)
        if drug is None:
            return None
        return self._detail(drug)

    def search(
        self,
        *,
        imprint: str | None = None,
        shape: str | None = None,
        color: str | None = None,
        dosage_form: str | None = None,
    ) -> list[dict]:
        return [
            self._summary(drug)
            for drug in self.repository.find_by_visual_features(
                imprint=imprint,
                shape=shape,
                color=color,
                dosage_form=dosage_form,
            )
        ]

    @staticmethod
    def _summary(drug: DrugProduct) -> dict:
        return {
            "drug_id": drug.drug_id,
            "product_code": drug.product_code,
            "name": drug.name,
            "manufacturer": drug.manufacturer,
            "dosage_form": drug.dosage_form,
            "generic_name": drug.generic_name,
            "route": drug.route,
            "market": drug.market,
            "source": {
                "source_name": drug.source_name,
                "source_reference": drug.source_reference,
                "source_set_id": drug.source_set_id,
                "spl_version": drug.spl_version,
                "published_date": drug.published_date,
            },
        }

    def _detail(self, drug: DrugProduct) -> dict:
        payload = self._summary(drug)
        payload["product_image"] = drug.product_image
        payload["appearances"] = [
            {
                "appearance_id": item.appearance_id,
                "imprint": item.imprint,
                "imprint_raw": item.imprint_raw,
                "imprint_normalized": item.imprint_normalized,
                "imprint_side_a": item.imprint_side_a,
                "imprint_side_b": item.imprint_side_b,
                "shape": item.shape,
                "color": item.color,
                "primary_color": item.primary_color,
                "secondary_color": item.secondary_color,
                "color_pattern": item.color_pattern,
                "size_mm": float(item.size_mm) if item.size_mm is not None else None,
                "score_line": item.score_line,
                "logo_or_symbol": item.logo_or_symbol,
                "coating": item.coating,
                "source": {
                    "source_name": item.source_name,
                    "source_reference": item.source_reference,
                },
            }
            for item in drug.appearances
        ]
        payload["active_ingredients"] = [
            {
                "ingredient_id": item.ingredient.ingredient_id,
                "name": item.ingredient.name,
                "normalized_name": item.ingredient.normalized_name,
                "rxcui": item.ingredient.rxcui,
                "unii": item.ingredient.unii,
                "strength": item.strength,
                "strength_value": float(item.strength_value) if item.strength_value is not None else None,
                "strength_unit": item.strength_unit,
                "source_ingredient_name": item.source_ingredient_name,
            }
            for item in drug.product_ingredients
        ]
        return payload
