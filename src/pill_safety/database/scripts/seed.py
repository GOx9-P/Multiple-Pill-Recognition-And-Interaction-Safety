from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from pill_safety.database.models import (
    DrugAppearance,
    DrugInteraction,
    DrugProduct,
    Ingredient,
    ProductIngredient,
)
from pill_safety.database.session import SessionLocal


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SEED_DIR = PROJECT_ROOT / "database_seed"


def read_json(filename: str) -> list[dict[str, Any]]:
    filepath = SEED_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy file seed: {filepath}")

    try:
        payload = json.loads(filepath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON không hợp lệ trong {filename}: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError(f"{filename}: root JSON phải là một list.")
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{filename}[{index}] phải là một object/dictionary.")
    return payload


def require_text(row: dict[str, Any], field: str, filename: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{filename}[{index}].{field} phải là chuỗi không rỗng.")
    return value.strip()


def normalize_imprint(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def get_product_by_code(db: Session, product_code: str) -> DrugProduct:
    product = db.scalars(select(DrugProduct).where(DrugProduct.product_code == product_code)).first()
    if product is None:
        raise ValueError(f"Không tìm thấy drug product với product_code: {product_code}")
    return product


def get_ingredient_by_name(db: Session, name: str) -> Ingredient:
    ingredient = db.scalars(select(Ingredient).where(Ingredient.name == name)).first()
    if ingredient is None:
        raise ValueError(f"Không tìm thấy ingredient với name: {name}")
    return ingredient


def seed_ingredients(db: Session) -> None:
    for index, row in enumerate(read_json("ingredients.json")):
        name = require_text(row, "name", "ingredients.json", index)
        ingredient = db.scalars(select(Ingredient).where(Ingredient.name == name)).first()
        if ingredient is None:
            ingredient = Ingredient(name=name)
            db.add(ingredient)
        ingredient.normalized_name = row.get("normalized_name") or name.lower()
        ingredient.rxcui = row.get("rxcui")
        ingredient.unii = row.get("unii")
        ingredient.source_name = row.get("source_name")
        ingredient.source_reference = row.get("source_reference")


def seed_drug_products(db: Session) -> None:
    for index, row in enumerate(read_json("drug_products.json")):
        product_code = require_text(row, "product_code", "drug_products.json", index)
        name = require_text(row, "name", "drug_products.json", index)
        drug = db.scalars(select(DrugProduct).where(DrugProduct.product_code == product_code)).first()
        if drug is None:
            drug = DrugProduct(product_code=product_code, name=name)
            db.add(drug)
        drug.name = name
        drug.manufacturer = row.get("manufacturer")
        drug.dosage_form = row.get("dosage_form")
        drug.product_image = row.get("product_image")
        drug.generic_name = row.get("generic_name")
        drug.route = row.get("route")
        drug.market = row.get("market") or "US"
        drug.product_rxcui = row.get("product_rxcui")
        drug.source_name = row.get("source_name")
        drug.source_reference = row.get("source_reference")
        drug.source_set_id = row.get("source_set_id")
        drug.spl_version = row.get("spl_version")
        drug.published_date = row.get("published_date")
        drug.active = bool(row.get("active", True))


def seed_drug_appearances(db: Session) -> None:
    for index, row in enumerate(read_json("drug_appearances.json")):
        product_code = require_text(row, "product_code", "drug_appearances.json", index)
        product = get_product_by_code(db, product_code)
        imprint = str(row.get("imprint") or "")
        shape = str(row.get("shape") or "")
        color = str(row.get("color") or "")
        imprint_normalized = row.get("imprint_normalized") or normalize_imprint(imprint)
        appearance = db.scalars(
            select(DrugAppearance).where(
                DrugAppearance.drug_id == product.drug_id,
                DrugAppearance.imprint == imprint,
                DrugAppearance.shape == shape,
                DrugAppearance.color == color,
            )
        ).first()
        if appearance is None:
            appearance = DrugAppearance(
                drug_id=product.drug_id,
                imprint=imprint,
                shape=shape,
                color=color,
            )
            db.add(appearance)
        appearance.size_mm = row.get("size_mm")
        appearance.score_line = bool(row.get("score_line", False))
        appearance.logo_or_symbol = bool(row.get("logo_or_symbol", False))
        appearance.imprint_raw = row.get("imprint_raw") or imprint
        appearance.imprint_normalized = imprint_normalized
        appearance.imprint_side_a = row.get("imprint_side_a")
        appearance.imprint_side_b = row.get("imprint_side_b")
        appearance.primary_color = row.get("primary_color") or color
        appearance.secondary_color = row.get("secondary_color")
        appearance.color_pattern = row.get("color_pattern")
        appearance.coating = row.get("coating")
        appearance.source_name = row.get("source_name")
        appearance.source_reference = row.get("source_reference")


def seed_product_ingredients(db: Session) -> None:
    for index, row in enumerate(read_json("product_ingredients.json")):
        product_code = require_text(row, "product_code", "product_ingredients.json", index)
        ingredient_name = require_text(row, "ingredient_name", "product_ingredients.json", index)
        product = get_product_by_code(db, product_code)
        ingredient = get_ingredient_by_name(db, ingredient_name)
        product_ingredient = db.get(ProductIngredient, (product.drug_id, ingredient.ingredient_id))
        if product_ingredient is None:
            product_ingredient = ProductIngredient(
                drug_id=product.drug_id,
                ingredient_id=ingredient.ingredient_id,
            )
            db.add(product_ingredient)
        product_ingredient.strength = row.get("strength")
        product_ingredient.strength_value = row.get("strength_value")
        product_ingredient.strength_unit = row.get("strength_unit")
        product_ingredient.numerator_text = row.get("numerator_text")
        product_ingredient.source_ingredient_name = row.get("source_ingredient_name")


def seed_drug_interactions(db: Session) -> None:
    allowed_severity = {"minor", "moderate", "major", "contraindicated"}
    for index, row in enumerate(read_json("drug_interactions.json")):
        ingredient_a = get_ingredient_by_name(
            db,
            require_text(row, "ingredient_a", "drug_interactions.json", index),
        )
        ingredient_b = get_ingredient_by_name(
            db,
            require_text(row, "ingredient_b", "drug_interactions.json", index),
        )
        if ingredient_a.ingredient_id == ingredient_b.ingredient_id:
            raise ValueError(f"drug_interactions.json[{index}] không được dùng cùng một ingredient.")

        first_id, second_id = sorted((ingredient_a.ingredient_id, ingredient_b.ingredient_id))
        severity = require_text(row, "severity", "drug_interactions.json", index).lower()
        if severity not in allowed_severity:
            raise ValueError(f"drug_interactions.json[{index}].severity không hợp lệ: {severity}")
        description = require_text(row, "description", "drug_interactions.json", index)

        interaction = db.scalars(
            select(DrugInteraction).where(
                DrugInteraction.ingredient_a_id == first_id,
                DrugInteraction.ingredient_b_id == second_id,
            )
        ).first()
        if interaction is None:
            interaction = DrugInteraction(
                ingredient_a_id=first_id,
                ingredient_b_id=second_id,
                severity=severity,
                description=description,
            )
            db.add(interaction)
        interaction.severity = severity
        interaction.description = description
        interaction.recommendation = row.get("recommendation")
        interaction.source = row.get("source")
        interaction.clinical_risk = row.get("clinical_risk")
        interaction.mechanism = row.get("mechanism")
        interaction.management = row.get("management")
        interaction.alternative = row.get("alternative")
        interaction.source_name = row.get("source_name") or row.get("source")
        interaction.source_reference = row.get("source_reference")
        interaction.source_level = row.get("source_level")
        interaction.evidence_text = row.get("evidence_text")
        interaction.last_reviewed = row.get("last_reviewed")


def seed_database() -> None:
    db = SessionLocal()
    try:
        read_json("patient_profiles.json")
        seed_ingredients(db)
        seed_drug_products(db)
        db.flush()
        seed_drug_appearances(db)
        seed_product_ingredients(db)
        seed_drug_interactions(db)
        read_json("scan_sessions.json")
        read_json("scan_items.json")
        read_json("scan_interaction_results.json")
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"Seed database thất bại: {exc}")
        raise
    finally:
        db.close()

    print("Seed database thành công.")


if __name__ == "__main__":
    seed_database()
