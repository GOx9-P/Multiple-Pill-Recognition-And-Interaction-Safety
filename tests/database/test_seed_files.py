from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = PROJECT_ROOT / "database_seed"

REQUIRED_SEED_FILES = [
    "patient_profiles.json",
    "drug_products.json",
    "drug_appearances.json",
    "ingredients.json",
    "product_ingredients.json",
    "drug_interactions.json",
    "scan_sessions.json",
    "scan_items.json",
    "scan_interaction_results.json",
]


def test_required_seed_files_are_json_lists() -> None:
    for filename in REQUIRED_SEED_FILES:
        path = SEED_DIR / filename
        assert path.exists(), f"Missing seed file: {filename}"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, list), f"{filename} must contain a list"
        assert all(isinstance(item, dict) for item in payload), f"{filename} items must be objects"


def test_seed_references_use_business_keys() -> None:
    products = json.loads((SEED_DIR / "drug_products.json").read_text(encoding="utf-8"))
    ingredients = json.loads((SEED_DIR / "ingredients.json").read_text(encoding="utf-8"))
    product_codes = {row["product_code"] for row in products}
    ingredient_names = {row["name"] for row in ingredients}

    for row in json.loads((SEED_DIR / "drug_appearances.json").read_text(encoding="utf-8")):
        assert row["product_code"] in product_codes
        assert "drug_id" not in row

    for row in json.loads((SEED_DIR / "product_ingredients.json").read_text(encoding="utf-8")):
        assert row["product_code"] in product_codes
        assert row["ingredient_name"] in ingredient_names
        assert "drug_id" not in row
        assert "ingredient_id" not in row

    for row in json.loads((SEED_DIR / "drug_interactions.json").read_text(encoding="utf-8")):
        assert row["ingredient_a"] in ingredient_names
        assert row["ingredient_b"] in ingredient_names
        assert row["ingredient_a"] != row["ingredient_b"]
        assert row["severity"] in {"minor", "moderate", "major", "contraindicated"}
        assert "ingredient_a_id" not in row
        assert "ingredient_b_id" not in row


def test_db_document_mentions_all_required_tables_and_seed_files() -> None:
    db_doc = (SEED_DIR / "db.md").read_text(encoding="utf-8")
    table_names = [filename.removesuffix(".json") for filename in REQUIRED_SEED_FILES]
    for table_name in table_names:
        assert table_name in db_doc
    for filename in REQUIRED_SEED_FILES:
        assert filename in db_doc

