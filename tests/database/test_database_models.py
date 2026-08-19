from __future__ import annotations

from pill_safety.database.base import Base


EXPECTED_TABLES = {
    "patient_profiles",
    "scan_sessions",
    "scan_items",
    "drug_products",
    "drug_appearances",
    "ingredients",
    "product_ingredients",
    "drug_interactions",
    "scan_interaction_results",
}


def test_metadata_contains_expected_tables() -> None:
    import pill_safety.database.models  # noqa: F401

    assert EXPECTED_TABLES.issubset(set(Base.metadata.tables))


def test_drug_interaction_constraints_are_declared() -> None:
    import pill_safety.database.models  # noqa: F401

    table = Base.metadata.tables["drug_interactions"]
    constraint_names = {constraint.name for constraint in table.constraints}
    assert "ck_drug_interactions_severity" in constraint_names
    assert "ck_drug_interactions_distinct_ingredients" in constraint_names
    assert "ck_drug_interactions_normalized_pair" in constraint_names
    assert "uq_drug_interactions_ingredient_pair" in constraint_names


def test_scan_item_constraints_are_declared() -> None:
    import pill_safety.database.models  # noqa: F401

    table = Base.metadata.tables["scan_items"]
    constraint_names = {constraint.name for constraint in table.constraints}
    assert "ck_scan_items_ocr_confidence_range" in constraint_names
    assert "ck_scan_items_match_probability_range" in constraint_names
    assert "ck_scan_items_identification_status" in constraint_names

