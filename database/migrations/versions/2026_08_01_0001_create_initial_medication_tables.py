"""create initial medication tables

Revision ID: 202608010001
Revises:
Create Date: 2026-08-01 00:01:00.000000+00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "202608010001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "drug_products",
        sa.Column("drug_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("product_code", sa.String(length=100), nullable=False),
        sa.Column("manufacturer", sa.String(length=255), nullable=True),
        sa.Column("dosage_form", sa.String(length=100), nullable=True),
        sa.Column("product_image", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("drug_id"),
        sa.UniqueConstraint("product_code"),
    )
    op.create_index(op.f("ix_drug_products_product_code"), "drug_products", ["product_code"], unique=False)
    op.create_index(op.f("ix_drug_products_dosage_form"), "drug_products", ["dosage_form"], unique=False)

    op.create_table(
        "ingredients",
        sa.Column("ingredient_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("rxcui", sa.String(length=50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("ingredient_id"),
        sa.UniqueConstraint("name"),
        sa.UniqueConstraint("rxcui"),
    )
    op.create_index(op.f("ix_ingredients_name"), "ingredients", ["name"], unique=False)
    op.create_index(op.f("ix_ingredients_rxcui"), "ingredients", ["rxcui"], unique=False)

    op.create_table(
        "patient_profiles",
        sa.Column("patient_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column("age", sa.Integer(), nullable=True),
        sa.Column("sex", sa.String(length=50), nullable=True),
        sa.Column("medical_history", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("allergies", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("age IS NULL OR age >= 0", name="ck_patient_profiles_age_non_negative"),
        sa.PrimaryKeyConstraint("patient_id"),
    )

    op.create_table(
        "drug_appearances",
        sa.Column("appearance_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("drug_id", sa.BigInteger(), nullable=False),
        sa.Column("imprint", sa.String(length=100), server_default="", nullable=False),
        sa.Column("shape", sa.String(length=100), server_default="", nullable=False),
        sa.Column("color", sa.String(length=100), server_default="", nullable=False),
        sa.Column("size_mm", sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column("score_line", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("logo_or_symbol", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.ForeignKeyConstraint(["drug_id"], ["drug_products.drug_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("appearance_id"),
        sa.UniqueConstraint("drug_id", "imprint", "shape", "color", name="uq_drug_appearances_visual_signature"),
    )
    op.create_index("ix_drug_appearances_drug_id", "drug_appearances", ["drug_id"], unique=False)
    op.create_index("ix_drug_appearances_imprint", "drug_appearances", ["imprint"], unique=False)
    op.create_index("ix_drug_appearances_shape", "drug_appearances", ["shape"], unique=False)
    op.create_index("ix_drug_appearances_color", "drug_appearances", ["color"], unique=False)

    op.create_table(
        "product_ingredients",
        sa.Column("drug_id", sa.BigInteger(), nullable=False),
        sa.Column("ingredient_id", sa.BigInteger(), nullable=False),
        sa.Column("strength", sa.String(length=100), nullable=True),
        sa.ForeignKeyConstraint(["drug_id"], ["drug_products.drug_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["ingredient_id"], ["ingredients.ingredient_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("drug_id", "ingredient_id"),
    )

    op.create_table(
        "drug_interactions",
        sa.Column("interaction_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("ingredient_a_id", sa.BigInteger(), nullable=False),
        sa.Column("ingredient_b_id", sa.BigInteger(), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("recommendation", sa.Text(), nullable=True),
        sa.Column("source", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "severity IN ('minor', 'moderate', 'major', 'contraindicated')",
            name="ck_drug_interactions_severity",
        ),
        sa.CheckConstraint(
            "ingredient_a_id <> ingredient_b_id",
            name="ck_drug_interactions_distinct_ingredients",
        ),
        sa.CheckConstraint(
            "ingredient_a_id < ingredient_b_id",
            name="ck_drug_interactions_normalized_pair",
        ),
        sa.ForeignKeyConstraint(["ingredient_a_id"], ["ingredients.ingredient_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["ingredient_b_id"], ["ingredients.ingredient_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("interaction_id"),
        sa.UniqueConstraint("ingredient_a_id", "ingredient_b_id", name="uq_drug_interactions_ingredient_pair"),
    )
    op.create_index("ix_drug_interactions_ingredient_a_id", "drug_interactions", ["ingredient_a_id"], unique=False)
    op.create_index("ix_drug_interactions_ingredient_b_id", "drug_interactions", ["ingredient_b_id"], unique=False)

    op.create_table(
        "scan_sessions",
        sa.Column("scan_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("patient_id", sa.BigInteger(), nullable=True),
        sa.Column("image_uri", sa.String(length=500), nullable=True),
        sa.Column("overall_severity", sa.String(length=20), nullable=True),
        sa.Column("llm_report", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["patient_id"], ["patient_profiles.patient_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("scan_id"),
    )
    op.create_index("ix_scan_sessions_patient_id", "scan_sessions", ["patient_id"], unique=False)

    op.create_table(
        "scan_items",
        sa.Column("scan_item_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("scan_id", sa.BigInteger(), nullable=False),
        sa.Column("matched_drug_id", sa.BigInteger(), nullable=True),
        sa.Column("instance_id", sa.String(length=100), nullable=True),
        sa.Column("instance_token", sa.String(length=100), nullable=True),
        sa.Column("crop_path", sa.String(length=500), nullable=True),
        sa.Column("detected_imprint", sa.String(length=100), nullable=True),
        sa.Column("detected_shape", sa.String(length=100), nullable=True),
        sa.Column("detected_color", sa.String(length=100), nullable=True),
        sa.Column("dosage_form", sa.String(length=100), nullable=True),
        sa.Column("ocr_confidence", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("match_probability", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("identification_status", sa.String(length=20), server_default="AMBIGUOUS", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "ocr_confidence IS NULL OR (ocr_confidence >= 0 AND ocr_confidence <= 1)",
            name="ck_scan_items_ocr_confidence_range",
        ),
        sa.CheckConstraint(
            "match_probability IS NULL OR (match_probability >= 0 AND match_probability <= 1)",
            name="ck_scan_items_match_probability_range",
        ),
        sa.CheckConstraint(
            "identification_status IN ('IDENTIFIED', 'AMBIGUOUS', 'UNKNOWN')",
            name="ck_scan_items_identification_status",
        ),
        sa.ForeignKeyConstraint(["matched_drug_id"], ["drug_products.drug_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["scan_id"], ["scan_sessions.scan_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("scan_item_id"),
    )
    op.create_index("ix_scan_items_scan_id", "scan_items", ["scan_id"], unique=False)
    op.create_index("ix_scan_items_matched_drug_id", "scan_items", ["matched_drug_id"], unique=False)

    op.create_table(
        "scan_interaction_results",
        sa.Column("result_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("scan_id", sa.BigInteger(), nullable=False),
        sa.Column("interaction_id", sa.BigInteger(), nullable=False),
        sa.Column("ingredient_a_id", sa.BigInteger(), nullable=False),
        sa.Column("ingredient_b_id", sa.BigInteger(), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("warning_message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["ingredient_a_id"], ["ingredients.ingredient_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["ingredient_b_id"], ["ingredients.ingredient_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["interaction_id"], ["drug_interactions.interaction_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["scan_id"], ["scan_sessions.scan_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("result_id"),
    )
    op.create_index("ix_scan_interaction_results_scan_id", "scan_interaction_results", ["scan_id"], unique=False)
    op.create_index(
        "ix_scan_interaction_results_interaction_id",
        "scan_interaction_results",
        ["interaction_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_scan_interaction_results_interaction_id", table_name="scan_interaction_results")
    op.drop_index("ix_scan_interaction_results_scan_id", table_name="scan_interaction_results")
    op.drop_table("scan_interaction_results")
    op.drop_index("ix_scan_items_matched_drug_id", table_name="scan_items")
    op.drop_index("ix_scan_items_scan_id", table_name="scan_items")
    op.drop_table("scan_items")
    op.drop_index("ix_scan_sessions_patient_id", table_name="scan_sessions")
    op.drop_table("scan_sessions")
    op.drop_index("ix_drug_interactions_ingredient_b_id", table_name="drug_interactions")
    op.drop_index("ix_drug_interactions_ingredient_a_id", table_name="drug_interactions")
    op.drop_table("drug_interactions")
    op.drop_table("product_ingredients")
    op.drop_index("ix_drug_appearances_color", table_name="drug_appearances")
    op.drop_index("ix_drug_appearances_shape", table_name="drug_appearances")
    op.drop_index("ix_drug_appearances_imprint", table_name="drug_appearances")
    op.drop_index("ix_drug_appearances_drug_id", table_name="drug_appearances")
    op.drop_table("drug_appearances")
    op.drop_table("patient_profiles")
    op.drop_index(op.f("ix_ingredients_rxcui"), table_name="ingredients")
    op.drop_index(op.f("ix_ingredients_name"), table_name="ingredients")
    op.drop_table("ingredients")
    op.drop_index(op.f("ix_drug_products_dosage_form"), table_name="drug_products")
    op.drop_index(op.f("ix_drug_products_product_code"), table_name="drug_products")
    op.drop_table("drug_products")
