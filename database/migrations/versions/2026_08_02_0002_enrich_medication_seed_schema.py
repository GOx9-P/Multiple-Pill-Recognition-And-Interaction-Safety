"""enrich medication seed schema

Revision ID: 202608020002
Revises: 202608010001
Create Date: 2026-08-02 00:02:00.000000+00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "202608020002"
down_revision = "202608010001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("drug_products", sa.Column("generic_name", sa.String(length=255), nullable=True))
    op.add_column("drug_products", sa.Column("route", sa.String(length=100), nullable=True))
    op.add_column("drug_products", sa.Column("market", sa.String(length=20), server_default="US", nullable=False))
    op.add_column("drug_products", sa.Column("product_rxcui", sa.String(length=50), nullable=True))
    op.add_column("drug_products", sa.Column("source_name", sa.String(length=100), nullable=True))
    op.add_column("drug_products", sa.Column("source_reference", sa.String(length=500), nullable=True))
    op.add_column("drug_products", sa.Column("source_set_id", sa.String(length=100), nullable=True))
    op.add_column("drug_products", sa.Column("spl_version", sa.String(length=50), nullable=True))
    op.add_column("drug_products", sa.Column("published_date", sa.String(length=50), nullable=True))
    op.add_column("drug_products", sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False))
    op.add_column(
        "drug_products",
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index(op.f("ix_drug_products_product_rxcui"), "drug_products", ["product_rxcui"], unique=False)
    op.create_index(op.f("ix_drug_products_source_set_id"), "drug_products", ["source_set_id"], unique=False)

    op.add_column("ingredients", sa.Column("normalized_name", sa.String(length=255), nullable=True))
    op.add_column("ingredients", sa.Column("unii", sa.String(length=50), nullable=True))
    op.add_column("ingredients", sa.Column("source_name", sa.String(length=100), nullable=True))
    op.add_column("ingredients", sa.Column("source_reference", sa.String(length=500), nullable=True))
    op.create_index(op.f("ix_ingredients_normalized_name"), "ingredients", ["normalized_name"], unique=False)
    op.create_index(op.f("ix_ingredients_unii"), "ingredients", ["unii"], unique=False)

    op.add_column("drug_appearances", sa.Column("imprint_raw", sa.String(length=255), nullable=True))
    op.add_column("drug_appearances", sa.Column("imprint_normalized", sa.String(length=255), nullable=True))
    op.add_column("drug_appearances", sa.Column("imprint_side_a", sa.String(length=100), nullable=True))
    op.add_column("drug_appearances", sa.Column("imprint_side_b", sa.String(length=100), nullable=True))
    op.add_column("drug_appearances", sa.Column("primary_color", sa.String(length=100), nullable=True))
    op.add_column("drug_appearances", sa.Column("secondary_color", sa.String(length=100), nullable=True))
    op.add_column("drug_appearances", sa.Column("color_pattern", sa.String(length=100), nullable=True))
    op.add_column("drug_appearances", sa.Column("coating", sa.String(length=100), nullable=True))
    op.add_column("drug_appearances", sa.Column("source_name", sa.String(length=100), nullable=True))
    op.add_column("drug_appearances", sa.Column("source_reference", sa.String(length=500), nullable=True))
    op.create_index(
        op.f("ix_drug_appearances_imprint_normalized"),
        "drug_appearances",
        ["imprint_normalized"],
        unique=False,
    )

    op.add_column("product_ingredients", sa.Column("strength_value", sa.Numeric(precision=12, scale=4), nullable=True))
    op.add_column("product_ingredients", sa.Column("strength_unit", sa.String(length=50), nullable=True))
    op.add_column("product_ingredients", sa.Column("numerator_text", sa.String(length=100), nullable=True))
    op.add_column("product_ingredients", sa.Column("source_ingredient_name", sa.String(length=255), nullable=True))

    op.add_column("drug_interactions", sa.Column("clinical_risk", sa.Text(), nullable=True))
    op.add_column("drug_interactions", sa.Column("mechanism", sa.Text(), nullable=True))
    op.add_column("drug_interactions", sa.Column("management", sa.Text(), nullable=True))
    op.add_column("drug_interactions", sa.Column("alternative", sa.Text(), nullable=True))
    op.add_column("drug_interactions", sa.Column("source_name", sa.String(length=100), nullable=True))
    op.add_column("drug_interactions", sa.Column("source_reference", sa.String(length=500), nullable=True))
    op.add_column("drug_interactions", sa.Column("source_level", sa.String(length=50), nullable=True))
    op.add_column("drug_interactions", sa.Column("evidence_text", sa.Text(), nullable=True))
    op.add_column("drug_interactions", sa.Column("last_reviewed", sa.String(length=50), nullable=True))

    op.add_column("scan_sessions", sa.Column("request_id", sa.String(length=100), nullable=True))
    op.add_column("scan_sessions", sa.Column("session_key", sa.String(length=100), nullable=True))
    op.add_column("scan_sessions", sa.Column("image_id", sa.String(length=100), nullable=True))
    op.add_column("scan_sessions", sa.Column("image_quality", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("scan_sessions", sa.Column("scope_warnings", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("scan_sessions", sa.Column("llm_context", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("scan_sessions", sa.Column("llm_report_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.create_index(op.f("ix_scan_sessions_request_id"), "scan_sessions", ["request_id"], unique=False)
    op.create_index(op.f("ix_scan_sessions_session_key"), "scan_sessions", ["session_key"], unique=False)

    op.add_column("scan_items", sa.Column("mask_path", sa.String(length=500), nullable=True))
    op.add_column("scan_items", sa.Column("cv_status", sa.String(length=50), nullable=True))
    op.add_column("scan_items", sa.Column("cv_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("scan_items", sa.Column("candidate_generation", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("scan_items", sa.Column("top_candidates", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("scan_items", sa.Column("ranking_evidence", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("scan_items", sa.Column("required_action", sa.String(length=100), nullable=True))
    op.add_column("scan_items", sa.Column("scope_warning", sa.String(length=255), nullable=True))
    op.add_column("scan_items", sa.Column("top2_margin", sa.Numeric(precision=5, scale=4), nullable=True))
    op.drop_constraint("ck_scan_items_identification_status", "scan_items", type_="check")
    op.create_check_constraint(
        "ck_scan_items_identification_status",
        "scan_items",
        "identification_status IN ('IDENTIFIED', 'AMBIGUOUS', 'UNKNOWN', 'INSUFFICIENT_EVIDENCE')",
    )

    op.alter_column("scan_interaction_results", "interaction_id", existing_type=sa.BigInteger(), nullable=True)
    op.add_column("scan_interaction_results", sa.Column("result_type", sa.String(length=50), nullable=True))
    op.add_column(
        "scan_interaction_results",
        sa.Column("source_instance_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column("scan_interaction_results", sa.Column("ingredient_a_name", sa.String(length=255), nullable=True))
    op.add_column("scan_interaction_results", sa.Column("ingredient_b_name", sa.String(length=255), nullable=True))
    op.add_column("scan_interaction_results", sa.Column("clinical_risk", sa.Text(), nullable=True))
    op.add_column("scan_interaction_results", sa.Column("mechanism", sa.Text(), nullable=True))
    op.add_column("scan_interaction_results", sa.Column("management", sa.Text(), nullable=True))
    op.add_column("scan_interaction_results", sa.Column("source", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column(
        "scan_interaction_results",
        sa.Column("evidence_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("scan_interaction_results", "evidence_snapshot")
    op.drop_column("scan_interaction_results", "source")
    op.drop_column("scan_interaction_results", "management")
    op.drop_column("scan_interaction_results", "mechanism")
    op.drop_column("scan_interaction_results", "clinical_risk")
    op.drop_column("scan_interaction_results", "ingredient_b_name")
    op.drop_column("scan_interaction_results", "ingredient_a_name")
    op.drop_column("scan_interaction_results", "source_instance_ids")
    op.drop_column("scan_interaction_results", "result_type")
    op.alter_column("scan_interaction_results", "interaction_id", existing_type=sa.BigInteger(), nullable=False)

    op.drop_constraint("ck_scan_items_identification_status", "scan_items", type_="check")
    op.create_check_constraint(
        "ck_scan_items_identification_status",
        "scan_items",
        "identification_status IN ('IDENTIFIED', 'AMBIGUOUS', 'UNKNOWN')",
    )
    op.drop_column("scan_items", "top2_margin")
    op.drop_column("scan_items", "scope_warning")
    op.drop_column("scan_items", "required_action")
    op.drop_column("scan_items", "ranking_evidence")
    op.drop_column("scan_items", "top_candidates")
    op.drop_column("scan_items", "candidate_generation")
    op.drop_column("scan_items", "cv_payload")
    op.drop_column("scan_items", "cv_status")
    op.drop_column("scan_items", "mask_path")

    op.drop_index(op.f("ix_scan_sessions_session_key"), table_name="scan_sessions")
    op.drop_index(op.f("ix_scan_sessions_request_id"), table_name="scan_sessions")
    op.drop_column("scan_sessions", "llm_report_payload")
    op.drop_column("scan_sessions", "llm_context")
    op.drop_column("scan_sessions", "scope_warnings")
    op.drop_column("scan_sessions", "image_quality")
    op.drop_column("scan_sessions", "image_id")
    op.drop_column("scan_sessions", "session_key")
    op.drop_column("scan_sessions", "request_id")

    op.drop_column("drug_interactions", "last_reviewed")
    op.drop_column("drug_interactions", "evidence_text")
    op.drop_column("drug_interactions", "source_level")
    op.drop_column("drug_interactions", "source_reference")
    op.drop_column("drug_interactions", "source_name")
    op.drop_column("drug_interactions", "alternative")
    op.drop_column("drug_interactions", "management")
    op.drop_column("drug_interactions", "mechanism")
    op.drop_column("drug_interactions", "clinical_risk")

    op.drop_column("product_ingredients", "source_ingredient_name")
    op.drop_column("product_ingredients", "numerator_text")
    op.drop_column("product_ingredients", "strength_unit")
    op.drop_column("product_ingredients", "strength_value")

    op.drop_index(op.f("ix_drug_appearances_imprint_normalized"), table_name="drug_appearances")
    op.drop_column("drug_appearances", "source_reference")
    op.drop_column("drug_appearances", "source_name")
    op.drop_column("drug_appearances", "coating")
    op.drop_column("drug_appearances", "color_pattern")
    op.drop_column("drug_appearances", "secondary_color")
    op.drop_column("drug_appearances", "primary_color")
    op.drop_column("drug_appearances", "imprint_side_b")
    op.drop_column("drug_appearances", "imprint_side_a")
    op.drop_column("drug_appearances", "imprint_normalized")
    op.drop_column("drug_appearances", "imprint_raw")

    op.drop_index(op.f("ix_ingredients_unii"), table_name="ingredients")
    op.drop_index(op.f("ix_ingredients_normalized_name"), table_name="ingredients")
    op.drop_column("ingredients", "source_reference")
    op.drop_column("ingredients", "source_name")
    op.drop_column("ingredients", "unii")
    op.drop_column("ingredients", "normalized_name")

    op.drop_index(op.f("ix_drug_products_source_set_id"), table_name="drug_products")
    op.drop_index(op.f("ix_drug_products_product_rxcui"), table_name="drug_products")
    op.drop_column("drug_products", "updated_at")
    op.drop_column("drug_products", "active")
    op.drop_column("drug_products", "published_date")
    op.drop_column("drug_products", "spl_version")
    op.drop_column("drug_products", "source_set_id")
    op.drop_column("drug_products", "source_reference")
    op.drop_column("drug_products", "source_name")
    op.drop_column("drug_products", "product_rxcui")
    op.drop_column("drug_products", "market")
    op.drop_column("drug_products", "route")
    op.drop_column("drug_products", "generic_name")
