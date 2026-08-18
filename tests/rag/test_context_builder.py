from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from pill_safety.database.models import (
    DrugAppearance,
    DrugProduct,
    Ingredient,
    ProductIngredient,
    DrugInteraction,
)
from pill_safety.rag.ddi.ddi_lookup_service import DdiLookupService
from pill_safety.rag.reporting.context_builder import ContextBuilderService


def make_test_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    # Create all tables
    DrugProduct.__table__.create(engine)
    DrugAppearance.__table__.create(engine)
    Ingredient.__table__.create(engine)
    ProductIngredient.__table__.create(engine)
    DrugInteraction.__table__.create(engine)
    
    session = Session(engine)
    
    # 1. Seed Ingredients
    ing1 = Ingredient(ingredient_id=1, name="Amiodarone", normalized_name="amiodarone")
    ing2 = Ingredient(ingredient_id=2, name="Levofloxacin", normalized_name="levofloxacin")
    ing3 = Ingredient(ingredient_id=3, name="Paracetamol", normalized_name="paracetamol")
    session.add_all([ing1, ing2, ing3])
    
    # 2. Seed Drug Products
    prod1 = DrugProduct(
        drug_id=1,
        product_code="0093-5056",
        name="Cordarone 200mg",
        dosage_form="TABLET",
        market="US",
        active=True,
    )
    prod2 = DrugProduct(
        drug_id=2,
        product_code="72789-401",
        name="Tavanic 500mg",
        dosage_form="TABLET",
        market="US",
        active=True,
    )
    prod3 = DrugProduct(
        drug_id=3,
        product_code="61919-448",
        name="Panadol Extra",
        dosage_form="TABLET",
        market="US",
        active=True,
    )
    session.add_all([prod1, prod2, prod3])
    session.flush()
    
    # 3. Seed Product Ingredients (mapping)
    session.add_all([
        ProductIngredient(drug_id=1, ingredient_id=1, strength="200 mg"),
        ProductIngredient(drug_id=2, ingredient_id=2, strength="500 mg"),
        ProductIngredient(drug_id=3, ingredient_id=3, strength="500 mg"), # Panadol has paracetamol
        ProductIngredient(drug_id=3, ingredient_id=1, strength="10 mg"),  # Let's mock Panadol as having Amiodarone too for duplicate warning testing!
    ])
    
    # 4. Seed Interactions
    # ingredient_a_id < ingredient_b_id (1 < 2)
    session.add(
        DrugInteraction(
            interaction_id=1,
            ingredient_a_id=1,
            ingredient_b_id=2,
            severity="contraindicated",
            description="High risk of QT prolongation and torsades de pointes.",
            recommendation="Avoid combination.",
            source="DDInter",
            clinical_risk="Cardiac arrest",
            mechanism="Additive effects",
            management="Do not co-administer",
            alternative="Use alternative antibiotic"
        )
    )
    
    session.commit()
    return session


def test_ddi_lookup_and_duplicates() -> None:
    session = make_test_session()
    service = DdiLookupService(session)

    # Test request matching ddi_request_v0
    ddi_req = {
        "schema_version": "ddi_request_v0",
        "request_id": "req_001",
        "session_id": "sess_001",
        "identified_products": [
            {"instance_id": "pill_001", "product_id": "drug_1"},
            {"instance_id": "pill_002", "product_id": "drug_2"},
            {"instance_id": "pill_003", "product_id": "drug_3"}, # Contains Amiodarone and Paracetamol
        ]
    }

    res = service.lookup_ddi(ddi_req)

    assert res["schema_version"] == "ddi_output_v0"
    assert res["request_id"] == "req_001"
    assert len(res["identified_drugs"]) == 3
    
    # Verify overall severity (should be contraindicated)
    assert res["overall_severity"] == "contraindicated"

    # Verify DDI interaction between pill_001 (Amiodarone) and pill_002 (Levofloxacin)
    # also pill_003 has Amiodarone, so source_instance_ids should include it
    assert len(res["interactions"]) == 1
    interaction = res["interactions"][0]
    assert interaction["severity"] == "contraindicated"
    assert "pill_001" in interaction["source_instance_ids"]
    assert "pill_002" in interaction["source_instance_ids"]
    assert "pill_003" in interaction["source_instance_ids"]

    # Verify duplicate ingredient warning on Amiodarone (present in pill_001 and pill_003)
    assert len(res["duplicate_ingredient_warnings"]) == 1
    dup = res["duplicate_ingredient_warnings"][0]
    assert dup["ingredient_name"] == "Amiodarone"
    assert dup["source_instance_ids"] == ["pill_001", "pill_003"]
    assert dup["severity"] == "major"


def test_ddi_lookup_not_found() -> None:
    session = make_test_session()
    service = DdiLookupService(session)

    ddi_req = {
        "schema_version": "ddi_request_v0",
        "request_id": "req_002",
        "identified_products": [
            {"instance_id": "pill_001", "product_id": "drug_9999"} # Non-existent drug
        ]
    }

    with pytest.raises(ValueError, match="Drug product not found"):
        service.lookup_ddi(ddi_req)


def test_context_builder() -> None:
    session = make_test_session()
    ddi_service = DdiLookupService(session)
    builder = ContextBuilderService()

    ddi_req = {
        "request_id": "req_100",
        "session_id": "sess_100",
        "identified_products": [
            {"instance_id": "pill_001", "product_id": "drug_1"},
            {"instance_id": "pill_002", "product_id": "drug_2"}
        ]
    }
    ddi_res = ddi_service.lookup_ddi(ddi_req)

    # Mock RAG identification output containing an ambiguous pill
    rag_res = {
        "schema_version": "rag_identification_v0",
        "pill_results": [
            {
                "instance_id": "pill_001",
                "identification_status": "identified",
            },
            {
                "instance_id": "pill_002",
                "identification_status": "identified",
            },
            {
                "instance_id": "pill_003",
                "identification_status": "ambiguous",
                "required_action": "capture_reverse_side",
                "decision_reasons": ["top_candidates_too_close"]
            }
        ]
    }

    context_input = {
        "schema_version": "context_builder_input_v0",
        "request_id": "req_100",
        "session_id": "sess_100",
        "cv_output": {},
        "rag_identification": rag_res,
        "ddi_output": ddi_res
    }

    context = builder.build_context(context_input)

    assert context["schema_version"] == "llm_context_v0"
    assert context["request_id"] == "req_100"
    assert context["task"] == "format_grounded_medication_safety_report"
    
    # 2 resolved drugs, 1 unresolved pill
    assert len(context["identified_drugs"]) == 2
    assert len(context["unresolved_pills"]) == 1
    
    unresolved = context["unresolved_pills"][0]
    assert unresolved["instance_id"] == "pill_003"
    assert unresolved["identification_status"] == "ambiguous"
    assert unresolved["reason"] == "top_candidates_too_close"
    assert unresolved["required_action"] == "capture_reverse_side"

    # Verify scope warnings: should have both because of pill_003
    assert "only_identified_drugs_checked" in context["scope_warnings"]
    assert "no_interaction_found_does_not_mean_safe" in context["scope_warnings"]

    # Verify sources list (contains DailyMed and DDInter)
    assert len(context["sources"]) == 2
    source_names = [s["source_name"] for s in context["sources"]]
    assert "DailyMed" in source_names
    assert "DDInter" in source_names


def test_safety_limit_validation() -> None:
    from pydantic import ValidationError
    from pill_safety.schemas.rag import CvOutput, DdiRequest

    # 1. Test DdiRequest validation limit (> 15 products)
    products_over_limit = [{"instance_id": f"pill_{i}", "product_id": f"drug_{i}"} for i in range(16)]
    with pytest.raises(ValidationError, match="vượt quá giới hạn an toàn là 15 viên"):
        DdiRequest(
            schema_version="ddi_request_v0",
            request_id="req_limit",
            identified_products=products_over_limit
        )

    # 2. Test CvOutput validation limit (> 15 pills)
    pills_over_limit = [{"instance_id": f"pill_{i}"} for i in range(16)]
    with pytest.raises(ValidationError, match="vượt quá giới hạn an toàn là 15 viên"):
        CvOutput(
            schema_version="cv_output_v0",
            request_id="req_limit",
            pills=pills_over_limit
        )

