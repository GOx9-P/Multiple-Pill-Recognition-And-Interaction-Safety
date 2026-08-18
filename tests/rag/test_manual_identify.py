from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from pill_safety.database.base import Base
from pill_safety.database.models import DrugAppearance, DrugProduct, Ingredient, ProductIngredient
from pill_safety.rag.identification_service import IdentificationService


def make_test_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    DrugProduct.__table__.create(engine)
    DrugAppearance.__table__.create(engine)
    Ingredient.__table__.create(engine)
    ProductIngredient.__table__.create(engine)
    session = Session(engine)


    
    product = DrugProduct(
        drug_id=1,
        product_code="0093-5056",
        name="Cordarone 200mg",
        dosage_form="TABLET",
        market="US",
        active=True,
    )
    session.add(product)
    
    appearance = DrugAppearance(
        appearance_id=1,
        drug_id=1,
        imprint="TV5056",
        imprint_normalized="TV5056",
        shape="OVAL",
        color="WHITE",
    )
    session.add(appearance)
    session.commit()
    return session


def test_manual_bind_unresolved_pill_success() -> None:
    db = make_test_session()
    try:
        service = IdentificationService(db)
        result = service.manual_bind_unresolved_pill(
            instance_id="pill_manual_1",
            manual_drug_name="Cordarone"
        )
        assert result["instance_id"] == "pill_manual_1"
        assert result["identification_status"] == "identified"
        assert result["accepted_product"]["drug_id"] == 1
        assert result["accepted_product"]["product_name"] == "Cordarone 200mg"
        assert "manual_user_override" in result["decision_reasons"]
    finally:
        db.close()


def test_manual_bind_unresolved_pill_not_found() -> None:
    db = make_test_session()
    try:
        service = IdentificationService(db)
        with pytest.raises(ValueError, match="Không tìm thấy sản phẩm thuốc phù hợp"):
            service.manual_bind_unresolved_pill(
                instance_id="pill_manual_2",
                manual_drug_name="NON_EXISTENT_DRUG_XYZ_12345"
            )
    finally:
        db.close()
