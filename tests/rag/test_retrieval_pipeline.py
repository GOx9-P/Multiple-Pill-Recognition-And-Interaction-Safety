from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from pill_safety.database.models import DrugAppearance, DrugProduct
from pill_safety.rag.retrieval.candidate_retriever import CandidateRetriever
from pill_safety.rag.retrieval.cv_input_adapter import adapt_cv_pill_to_recognition_input
from pill_safety.rag.retrieval.idf_statistics import IdfStatisticsBuilder
from pill_safety.rag.identification_service import IdentificationService
from pill_safety.rag.retrieval.normalization import normalize_dosage_form, normalize_imprint
from pill_safety.rag.retrieval.similarity import weighted_edit_similarity


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    DrugProduct.__table__.create(engine)
    DrugAppearance.__table__.create(engine)
    session = Session(engine)
    seed_products(session)
    return session


def seed_products(session: Session) -> None:
    products = [
        DrugProduct(
            drug_id=1,
            product_code="0093-5056",
            name="ATORVASTATIN CALCIUM 10 mg tablet, film coated",
            dosage_form="TABLET, FILM COATED",
            market="US",
            active=True,
        ),
        DrugProduct(
            drug_id=2,
            product_code="72789-401",
            name="PREDNISONE 10 mg tablet",
            dosage_form="TABLET",
            market="US",
            active=True,
        ),
        DrugProduct(
            drug_id=3,
            product_code="61919-448",
            name="GLIMEPIRIDE 2 mg tablet",
            dosage_form="TABLET",
            market="US",
            active=True,
        ),
        DrugProduct(
            drug_id=4,
            product_code="67046-520",
            name="OMEPRAZOLE 20 mg capsule",
            dosage_form="CAPSULE, DELAYED RELEASE PELLETS",
            market="US",
            active=True,
        ),
    ]
    appearances = [
        DrugAppearance(
            appearance_id=1,
            drug_id=1,
            imprint="TV5056",
            imprint_normalized="TV5056",
            shape="OVAL",
            color="WHITE",
            primary_color="WHITE",
            score_line=False,
            logo_or_symbol=False,
        ),
        DrugAppearance(
            appearance_id=2,
            drug_id=2,
            imprint="059",
            imprint_normalized="059",
            shape="ROUND",
            color="WHITE",
            primary_color="WHITE",
            score_line=True,
            logo_or_symbol=False,
        ),
        DrugAppearance(
            appearance_id=3,
            drug_id=3,
            imprint="AHI2",
            imprint_normalized="AHI2",
            shape="OVAL",
            color="GREEN",
            primary_color="GREEN",
            score_line=True,
            logo_or_symbol=False,
        ),
        DrugAppearance(
            appearance_id=4,
            drug_id=4,
            imprint="KU118",
            imprint_normalized="KU118",
            shape="CAPSULE",
            color="WHITE",
            primary_color="WHITE",
            score_line=False,
            logo_or_symbol=False,
        ),
    ]
    session.add_all([*products, *appearances])
    session.commit()


def cv_request_for_pill(pill: dict) -> dict:
    return {
        "schema_version": "rag_request_v0",
        "request_id": "req_001",
        "session_id": "sess_001",
        "market": "US",
        "known_drug_names": [],
        "cv_output": {
            "schema_version": "cv_output_v0",
            "request_id": "req_001",
            "session_id": "sess_001",
            "image_id": "img_001",
            "image_quality": {
                "status": "usable",
                "blur_score": 0.1,
                "glare_detected": False,
                "lighting_warning": False,
            },
            "pills": [pill],
        },
    }


def base_pill(**overrides: object) -> dict:
    pill = {
        "instance_id": "pill_001",
        "instance_token": "pill_token_001",
        "side_hint": "unknown",
        "cv_status": "features_ready",
        "segmentation": {
            "confidence": 0.96,
            "occlusion_estimate": 0.1,
            "possible_merged_instance": False,
            "possible_non_pill": False,
        },
        "shape": {"label": "oval", "confidence": 0.91, "alternatives": []},
        "color": {
            "primary": "white",
            "secondary": None,
            "distribution": {"white": 0.8, "green": 0.05},
            "confidence": 0.88,
            "lighting_warning": False,
        },
        "dosage_form": {"label": "tablet", "confidence": 0.89},
        "scoreline": {"label": "none", "visible": False, "confidence": 0.81},
        "logo_or_symbol": {"visible": False, "confidence": 0.63},
        "imprint_visibility": {"visible": True, "confidence": 0.86},
        "imprint": {
            "visible": True,
            "raw": "TV5056",
            "confidence": 0.9,
            "normalized_candidates": [{"text": "TV5056", "score": 0.95, "source": "test"}],
        },
        "quality_flags": [],
    }
    pill.update(overrides)
    return pill


def test_normalization_matches_cv_and_database_contract() -> None:
    assert normalize_imprint("TV; 5056") == "TV5056"
    assert normalize_dosage_form("TABLET, FILM COATED") == "TABLET"


def test_weighted_edit_similarity_discounts_common_ocr_confusions() -> None:
    assert weighted_edit_similarity("A01", "AO1") > weighted_edit_similarity("A01", "P50")


def test_adapter_maps_nested_cv_schema_to_recognition_input() -> None:
    request = cv_request_for_pill(base_pill())
    cv_output = request["cv_output"]
    normalized = adapt_cv_pill_to_recognition_input(
        rag_request=request,
        cv_output=cv_output,
        pill=cv_output["pills"][0],
    )

    assert normalized.imprint_candidates[0].text == "TV5056"
    assert normalized.shape.label == "OVAL"
    assert normalized.color.distribution["WHITE"] == 0.8
    assert normalized.dosage_form.label == "TABLET"
    assert normalized.scoreline.visible is False


def test_idf_weights_make_common_color_weaker_than_rare_color() -> None:
    session = make_session()
    try:
        stats = IdfStatisticsBuilder.from_database(session)
        assert stats.get_weight("primary_color", "GREEN") > stats.get_weight("primary_color", "WHITE")
    finally:
        session.close()


def test_identification_service_identifies_exact_imprint() -> None:
    session = make_session()
    try:
        result = IdentificationService(session).identify(cv_request_for_pill(base_pill()))
    finally:
        session.close()

    pill_result = result["pill_results"][0]
    assert pill_result["identification_status"] == "identified"
    assert pill_result["accepted_product"]["product_code"] == "0093-5056"
    assert pill_result["candidate_generation"]["strategy"] == "imprint_first"


def test_identification_service_handles_ocr_confusion_variant() -> None:
    pill = base_pill(
        imprint={
            "visible": True,
            "raw": "TV5O56",
            "confidence": 0.9,
            "normalized_candidates": [{"text": "TV5O56", "score": 0.95, "source": "test"}],
        }
    )
    session = make_session()
    try:
        result = IdentificationService(session).identify(cv_request_for_pill(pill))
    finally:
        session.close()

    pill_result = result["pill_results"][0]
    assert pill_result["top_candidates"][0]["product_code"] == "0093-5056"
    assert pill_result["identification_status"] in {"identified", "ambiguous"}


def test_identification_service_does_not_identify_without_usable_imprint() -> None:
    pill = base_pill(
        imprint_visibility={"visible": False, "confidence": 0.9},
        imprint={"visible": False, "raw": "", "confidence": 0.0, "normalized_candidates": []},
    )
    session = make_session()
    try:
        result = IdentificationService(session).identify(cv_request_for_pill(pill))
    finally:
        session.close()

    pill_result = result["pill_results"][0]
    assert pill_result["identification_status"] != "identified"
    assert "no_usable_imprint" in pill_result["decision_reasons"]


def test_identification_service_rejects_merged_instance_for_identification() -> None:
    pill = base_pill(
        segmentation={
            "confidence": 0.96,
            "occlusion_estimate": 0.1,
            "possible_merged_instance": True,
            "possible_non_pill": False,
        }
    )
    session = make_session()
    try:
        result = IdentificationService(session).identify(cv_request_for_pill(pill))
    finally:
        session.close()

    pill_result = result["pill_results"][0]
    assert pill_result["identification_status"] != "identified"
    assert "possible_merged_instance" in pill_result["decision_reasons"]


def test_identification_service_returns_unknown_for_out_of_database_imprint() -> None:
    pill = base_pill(
        imprint={
            "visible": True,
            "raw": "XXXX999",
            "confidence": 0.95,
            "normalized_candidates": [{"text": "XXXX999", "score": 0.95, "source": "test"}],
        }
    )
    session = make_session()
    try:
        result = IdentificationService(session).identify(cv_request_for_pill(pill))
    finally:
        session.close()

    pill_result = result["pill_results"][0]
    assert pill_result["identification_status"] == "unknown"
    assert pill_result["accepted_product"] is None


def test_cv_input_adapter_market_normalization() -> None:
    request = cv_request_for_pill(base_pill())
    request["market"] = "  us  "
    cv_output = request["cv_output"]
    normalized = adapt_cv_pill_to_recognition_input(
        rag_request=request,
        cv_output=cv_output,
        pill=cv_output["pills"][0],
    )
    assert normalized.market == "US"


def test_identification_service_dosage_form_hard_reject() -> None:
    pill = base_pill(
        dosage_form={"label": "capsule", "confidence": 0.96}
    )
    session = make_session()
    try:
        result = IdentificationService(session).identify(cv_request_for_pill(pill))
    finally:
        session.close()
    
    pill_result = result["pill_results"][0]
    assert pill_result["identification_status"] != "identified"
    assert any(c["evidence"]["hard_reject"] for c in pill_result["top_candidates"])
    assert any("dosage_form_contradiction" in c["evidence"]["hard_reject_reasons"] for c in pill_result["top_candidates"])


def test_identification_service_dosage_form_no_hard_reject_on_low_confidence() -> None:
    pill = base_pill(
        dosage_form={"label": "capsule", "confidence": 0.9}
    )
    session = make_session()
    try:
        result = IdentificationService(session).identify(cv_request_for_pill(pill))
    finally:
        session.close()
    
    pill_result = result["pill_results"][0]
    assert not any(c["evidence"]["hard_reject"] for c in pill_result["top_candidates"])


def make_custom_session(products: list[DrugProduct], appearances: list[DrugAppearance]) -> Session:
    engine = create_engine("sqlite:///:memory:")
    DrugProduct.__table__.create(engine)
    DrugAppearance.__table__.create(engine)
    session = Session(engine)
    session.add_all([*products, *appearances])
    session.commit()
    return session


def test_evidence_scorer_secondary_color_matching() -> None:
    products = [
        DrugProduct(
            drug_id=10,
            product_code="10-10",
            name="TEST DRUG",
            dosage_form="TABLET",
            market="US",
            active=True,
        )
    ]
    appearances = [
        DrugAppearance(
            appearance_id=10,
            drug_id=10,
            imprint="TEST1",
            imprint_normalized="TEST1",
            shape="ROUND",
            color="RED/WHITE",
            primary_color="RED",
            secondary_color="WHITE",
            score_line=False,
            logo_or_symbol=False,
        )
    ]
    session = make_custom_session(products, appearances)
    try:
        pill = base_pill(
            imprint={
                "visible": True,
                "raw": "TEST1",
                "confidence": 0.95,
                "normalized_candidates": [{"text": "TEST1", "score": 0.95, "source": "test"}],
            },
            shape={"label": "round", "confidence": 0.9},
            color={
                "primary": "red",
                "secondary": "white",
                "distribution": {"RED": 0.6, "WHITE": 0.4},
                "confidence": 0.9,
                "lighting_warning": False,
            }
        )
        result = IdentificationService(session).identify(cv_request_for_pill(pill))
        pill_result = result["pill_results"][0]
        assert pill_result["identification_status"] == "identified"
        top_cand = pill_result["top_candidates"][0]
        color_field = top_cand["evidence"]["fields"]["color"]
        assert abs(color_field["match_score"] - 0.54) < 1e-4
    finally:
        session.close()


def test_evidence_scorer_color_quality_multiplier() -> None:
    session = make_session()
    try:
        pill_lw = base_pill(
            color={
                "primary": "white",
                "secondary": None,
                "distribution": {"WHITE": 0.8},
                "confidence": 0.9,
                "lighting_warning": True,
            }
        )
        result_lw = IdentificationService(session).identify(cv_request_for_pill(pill_lw))
        color_field_lw = result_lw["pill_results"][0]["top_candidates"][0]["evidence"]["fields"]["color"]
        assert color_field_lw["quality_multiplier"] == 0.5

        pill_glare = base_pill()
        req_glare = cv_request_for_pill(pill_glare)
        req_glare["cv_output"]["image_quality"]["glare_detected"] = True
        result_glare = IdentificationService(session).identify(req_glare)
        color_field_glare = result_glare["pill_results"][0]["top_candidates"][0]["evidence"]["fields"]["color"]
        assert color_field_glare["quality_multiplier"] == 0.7

        pill_mf = base_pill(quality_flags=["minor_glare"])
        result_mf = IdentificationService(session).identify(cv_request_for_pill(pill_mf))
        color_field_mf = result_mf["pill_results"][0]["top_candidates"][0]["evidence"]["fields"]["color"]
        assert abs(color_field_mf["quality_multiplier"] - 0.85) < 1e-4
    finally:
        session.close()


def test_evidence_scorer_auxiliary_evidence_low_confidence() -> None:
    pill = base_pill(
        scoreline={"label": "single", "visible": True, "confidence": 0.3},
        logo_or_symbol={"visible": True, "confidence": 0.3}
    )
    session = make_session()
    try:
        result = IdentificationService(session).identify(cv_request_for_pill(pill))
        pill_result = result["pill_results"][0]
        top_cand = pill_result["top_candidates"][0]
        
        scoreline_field = top_cand["evidence"]["fields"]["scoreline"]
        assert scoreline_field["match_score"] == 0.5
        
        logo_field = top_cand["evidence"]["fields"]["logo_or_symbol"]
        assert logo_field["match_score"] == 0.5
    finally:
        session.close()


def test_safety_gate_pre_retrieval_decisions() -> None:
    session = make_session()
    try:
        pill_non = base_pill(
            segmentation={
                "confidence": 0.96,
                "occlusion_estimate": 0.1,
                "possible_merged_instance": False,
                "possible_non_pill": True,
            }
        )
        result_non = IdentificationService(session).identify(cv_request_for_pill(pill_non))
        res_non = result_non["pill_results"][0]
        assert res_non["identification_status"] == "unknown"
        assert res_non["required_action"] == "capture_clear_pill_image"
        assert "segmentation_possible_non_pill" in res_non["decision_reasons"]
        assert res_non["candidate_generation"]["strategy"] == "not_queried"

        pill_insuf = base_pill(cv_status="insufficient_visual_evidence")
        result_insuf = IdentificationService(session).identify(cv_request_for_pill(pill_insuf))
        res_insuf = result_insuf["pill_results"][0]
        assert res_insuf["identification_status"] == "insufficient_visual_evidence"
        assert res_insuf["required_action"] == "recapture_clear_image"
        assert "cv_status_insufficient_visual_evidence" in res_insuf["decision_reasons"]
        assert res_insuf["candidate_generation"]["strategy"] == "not_queried"
    finally:
        session.close()


def test_idf_statistics_caching_and_invalidation() -> None:
    session = make_session()
    try:
        IdfStatisticsBuilder.invalidate_cache()
        stats1 = IdfStatisticsBuilder.get_cached_statistics(session)
        stats2 = IdfStatisticsBuilder.get_cached_statistics(session)
        assert stats1 is stats2

        IdfStatisticsBuilder.invalidate_cache()
        stats3 = IdfStatisticsBuilder.get_cached_statistics(session)
        assert stats1 is not stats3
    finally:
        session.close()


def test_candidate_retriever_length_filtering() -> None:
    session = make_session()
    try:
        retriever = CandidateRetriever(session)
        pill = base_pill(
            imprint={
                "visible": True,
                "raw": "TV5056",
                "confidence": 0.9,
                "normalized_candidates": [{"text": "TV5056", "score": 0.95, "source": "test"}],
            }
        )
        request = cv_request_for_pill(pill)
        normalized = adapt_cv_pill_to_recognition_input(
            rag_request=request,
            cv_output=request["cv_output"],
            pill=pill,
        )
        diagnostics, candidates = retriever.retrieve(normalized)
        assert diagnostics.strategy == "imprint_first"
        assert len(candidates) > 0
    finally:
        session.close()


def test_candidate_retriever_fallback_by_attributes_uses_idf() -> None:
    pill = base_pill(
        imprint_visibility={"visible": False, "confidence": 0.0},
        imprint={"visible": False, "raw": "", "confidence": 0.0, "normalized_candidates": []},
        shape={"label": "oval", "confidence": 0.9, "alternatives": []},
        color={
            "primary": "green",
            "secondary": None,
            "distribution": {"GREEN": 0.8},
            "confidence": 0.9,
            "lighting_warning": False,
        },
        dosage_form={"label": "tablet", "confidence": 0.9}
    )
    request = cv_request_for_pill(pill)
    normalized = adapt_cv_pill_to_recognition_input(
        rag_request=request,
        cv_output=request["cv_output"],
        pill=pill,
    )
    
    session = make_session()
    try:
        idf_stats = IdfStatisticsBuilder.get_cached_statistics(session)
        retriever = CandidateRetriever(session)
        diagnostics, candidates = retriever.retrieve(normalized, idf_statistics=idf_stats)
        assert diagnostics.strategy == "attribute_fallback"
        assert candidates[0].appearance_id == 3
    finally:
        session.close()

