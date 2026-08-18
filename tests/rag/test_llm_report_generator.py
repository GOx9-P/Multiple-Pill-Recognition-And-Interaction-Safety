from __future__ import annotations

import pytest
from pill_safety.rag.reporting.llm_report_generator import LlmReportGenerator, FallbackLlmProvider


@pytest.fixture
def sample_context_major() -> dict:
    return {
        "schema_version": "llm_context_v0",
        "request_id": "req_test_001",
        "session_id": "sess_test_001",
        "task": "format_grounded_medication_safety_report",
        "identified_drugs": [
            {
                "instance_id": "pill_1",
                "product_id": "drug_1",
                "product_name": "Cordarone 200mg",
                "brand_name": "Cordarone 200mg",
                "generic_name": "Amiodarone",
                "dosage_form": "Tablet",
                "active_ingredients": [
                    {"ingredient_id": "ing_1", "name": "amiodarone", "strength": "200mg"}
                ]
            },
            {
                "instance_id": "pill_2",
                "product_id": "drug_2",
                "product_name": "Tavanic 500mg",
                "brand_name": "Tavanic 500mg",
                "generic_name": "Levofloxacin",
                "dosage_form": "Tablet",
                "active_ingredients": [
                    {"ingredient_id": "ing_2", "name": "levofloxacin", "strength": "500mg"}
                ]
            }
        ],
        "unresolved_pills": [
            {
                "instance_id": "pill_3",
                "identification_status": "unknown",
                "reason": "insufficient_visual_evidence",
                "required_action": "manual_input"
            }
        ],
        "interactions": [
            {
                "interaction_id": "ddi_100",
                "ingredient_a_id": "ing_1",
                "ingredient_b_id": "ing_2",
                "ingredient_a_name": "amiodarone",
                "ingredient_b_name": "levofloxacin",
                "severity": "major",
                "clinical_risk": "Gây kéo dài khoảng QT tim nghiêm trọng dẫn đến Xoắn đỉnh (Torsades de Pointes).",
                "mechanism": "Cộng dồn tác dụng kéo dài tái cực cơ tim.",
                "management": "Không uống kết hợp. Yêu cầu bác sĩ thay thế kháng sinh."
            }
        ],
        "duplicate_ingredient_warnings": [],
        "scope_warnings": ["only_identified_drugs_checked"]
    }


@pytest.fixture
def sample_context_safe() -> dict:
    return {
        "schema_version": "llm_context_v0",
        "request_id": "req_test_002",
        "session_id": "sess_test_002",
        "task": "format_grounded_medication_safety_report",
        "identified_drugs": [
            {
                "instance_id": "pill_1",
                "product_id": "drug_1",
                "product_name": "Panadol 500mg",
                "brand_name": "Panadol 500mg",
                "active_ingredients": [
                    {"ingredient_id": "ing_10", "name": "paracetamol", "strength": "500mg"}
                ]
            }
        ],
        "unresolved_pills": [],
        "interactions": [],
        "duplicate_ingredient_warnings": [],
        "scope_warnings": []
    }


def test_llm_report_generator_major_severity(sample_context_major: dict) -> None:
    generator = LlmReportGenerator(provider_name="fallback")
    result = generator.generate_report(sample_context_major)

    assert result["schema_version"] == "llm_report_v0"
    assert result["request_id"] == "req_test_001"
    assert result["overall_severity"] == "major"
    assert "fallback" in result["provider_used"]
    
    text = result["formatted_report_text"]
    assert "[🔴 MỨC ĐỘ BÁO ĐỘNG: CỰC KỲ NGUY HIỂM]" in text
    assert "1. KẾT QUẢ NHẬN DIỆN THUỐC" in text
    assert "Cordarone 200mg" in text
    assert "Tavanic 500mg" in text
    assert "pill_3" in text
    assert "2. CHI TIẾT CÁC TƯƠNG TÁC GÂY HẠI" in text
    assert "amiodarone" in text
    assert "levofloxacin" in text
    assert "3. TỔNG KẾT KHUYẾN CÁO VÀ HƯỚNG XỬ LÝ" in text
    assert "⛔ HỆ THỐNG ĐỀ XUẤT: KHÔNG NÊN UỐNG ĐƠN THUỐC NÀY." in text


def test_llm_report_generator_safe(sample_context_safe: dict) -> None:
    generator = LlmReportGenerator(provider_name="fallback")
    result = generator.generate_report(sample_context_safe)

    assert result["overall_severity"] == "none"
    text = result["formatted_report_text"]
    assert "[🟢 TÌNH TRẠNG: AN TOÀN - KHÔNG PHÁT HIỆN TƯƠNG TÁC XUNG ĐỘT]" in text
    assert "Panadol 500mg" in text
    assert "✅ HỆ THỐNG ĐỀ XUẤT: ĐƠN THUỐC KHÔNG PHÁT HIỆN TƯƠNG TÁC XUNG ĐỘT GÂY HẠI." in text


def test_fallback_provider_direct(sample_context_major: dict) -> None:
    provider = FallbackLlmProvider()
    text = provider.generate(sample_context_major)

    assert "KẾT QUẢ PHÂN TÍCH TƯƠNG TÁC THUỐC" in text
    assert "CỰC KỲ NGUY HIỂM" in text
    assert "Disclaimer" in text
