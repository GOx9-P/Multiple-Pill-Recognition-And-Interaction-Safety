"""Tests for the CV to RAG/DDI/LLM orchestration boundary."""

from __future__ import annotations

import csv

from pill_safety.rag.orchestration import EndToEndPostCvPipeline


def _cv_output() -> dict:
    """Build a valid cv_output_v1 without loading any CV model."""

    def pill(instance_id: str, token: str, imprint: str) -> dict:
        return {
            "instance_id": instance_id,
            "instance_token": token,
            "side_hint": "unknown",
            "cv_status": "features_ready",
            "bbox_xyxy": [10, 20, 80, 90],
            "mask_path": f"outputs/masks/{instance_id}.png",
            "crop_path": f"outputs/crops/{instance_id}.png",
            "segmentation": {
                "confidence": 0.95,
                "occlusion_estimate": 0.0,
                "possible_merged_instance": False,
                "possible_non_pill": False,
            },
            "shape": {"label": "round", "confidence": 0.9, "alternatives": []},
            "color": {
                "primary": "white",
                "secondary": None,
                "distribution": {"white": 0.9},
                "confidence": 0.9,
                "lighting_warning": False,
            },
            "dosage_form": {
                "label": "unknown",
                "confidence": None,
                "source": "not_predicted_by_attribute",
            },
            "scoreline": {
                "visible": False,
                "confidence": 0.0,
                "angle_degrees": None,
                "orientation": "unknown",
                "line_xyxy": None,
                "support_count": 0,
                "rotation_degrees": None,
                "preprocessing": None,
                "source": "ocr_hough_consensus",
            },
            "logo_or_symbol": {
                "visible": None,
                "confidence": None,
                "source": "not_predicted_by_attribute",
            },
            "damage_or_occlusion": {
                "visible": None,
                "confidence": None,
                "source": "not_predicted_by_attribute",
            },
            "imprint_visibility": {"visible": True, "confidence": 0.9},
            "imprint": {
                "visible": True,
                "raw": imprint,
                "confidence": 0.9,
                "ocr_observations": [],
                "normalized_candidates": [
                    {
                        "text": imprint,
                        "score": 0.9,
                        "source": "raw_ocr",
                        "evidence": ["test"],
                    }
                ],
            },
            "quality_flags": [],
        }

    return {
        "schema_version": "cv_output_v1",
        "request_id": "req_001",
        "session_id": "sess_001",
        "image_id": "img_001",
        "image_quality": {
            "status": "usable",
            "blur_score": 0.1,
            "glare_detected": False,
            "lighting_warning": False,
        },
        "pills": [pill("pill_001", "token_001", "A1"), pill("pill_002", "token_002", "B2")],
    }


class _IdentificationService:
    """Fake RAG service that exposes the payload sent by the orchestrator."""

    def identify(self, request: dict) -> dict:
        assert request["schema_version"] == "rag_request_v1"
        assert request["market"] == "US"
        assert request["known_drug_names"] == ["Known medicine"]
        return {
            "schema_version": "rag_identification_v1",
            "request_id": request["request_id"],
            "session_id": request["session_id"],
            "pill_results": [
                {
                    "instance_id": "pill_001",
                    "instance_token": "token_001",
                    "identification_status": "identified",
                    "required_action": "none",
                    "top_candidates": [
                        {
                            "product_name": "Drug A",
                            "final_score": 0.95,
                            "evidence": {"top1_top2_margin": 0.2},
                        }
                    ],
                    "accepted_product": {
                        "drug_id": 42,
                        "product_code": "00042-0001",
                        "product_name": "Drug A",
                    },
                },
                {
                    "instance_id": "pill_002",
                    "instance_token": "token_002",
                    "identification_status": "ambiguous",
                    "required_action": "capture_reverse_side",
                    "top_candidates": [],
                    "accepted_product": None,
                },
            ],
        }


class _DdiLookupService:
    """Fake DDI service that verifies only accepted products are submitted."""

    def lookup_ddi(self, request: dict) -> dict:
        assert request["identified_products"] == [
            {"instance_id": "pill_001", "product_id": "drug_42"}
        ]
        return {
            "schema_version": "ddi_output_v0",
            "request_id": request["request_id"],
            "session_id": request["session_id"],
            "identified_drugs": [{"instance_id": "pill_001", "product_name": "Drug A"}],
            "duplicate_ingredient_warnings": [],
            "interactions": [
                {
                    "interaction_id": "ddi_8",
                    "source_instance_ids": ["pill_001"],
                    "severity": "moderate",
                }
            ],
            "overall_severity": "moderate",
            "scope_warnings": [],
        }


class _ContextBuilder:
    """Fake context builder for checking the orchestrator join contract."""

    def build_context(self, payload: dict) -> dict:
        assert payload["rag_identification"]["schema_version"] == "rag_identification_v1"
        return {
            "schema_version": "llm_context_v0",
            "request_id": payload["request_id"],
            "session_id": payload["session_id"],
            "interactions": payload["ddi_output"]["interactions"],
        }


class _ReportGenerator:
    """Fake report generator so the test does not call a network provider."""

    def generate_report(self, context: dict) -> dict:
        return {
            "schema_version": "llm_report_v0",
            "request_id": context["request_id"],
            "session_id": context["session_id"],
            "overall_severity": "moderate",
            "provider_used": "test",
            "formatted_report_text": "grounded report",
            "structured_context": context,
        }


def test_end_to_end_pipeline_joins_by_instance_and_writes_audit_artifacts(tmp_path):
    """An accepted pill reaches DDI while an ambiguous pill remains unresolved."""

    pipeline = EndToEndPostCvPipeline(
        identification_service=_IdentificationService(),
        ddi_lookup_service=_DdiLookupService(),
        context_builder=_ContextBuilder(),
        report_generator=_ReportGenerator(),
    )
    artifacts = pipeline.run_with_artifacts(
        _cv_output(),
        output_dir=tmp_path,
        market="US",
        known_drug_names=["Known medicine"],
    )

    assert artifacts.output["schema_version"] == "end_to_end_result_v1"
    assert [row["instance_id"] for row in artifacts.output["pill_summary"]] == [
        "pill_001",
        "pill_002",
    ]
    assert artifacts.output["pill_summary"][0]["interaction_ids"] == ["ddi_8"]
    assert artifacts.output["pill_summary"][1]["identification_status"] == "ambiguous"
    assert artifacts.paths["end_to_end_result"].is_file()
    assert artifacts.paths["llm_report_text"].read_text(encoding="utf-8") == "grounded report"

    with artifacts.paths["pill_summary"].open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["top_candidate_name"] == "Drug A"
