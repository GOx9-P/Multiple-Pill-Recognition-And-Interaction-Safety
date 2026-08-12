"""Kiem thu fusion Module 4 khong can chay model CV that."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from pill_safety.cv.pipeline import CVPipelineAssembler, CVPipelineConfig
from pill_safety.schemas import (
    AttributeInferenceOutput,
    CVPipelineInput,
    OCRInferenceOutput,
    SegmentationInferenceOutput,
)


def _schema_shape(value):
    """Rut gon JSON thanh cau truc type de doi chieu voi schema.md."""

    if isinstance(value, dict):
        return {key: _schema_shape(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_schema_shape(value[0])] if value else []
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        return "null"
    return "string"


def _segmentation_output(
    *,
    possible_non_pill: bool = False,
    possible_merged: bool = False,
    status: str = "usable",
) -> SegmentationInferenceOutput:
    """Tao output Module 1 hop le de kiem thu quy tac fusion."""

    return SegmentationInferenceOutput.model_validate(
        {
            "request_id": "req_001",
            "session_id": "sess_001",
            "image_id": "img_001",
            "image_quality": {
                "status": status,
                "blur_score": 0.12,
                "glare_detected": False,
                "lighting_warning": False,
            },
            "instances": [
                {
                    "instance_id": "pill_001",
                    "instance_token": "pill_token_001",
                    "bbox_xyxy": [10, 20, 70, 80],
                    "mask_path": "outputs/masks/pill_001_mask.png",
                    "crop_path": "outputs/crops/pill_001_crop.png",
                    "segmentation": {
                        "confidence": 0.96,
                        "occlusion_estimate": 0.0,
                        "possible_merged_instance": possible_merged,
                        "possible_non_pill": possible_non_pill,
                    },
                    "quality_flags": ["minor_glare"],
                }
            ],
        }
    )


def _attribute_output() -> AttributeInferenceOutput:
    """Tao output Module 2 voi scoreline placeholder de xac nhan no bi thay the."""

    return AttributeInferenceOutput.model_validate(
        {
            "request_id": "req_001",
            "session_id": "sess_001",
            "image_id": "img_001",
            "instance_id": "pill_001",
            "instance_token": "pill_token_001",
            "shape": {
                "label": "oval",
                "confidence": 0.91,
                "alternatives": [{"label": "round", "confidence": 0.05}],
            },
            "color": {
                "primary": "white",
                "secondary": None,
                "distribution": {"white": 0.88, "gray": 0.08, "yellow": 0.03},
                "confidence": 0.88,
                "lighting_warning": False,
            },
            "dosage_form": {
                "label": "unknown",
                "confidence": None,
                "source": "not_predicted_by_attribute",
            },
            "scoreline": {
                "label": "unknown",
                "visible": None,
                "confidence": None,
                "source": "not_predicted_by_attribute",
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
        }
    )


def _ocr_output(*, visible: bool = True) -> OCRInferenceOutput:
    """Tao output Module 3 va cung cap scoreline chinh thuc cho Module 4."""

    return OCRInferenceOutput.model_validate(
        {
            "request_id": "req_001",
            "session_id": "sess_001",
            "image_id": "img_001",
            "instance_id": "pill_001",
            "instance_token": "pill_token_001",
            "scoreline": {
                "visible": True,
                "confidence": 0.77,
                "angle_degrees": 90.0,
                "orientation": "vertical",
                "line_xyxy": [40.0, 5.0, 40.0, 95.0],
                "support_count": 2,
                "rotation_degrees": 0,
                "preprocessing": "original",
                "source": "ocr_hough_consensus",
            },
            "imprint_visibility": {
                "visible": visible,
                "confidence": 0.86 if visible else 0.0,
            },
            "imprint": {
                "visible": visible,
                "raw": "K 56" if visible else "",
                "confidence": 0.86 if visible else 0.0,
                "text_regions": [],
                "ocr_observations": [
                    {
                        "region_id": "region_01",
                        "rotation_degrees": 0,
                        "preprocessing": "original",
                        "text": "K 56",
                        "confidence": 0.86,
                    }
                ],
                "normalized_candidates": [
                    {
                        "text": "K 56",
                        "score": 0.86,
                        "source": "raw_ocr",
                        "evidence": ["legacy_priority_confidence"],
                    }
                ],
            },
        }
    )


def _input(**segmentation_kwargs) -> CVPipelineInput:
    """Tao input Module 4 day du cho mot instance."""

    return CVPipelineInput(
        segmentation_output=_segmentation_output(**segmentation_kwargs),
        attribute_outputs=[_attribute_output()],
        ocr_outputs=[_ocr_output()],
    )


def test_fusion_uses_ocr_scoreline_and_exports_documented_cv_schema(tmp_path):
    """Scoreline OCR phai thay placeholder Attribute va JSON phai khop schema.md."""

    artifacts = CVPipelineAssembler(
        CVPipelineConfig(output_dir=tmp_path / "outputs")
    ).predict_with_artifacts(_input())
    payload = artifacts.output.model_dump(mode="json")
    pill = payload["pills"][0]

    assert payload["schema_version"] == "cv_output_v1"
    assert pill["scoreline"]["visible"] is True
    assert pill["scoreline"]["source"] == "ocr_hough_consensus"
    assert pill["cv_status"] == "features_ready"
    assert "text_regions" not in pill["imprint"]
    assert artifacts.schema_json_path == (
        tmp_path
        / "outputs"
        / "predictions"
        / "cv_pipeline"
        / "req_001"
        / "img_001"
        / "cv_output.json"
    )
    with artifacts.schema_json_path.open("r", encoding="utf-8") as file:
        assert json.load(file) == payload

    document = (PROJECT_ROOT / "docs" / "schema.md").read_text(encoding="utf-8")
    module_4 = document.split("## 5. Module 4", 1)[1].split("## 6. Module 5", 1)[0]
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", module_4, re.S)
    documented_output = json.loads(blocks[1])
    assert _schema_shape(payload) == _schema_shape(documented_output)


@pytest.mark.parametrize(
    ("segmentation_kwargs", "ocr_visible", "expected"),
    [
        ({"possible_non_pill": True}, True, "unknown_object"),
        ({"status": "unusable"}, True, "insufficient_visual_evidence"),
        ({"possible_merged": True}, True, "partial_features"),
        ({}, False, "partial_features"),
    ],
)
def test_fusion_derives_cv_status_from_documented_safety_signals(
    segmentation_kwargs, ocr_visible, expected
):
    """Fusion ha trang thai khi object, chat luong anh hoac OCR khong du evidence."""

    pipeline_input = _input(**segmentation_kwargs).model_copy(
        update={"ocr_outputs": [_ocr_output(visible=ocr_visible)]}
    )
    output = CVPipelineAssembler().predict(pipeline_input)
    assert output.pills[0].cv_status == expected


def test_fusion_rejects_missing_or_wrong_instance_token():
    """Khong cho phep ghep thieu hoac ghep OCR/Attribute cua vien thuoc khac."""

    with pytest.raises(ValueError, match="missing ocr output"):
        CVPipelineAssembler().predict(_input().model_copy(update={"ocr_outputs": []}))

    wrong_ocr = _ocr_output().model_copy(update={"instance_token": "pill_token_other"})
    with pytest.raises(ValueError, match="not emitted by segmentation"):
        CVPipelineAssembler().predict(
            _input().model_copy(update={"ocr_outputs": [wrong_ocr]})
        )

    wrong_instance = _ocr_output().model_copy(update={"instance_id": "pill_999"})
    with pytest.raises(ValueError, match="instance_id does not match token"):
        CVPipelineAssembler().predict(
            _input().model_copy(update={"ocr_outputs": [wrong_instance]})
        )
