"""Kiểm tra contract Module 2 khi model mới chỉ dự đoán shape và color."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from pill_safety.schemas import AttributeInferenceOutput


def _load_attribute_formatter():
    """Nạp riêng formatter để test contract mà không khởi tạo model PyTorch."""

    path = (
        SRC_DIRECTORY
        / "pill_safety"
        / "cv"
        / "attribute"
        / "postprocessing"
        / "formatter.py"
    )
    spec = importlib.util.spec_from_file_location("attribute_formatter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không thể nạp formatter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.format_attribute_predictions


def test_untrained_attribute_fields_are_explicitly_unknown():
    """Ngăn các giá trị mặc định bị hiểu nhầm là dự đoán từ model."""

    output = AttributeInferenceOutput.model_validate(
        {
            "request_id": "req_001",
            "session_id": "sess_001",
            "image_id": "img_001",
            "instance_id": "pill_001",
            "instance_token": "pill_token_001",
            "shape": {
                "label": "oval",
                "confidence": 0.91,
                "alternatives": [],
            },
            "color": {
                "primary": "white",
                "secondary": None,
                "distribution": {"white": 0.88},
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

    assert output.scoreline.visible is None
    assert output.scoreline.source == "not_predicted_by_attribute"
    assert output.dosage_form.label == "unknown"


def test_formatter_output_can_be_inserted_into_module_2_schema():
    """Bảo đảm formatter hiện tại tạo đúng phần payload dùng cho schema Module 2."""

    format_predictions = _load_attribute_formatter()
    prediction = format_predictions(
        shape_label="OVAL",
        shape_conf=0.91,
        color_labels=["color_WHITE", "color_GRAY"],
        color_probs={
            "color_WHITE": 0.88,
            "color_GRAY": 0.61,
            "color_RED": 0.05,
        },
        shape_alternatives=[("ROUND", 0.06)],
    )
    output = AttributeInferenceOutput.model_validate(
        {
            "request_id": "req_001",
            "session_id": "sess_001",
            "image_id": "img_001",
            "instance_id": "pill_001",
            "instance_token": "pill_token_001",
            **prediction,
        }
    )

    assert output.color.primary == "white"
    assert output.color.secondary == "gray"
    assert output.color.distribution["red"] == 0.05
    assert output.shape.alternatives[0].label == "round"
    assert output.scoreline.visible is None
