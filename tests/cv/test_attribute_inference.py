"""Kiểm tra cấu hình và schema mapper của inference Module 2 không cần weight thật."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from pill_safety.cv.attribute.config import AttributeInferenceConfig
from pill_safety.cv.attribute.labels.label_mapping import (
    load_color_threshold_values,
    load_label_mapping,
)
from pill_safety.cv.attribute.postprocessing.schema_mapper import (
    build_attribute_output,
)
from pill_safety.schemas import AttributeInferenceRequest


def test_attribute_yaml_selects_last_blocks_by_default():
    """Bảo đảm config chính thức mặc định dùng checkpoint last-block đã chọn."""

    config = AttributeInferenceConfig.from_yaml(
        PROJECT_ROOT / "configs" / "inference" / "attribute.yaml"
    )

    assert config.selected_model == "attribute_resnet18_last_blocks_finetune"
    assert config.weights_path == Path(
        "models/attribute_resnet18_last_blocks_finetune/best.pt"
    )
    assert config.image_size == 224
    assert config.shape_top_k == 3


def test_promoted_last_block_artifacts_use_the_expected_label_and_threshold_order():
    """Kiểm tra đúng format artifact thực tế của run last-block được chọn."""

    artifact_directory = (
        PROJECT_ROOT / "models" / "attribute_resnet18_last_blocks_finetune"
    )
    mapping, shape_count, color_count, _ = load_label_mapping(
        artifact_directory / "label_mapping.json"
    )
    thresholds = load_color_threshold_values(
        artifact_directory / "optimal_thresholds.json",
        mapping["color"],
    )

    assert mapping["shape"] == ["CAPSULE", "IRREGULAR", "OVAL", "POLYGON", "ROUND"]
    assert shape_count == 5
    assert color_count == 12
    assert thresholds[mapping["color"].index("PINK")] == 0.4
    assert thresholds[mapping["color"].index("ORANGE")] == 0.95


def test_schema_mapper_preserves_ids_and_marks_untrained_fields_unknown():
    """Bảo đảm Module 2 không làm rơi ID và không tự nhận scoreline là dự đoán."""

    request = AttributeInferenceRequest(
        request_id="req_001",
        session_id="sess_001",
        image_id="img_001",
        instance_id="pill_001",
        instance_token="pill_token_001",
        crop_path="outputs/crops/pill_001_crop.png",
        mask_path="outputs/masks/pill_001_mask.png",
    )
    output = build_attribute_output(
        request,
        {
            "shape": {
                "label": "oval",
                "confidence": 0.91,
                "alternatives": [{"label": "round", "confidence": 0.06}],
            },
            "color": {
                "primary": "white",
                "secondary": None,
                "distribution": {"white": 0.88, "gray": 0.11},
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
        },
    )

    assert output.instance_token == "pill_token_001"
    assert output.shape.alternatives[0].label == "round"
    assert output.scoreline.visible is None
    assert output.scoreline.source == "not_predicted_by_attribute"


def test_attribute_model_architectures_both_loadable(tmp_path: Path):
    """Bảo đảm AttributePredictor nhận diện và tải thành công cả 2 cấu trúc checkpoint."""
    import torch
    from pill_safety.cv.attribute.models.resnet18_multitask import (
        MultiTaskResNet18 as LinearMultiTaskResNet18,
    )
    from pill_safety.cv.attribute.models.resnet_multitask import (
        MultiTaskResNet18 as SequentialMultiTaskResNet18,
    )

    # 1. Linear head model
    linear_model = LinearMultiTaskResNet18(num_shape_classes=16, num_color_classes=12, pretrained=False)
    linear_sd = linear_model.state_dict()
    assert "shape_head.weight" in linear_sd
    assert "color_head.weight" in linear_sd

    # 2. Sequential head model
    seq_model = SequentialMultiTaskResNet18(num_shape_classes=16, num_color_classes=12, pretrained=False)
    seq_sd = seq_model.state_dict()
    assert "fc_shape.1.weight" in seq_sd
    assert "fc_color.4.weight" in seq_sd

