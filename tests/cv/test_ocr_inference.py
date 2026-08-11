from __future__ import annotations

import json
import re
import sys
import types
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from pill_safety.cv.ocr import OCRConfig, OCRPredictor, RotationTier
from pill_safety.cv.ocr.engines import PaddleOCREngine
from pill_safety.cv.ocr.postprocessing.candidates import (
    select_baseline_observation,
)
from pill_safety.cv.ocr.postprocessing.ordering import sequence_confidence
from pill_safety.cv.ocr.postprocessing.schema_mapper import build_ocr_output
from pill_safety.cv.ocr.postprocessing.scoreline import run_scoreline_side_split
from pill_safety.cv.ocr.preprocessing import map_polygon_to_original
from pill_safety.schemas import OCRInferenceRequest


class StaticEngine:
    def __init__(self, items):
        self.items = items

    def predict(self, image_path, output_json_dir, step_id):
        del image_path, output_json_dir, step_id
        return [dict(item) for item in self.items]


class SideEngine:
    def predict(self, image_path, output_json_dir, step_id):
        del image_path, output_json_dir
        if step_id.endswith("side_a"):
            return [
                {
                    "text": "K",
                    "confidence": 0.82,
                    "polygon": [[15, 30], [35, 30], [35, 60], [15, 60]],
                }
            ]
        return [
            {
                "text": "56",
                "confidence": 0.99,
                "polygon": [[65, 30], [90, 30], [90, 60], [65, 60]],
            }
        ]


def make_request(image_path: Path, instance_id: str = "pill_007"):
    return OCRInferenceRequest(
        request_id="req_123",
        session_id="sess_456",
        image_id="img_003",
        instance_id=instance_id,
        instance_token="pill_token_007",
        crop_path=str(image_path),
        mask_path="outputs/masks/pill_007_mask.png",
    )


def schema_shape(value):
    if isinstance(value, dict):
        return {key: schema_shape(child) for key, child in value.items()}
    if isinstance(value, list):
        return [schema_shape(value[0])] if value else []
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        return "null"
    return "string"


def test_predictor_preserves_ids_and_exports_exact_module_3_schema(tmp_path):
    image_path = tmp_path / "crop.png"
    cv2.imwrite(str(image_path), np.full((180, 260, 3), 220, dtype=np.uint8))
    engine = StaticEngine(
        [
            {
                "text": "K",
                "confidence": 0.82,
                "polygon": [[30, 60], [80, 60], [80, 120], [30, 120]],
            },
            {
                "text": "56",
                "confidence": 0.99,
                "polygon": [[140, 60], [220, 60], [220, 120], [140, 120]],
            },
        ]
    )
    config = replace(
        OCRConfig(),
        preprocessing_steps=("original",),
        rotation_tiers=(RotationTier("tier1_0_180", (0,)),),
        enable_scoreline_side_split=False,
        force_run_all_rotation_tiers=True,
        output_dir=tmp_path / "outputs",
    )

    artifacts = OCRPredictor(config=config, engine=engine).predict_with_artifacts(
        make_request(image_path)
    )
    payload = artifacts.output.model_dump(mode="json")

    assert set(payload) == {
        "request_id",
        "session_id",
        "image_id",
        "instance_id",
        "instance_token",
        "imprint_visibility",
        "imprint",
    }
    assert payload["request_id"] == "req_123"
    assert payload["image_id"] == "img_003"
    assert payload["instance_id"] == "pill_007"
    assert payload["instance_token"] == "pill_token_007"
    assert payload["imprint"]["raw"] == "K 56"
    assert payload["imprint"]["confidence"] == 0.8625
    assert payload["imprint"]["normalized_candidates"][0]["source"] == "raw_ocr"
    assert set(payload["imprint"]) == {
        "visible",
        "raw",
        "confidence",
        "text_regions",
        "ocr_observations",
        "normalized_candidates",
    }
    assert artifacts.overlay_path is not None and artifacts.overlay_path.exists()
    assert artifacts.schema_json_path.exists()
    assert artifacts.debug_json_path.exists()
    assert artifacts.schema_json_path.parent == (
        tmp_path / "outputs" / "req_123" / "img_003" / "pill_007"
    )
    region_ids = {
        region["region_id"] for region in payload["imprint"]["text_regions"]
    }
    observation_region_ids = {
        observation["region_id"]
        for observation in payload["imprint"]["ocr_observations"]
    }
    assert observation_region_ids <= region_ids
    assert payload["imprint"]["text_regions"][0]["polygon"] == [
        [20.0, 50.0],
        [70.0, 50.0],
        [70.0, 110.0],
        [20.0, 110.0],
    ]
    with artifacts.schema_json_path.open("r", encoding="utf-8") as file:
        assert json.load(file) == payload

    schema_document = (PROJECT_ROOT / "docs" / "schema.md").read_text(
        encoding="utf-8"
    )
    module_3 = schema_document.split("## 4. Module 3", 1)[1].split(
        "## 5. Module 4", 1
    )[0]
    json_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", module_3, re.S)
    documented_output = json.loads(json_blocks[1])
    assert schema_shape(payload) == schema_shape(documented_output)


def test_no_text_still_returns_valid_module_3_output(tmp_path):
    image_path = tmp_path / "crop.png"
    cv2.imwrite(str(image_path), np.full((100, 100, 3), 255, dtype=np.uint8))
    config = replace(
        OCRConfig(),
        preprocessing_steps=("original",),
        rotation_tiers=(RotationTier("tier1_0_180", (0,)),),
        enable_scoreline_side_split=False,
        output_dir=tmp_path / "outputs",
    )
    output = OCRPredictor(config=config, engine=StaticEngine([])).predict(
        make_request(image_path)
    )

    assert output.imprint_visibility.model_dump() == {
        "visible": False,
        "confidence": 0.0,
    }
    assert output.imprint.model_dump() == {
        "visible": False,
        "raw": "",
        "confidence": 0.0,
        "text_regions": [],
        "ocr_observations": [],
        "normalized_candidates": [],
    }


def test_notebook_priority_then_confidence_rule_is_preserved():
    one_box_high_confidence = {"priority": 1, "best_confidence": 0.99}
    two_boxes_lower_confidence = {"priority": 2, "best_confidence": 0.90}
    reliable_split = {"priority": 3, "best_confidence": 0.80}

    assert (
        select_baseline_observation(
            [one_box_high_confidence, two_boxes_lower_confidence, reliable_split]
        )
        is reliable_split
    )


def test_notebook_sequence_confidence_formula_is_preserved():
    items = [{"confidence": 0.82}, {"confidence": 0.99}]
    expected = 0.5 * np.mean([0.82, 0.99]) + 0.5 * 0.82
    assert sequence_confidence(items) == expected


def test_yaml_defaults_match_the_notebook_configuration():
    config = OCRConfig.from_yaml(PROJECT_ROOT / "configs" / "inference" / "ocr.yaml")
    assert config.ocr_version == "PP-OCRv5"
    assert config.det_db_thresh == 0.2
    assert config.det_db_unclip_ratio == 2.0
    assert config.preprocessing_steps == (
        "original",
        "clahe",
        "blackhat",
        "blackhat_bold",
    )
    assert [tier.rotations for tier in config.rotation_tiers] == [
        (0, 180),
        (90, 270),
        (-45, -30, -15, 15, 30, 45),
    ]
    assert config.force_run_all_rotation_tiers is True
    assert config.min_usable_confidence == 0.50
    assert config.min_scoreline_detection_confidence == 0.45
    assert config.min_scoreline_support == 2
    assert config.min_side_confidence == 0.60


def test_vertical_scoreline_split_merges_left_then_right(tmp_path):
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    split_items, split_info = run_scoreline_side_split(
        variant=image,
        rotation_degrees=0,
        step_id="tier1_rot0_original",
        split_dir=tmp_path / "split",
        json_dir=tmp_path / "json",
        scoreline={
            "visible": True,
            "confidence": 0.8,
            "orientation": "vertical",
            "line_xyxy": [50.0, 0.0, 50.0, 100.0],
        },
        engine=SideEngine(),
        config=OCRConfig(),
    )

    assert [item["text"] for item in split_items] == ["K", "56"]
    assert split_info["reliable"] is True


def test_paddle_adapter_uses_the_same_model_arguments_as_notebook(
    monkeypatch, tmp_path
):
    captured = {}

    class FakeDevice:
        @staticmethod
        def is_compiled_with_cuda():
            return True

    fake_paddle = types.ModuleType("paddle")
    fake_paddle.device = FakeDevice()
    fake_paddle.set_device = lambda device: captured.update(device=device)

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured.update(kwargs=kwargs)

        def predict(self, input):
            captured["input"] = input
            return [
                {
                    "rec_texts": ["K", "56"],
                    "rec_scores": [0.82, 0.99],
                    "rec_polys": [
                        [[1, 1], [2, 1], [2, 2], [1, 2]],
                        [[3, 1], [5, 1], [5, 2], [3, 2]],
                    ],
                }
            ]

    fake_paddleocr = types.ModuleType("paddleocr")
    fake_paddleocr.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    engine = PaddleOCREngine(OCRConfig())
    items = engine.predict(tmp_path / "crop.png", tmp_path / "json", "step_1")

    assert captured["device"] == "gpu:0"
    assert captured["kwargs"] == {
        "ocr_version": "PP-OCRv5",
        "lang": "en",
        "device": "gpu:0",
        "det_db_thresh": 0.2,
        "det_db_unclip_ratio": 2.0,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    assert [item["text"] for item in items] == ["K", "56"]
    assert [item["confidence"] for item in items] == [0.82, 0.99]


def test_rotated_polygon_is_mapped_back_to_original_crop_coordinates():
    # Original crop is 100x80, padded by 10 pixels to 120x100.
    padded_polygon = np.asarray(
        [[20.0, 30.0], [40.0, 30.0], [40.0, 50.0], [20.0, 50.0]],
        dtype=np.float32,
    )
    padded_height = 120
    rotated_90 = np.column_stack(
        (padded_height - 1.0 - padded_polygon[:, 1], padded_polygon[:, 0])
    ).tolist()

    restored = map_polygon_to_original(
        rotated_90,
        padded_shape=(120, 100, 3),
        rotation_degrees=90,
        pad_px=10,
        original_height=100,
        original_width=80,
    )

    assert restored == [
        [10.0, 20.0],
        [30.0, 20.0],
        [30.0, 40.0],
        [10.0, 40.0],
    ]


def test_oblique_polygons_are_inverse_transformed_to_original_crop():
    padded_shape = (120, 100, 3)
    original_polygon = np.asarray(
        [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]],
        dtype=np.float32,
    )
    padded_polygon = original_polygon + np.asarray([10.0, 10.0])
    height, width = padded_shape[:2]
    center = (width / 2, height / 2)

    for angle in (-45, -30, -15, 15, 30, 45):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_value, sin_value = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_width = int((height * sin_value) + (width * cos_value))
        new_height = int((height * cos_value) + (width * sin_value))
        matrix[0, 2] += (new_width / 2) - center[0]
        matrix[1, 2] += (new_height / 2) - center[1]
        homogeneous = np.column_stack(
            (padded_polygon, np.ones(len(padded_polygon), dtype=np.float32))
        )
        rotated_polygon = homogeneous @ matrix.T

        restored = map_polygon_to_original(
            rotated_polygon.tolist(),
            padded_shape=padded_shape,
            rotation_degrees=angle,
            pad_px=10,
            original_height=100,
            original_width=80,
        )
        assert np.allclose(restored, original_polygon, atol=0.02)


def test_all_observation_region_ids_reference_selected_text_regions():
    request = OCRInferenceRequest(
        request_id="req_1",
        session_id="sess_1",
        image_id="img_1",
        instance_id="pill_1",
        instance_token="token_1",
        crop_path="crop.png",
        mask_path="mask.png",
    )
    canonical_item = {
        "text": "K 56",
        "confidence": 0.90,
        "polygon_original": [[10, 10], [90, 10], [90, 40], [10, 40]],
    }
    observation = {
        "priority": 2,
        "best_confidence": 0.95,
        "rotation_degrees": 180,
        "preprocessing": "clahe",
        "ordered_items": [
            {
                "text": "K",
                "confidence": 0.90,
                "polygon_original": [[10, 10], [40, 10], [40, 40], [10, 40]],
            },
            {
                "text": "56",
                "confidence": 0.95,
                "polygon_original": [[50, 10], [90, 10], [90, 40], [50, 40]],
            },
        ],
    }
    output = build_ocr_output(
        request=request,
        config=OCRConfig(),
        final_candidate={"text": "K 56", "score": 0.90},
        best_observation={
            "mode": "full_image",
            "rotation_degrees": 0,
            "preprocessing": "original",
        },
        best_items=[canonical_item],
        valid_observations=[observation],
        ranked_candidates=[],
    )

    region_ids = {region.region_id for region in output.imprint.text_regions}
    observation_ids = {
        item.region_id for item in output.imprint.ocr_observations
    }
    assert observation_ids == {"region_01"}
    assert observation_ids <= region_ids


def test_different_requests_do_not_overwrite_the_same_instance_artifacts(tmp_path):
    image_path = tmp_path / "crop.png"
    cv2.imwrite(str(image_path), np.full((80, 80, 3), 255, dtype=np.uint8))
    config = replace(
        OCRConfig(),
        preprocessing_steps=("original",),
        rotation_tiers=(RotationTier("tier1", (0,)),),
        enable_scoreline_side_split=False,
        output_dir=tmp_path / "outputs",
    )
    predictor = OCRPredictor(config=config, engine=StaticEngine([]))
    request_a = make_request(image_path)
    request_b = request_a.model_copy(update={"request_id": "req_999"})

    artifacts_a = predictor.predict_with_artifacts(request_a)
    artifacts_b = predictor.predict_with_artifacts(request_b)

    assert artifacts_a.schema_json_path != artifacts_b.schema_json_path
    assert artifacts_a.schema_json_path.exists()
    assert artifacts_b.schema_json_path.exists()
