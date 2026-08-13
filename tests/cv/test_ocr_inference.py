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

import pill_safety.cv.ocr.predictors.ocr_predictor as ocr_predictor_module
from pill_safety.cv.ocr import OCRConfig, OCRPredictor, RotationTier
from pill_safety.cv.ocr.engines import PaddleOCREngine
from pill_safety.cv.ocr.engines.paddleocr_engine import parse_prediction_result
from pill_safety.cv.ocr.postprocessing.candidates import (
    build_final_candidate,
    finalize_scoreline,
    select_baseline_observation,
)
from pill_safety.cv.ocr.postprocessing.ordering import sequence_confidence
from pill_safety.cv.ocr.postprocessing.schema_mapper import build_ocr_output
from pill_safety.cv.ocr.postprocessing.region_filter import filter_text_regions
from pill_safety.cv.ocr.postprocessing.scoreline import (
    detect_scoreline_for_split,
    map_scoreline_to_original,
    run_scoreline_side_split,
)
from pill_safety.cv.ocr.preprocessing.image_ops import PreparedImage
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


def test_paddle_v3_json_property_is_parsed_into_ocr_items():
    """PaddleOCR v3 exposes Result.json as a dictionary property."""

    class PaddleV3Result:
        json = {
            "res": {
                "rec_texts": ["K", "56"],
                "rec_scores": [0.82, 0.99],
                "rec_polys": [
                    [[1, 1], [2, 1], [2, 2], [1, 2]],
                    [[3, 1], [5, 1], [5, 2], [3, 2]],
                ],
            }
        }

    items = parse_prediction_result([PaddleV3Result()])

    assert [item["text"] for item in items] == ["K", "56"]
    assert [item["confidence"] for item in items] == [0.82, 0.99]


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
        "scoreline",
        "imprint_visibility",
        "imprint",
    }
    assert payload["request_id"] == "req_123"
    assert payload["image_id"] == "img_003"
    assert payload["instance_id"] == "pill_007"
    assert payload["instance_token"] == "pill_token_007"
    assert payload["scoreline"] == {
        "visible": False,
        "confidence": 0.0,
        "angle_degrees": None,
        "orientation": "unknown",
        "line_xyxy": None,
        "support_count": 0,
        "rotation_degrees": None,
        "preprocessing": None,
        "source": "ocr_hough_consensus",
    }
    assert payload["imprint"]["raw"] == "K 56"
    assert payload["imprint"]["confidence"] == 0.694
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
    payload_shape = schema_shape(payload)
    documented_shape = schema_shape(documented_output)
    assert set(payload_shape["scoreline"]) == set(documented_shape["scoreline"])
    payload_shape.pop("scoreline")
    documented_shape.pop("scoreline")
    assert payload_shape == documented_shape


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


def test_unrankable_observation_returns_empty_imprint_instead_of_crashing(tmp_path):
    """OCR co box yeu nhung khong co candidate hop le phai tra unknown an toan."""

    image_path = tmp_path / "crop.png"
    cv2.imwrite(str(image_path), np.full((100, 100, 3), 180, dtype=np.uint8))
    engine = StaticEngine(
        [
            {
                "text": "A",
                "confidence": 0.99,
                "polygon": [[20, 30], [40, 30], [40, 60], [20, 60]],
            },
            {
                "text": "B",
                "confidence": 0.01,
                "polygon": [[50, 30], [70, 30], [70, 60], [50, 60]],
            },
        ]
    )
    config = replace(
        OCRConfig(),
        preprocessing_steps=("original",),
        rotation_tiers=(RotationTier("tier1", (0,)),),
        enable_scoreline_side_split=False,
        output_dir=tmp_path / "outputs",
    )

    artifacts = OCRPredictor(config=config, engine=engine).predict_with_artifacts(
        make_request(image_path)
    )
    debug_payload = json.loads(artifacts.debug_json_path.read_text(encoding="utf-8"))

    assert artifacts.output.imprint.visible is False
    assert artifacts.output.imprint.raw == ""
    assert debug_payload["rejection_reason"] == "no_ranked_candidate"


def test_no_text_still_exports_scoreline_owned_by_ocr(tmp_path, monkeypatch):
    """Bảo đảm scoreline không bị mất khi OCR không đọc được imprint."""

    image_path = tmp_path / "crop.png"
    cv2.imwrite(str(image_path), np.full((100, 100, 3), 255, dtype=np.uint8))
    config = replace(
        OCRConfig(),
        preprocessing_steps=("original",),
        rotation_tiers=(RotationTier("tier1_0_180", (0,)),),
        enable_scoreline_side_split=False,
        output_dir=tmp_path / "outputs",
    )
    monkeypatch.setattr(
        ocr_predictor_module,
        "finalize_scoreline",
        lambda observations, current_config: {
            "visible": True,
            "confidence": 0.77,
            "angle_degrees": 79.99,
            "orientation": "vertical",
            "line_xyxy": [10.0, 20.0, 30.0, 80.0],
            "support_count": 3,
            "rotation_degrees": 0,
            "preprocessing": "original",
            "source": "ocr_hough_consensus",
        },
    )

    output = OCRPredictor(config=config, engine=StaticEngine([])).predict(
        make_request(image_path)
    )

    assert output.imprint.visible is False
    assert output.scoreline.model_dump() == {
        "visible": True,
        "confidence": 0.77,
        "angle_degrees": 79.99,
        "orientation": "vertical",
        "line_xyxy": [10.0, 20.0, 30.0, 80.0],
        "support_count": 3,
        "rotation_degrees": 0,
        "preprocessing": "original",
        "source": "ocr_hough_consensus",
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


def test_final_answer_uses_top_ranked_candidate_not_legacy_observation():
    """Dung candidate consensus tot hon thay vi mot lan OCR co priority cao."""

    legacy_observation = {
        "detected_text": "829 AH12 18332",
        "ordered_items": [{"confidence": 0.46}],
        "mode": "full_image",
        "rotation_degrees": 90,
        "preprocessing": "blackhat_bold",
    }
    winner_observation = {
        "detected_text": "AH12",
        "ordered_items": [{"confidence": 0.90}],
        "mode": "full_image",
        "rotation_degrees": 0,
        "preprocessing": "original",
    }
    ranked_candidates = [
        {
            "text": "AH12",
            "normalized_text": "AH12",
            "score": 0.7732,
            "mean_ocr_confidence": 0.90,
            "support_count": 3,
            "modes": ["full_image"],
            "rotations": [0, 180],
            "preprocessings": ["original", "blackhat_bold"],
            "_best_observation": winner_observation,
            "_best_items": winner_observation["ordered_items"],
        }
    ]

    final_candidate = build_final_candidate(
        legacy_observation, ranked_candidates
    )

    assert final_candidate["text"] == "AH12"
    assert final_candidate["score"] == 0.7732
    assert final_candidate["selection_method"] == "ranked_candidate_consensus"


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
    assert config.min_text_region_foreground_coverage == 0.70
    assert config.max_text_region_area_ratio == 0.75
    assert config.text_region_edge_margin_ratio == 0.02
    assert config.min_scoreline_detection_confidence == 0.45
    assert config.min_scoreline_support == 2
    assert config.scoreline_angle_consensus_tolerance_degrees == 12.0
    assert config.scoreline_consensus_distance_ratio == 0.08
    assert config.scoreline_min_foreground_coverage == 0.75
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


def test_rotated_scoreline_is_mapped_back_to_original_crop_coordinates():
    prepared_image = PreparedImage(
        bgr=np.zeros((120, 100, 3), dtype=np.uint8),
        original_height=100,
        original_width=80,
        pad_px=10,
    )
    scoreline = map_scoreline_to_original(
        {
            "visible": True,
            "confidence": 0.80,
            "line_xyxy": [99.0, 50.0, 19.0, 50.0],
            "angle_degrees": 0.0,
            "orientation": "horizontal",
        },
        padded_shape=prepared_image.bgr.shape,
        rotation_degrees=90,
        prepared_image=prepared_image,
    )

    assert scoreline["line_xyxy"] == [40.0, 10.0, 40.0, 90.0]
    assert scoreline["angle_degrees"] == 90.0
    assert scoreline["orientation"] == "vertical"


def test_predictor_exports_scoreline_in_original_crop_coordinates(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "crop.png"
    cv2.imwrite(str(image_path), np.full((100, 80, 3), 255, dtype=np.uint8))
    config = replace(
        OCRConfig(),
        preprocessing_steps=("original",),
        rotation_tiers=(RotationTier("tier2_90", (90,)),),
        enable_scoreline_side_split=False,
        min_scoreline_support=1,
        output_dir=tmp_path / "outputs",
    )
    monkeypatch.setattr(
        ocr_predictor_module,
        "detect_scoreline_for_split",
        lambda variant, current_config, foreground_mask=None: {
            "visible": True,
            "confidence": 0.80,
            "line_xyxy": [99.0, 50.0, 19.0, 50.0],
            "angle_degrees": 0.0,
            "orientation": "horizontal",
        },
    )

    output = OCRPredictor(config=config, engine=StaticEngine([])).predict(
        make_request(image_path)
    )

    assert output.scoreline.line_xyxy == [40.0, 10.0, 40.0, 90.0]
    assert output.scoreline.angle_degrees == 90.0
    assert output.scoreline.orientation == "vertical"


def test_scoreline_consensus_rejects_unrelated_hough_lines():
    config = replace(OCRConfig(), min_scoreline_support=2)
    scoreline = finalize_scoreline(
        [
            {
                "visible": True,
                "confidence": 0.90,
                "line_xyxy": [30.0, 0.0, 30.0, 100.0],
                "angle_degrees": 90.0,
                "orientation": "vertical",
            },
            {
                "visible": True,
                "confidence": 0.90,
                "line_xyxy": [0.0, 70.0, 100.0, 70.0],
                "angle_degrees": 0.0,
                "orientation": "horizontal",
            },
        ],
        config,
    )

    assert scoreline["visible"] is False
    assert scoreline["support_count"] == 1


def test_scoreline_consensus_accepts_same_geometric_line():
    config = replace(OCRConfig(), min_scoreline_support=2)
    scoreline = finalize_scoreline(
        [
            {
                "visible": True,
                "confidence": 0.80,
                "line_xyxy": [50.0, 0.0, 50.0, 100.0],
                "angle_degrees": 90.0,
                "orientation": "vertical",
            },
            {
                "visible": True,
                "confidence": 0.90,
                "line_xyxy": [53.0, 0.0, 53.0, 100.0],
                "angle_degrees": 90.0,
                "orientation": "vertical",
            },
        ],
        config,
    )

    assert scoreline["visible"] is True
    assert scoreline["support_count"] == 2
    assert scoreline["line_xyxy"] == [53.0, 0.0, 53.0, 100.0]


def test_scoreline_detector_rejects_line_outside_pill_foreground(monkeypatch):
    """Line tren nen/padding khong duoc phep tro thanh scoreline."""

    image = np.full((100, 100, 3), 180, dtype=np.uint8)
    foreground_mask = np.zeros((100, 100), dtype=np.uint8)
    foreground_mask[20:80, 20:80] = 1
    monkeypatch.setattr(
        cv2,
        "HoughLinesP",
        lambda *args, **kwargs: np.asarray([[[0, 50, 99, 50]]]),
    )

    scoreline = detect_scoreline_for_split(
        image, OCRConfig(), foreground_mask
    )

    assert scoreline["visible"] is False


def test_scoreline_detector_accepts_line_inside_pill_foreground(monkeypatch):
    """Line nam trong mask vien duoc giu lai de xu ly side split."""

    image = np.full((100, 100, 3), 180, dtype=np.uint8)
    foreground_mask = np.zeros((100, 100), dtype=np.uint8)
    foreground_mask[10:90, 10:90] = 1
    monkeypatch.setattr(
        cv2,
        "HoughLinesP",
        lambda *args, **kwargs: np.asarray([[[50, 10, 50, 89]]]),
    )

    scoreline = detect_scoreline_for_split(
        image, OCRConfig(), foreground_mask
    )

    assert scoreline["visible"] is True
    assert scoreline["foreground_coverage"] == 1.0


def test_region_filter_keeps_imprint_and_rejects_canvas_false_regions():
    """Box AH12 trong vien duoc giu, box nam ngoai vien bi loai."""

    foreground_mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(foreground_mask, (60, 60), 48, 1, -1)
    items = [
        {
            "text": "AH12",
            "confidence": 0.90,
            "polygon": [[38, 44], [82, 44], [82, 76], [38, 76]],
        },
        {
            "text": "829",
            "confidence": 0.36,
            "polygon": [[0, 45], [12, 45], [12, 75], [0, 75]],
        },
    ]

    accepted, rejected = filter_text_regions(items, foreground_mask, OCRConfig())

    assert [item["text"] for item in accepted] == ["AH12"]
    assert [item["rejection_reason"] for item in rejected] == [
        "outside_pill_mask"
    ]


def test_region_filter_rejects_canvas_edge_and_oversized_regions():
    """Hai gate hinh hoc con lai phai loai dung text region bat thuong."""

    foreground_mask = np.ones((120, 120), dtype=np.uint8)
    items = [
        {
            "text": "edge",
            "confidence": 0.80,
            "polygon": [[0, 45], [12, 45], [12, 75], [0, 75]],
        },
        {
            "text": "large",
            "confidence": 0.80,
            "polygon": [[3, 3], [116, 3], [116, 116], [3, 116]],
        },
    ]

    accepted, rejected = filter_text_regions(items, foreground_mask, OCRConfig())

    assert accepted == []
    assert [item["rejection_reason"] for item in rejected] == [
        "touches_canvas_edge",
        "text_region_too_large",
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
