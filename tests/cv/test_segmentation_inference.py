"""Kiểm thử hợp đồng schema, artifact và hậu xử lý của inference phân đoạn."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from pill_safety.cv.segmentation import SegmentationConfig, SegmentationPredictor
from pill_safety.cv.segmentation.models import (
    RawSegmentationPrediction,
    SegmentationModel,
)
from pill_safety.cv.segmentation.postprocessing import process_prediction
from pill_safety.schemas import SegmentationInferenceRequest


class StaticSegmentationEngine:
    """Trả mask định sẵn và ghi nhận tham số mà predictor truyền vào model."""

    def __init__(self, predictions):
        """Khởi tạo model giả với danh sách dự đoán cố định."""

        self.predictions = predictions
        self.calls = []

    def predict(self, image_bgr, **kwargs):
        """Ghi nhận lời gọi rồi trả bản sao danh sách dự đoán đã cấu hình."""

        self.calls.append({"shape": image_bgr.shape, **kwargs})
        return list(self.predictions)


def _request(image_path: Path) -> SegmentationInferenceRequest:
    """Tạo request phân đoạn hợp lệ dùng chung trong các test."""

    return SegmentationInferenceRequest(
        request_id="req_123",
        session_id="sess_456",
        image_id="img_003",
        image_path=str(image_path),
    )


def _schema_shape(value):
    """Rút gọn dữ liệu thành cấu trúc kiểu để so sánh với schema tài liệu."""

    if isinstance(value, dict):
        return {key: _schema_shape(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_schema_shape(value[0])] if value else []
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def test_predictor_exports_exact_module_1_schema_and_artifacts(tmp_path):
    """Kiểm tra predictor xuất đúng schema, thứ tự viên và toàn bộ artifact."""

    rng = np.random.default_rng(42)
    image = rng.integers(30, 220, size=(180, 300, 3), dtype=np.uint8)
    image_path = tmp_path / "input.jpg"
    cv2.imwrite(str(image_path), image)

    left_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    right_mask = np.zeros_like(left_mask)
    cv2.ellipse(left_mask, (70, 90), (35, 50), 0, 0, 360, 1, -1)
    cv2.ellipse(right_mask, (230, 90), (40, 45), 0, 0, 360, 1, -1)
    # Trả viên bên phải trước để chứng minh ID chỉ được gán sau khi sắp xếp vị trí.
    engine = StaticSegmentationEngine(
        [
            RawSegmentationPrediction((190, 40, 270, 140), 0.93, right_mask),
            RawSegmentationPrediction((35, 35, 105, 145), 0.96, left_mask),
        ]
    )
    config = replace(
        SegmentationConfig(),
        output_dir=tmp_path / "outputs",
        crop_size=224,
    )

    artifacts = SegmentationPredictor(config=config, engine=engine).predict_with_artifacts(
        _request(image_path)
    )
    payload = artifacts.output.model_dump(mode="json")

    assert set(payload) == {
        "request_id",
        "session_id",
        "image_id",
        "image_quality",
        "instances",
    }
    assert [item["instance_id"] for item in payload["instances"]] == [
        "pill_001",
        "pill_002",
    ]
    assert payload["instances"][0]["bbox_xyxy"] == [35, 35, 105, 145]
    assert payload["instances"][0]["instance_token"].startswith("pill_token_")
    assert payload["instances"][0]["instance_token"] != payload["instances"][1]["instance_token"]
    assert engine.calls[0]["image_size"] == 640
    assert engine.calls[0]["device"] is None

    for instance in payload["instances"]:
        mask = cv2.imread(instance["mask_path"], cv2.IMREAD_GRAYSCALE)
        crop = cv2.imread(instance["crop_path"], cv2.IMREAD_COLOR)
        assert mask.shape == (224, 224)
        assert crop.shape == (224, 224, 3)
        assert np.any(mask > 0)
    assert artifacts.schema_json_path == (
        tmp_path
        / "outputs"
        / "predictions"
        / "segmentation"
        / "req_123"
        / "img_003"
        / "segmentation_output.json"
    )
    assert artifacts.overlay_path is not None and artifacts.overlay_path.exists()
    with artifacts.schema_json_path.open("r", encoding="utf-8") as file:
        assert json.load(file) == payload

    schema_document = (PROJECT_ROOT / "docs" / "schema.md").read_text(encoding="utf-8")
    module_1 = schema_document.split("## 2. Module 1", 1)[1].split("## 3. Module 2", 1)[0]
    json_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", module_1, re.S)
    documented_output = json.loads(json_blocks[1])
    # Ảnh không có cảnh báo được phép có danh sách quality_flags rỗng.
    payload_shape_sample = json.loads(json.dumps(payload))
    payload_shape_sample["instances"][0]["quality_flags"] = ["example_flag"]
    assert _schema_shape(payload_shape_sample) == _schema_shape(documented_output)


def test_no_detection_returns_valid_empty_output(tmp_path):
    """Kiểm tra trường hợp không phát hiện viên vẫn trả output rỗng hợp lệ."""

    image_path = tmp_path / "empty.jpg"
    cv2.imwrite(str(image_path), np.full((100, 140, 3), 127, dtype=np.uint8))
    config = replace(SegmentationConfig(), output_dir=tmp_path / "outputs")

    output = SegmentationPredictor(
        config=config,
        engine=StaticSegmentationEngine([]),
    ).predict(_request(image_path))

    assert output.instances == []
    assert output.request_id == "req_123"


def test_fragmented_border_mask_sets_safety_flags():
    """Kiểm tra mask vỡ và chạm biên sinh đủ các cờ cảnh báo an toàn."""

    image = np.full((100, 120, 3), 180, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (0, 50), 20, 1, -1)
    cv2.circle(mask, (90, 50), 12, 1, -1)
    prediction = RawSegmentationPrediction((0, 25, 105, 75), 0.40, mask)

    processed = process_prediction(image, prediction, SegmentationConfig())

    assert processed is not None
    assert processed.possible_merged_instance is True
    assert processed.possible_non_pill is True
    assert processed.occlusion_estimate >= 0.25
    assert "fragmented_mask" in processed.quality_flags
    assert "touches_image_border" in processed.quality_flags


def test_prediction_mask_must_use_original_image_coordinates():
    """Kiểm tra hậu xử lý từ chối mask không cùng hệ tọa độ với ảnh gốc."""

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    prediction = RawSegmentationPrediction(
        (10, 10, 50, 50),
        0.9,
        np.zeros((50, 50), dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="original image coordinates"):
        process_prediction(image, prediction, SegmentationConfig())


def test_model_adapter_resizes_masks_to_original_image_coordinates():
    """Kiểm tra adapter YOLO đưa mask về đúng kích thước của ảnh đầu vào."""

    class ArrayAdapter:
        def __init__(self, value):
            """Bọc mảng NumPy bằng giao diện tối thiểu giống tensor."""

            self.value = np.asarray(value)

        def detach(self):
            """Mô phỏng thao tác tách tensor khỏi đồ thị tính toán."""

            return self

        def cpu(self):
            """Mô phỏng thao tác chuyển tensor về CPU."""

            return self

        def numpy(self):
            """Trả dữ liệu dưới dạng mảng NumPy."""

            return self.value

    class Boxes:
        xyxy = ArrayAdapter([[10, 20, 90, 70]])
        conf = ArrayAdapter([0.88])

    class Masks:
        data = ArrayAdapter([np.ones((20, 40), dtype=np.float32)])

    class FakeUltralyticsModel:
        def predict(self, **kwargs):
            """Trả kết quả YOLO giả để kiểm tra lớp chuyển đổi output."""

            self.kwargs = kwargs
            result = type("Result", (), {"boxes": Boxes(), "masks": Masks()})()
            return [result]

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    wrapper = SegmentationModel("unused.pt")
    wrapper._model = FakeUltralyticsModel()
    predictions = wrapper.predict(
        image,
        image_size=640,
        confidence_threshold=0.25,
        iou_threshold=0.6,
        mask_threshold=0.5,
        device=None,
    )

    assert len(predictions) == 1
    assert predictions[0].mask.shape == image.shape[:2]
    assert predictions[0].mask.dtype == np.uint8
    assert predictions[0].confidence == pytest.approx(0.88)


def test_yaml_defaults_match_segmentation_contract():
    """Kiểm tra cấu hình YAML mặc định khớp hợp đồng inference đã chốt."""

    config = SegmentationConfig.from_yaml(
        PROJECT_ROOT / "configs" / "inference" / "segmentation.yaml"
    )
    assert config.image_size == 640
    assert config.confidence_threshold == 0.25
    assert config.iou_threshold == 0.60
    assert config.mask_threshold == 0.50
    assert config.crop_size == 640
    assert config.align_long_axis is True
    assert config.weights_path.name == "yolov11m_seg_mediseg_full_finetune_v1.pt"
    assert config.output_dir == Path("outputs")


@pytest.mark.skipif(
    os.getenv("RUN_SEGMENTATION_SMOKE") != "1"
    or importlib.util.find_spec("ultralytics") is None,
    reason="Set RUN_SEGMENTATION_SMOKE=1 in an Ultralytics environment.",
)
def test_real_checkpoint_smoke(tmp_path):
    """Chạy smoke test tùy chọn bằng checkpoint thật trên CPU hoặc GPU."""

    candidates = [
        PROJECT_ROOT
        / "models"
        / "segmentation_yolov11_full_finetune"
        / "yolov11m_seg_mediseg_full_finetune_v1.pt",
        PROJECT_ROOT / "experiments" / "segmentation_yolov11_full_finetune" / "checkpoints" / "best.pt",
    ]
    weights = next((path for path in candidates if path.is_file()), None)
    if weights is None:
        pytest.skip("No segmentation checkpoint is available.")

    image_path = tmp_path / "smoke.jpg"
    cv2.imwrite(str(image_path), np.full((256, 256, 3), 180, dtype=np.uint8))
    config = replace(
        SegmentationConfig(),
        weights_path=weights,
        output_dir=tmp_path / "outputs",
        device="cpu",
    )
    output = SegmentationPredictor(config=config).predict(_request(image_path))
    assert output.request_id == "req_123"
