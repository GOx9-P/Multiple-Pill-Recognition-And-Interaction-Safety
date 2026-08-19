"""Kiem thu dieu phoi end-to-end CV bang predictor gia, khong nap model that."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from pill_safety.cv.pipeline import (
    CVPipelineAssembler,
    CVPipelineConfig,
    FullCVPipeline,
)
from pill_safety.schemas import (
    AttributeInferenceOutput,
    OCRInferenceOutput,
    SegmentationInferenceOutput,
    SegmentationInferenceRequest,
)


def _segmentation_output() -> SegmentationInferenceOutput:
    """Tao hai crop de kiem tra Attribute va OCR chay tren tung instance."""

    return SegmentationInferenceOutput.model_validate(
        {
            "request_id": "req_001",
            "session_id": "sess_001",
            "image_id": "img_001",
            "image_quality": {
                "status": "usable",
                "blur_score": 0.1,
                "glare_detected": False,
                "lighting_warning": False,
            },
            "instances": [
                {
                    "instance_id": "pill_001",
                    "instance_token": "token_001",
                    "bbox_xyxy": [10, 20, 70, 80],
                    "mask_path": "outputs/masks/pill_001_mask.png",
                    "color_crop_path": "outputs/crops/pill_001_color_crop.png",
                    "shape_crop_path": "outputs/crops/pill_001_shape_crop.png",
                    "ocr_crop_path": "outputs/crops/pill_001_ocr_crop.png",
                    "crop_path": "outputs/crops/pill_001_crop.png",
                    "segmentation": {
                        "confidence": 0.96,
                        "occlusion_estimate": 0.0,
                        "possible_merged_instance": False,
                        "possible_non_pill": False,
                    },
                    "quality_flags": [],
                },
                {
                    "instance_id": "pill_002",
                    "instance_token": "token_002",
                    "bbox_xyxy": [90, 20, 150, 80],
                    "mask_path": "outputs/masks/pill_002_mask.png",
                    "color_crop_path": "outputs/crops/pill_002_color_crop.png",
                    "shape_crop_path": "outputs/crops/pill_002_shape_crop.png",
                    "ocr_crop_path": "outputs/crops/pill_002_ocr_crop.png",
                    "crop_path": "outputs/crops/pill_002_crop.png",
                    "segmentation": {
                        "confidence": 0.95,
                        "occlusion_estimate": 0.0,
                        "possible_merged_instance": False,
                        "possible_non_pill": False,
                    },
                    "quality_flags": [],
                },
            ],
        }
    )


def _attribute_output(request) -> AttributeInferenceOutput:
    """Tao output Attribute hop le, giu scoreline placeholder cho OCR ghi de."""

    return AttributeInferenceOutput.model_validate(
        {
            "request_id": request.request_id,
            "session_id": request.session_id,
            "image_id": request.image_id,
            "instance_id": request.instance_id,
            "instance_token": request.instance_token,
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


def _ocr_output(request) -> OCRInferenceOutput:
    """Tao output OCR hop le de kiem tra fusion lay scoreline tu Module 3."""

    return OCRInferenceOutput.model_validate(
        {
            "request_id": request.request_id,
            "session_id": request.session_id,
            "image_id": request.image_id,
            "instance_id": request.instance_id,
            "instance_token": request.instance_token,
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
            "imprint_visibility": {"visible": True, "confidence": 0.9},
            "imprint": {
                "visible": True,
                "raw": "K 56",
                "confidence": 0.9,
                "text_regions": [],
                "ocr_observations": [],
                "normalized_candidates": [
                    {
                        "text": "K 56",
                        "score": 0.9,
                        "source": "raw_ocr",
                        "evidence": ["test"],
                    }
                ],
            },
        }
    )


class _SegmentationPredictor:
    """Predictor gia tra ket qua segmentation da chuan bi."""

    def predict_with_artifacts(self, request):
        """Kiem tra request goc va tra output Module 1."""

        assert request.image_id == "img_001"
        return SimpleNamespace(output=_segmentation_output())


class _AttributePredictor:
    """Predictor gia ghi lai cac request Attribute de kiem tra flow crop/mask."""

    def __init__(self):
        """Khoi tao danh sach request da nhan."""

        self.requests = []

    def predict_with_artifacts(self, request):
        """Ghi request va tra output Attribute cung instance token."""

        self.requests.append(request)
        return SimpleNamespace(output=_attribute_output(request))


class _OCRPredictor:
    """Predictor gia ghi lai cac request OCR de kiem tra flow crop/mask."""

    def __init__(self):
        """Khoi tao danh sach request da nhan."""

        self.requests = []

    def predict_with_artifacts(self, request):
        """Ghi request va tra output OCR cung instance token."""

        self.requests.append(request)
        return SimpleNamespace(output=_ocr_output(request))


def test_full_cv_pipeline_runs_modules_in_dependency_order(tmp_path):
    """Segmentation phai tao crop/mask truoc khi Attribute va OCR chay tung vien."""

    attribute_predictor = _AttributePredictor()
    ocr_predictor = _OCRPredictor()
    pipeline = FullCVPipeline(
        segmentation_predictor=_SegmentationPredictor(),
        attribute_predictor=attribute_predictor,
        ocr_predictor=ocr_predictor,
        pipeline_assembler=CVPipelineAssembler(
            CVPipelineConfig(output_dir=tmp_path / "outputs")
        ),
    )
    request = SegmentationInferenceRequest(
        request_id="req_001",
        session_id="sess_001",
        image_id="img_001",
        image_path="input.jpg",
    )

    artifacts = pipeline.predict_with_artifacts(request)

    assert [item.instance_id for item in attribute_predictor.requests] == [
        "pill_001",
        "pill_002",
    ]
    assert [item.instance_token for item in ocr_predictor.requests] == [
        "token_001",
        "token_002",
    ]
    assert attribute_predictor.requests[0].crop_path.endswith("pill_001_color_crop.png")
    assert attribute_predictor.requests[0].shape_crop_path.endswith("pill_001_shape_crop.png")
    assert ocr_predictor.requests[0].crop_path.endswith("pill_001_ocr_crop.png")
    assert ocr_predictor.requests[1].mask_path.endswith("pill_002_mask.png")
    assert artifacts.output.schema_version == "cv_output_v1"
    assert len(artifacts.output.pills) == 2
    assert artifacts.pipeline_artifacts.schema_json_path.exists()
