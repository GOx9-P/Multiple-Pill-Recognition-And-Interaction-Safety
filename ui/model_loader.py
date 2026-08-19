"""Cached construction of the project's real CV pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CVPipelineLoadResult:
    """Outcome of attempting to build the real FullCVPipeline."""

    pipeline: Any | None
    error: str | None

    @property
    def available(self) -> bool:
        return self.pipeline is not None


def _find_path(preferred_path: Path, patterns: list[str]) -> Path:
    """Find file from preferred path or search in models/ and kaggle input dirs."""
    if preferred_path.exists():
        return preferred_path
    search_dirs = [
        PROJECT_ROOT / "models",
        Path("/kaggle/input"),
        Path.cwd() / "models",
        Path.cwd(),
    ]
    for s_dir in search_dirs:
        if s_dir.exists():
            for pat in patterns:
                matches = list(s_dir.rglob(pat))
                if matches:
                    return matches[0]
    return preferred_path


def _build_cv_pipeline() -> Any:
    """Build the actual predictors and full orchestrator from inference YAML files."""

    from pill_safety.cv.attribute.config import AttributeInferenceConfig
    from pill_safety.cv.attribute.predictors import AttributePredictor
    from pill_safety.cv.ocr import OCRConfig, OCRPredictor
    from pill_safety.cv.pipeline import CVPipelineAssembler, CVPipelineConfig, FullCVPipeline
    from pill_safety.cv.segmentation import SegmentationConfig, SegmentationPredictor

    config_directory = PROJECT_ROOT / "configs" / "inference"
    segmentation_config = SegmentationConfig.from_yaml(
        config_directory / "segmentation.yaml"
    )
    attribute_config = AttributeInferenceConfig.from_yaml(
        config_directory / "attribute.yaml"
    ).resolve_paths(PROJECT_ROOT)
    ocr_config = OCRConfig.from_yaml(config_directory / "ocr.yaml")
    pipeline_config = CVPipelineConfig.from_yaml(config_directory / "cv_pipeline.yaml")

    def project_path(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    output_root = project_path(pipeline_config.output_dir)

    # Auto-resolve segmentation weights
    seg_weights = _find_path(
        project_path(segmentation_config.weights_path),
        ["*yolo*.pt", "*seg*.pt", "best.pt", "*.pt"]
    )
    segmentation_config = segmentation_config.with_weights_path(seg_weights).with_output_dir(output_root)

    # Auto-resolve attribute weights & configs
    attr_weights = _find_path(
        attribute_config.weights_path,
        ["*resnet*.pt", "*attribute*.pt", "best.pt"]
    )
    attr_labels = _find_path(
        attribute_config.label_mapping_path,
        ["label_mapping.json", "*mapping*.json"]
    )
    attr_thresh = _find_path(
        attribute_config.color_thresholds_path,
        ["optimal_thresholds.json", "*thresholds*.json"]
    )
    attr_model_cfg = _find_path(
        attribute_config.model_config_path,
        ["model_config.yaml", "*config*.yaml"]
    )

    from dataclasses import replace
    attribute_config = replace(
        attribute_config,
        weights_path=attr_weights,
        label_mapping_path=attr_labels,
        color_thresholds_path=attr_thresh,
        model_config_path=attr_model_cfg,
    ).with_output_dir(output_root)

    valid_ocr_versions = {"PP-OCR", "PP-OCRv2", "PP-OCRv3", "PP-OCRv4"}
    if ocr_config.ocr_version not in valid_ocr_versions:
        from dataclasses import replace
        ocr_config = replace(ocr_config, ocr_version="PP-OCRv4")
    ocr_config = ocr_config.with_output_dir(output_root / "predictions" / "ocr")
    pipeline_config = pipeline_config.with_output_dir(output_root)

    return FullCVPipeline(
        segmentation_predictor=SegmentationPredictor(segmentation_config),
        attribute_predictor=AttributePredictor(attribute_config),
        ocr_predictor=OCRPredictor(ocr_config),
        pipeline_assembler=CVPipelineAssembler(pipeline_config),
    )


@st.cache_resource(show_spinner="Đang khởi tạo các mô hình AI (YOLOv11, ResNet-18, PaddleOCR)...")
def load_cv_pipeline() -> CVPipelineLoadResult:
    """Try to build the real CV pipeline; return unavailable state on failure."""

    try:
        return CVPipelineLoadResult(pipeline=_build_cv_pipeline(), error=None)
    except Exception as exc:
        import traceback
        return CVPipelineLoadResult(pipeline=None, error=f"{exc}\n{traceback.format_exc()}")


@st.cache_resource
def load_interaction_checker():
    """Return the deterministic temporary interaction-checker entry point."""

    from pill_safety.rag.interaction_checker import check_interactions

    return check_interactions
