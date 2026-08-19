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
    segmentation_config = segmentation_config.with_weights_path(
        project_path(segmentation_config.weights_path)
    ).with_output_dir(output_root)
    ocr_config = ocr_config.with_output_dir(output_root / "predictions" / "ocr")
    pipeline_config = pipeline_config.with_output_dir(output_root)

    return FullCVPipeline(
        segmentation_predictor=SegmentationPredictor(segmentation_config),
        attribute_predictor=AttributePredictor(attribute_config),
        ocr_predictor=OCRPredictor(ocr_config),
        pipeline_assembler=CVPipelineAssembler(pipeline_config),
    )


@st.cache_resource(show_spinner="Đang tải các mô hình CV...")
def load_cv_pipeline() -> CVPipelineLoadResult:
    """Try to build the real CV pipeline; return unavailable state on failure."""

    try:
        return CVPipelineLoadResult(pipeline=_build_cv_pipeline(), error=None)
    except Exception as exc:
        return CVPipelineLoadResult(pipeline=None, error=str(exc))


@st.cache_resource
def load_interaction_checker():
    """Return the deterministic temporary interaction-checker entry point."""

    from pill_safety.rag.interaction_checker import check_interactions

    return check_interactions
