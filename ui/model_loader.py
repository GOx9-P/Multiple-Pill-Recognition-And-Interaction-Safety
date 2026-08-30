"""Cached construction of the project's real CV pipeline."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

try:
    import streamlit as st
    cache_resource = st.cache_resource
except ImportError:
    st = None

    def cache_resource(*args, **kwargs):
        def decorator(f):
            return f
        if args and callable(args[0]):
            return args[0]
        return decorator


@dataclass(frozen=True)
class CVPipelineLoadResult:
    """Outcome of attempting to build the real FullCVPipeline."""

    pipeline: Any | None
    error: str | None

    @property
    def available(self) -> bool:
        return self.pipeline is not None


def _find_path(
    preferred_path: Path,
    patterns: list[str],
    scope_token: str | None = None,
) -> Path:
    """Find an artifact without crossing into another model family.

    A broad ``*.pt`` fallback can otherwise select an unrelated checkpoint
    (for example the legacy 16-class attribute model under the segmentation
    experiment).  When ``scope_token`` is supplied, every fallback candidate
    must live under a path containing that selected model name.
    """
    if preferred_path.exists():
        return preferred_path
    search_dirs = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "experiments",
        PROJECT_ROOT / "kaggle_uploads",
        Path("/kaggle/input"),
    ]
    for s_dir in search_dirs:
        if s_dir.exists():
            for pat in patterns:
                try:
                    for match in s_dir.rglob(pat):
                        if match.is_file() and ".git" not in str(match):
                            if (
                                scope_token
                                and scope_token.casefold() not in str(match).casefold()
                            ):
                                continue
                            return match
                except Exception:
                    continue
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
        ["*yolo*.pt", "*seg*.pt", "best.pt", "*.pt"],
        scope_token=project_path(segmentation_config.weights_path).parent.name,
    )
    segmentation_config = segmentation_config.with_weights_path(seg_weights).with_output_dir(output_root)

    # Auto-resolve attribute weights & configs
    attr_weights = _find_path(
        attribute_config.weights_path,
        ["*attr*.pt", "attr_last_v1_best.pt", "*resnet*.pt", "*attribute*.pt"],
        scope_token=attribute_config.selected_model,
    )
    attr_labels = _find_path(
        attribute_config.label_mapping_path,
        ["label_mapping.json", "*mapping*.json"],
        scope_token=attribute_config.selected_model,
    )
    attr_thresh = _find_path(
        attribute_config.color_thresholds_path,
        ["optimal_thresholds.json", "*thresholds*.json"],
        scope_token=attribute_config.selected_model,
    )
    attr_model_cfg = _find_path(
        attribute_config.model_config_path,
        ["model_config.yaml", "*config*.yaml"],
        scope_token=attribute_config.selected_model,
    )

    from dataclasses import replace
    attribute_config = replace(
        attribute_config,
        weights_path=attr_weights,
        label_mapping_path=attr_labels,
        color_thresholds_path=attr_thresh,
        model_config_path=attr_model_cfg,
    ).with_output_dir(output_root)

    ocr_config = ocr_config.with_output_dir(output_root / "predictions" / "ocr")
    pipeline_config = pipeline_config.with_output_dir(output_root)

    return FullCVPipeline(
        segmentation_predictor=SegmentationPredictor(segmentation_config),
        attribute_predictor=AttributePredictor(attribute_config),
        ocr_predictor=OCRPredictor(ocr_config),
        pipeline_assembler=CVPipelineAssembler(pipeline_config),
    )


@cache_resource(show_spinner="Đang khởi tạo các mô hình AI (YOLOv11, ResNet-18, PaddleOCR)...")
def load_cv_pipeline() -> CVPipelineLoadResult:
    """Try to build the real CV pipeline; return unavailable state on failure."""

    try:
        return CVPipelineLoadResult(pipeline=_build_cv_pipeline(), error=None)
    except Exception as exc:
        import traceback
        return CVPipelineLoadResult(pipeline=None, error=f"{exc}\n{traceback.format_exc()}")


@cache_resource
def load_interaction_checker():
    """Return the deterministic temporary interaction-checker entry point."""

    from pill_safety.rag.interaction_checker import check_interactions

    return check_interactions
