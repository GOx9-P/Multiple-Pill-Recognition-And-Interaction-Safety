#!/usr/bin/env python3
"""Entrypoint dong lenh chay lien mach Module 1, 2, 3 va Module 4 CV."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from pill_safety.cv.ocr import OCRConfig, OCRPredictor
from pill_safety.cv.pipeline import (
    CVPipelineAssembler,
    CVPipelineConfig,
    FullCVPipeline,
)
from pill_safety.cv.segmentation import SegmentationConfig, SegmentationPredictor
from pill_safety.schemas import SegmentationInferenceRequest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Doc request anh va cac config cua ba model CV."""

    parser = argparse.ArgumentParser(
        description="Run segmentation, attribute, OCR and CV fusion for one image."
    )
    parser.add_argument(
        "--request",
        type=Path,
        required=True,
        help="JSON file matching Module 1 input in docs/schema.md.",
    )
    parser.add_argument(
        "--segmentation-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "segmentation.yaml",
        help="Module 1 YAML configuration.",
    )
    parser.add_argument(
        "--attribute-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "attribute.yaml",
        help="Module 2 YAML configuration.",
    )
    parser.add_argument(
        "--ocr-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "ocr.yaml",
        help="Module 3 YAML configuration.",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "cv_pipeline.yaml",
        help="Module 4 YAML configuration.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional Module 1 checkpoint override.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Shared output root for artifacts from all CV modules.",
    )
    return parser.parse_args(argv)


def _resolve_from_project(path: Path) -> Path:
    """Doi path tuong doi thanh path tuyet doi tinh tu project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> int:
    """Chay day du CV tu mot anh va in cv_output_v1 ra stdout."""

    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    with args.request.open("r", encoding="utf-8") as file:
        request = SegmentationInferenceRequest.model_validate(json.load(file))
    if not Path(request.image_path).is_absolute():
        request = request.model_copy(
            update={"image_path": str(PROJECT_ROOT / request.image_path)}
        )

    pipeline_config = CVPipelineConfig.from_yaml(args.pipeline_config)
    output_root = _resolve_from_project(args.output_dir or pipeline_config.output_dir)

    segmentation_config = SegmentationConfig.from_yaml(args.segmentation_config)
    segmentation_weights = _resolve_from_project(
        args.weights or segmentation_config.weights_path
    )
    segmentation_config = segmentation_config.with_weights_path(
        segmentation_weights
    ).with_output_dir(output_root)

    from pill_safety.cv.attribute.config import AttributeInferenceConfig
    from pill_safety.cv.attribute.predictors import AttributePredictor

    attribute_config = AttributeInferenceConfig.from_yaml(
        args.attribute_config
    ).resolve_paths(PROJECT_ROOT).with_output_dir(output_root)
    ocr_config = OCRConfig.from_yaml(args.ocr_config).with_output_dir(
        output_root / "predictions" / "ocr"
    )

    artifacts = FullCVPipeline(
        segmentation_predictor=SegmentationPredictor(config=segmentation_config),
        attribute_predictor=AttributePredictor(config=attribute_config),
        ocr_predictor=OCRPredictor(config=ocr_config),
        pipeline_assembler=CVPipelineAssembler(
            config=pipeline_config.with_output_dir(output_root)
        ),
    ).predict_with_artifacts(request)
    print(json.dumps(artifacts.output.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
