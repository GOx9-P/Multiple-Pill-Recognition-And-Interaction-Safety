#!/usr/bin/env python3
"""Run the complete image-to-report medication safety pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the image request, all model configs, and the safety-report options."""

    parser = argparse.ArgumentParser(
        description="Run CV, retrieval, DDI lookup, and grounded LLM reporting."
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument(
        "--segmentation-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "segmentation.yaml",
    )
    parser.add_argument(
        "--attribute-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "attribute.yaml",
    )
    parser.add_argument(
        "--ocr-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "ocr.yaml",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "cv_pipeline.yaml",
    )
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "end_to_end",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--market", default="US")
    parser.add_argument("--known-drug-name", action="append", default=[])
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args(argv)


def _resolve_from_project(path: Path) -> Path:
    """Resolve relative paths from the project root for reproducible CLI runs."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> int:
    """Run CV once, then create the RAG/DDI/LLM result and audit artifacts."""

    args = parse_args(argv)
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
    if args.llm_provider:
        os.environ["LLM_PROVIDER"] = args.llm_provider

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    from pill_safety.core.config import get_settings

    get_settings.cache_clear()
    from pill_safety.cv.attribute.config import AttributeInferenceConfig
    from pill_safety.cv.attribute.predictors import AttributePredictor
    from pill_safety.cv.ocr import OCRConfig, OCRPredictor
    from pill_safety.cv.pipeline import CVPipelineAssembler, CVPipelineConfig, FullCVPipeline
    from pill_safety.cv.segmentation import SegmentationConfig, SegmentationPredictor
    from pill_safety.database.session import SessionLocal
    from pill_safety.rag.orchestration import EndToEndPostCvPipeline
    from pill_safety.schemas import SegmentationInferenceRequest

    with args.request.open("r", encoding="utf-8") as file:
        request = SegmentationInferenceRequest.model_validate(json.load(file))
    if not Path(request.image_path).is_absolute():
        request = request.model_copy(
            update={"image_path": str(PROJECT_ROOT / request.image_path)}
        )

    output_root = _resolve_from_project(args.output_dir)
    pipeline_config = CVPipelineConfig.from_yaml(args.pipeline_config)
    segmentation_config = SegmentationConfig.from_yaml(args.segmentation_config)
    segmentation_weights = _resolve_from_project(
        args.weights or segmentation_config.weights_path
    )
    segmentation_config = segmentation_config.with_weights_path(
        segmentation_weights
    ).with_output_dir(output_root)
    attribute_config = AttributeInferenceConfig.from_yaml(
        args.attribute_config
    ).resolve_paths(PROJECT_ROOT).with_output_dir(output_root)
    ocr_config = OCRConfig.from_yaml(args.ocr_config).with_output_dir(
        output_root / "predictions" / "ocr"
    )

    cv_artifacts = FullCVPipeline(
        segmentation_predictor=SegmentationPredictor(config=segmentation_config),
        attribute_predictor=AttributePredictor(config=attribute_config),
        ocr_predictor=OCRPredictor(config=ocr_config),
        pipeline_assembler=CVPipelineAssembler(
            config=pipeline_config.with_output_dir(output_root)
        ),
    ).predict_with_artifacts(request)

    db = SessionLocal()
    try:
        artifacts = EndToEndPostCvPipeline.from_database_session(
            db,
            llm_provider=args.llm_provider,
        ).run_with_artifacts(
            cv_artifacts.output,
            output_dir=output_root / "reports" / request.request_id,
            market=args.market,
            known_drug_names=args.known_drug_name,
        )
    finally:
        db.close()

    print(f"End-to-end artifacts: {artifacts.output_dir}")
    print(f"Pill summary: {artifacts.paths['pill_summary']}")
    if args.print_json:
        print(json.dumps(artifacts.output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
