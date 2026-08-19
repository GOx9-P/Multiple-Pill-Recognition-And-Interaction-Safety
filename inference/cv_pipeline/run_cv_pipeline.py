#!/usr/bin/env python3
"""Entrypoint dong lenh de fusion Module 1, 2, 3 thanh CV output v1."""

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

from pill_safety.cv.pipeline.config import CVPipelineConfig
from pill_safety.cv.pipeline.orchestration import CVPipelineAssembler
from pill_safety.schemas import CVPipelineInput


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Doc input JSON da chua ba output module va cau hinh artifact Module 4."""

    parser = argparse.ArgumentParser(
        description="Fuse Module 1, Module 2 and Module 3 outputs into cv_output_v1."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON object matching Module 4 input in docs/schema.md.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "cv_pipeline.yaml",
        help="CV Pipeline inference YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional outputs root override for cv_output.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Doc cac output CV, fusion strict va in cv_output_v1 ra stdout."""

    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    with args.input.open("r", encoding="utf-8") as file:
        pipeline_input = CVPipelineInput.model_validate(json.load(file))

    config = CVPipelineConfig.from_yaml(args.config)
    output_dir = args.output_dir or config.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    artifacts = CVPipelineAssembler(
        config=config.with_output_dir(output_dir)
    ).predict_with_artifacts(pipeline_input)
    print(json.dumps(artifacts.output.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
