#!/usr/bin/env python3
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
from pill_safety.schemas import OCRInferenceRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Module 3 imprint OCR for one segmented pill crop."
    )
    parser.add_argument(
        "--request",
        type=Path,
        required=True,
        help="Path to a JSON file matching Module 3 input in docs/schema.md.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "ocr.yaml",
        help="OCR inference YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional artifact directory override.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    with args.request.open("r", encoding="utf-8") as file:
        request = OCRInferenceRequest.model_validate(json.load(file))

    path_updates = {}
    for field_name in ("crop_path", "mask_path"):
        value = Path(getattr(request, field_name))
        if not value.is_absolute():
            path_updates[field_name] = str(PROJECT_ROOT / value)
    if path_updates:
        request = request.model_copy(update=path_updates)

    config = OCRConfig.from_yaml(args.config)
    output_dir = args.output_dir or config.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    config = config.with_output_dir(output_dir)

    artifacts = OCRPredictor(config=config).predict_with_artifacts(request)
    print(
        json.dumps(
            artifacts.output.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
