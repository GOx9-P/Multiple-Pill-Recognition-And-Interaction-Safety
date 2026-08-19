#!/usr/bin/env python3
"""Entrypoint dòng lệnh để chạy inference cho Module 1 segmentation."""

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

from pill_safety.cv.segmentation import SegmentationConfig, SegmentationPredictor
from pill_safety.schemas import SegmentationInferenceRequest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Đọc tham số CLI nhưng không triển khai lại logic inference tại đây."""

    parser = argparse.ArgumentParser(
        description="Run Module 1 YOLOv11-Seg inference for one input image."
    )
    parser.add_argument(
        "--request",
        type=Path,
        required=True,
        help="JSON file matching Module 1 input in docs/schema.md.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "segmentation.yaml",
        help="Segmentation inference YAML configuration.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Explicit checkpoint override, useful before a weight is promoted to models/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional outputs root override for masks/crops/predictions.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Đọc request/config, chạy predictor và in Module 1 JSON ra stdout."""

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

    config = SegmentationConfig.from_yaml(args.config)
    weights_path = args.weights or config.weights_path
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    config = config.with_weights_path(weights_path)

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = config.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    config = config.with_output_dir(output_dir)

    artifacts = SegmentationPredictor(config=config).predict_with_artifacts(
        request
    )
    # Giữ stdout ở dạng JSON để module khác đọc được; log runtime dùng stderr.
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
