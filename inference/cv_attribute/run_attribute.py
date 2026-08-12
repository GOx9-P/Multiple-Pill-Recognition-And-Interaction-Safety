#!/usr/bin/env python3
"""Entrypoint dòng lệnh để chạy inference Attribute Recognition của Module 2."""

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

from pill_safety.cv.attribute.config import AttributeInferenceConfig
from pill_safety.schemas import AttributeInferenceRequest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Đọc tham số CLI và giữ logic ResNet18 trong package ``src``."""

    parser = argparse.ArgumentParser(
        description="Run Module 2 ResNet18 attribute inference for one pill crop."
    )
    parser.add_argument(
        "--request",
        type=Path,
        required=True,
        help="JSON file matching Module 2 input in docs/schema.md.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "attribute.yaml",
        help="Attribute inference YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional outputs root override for the Module 2 JSON artifact.",
    )
    return parser.parse_args(argv)


def _resolve_request_paths(
    request: AttributeInferenceRequest,
) -> AttributeInferenceRequest:
    """Đổi crop/mask tương đối trong request sang đường dẫn tuyệt đối của project."""

    updates = {}
    for field_name in ("crop_path", "mask_path"):
        value = Path(getattr(request, field_name))
        if not value.is_absolute():
            updates[field_name] = str(PROJECT_ROOT / value)
    return request.model_copy(update=updates) if updates else request


def main(argv: list[str] | None = None) -> int:
    """Đọc request/config, chạy Module 2 và in JSON schema ra stdout."""

    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    with args.request.open("r", encoding="utf-8") as file:
        request = AttributeInferenceRequest.model_validate(json.load(file))
    request = _resolve_request_paths(request)

    config = AttributeInferenceConfig.from_yaml(args.config).resolve_paths(
        PROJECT_ROOT
    )
    if args.output_dir is not None:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        config = config.with_output_dir(output_dir)

    logging.info("Selected Attribute model: %s", config.selected_model)
    # Chỉ import runtime PyTorch khi thật sự chạy model để ``--help`` vẫn dùng được
    # ở môi trường chỉ kiểm tra cấu hình hoặc chưa cài dependency deep-learning.
    from pill_safety.cv.attribute.predictors import AttributePredictor

    artifacts = AttributePredictor(config=config).predict_with_artifacts(request)
    # Giữ stdout là JSON thuần để CV pipeline hoặc script khác có thể đọc trực tiếp.
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
