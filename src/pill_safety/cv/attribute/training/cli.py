"""CLI dung chung cho head-tune, last-block fine-tune, calibration va test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pill_safety.cv.attribute.training.workflow import (
    calibrate_color_thresholds,
    compare_validation_runs,
    evaluate_test,
    load_config,
    train,
)


def _parser(default_command: str | None = None, default_config: Path | None = None) -> argparse.ArgumentParser:
    """Tao parser co path override de notebook Kaggle khong can sua source code."""
    parser = argparse.ArgumentParser(description="Train and evaluate multi-task ResNet18 attributes.")
    parser.add_argument("command", choices=("train", "calibrate", "evaluate", "compare"), nargs="?", default=default_command)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--data-root", type=str)
    parser.add_argument("--output-root", type=str)
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--thresholds", type=str)
    parser.add_argument("--pretrained-from", type=str)
    parser.add_argument("--head-val-metrics", type=str)
    parser.add_argument("--last-val-metrics", type=str)
    parser.add_argument("--comparison-output", type=str)
    return parser


def main(argv: list[str] | None = None, default_command: str | None = None, default_config: Path | None = None) -> int:
    """Chay command va in JSON ket qua de shell/notebook co the tai su dung."""
    args = _parser(default_command, default_config).parse_args(argv)
    if args.command == "compare":
        if not all((args.head_val_metrics, args.last_val_metrics, args.comparison_output)):
            raise ValueError("compare requires --head-val-metrics, --last-val-metrics and --comparison-output.")
        result = compare_validation_runs(args.head_val_metrics, args.last_val_metrics, args.comparison_output)
    else:
        if not args.config:
            raise ValueError("--config is required for train, calibrate and evaluate.")
        config = load_config(args.config)
        common = {"data_root_override": args.data_root, "output_root_override": args.output_root, "run_id_override": args.run_id}
        if args.command == "train":
            result = train(config, pretrained_override=args.pretrained_from, **common)
        elif args.command == "calibrate":
            if not args.checkpoint:
                raise ValueError("calibrate requires --checkpoint.")
            result = calibrate_color_thresholds(config, args.checkpoint, **common)
        else:
            if not args.checkpoint or not args.thresholds:
                raise ValueError("evaluate requires --checkpoint and --thresholds.")
            result = evaluate_test(config, args.checkpoint, args.thresholds, **common)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
