from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.pill_safety.cv.segmentation.evaluators.evaluator import SegmentationEvaluator
from src.pill_safety.cv.segmentation.utils.config import (
    EVAL_SPLIT,
    OUTPUT_DIR,
    EXPERIMENTS_ROOT,
)

MODULE_EXPERIMENT_FOLDER = "segmentation_yolov11_full_finetune"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an explicit segmentation checkpoint.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--run-id",
        help="Resolve checkpoints/<run_id>_best.pt for this training run.",
    )
    selection.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to the exact checkpoint to evaluate.",
    )
    return parser.parse_args(argv)


def _resolve_checkpoint(args: argparse.Namespace, checkpoints_dir: Path) -> tuple[Path, str]:
    if args.checkpoint is not None:
        weights_path = args.checkpoint.expanduser().resolve()
        run_id = weights_path.stem.removesuffix("_best")
    else:
        run_id = args.run_id
        weights_path = checkpoints_dir / f"{run_id}_best.pt"

    if not weights_path.is_file():
        raise FileNotFoundError(f"Requested checkpoint does not exist: {weights_path}")
    return weights_path, run_id


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    experiment_dir = EXPERIMENTS_ROOT / MODULE_EXPERIMENT_FOLDER
    checkpoints_dir = experiment_dir / "checkpoints"
    weights_path, run_id = _resolve_checkpoint(args, checkpoints_dir)

    evaluator = SegmentationEvaluator(output_dir=OUTPUT_DIR, experiments_root=EXPERIMENTS_ROOT)
    evaluator.evaluate(weights_path=weights_path, split=EVAL_SPLIT, run_id=run_id)


if __name__ == "__main__":
    main()
