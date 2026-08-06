from __future__ import annotations

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


def _latest_best_weight(checkpoints_dir: Path) -> Path:
    candidates = sorted(checkpoints_dir.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy checkpoint tốt nhất trong {checkpoints_dir}.")
    return candidates[0]


def main():
    experiment_dir = EXPERIMENTS_ROOT / MODULE_EXPERIMENT_FOLDER
    checkpoints_dir = experiment_dir / "checkpoints"
    weights_path = _latest_best_weight(checkpoints_dir)
    run_id = weights_path.stem.removesuffix("_best")

    evaluator = SegmentationEvaluator(output_dir=OUTPUT_DIR, experiments_root=EXPERIMENTS_ROOT)
    evaluator.evaluate(weights_path=weights_path, split=EVAL_SPLIT, run_id=run_id)


if __name__ == "__main__":
    main()
