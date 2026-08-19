from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.pill_safety.cv.segmentation.models.yolo_model import SegmentationModel
from src.pill_safety.cv.segmentation.trainers.trainer import SegmentationTrainer
from src.pill_safety.cv.segmentation.utils.config import (
    BASE_WEIGHTS,
    BATCH,
    DEVICE,
    EPOCHS,
    EXPERIMENT_NAME,
    EXPERIMENTS_ROOT,
    FREEZE,
    IMGSZ,
    OUTPUT_DIR,
    PATIENCE,
    RANDOM_SEED,
)


def main():
    model = SegmentationModel(BASE_WEIGHTS)
    trainer = SegmentationTrainer(
        model=model,
        output_dir=OUTPUT_DIR,
        experiments_root=EXPERIMENTS_ROOT,
        experiment_name=EXPERIMENT_NAME,
        epochs=EPOCHS,
        batch=BATCH,
        patience=PATIENCE,
        device=DEVICE,
        freeze=FREEZE,
        imgsz=IMGSZ,
        seed=RANDOM_SEED,
    )
    trainer.train()


if __name__ == "__main__":
    main()
