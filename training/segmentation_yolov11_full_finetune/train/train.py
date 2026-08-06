from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from src.pill_safety.cv.segmentation.utils.config import (
    OUTPUT_DIR,
    BASE_WEIGHTS,
    IMGSZ,
    EPOCHS,
    BATCH,
    PATIENCE,
    DEVICE,
    FREEZE,
    RANDOM_SEED,
    EXPERIMENT_NAME,
    EXPERIMENTS_ROOT,
)


def main():
    data_yaml = OUTPUT_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Không thấy {data_yaml}. Chạy data_preparation/prepare_data.py trước."
        )

    train_images_dir = OUTPUT_DIR / "images" / "train"
    n_train = len(list(train_images_dir.glob("*.*")))
    print(f"[train.py] {n_train} ảnh train (gốc + augmented) tại {train_images_dir}")

    model = YOLO(BASE_WEIGHTS)

    print(
        f"[train.py] Bắt đầu FULL fine-tune {BASE_WEIGHTS} "
        f"({EPOCHS} epochs, imgsz={IMGSZ}, freeze={FREEZE})"
    )

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        device=DEVICE,
        freeze=FREEZE,          # None = full fine-tune, không freeze layer nào
        seed=RANDOM_SEED,
        project=str(EXPERIMENTS_ROOT),
        name=EXPERIMENT_NAME,
        exist_ok=True,
    )

    best_ckpt = EXPERIMENTS_ROOT / EXPERIMENT_NAME / "weights" / "best.pt"
    print(f"\n[train.py] Xong. Checkpoint tốt nhất: {best_ckpt}")
    print("Tiếp theo: chạy evaluation/evaluate.py để tính mask mAP trên tập test.")


if __name__ == "__main__":
    main()
