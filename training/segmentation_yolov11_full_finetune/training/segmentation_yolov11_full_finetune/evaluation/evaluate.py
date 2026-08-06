"""
evaluate.py
-------------
KHÔNG có trong script gốc — thêm để hoàn thiện pipeline: đánh giá checkpoint
đã train (weights/best.pt) trên split test, tính mask mAP@0.5 / mAP@0.5:0.95
(chỉ tiêu Report.pdf: Module 1 > 85%), lưu ra experiments/<EXPERIMENT_NAME>/metrics/.

Usage:
    python evaluation/evaluate.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from ultralytics import YOLO
from src.segmentation.config import (
    OUTPUT_DIR,
    EXPERIMENTS_ROOT,
    EXPERIMENT_NAME,
    EVAL_WEIGHTS_NAME,
    EVAL_SPLIT,
    EVAL_CONF_THRESHOLD,
    EVAL_IOU_THRESHOLD,
    TARGET_MASK_MAP50_95,
)


def main():
    weights_path = EXPERIMENTS_ROOT / EXPERIMENT_NAME / "weights" / EVAL_WEIGHTS_NAME
    data_yaml = OUTPUT_DIR / "data.yaml"

    if not weights_path.exists():
        raise FileNotFoundError(f"Không thấy checkpoint {weights_path}. Chạy train/train.py trước.")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Không thấy {data_yaml}. Chạy data_preparation/prepare_data.py trước.")

    model = YOLO(str(weights_path))

    print(f"[evaluate.py] Evaluate {weights_path} trên split '{EVAL_SPLIT}'...")
    results = model.val(
        data=str(data_yaml),
        split=EVAL_SPLIT,
        conf=EVAL_CONF_THRESHOLD,
        iou=EVAL_IOU_THRESHOLD,
        project=str(EXPERIMENTS_ROOT),
        name=f"{EXPERIMENT_NAME}_eval",
        exist_ok=True,
    )

    seg_metrics = results.seg  # metrics riêng cho mask (khác box mAP)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "weights": str(weights_path),
        "split": EVAL_SPLIT,
        "mask_mAP50": float(seg_metrics.map50),
        "mask_mAP50_95": float(seg_metrics.map),
        "mask_precision": float(seg_metrics.mp),
        "mask_recall": float(seg_metrics.mr),
        "target_mask_mAP50_95": TARGET_MASK_MAP50_95,
        "meets_target": bool(seg_metrics.map >= TARGET_MASK_MAP50_95),
    }

    metrics_dir = EXPERIMENTS_ROOT / EXPERIMENT_NAME / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"eval_{EVAL_SPLIT}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== KẾT QUẢ =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nĐã lưu metrics -> {out_path}")


if __name__ == "__main__":
    main()
