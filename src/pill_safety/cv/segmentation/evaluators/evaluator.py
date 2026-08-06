from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from src.pill_safety.cv.segmentation.utils.config import (
    EVAL_CONF_THRESHOLD,
    EVAL_IOU_THRESHOLD,
)

MODULE_EXPERIMENT_FOLDER = "segmentation_yolov11_full_finetune"


class SegmentationEvaluator:
    def __init__(self, output_dir: Path, experiments_root: Path):
        self.output_dir = Path(output_dir)
        self.experiment_dir = Path(experiments_root) / MODULE_EXPERIMENT_FOLDER
        self.predictions_dir = self.experiment_dir / "predictions"
        self.metrics_dir = self.experiment_dir / "metrics"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, weights_path: Path, split: str, run_id: str) -> dict[str, Any]:
        temp_dir = self.experiment_dir / ".tmp_eval" / run_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(weights_path))
        results = model.val(
            data=str(self.output_dir / "data.yaml"),
            split=split,
            conf=EVAL_CONF_THRESHOLD,
            iou=EVAL_IOU_THRESHOLD,
            save_dir=str(temp_dir),
            exist_ok=True,
            save_json=True,
        )

        metrics = self._summarize_results(results, weights_path, split, run_id)
        if split == "val":
            self._collect_error_cases(temp_dir, run_id)

        # Persist predictions JSON to the experiment predictions folder.
        predictions_file = temp_dir / "predictions.json"
        if predictions_file.exists():
            target_predictions = self.predictions_dir / f"{run_id}_{split}_predictions.json"
            shutil.copy2(predictions_file, target_predictions)

        shutil.rmtree(temp_dir, ignore_errors=True)
        return metrics

    def _summarize_results(self, results: Any, weights_path: Path, split: str, run_id: str) -> dict[str, Any]:
        seg_metrics = getattr(results, "seg", None)
        summary = {
            "timestamp": datetime.now().isoformat(),
            "weights": str(weights_path),
            "split": split,
            "run_id": run_id,
        }

        if seg_metrics is not None:
            summary.update(
                {
                    "mask_mAP50": float(seg_metrics.map50),
                    "mask_mAP50_95": float(seg_metrics.map),
                    "mask_precision": float(seg_metrics.mp),
                    "mask_recall": float(seg_metrics.mr),
                }
            )

        filename = f"{run_id}_{split}_metrics.json"
        out_path = self.metrics_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary

    def _collect_error_cases(self, temp_dir: Path, run_id: str) -> None:
        predictions_file = temp_dir / "predictions.json"
        if not predictions_file.exists():
            return

        with open(predictions_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        image_dir = self.output_dir / "images" / "val"
        label_dir = self.output_dir / "labels" / "val"
        target_dir = self.predictions_dir / run_id
        target_dir.mkdir(parents=True, exist_ok=True)

        grouped = {}
        for entry in predictions:
            grouped.setdefault(entry["file_name"], []).append(entry)

        error_cases = []
        for image_path in sorted(image_dir.glob("*.*")):
            image_name = image_path.name
            label_path = label_dir / f"{image_path.stem}.txt"
            gt = self._load_ground_truth(label_path)
            preds = grouped.get(image_name, [])

            if not preds and not gt:
                continue

            if not preds:
                reason = "false_negative"
            elif not gt:
                reason = "false_positive"
            elif len(preds) != len(gt):
                reason = "partial_match"
            else:
                continue

            dest_image = target_dir / image_name
            if not dest_image.exists():
                shutil.copy2(image_path, dest_image)

            error_cases.append(
                {
                    "image": str(dest_image.relative_to(self.predictions_dir)),
                    "ground_truth": gt,
                    "prediction": [
                        {
                            "category_id": int(p.get("category_id", -1)),
                            "bbox": p.get("bbox"),
                            "confidence": float(p.get("score", 0.0)),
                        }
                        for p in preds
                    ],
                    "failure_reason": reason,
                }
            )

        out_path = target_dir / f"{run_id}_error_cases.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(error_cases, f, indent=2, ensure_ascii=False)

    def _load_ground_truth(self, label_path: Path) -> list[str]:
        if not label_path.exists():
            return []
        with open(label_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
