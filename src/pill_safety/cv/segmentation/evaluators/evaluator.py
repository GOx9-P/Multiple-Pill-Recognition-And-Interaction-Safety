from __future__ import annotations

import itertools
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import ops
from src.pill_safety.cv.segmentation.transforms.augment_utils import yolo_seg_lines_to_masks
from src.pill_safety.cv.segmentation.utils.config import (
    EVAL_CONF_THRESHOLDS,
    EVAL_IOU_THRESHOLDS,
    EVAL_MASK_THRESHOLDS,
    EVAL_SELECTION_METRIC,
    RAW_ANN_PATH,
)


class ThresholdSegmentationValidator(SegmentationValidator):
    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        mask_threshold = float(getattr(self.args, "mask_threshold", 0.5))
        if self.args.save_json or self.args.save_txt:
            self.process = lambda proto, coeff, bboxes, shape: self._process_mask_native(
                proto, coeff, bboxes, shape, mask_threshold
            )
        else:
            self.process = lambda proto, coeff, bboxes, shape: self._process_mask(
                proto, coeff, bboxes, shape, mask_threshold
            )

    def _process_mask(self, protos, masks_in, bboxes, shape, mask_threshold: float) -> torch.Tensor:
        c, mh, mw = protos.shape
        if masks_in.shape[0] == 0:
            return torch.zeros((0, *(shape if True else (mh, mw))), dtype=torch.uint8, device=masks_in.device)

        masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
        masks = F.interpolate(masks[None], shape, mode="bilinear")[0]
        return ops.crop_mask(masks.gt_(mask_threshold).byte(), bboxes)

    def _process_mask_native(self, protos, masks_in, bboxes, shape, mask_threshold: float) -> torch.Tensor:
        c, mh, mw = protos.shape
        h, w = shape
        if masks_in.shape[0] == 0:
            return torch.zeros((0, h, w), dtype=torch.uint8, device=masks_in.device)

        coeffs = masks_in @ protos.float().view(c, -1)
        step = max(1, 32_000_000 // (h * w))
        masks = [
            ops.scale_masks(coeffs[i : i + step].view(-1, mh, mw)[None], shape)[0].gt_(mask_threshold).byte()
            for i in range(0, coeffs.shape[0], step)
        ]
        return ops.crop_mask(torch.cat(masks), bboxes)

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
        if split == "val":
            thresholds = self._tune_thresholds(model, temp_dir, weights_path)
            selected_thresholds = thresholds["thresholds"]
            results = self._run_evaluation(model, split, selected_thresholds, temp_dir)
            self._save_thresholds(selected_thresholds, run_id)
        else:
            selected_thresholds = self._load_thresholds(run_id)
            results = self._run_evaluation(model, split, selected_thresholds, temp_dir)

        metrics = self._summarize_results(
            results,
            weights_path,
            split,
            run_id,
            thresholds=selected_thresholds,
            temp_dir=temp_dir,
        )
        if split == "val":
            self._collect_error_cases(temp_dir, run_id)

        predictions_file = temp_dir / "predictions.json"
        if predictions_file.exists():
            target_predictions = self.predictions_dir / f"{run_id}_{split}_predictions.json"
            shutil.copy2(predictions_file, target_predictions)

        shutil.rmtree(temp_dir, ignore_errors=True)
        return metrics

    def _run_evaluation(self, model: YOLO, split: str, thresholds: dict[str, float], temp_dir: Path) -> Any:
        results = model.val(
            validator=ThresholdSegmentationValidator,
            data=str(self.output_dir / "data.yaml"),
            split=split,
            conf=thresholds["confidence"],
            iou=thresholds["iou"],
            mask_threshold=thresholds["mask"],
            save_dir=str(temp_dir),
            exist_ok=True,
            save_json=True,
        )
        return results

    def _load_thresholds(self, run_id: str) -> dict[str, float]:
        thresholds_path = self.metrics_dir / f"{run_id}_thresholds.json"
        if not thresholds_path.exists():
            raise FileNotFoundError(f"Threshold file not found for run {run_id}: {thresholds_path}")
        with open(thresholds_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj["thresholds"]

    def _save_thresholds(self, thresholds: dict[str, float], run_id: str) -> None:
        thresholds_path = self.metrics_dir / f"{run_id}_thresholds.json"
        with open(thresholds_path, "w", encoding="utf-8") as f:
            json.dump({"thresholds": thresholds}, f, indent=2, ensure_ascii=False)

    def _tune_thresholds(self, model: YOLO, temp_dir: Path, weights_path: Path) -> dict[str, Any]:
        best_score = float("-inf")
        best_thresholds = None
        results_log = []

        for confidence, iou, mask in itertools.product(
            EVAL_CONF_THRESHOLDS,
            EVAL_IOU_THRESHOLDS,
            EVAL_MASK_THRESHOLDS,
        ):
            current_temp = temp_dir / f"sweep_conf{confidence}_iou{iou}_mask{mask}"
            current_temp.mkdir(parents=True, exist_ok=True)
            results = model.val(
                validator=ThresholdSegmentationValidator,
                data=str(self.output_dir / "data.yaml"),
                split="val",
                conf=confidence,
                iou=iou,
                mask_threshold=mask,
                save_dir=str(current_temp),
                exist_ok=True,
                save_json=True,
            )

            metrics = self._summarize_results(
                results,
                weights_path,
                "val",
                f"sweep_{confidence}_{iou}_{mask}",
                thresholds={"confidence": confidence, "iou": iou, "mask": mask},
                temp_dir=current_temp,
            )
            results_score = metrics["metrics"].get(EVAL_SELECTION_METRIC)
            results_log.append({
                "confidence": confidence,
                "iou": iou,
                "mask": mask,
                "score": results_score,
            })
            if results_score is not None and results_score > best_score:
                best_score = results_score
                best_thresholds = {"confidence": confidence, "iou": iou, "mask": mask}

            shutil.rmtree(current_temp, ignore_errors=True)

        if best_thresholds is None:
            raise RuntimeError("Failed to select thresholds during validation sweep.")

        self._write_threshold_sweep_log(temp_dir, results_log)
        return {"thresholds": best_thresholds, "best_score": best_score}

    def _write_threshold_sweep_log(self, temp_dir: Path, results_log: list[dict[str, Any]]) -> None:
        log_path = self.metrics_dir / f"threshold_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"sweep": results_log}, f, indent=2, ensure_ascii=False)

    def _load_checkpoint_info(self, weights_path: Path) -> dict[str, Any]:
        info = {
            "best_checkpoint": str(weights_path),
            "best_epoch": None,
        }
        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict):
                epoch = checkpoint.get("epoch")
                if isinstance(epoch, (int, float)):
                    info["best_epoch"] = int(epoch)
                elif isinstance(epoch, str) and epoch.isdigit():
                    info["best_epoch"] = int(epoch)
        except Exception:
            pass
        return info

    def _decode_rle_string(self, rle_str: str) -> list[int]:
        counts: list[int] = []
        i = 0
        while i < len(rle_str):
            x = 0
            shift = 0
            while True:
                c = ord(rle_str[i]) - 48
                i += 1
                x |= (c & 0x1F) << (5 * shift)
                shift += 1
                if not (c & 0x20):
                    if c & 0x10:
                        x |= -1 << (5 * shift)
                    break
            if len(counts) > 2:
                x += counts[-2]
            counts.append(int(x))
        return counts

    def _rle_to_mask(self, rle: dict[str, Any]) -> np.ndarray:
        counts = self._decode_rle_string(rle["counts"])
        h, w = rle["size"]
        flat = np.zeros(h * w, dtype=np.uint8)
        value = 0
        idx = 0
        for count in counts:
            if value:
                flat[idx : idx + count] = 1
            idx += count
            value ^= 1
        if idx < flat.size:
            flat[idx:] = 0
        return flat.reshape((h, w), order="F")

    def _load_image_id_to_filename_map(self) -> dict[Any, str]:
        try:
            with open(RAW_ANN_PATH, "r", encoding="utf-8") as f:
                coco = json.load(f)
        except Exception as exc:
            print(
                f"[warn] unable to load COCO annotation metadata for image_id mapping: {RAW_ANN_PATH}: {exc}"
            )
            return {}

        images = coco.get("images")
        if not isinstance(images, list):
            print(f"[warn] COCO annotation file missing images list: {RAW_ANN_PATH}")
            return {}

        mapping: dict[Any, str] = {}
        for img in images:
            image_id = img.get("id")
            file_name = img.get("file_name")
            if image_id is None or not isinstance(file_name, str) or not file_name.strip():
                continue
            mapping[image_id] = file_name
            mapping[str(image_id)] = file_name
        return mapping

    def _resolve_prediction_image_name(self, entry: dict[str, Any], image_id_to_filename: dict[Any, str]) -> str | None:
        image_id = entry.get("image_id")
        if image_id is not None:
            file_name = image_id_to_filename.get(image_id)
            if file_name:
                return file_name
            print(
                f"[warn] prediction entry has image_id={image_id} but no file_name mapping was found; skipping entry."
            )
            return None

        file_name = entry.get("file_name")
        if isinstance(file_name, str) and file_name.strip():
            return file_name.strip()
        return None

    def _load_predicted_masks(self, predictions: list[dict[str, Any]]) -> dict[str, list[np.ndarray]]:
        image_id_to_filename = self._load_image_id_to_filename_map()
        grouped: dict[str, list[np.ndarray]] = {}
        for entry in predictions:
            file_name = self._resolve_prediction_image_name(entry, image_id_to_filename)
            if file_name is None:
                continue
            segmentation = entry.get("segmentation")
            if not isinstance(segmentation, dict):
                continue
            try:
                mask = self._rle_to_mask(segmentation)
            except Exception:
                continue
            grouped.setdefault(file_name, []).append(mask)
        return grouped

    def _load_ground_truth_masks(self, label_path: Path, image_path: Path) -> list[np.ndarray]:
        if not label_path.exists() or not image_path.exists():
            return []
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return []
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        h, w = image.shape[:2]
        masks, _ = yolo_seg_lines_to_masks(lines, w, h)
        return masks

    def _mask_iou_matrix(self, gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> np.ndarray:
        if not gt_masks or not pred_masks:
            return np.zeros((len(gt_masks), len(pred_masks)), dtype=float)
        gt_stack = np.stack([mask.astype(bool) for mask in gt_masks], axis=0)
        pred_stack = np.stack([mask.astype(bool) for mask in pred_masks], axis=0)
        intersection = np.logical_and(gt_stack[:, None], pred_stack[None]).sum(axis=(2, 3)).astype(float)
        union = np.logical_or(gt_stack[:, None], pred_stack[None]).sum(axis=(2, 3)).astype(float)
        union[union == 0] = 1.0
        return intersection / union

    def _compute_contract_metrics(self, split: str, temp_dir: Path) -> dict[str, float]:
        predictions_file = temp_dir / "predictions.json"
        if not predictions_file.exists():
            return {
                "instance_recall": 0.0,
                "merge_error_rate": 0.0,
                "false_positive_rate": 0.0,
            }

        with open(predictions_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        grouped_predictions = self._load_predicted_masks(predictions)
        image_dir = self.output_dir / "images" / split
        label_dir = self.output_dir / "labels" / split

        total_gt = 0
        total_tp = 0
        total_preds = 0
        merge_error_images = 0
        total_multi_pill_images = 0

        for image_path in sorted(image_dir.glob("*.*")):
            file_name = image_path.name
            label_path = label_dir / f"{image_path.stem}.txt"
            gt_masks = self._load_ground_truth_masks(label_path, image_path)
            pred_masks = grouped_predictions.get(file_name, [])
            total_gt += len(gt_masks)
            total_preds += len(pred_masks)

            if len(gt_masks) >= 2:
                total_multi_pill_images += 1

            if not gt_masks or not pred_masks:
                continue

            iou_matrix = self._mask_iou_matrix(gt_masks, pred_masks)
            iou_copy = iou_matrix.copy()
            matched_gt: set[int] = set()
            matched_pred: set[int] = set()
            while True:
                gt_idx, pred_idx = divmod(int(np.argmax(iou_copy)), iou_copy.shape[1])
                best_iou = float(iou_copy[gt_idx, pred_idx])
                if best_iou < 0.5:
                    break
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                iou_copy[gt_idx, :] = 0.0
                iou_copy[:, pred_idx] = 0.0

            total_tp += len(matched_gt)

            if len(gt_masks) >= 2:
                for pred_idx in range(len(pred_masks)):
                    if int((iou_matrix[:, pred_idx] >= 0.5).sum()) >= 2:
                        merge_error_images += 1
                        break

        false_positives = max(total_preds - total_tp, 0)
        instance_recall = float(total_tp / total_gt) if total_gt > 0 else 0.0
        merge_error_rate = float(merge_error_images / total_multi_pill_images) if total_multi_pill_images > 0 else 0.0
        false_positive_rate = float(false_positives / total_preds) if total_preds > 0 else 0.0

        return {
            "instance_recall": instance_recall,
            "merge_error_rate": merge_error_rate,
            "false_positive_rate": false_positive_rate,
        }

    def _summarize_results(
        self,
        results: Any,
        weights_path: Path,
        split: str,
        run_id: str,
        thresholds: dict[str, float] | None = None,
        temp_dir: Path | None = None,
    ) -> dict[str, Any]:
        seg_metrics = getattr(results, "seg", None)
        checkpoint_info = self._load_checkpoint_info(weights_path)
        summary: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "module": MODULE_EXPERIMENT_FOLDER,
            "weights": str(weights_path),
            "best_checkpoint": checkpoint_info["best_checkpoint"],
            "best_epoch": checkpoint_info["best_epoch"],
            "selection_metric": EVAL_SELECTION_METRIC,
            "thresholds": {
                "detection_confidence_threshold": thresholds["confidence"],
                "iou_threshold": thresholds["iou"],
                "mask_threshold": thresholds["mask"],
            }
            if thresholds is not None
            else {},
            "split": split,
            "run_id": run_id,
        }

        metrics: dict[str, float] = {}
        if seg_metrics is not None:
            metrics.update(
                {
                    "mask_mAP50": float(seg_metrics.map50),
                    "mask_mAP50_95": float(seg_metrics.map),
                    "mask_precision": float(seg_metrics.mp),
                    "mask_recall": float(seg_metrics.mr),
                }
            )

        if temp_dir is not None:
            contract_metrics = self._compute_contract_metrics(split, temp_dir)
            metrics.update(contract_metrics)
        else:
            metrics.update(
                {
                    "instance_recall": 0.0,
                    "merge_error_rate": 0.0,
                    "false_positive_rate": 0.0,
                }
            )

        summary.update(metrics)
        summary["metrics"] = metrics

        filename = f"{run_id}_{split}_metrics.json"
        out_path = self.metrics_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary

    def _ensure_evidence_dirs(self, run_id: str) -> dict[str, Path]:
        evidence_root = self.predictions_dir / run_id
        evidence_root.mkdir(parents=True, exist_ok=True)
        categories = ["easy", "touching", "overlap", "glare", "failed_cases"]
        dirs: dict[str, Path] = {}
        for category in categories:
            path = evidence_root / category
            path.mkdir(parents=True, exist_ok=True)
            dirs[category] = path
        return dirs

    def _make_overlay_image(
        self,
        image_path: Path,
        gt_masks: list[np.ndarray],
        pred_masks: list[np.ndarray],
        output_path: Path,
    ) -> None:
        image = cv2.imread(str(image_path))
        if image is None:
            return

        overlay = image.copy().astype(np.uint8)
        alpha = 0.4

        if gt_masks:
            gt_overlay = np.zeros_like(image, dtype=np.uint8)
            for mask in gt_masks:
                gt_overlay[mask.astype(bool)] = (0, 255, 0)
            overlay = cv2.addWeighted(overlay, 1.0, gt_overlay, alpha, 0)

        if pred_masks:
            pred_overlay = np.zeros_like(image, dtype=np.uint8)
            for mask in pred_masks:
                pred_overlay[mask.astype(bool)] = (0, 0, 255)
            overlay = cv2.addWeighted(overlay, 1.0, pred_overlay, alpha, 0)

        cv2.imwrite(str(output_path), overlay)

    def _classify_evidence_category(
        self,
        image_path: Path,
        gt_masks: list[np.ndarray],
        pred_masks: list[np.ndarray],
    ) -> str | None:
        # No reliable project logic for easy/touching/overlap/glare was found in the repo.
        # We keep these categories present but empty unless classification rules are added.
        return None

    def _collect_error_cases(self, temp_dir: Path, run_id: str) -> None:
        predictions_file = temp_dir / "predictions.json"
        if not predictions_file.exists():
            return

        with open(predictions_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        image_id_to_filename = self._load_image_id_to_filename_map()
        evidence_dirs = self._ensure_evidence_dirs(run_id)
        image_dir = self.output_dir / "images" / "val"
        label_dir = self.output_dir / "labels" / "val"

        grouped_predictions: dict[str, list[dict[str, Any]]] = {}
        for entry in predictions:
            image_name = self._resolve_prediction_image_name(entry, image_id_to_filename)
            if image_name is None:
                continue
            grouped_predictions.setdefault(image_name, []).append(entry)

        error_cases = []
        for image_path in sorted(image_dir.glob("*.*")):
            image_name = image_path.name
            label_path = label_dir / f"{image_path.stem}.txt"
            gt_masks = self._load_ground_truth_masks(label_path, image_path)
            preds = grouped_predictions.get(image_name, [])
            pred_masks: list[np.ndarray] = []
            if preds:
                for entry in preds:
                    segmentation = entry.get("segmentation")
                    if isinstance(segmentation, dict):
                        try:
                            pred_masks.append(self._rle_to_mask(segmentation))
                        except Exception:
                            continue

            failure_reason = None
            if not preds and gt_masks:
                failure_reason = "false_negative"
            elif preds and not gt_masks:
                failure_reason = "false_positive"
            elif gt_masks and pred_masks:
                iou_matrix = self._mask_iou_matrix(gt_masks, pred_masks)
                if iou_matrix.size > 0:
                    # false positive/negative are handled above; check merge/split
                    gt_matches = (iou_matrix >= 0.5).sum(axis=1)
                    pred_matches = (iou_matrix >= 0.5).sum(axis=0)
                    if int((pred_matches >= 2).sum()) > 0:
                        failure_reason = "merge"
                    elif int((gt_matches >= 2).sum()) > 0:
                        failure_reason = "split"
                    elif len(preds) != len(gt_masks):
                        failure_reason = "partial_match"
            elif preds and not gt_masks:
                failure_reason = "false_positive"
            elif gt_masks and not preds:
                failure_reason = "false_negative"

            if failure_reason is None:
                continue

            evidence_path = evidence_dirs["failed_cases"] / f"{image_path.stem}_{image_name}.png"
            self._make_overlay_image(image_path, gt_masks, pred_masks, evidence_path)

            error_cases.append(
                {
                    "image": str(evidence_path.relative_to(self.predictions_dir)),
                    "ground_truth": self._load_ground_truth(label_path),
                    "prediction": [
                        {
                            "category_id": int(p.get("category_id", -1)),
                            "bbox": p.get("bbox"),
                            "confidence": float(p.get("score", 0.0)),
                        }
                        for p in preds
                    ],
                    "failure_reason": failure_reason,
                }
            )

        out_path = self.predictions_dir / run_id / f"{run_id}_error_cases.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(error_cases, f, indent=2, ensure_ascii=False)

    def _load_ground_truth(self, label_path: Path) -> list[str]:
        if not label_path.exists():
            return []
        with open(label_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
