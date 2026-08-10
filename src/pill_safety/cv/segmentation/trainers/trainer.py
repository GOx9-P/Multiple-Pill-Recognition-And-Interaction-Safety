from __future__ import annotations

import csv
import json
import math
import platform
import shutil
import socket
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
import yaml

from src.pill_safety.cv.segmentation.models.yolo_model import SegmentationModel
from src.pill_safety.cv.segmentation.utils.config import (
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
    CLASS_AGNOSTIC,
)

MODULE_EXPERIMENT_FOLDER = "segmentation_yolov11_full_finetune"


class SegmentationTrainer:
    def __init__(
        self,
        model: SegmentationModel,
        output_dir: Path,
        experiments_root: Path,
        experiment_name: str,
        epochs: int,
        batch: int,
        patience: int,
        device: str | int,
        freeze: str | None,
        imgsz: int,
        seed: int,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.experiments_root = Path(experiments_root)
        self.experiment_name = experiment_name
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.patience = patience
        self.device = device
        self.freeze = freeze
        self.seed = seed
        self.data_yaml = self.output_dir / "data.yaml"
        self.module_dir = self.experiments_root / MODULE_EXPERIMENT_FOLDER

    def _generate_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.experiment_name}_{timestamp}"

    def _prepare_experiment_dirs(self, run_id: str) -> dict[str, Path]:
        checkpoint_dir = self.module_dir / "checkpoints"
        logs_dir = self.module_dir / "logs"
        metrics_dir = self.module_dir / "metrics"
        plots_dir = self.module_dir / "plots"
        predictions_dir = self.module_dir / "predictions"

        for directory in (checkpoint_dir, logs_dir, metrics_dir, plots_dir, predictions_dir):
            directory.mkdir(parents=True, exist_ok=True)

        return {
            "checkpoint_dir": checkpoint_dir,
            "logs_dir": logs_dir,
            "metrics_dir": metrics_dir,
            "plots_dir": plots_dir,
            "predictions_dir": predictions_dir,
        }

    def _write_config(self, run_id: str, logs_dir: Path) -> None:
        config = {
            "run_id": run_id,
            "module": MODULE_EXPERIMENT_FOLDER,
            "seed": self.seed,
            "dataset": {
                "output_dir": str(self.output_dir),
                "data_yaml": str(self.data_yaml),
            },
            "model": {
                "architecture": "YOLOv11-Seg",
                "pretrained_weight": self.model.base_weights,
            },
            "training": {
                "image_size": self.imgsz,
                "epochs": self.epochs,
                "batch_size": self.batch,
                "patience": self.patience,
                "device": self.device,
                "freeze": self.freeze,
                "seed": self.seed,
            },
        }

        out_path = logs_dir / f"{run_id}_config.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)

    def _write_dataset_manifest(self, run_id: str, logs_dir: Path) -> None:
        def _relative_image_paths(split: str) -> list[str]:
            image_dir = self.output_dir / "images" / split
            return [
                str(path.relative_to(self.output_dir))
                for path in sorted(image_dir.glob("*.*"))
            ]

        train_images = _relative_image_paths("train")
        val_images = _relative_image_paths("val")
        test_images = _relative_image_paths("test")

        manifest = {
            "run_id": run_id,
            "dataset_name": "MEDISEG",
            "data_yaml": str(self.data_yaml),
            "train_count": len(train_images),
            "val_count": len(val_images),
            "test_count": len(test_images),
            "train_images": train_images,
            "val_images": val_images,
            "test_images": test_images,
            # legacy fields kept for backward compatibility; values below will be evidence-backed
            "split_before_augmentation": None,
            "augmentation_train_only": None,
            "label_mapping_file": None,
        }

        split_strategy_path = self.output_dir / "conversion_stats.json"
        try:
            with open(split_strategy_path, "r", encoding="utf-8") as handle:
                manifest["split_strategy"] = json.load(handle).get("split_strategy")
        except (OSError, ValueError, AttributeError):
            manifest["split_strategy"] = None

        # Additional evidence and anti-leakage checks
        import re
        import hashlib
        import json as _json
        from src.pill_safety.cv.segmentation.utils.config import OUTPUT_DIR as _OUT

        def _sha256_of_file(p: Path) -> str | None:
            try:
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                return h.hexdigest()
            except Exception:
                return None

        # classify train images into original vs augmented based on pattern _augN
        aug_pattern = re.compile(r"_aug\d+$", re.IGNORECASE)
        train_names = [Path(p).name for p in train_images]
        original_train = [n for n in train_names if not aug_pattern.search(Path(n).stem)]
        augmented_train = [n for n in train_names if aug_pattern.search(Path(n).stem)]

        # counts
        orig_train_count = len(original_train)
        aug_train_count = len(augmented_train)

        # n_aug determinism: if aug_count divides evenly by num originals
        n_aug_determined = None
        n_aug_source = None
        if orig_train_count > 0 and aug_train_count > 0 and aug_train_count % orig_train_count == 0:
            n_aug_determined = aug_train_count // orig_train_count
            n_aug_source = f"filesystem:{self.output_dir / 'images' / 'train'}"

        # split source: attempt to record RAW_ANN_PATH if available
        split_source = None
        try:
            from src.pill_safety.cv.segmentation.utils.config import RAW_ANN_PATH
            if RAW_ANN_PATH and Path(RAW_ANN_PATH).exists():
                split_source = {"type": "coco_annotations", "path": str(RAW_ANN_PATH), "sha256": _sha256_of_file(Path(RAW_ANN_PATH))}
        except Exception:
            split_source = None

        # compute hashes of splits (evidence)
        def _hash_list(lst: list[str]) -> str:
            h = hashlib.sha256()
            for v in sorted(lst):
                h.update(v.encode("utf-8"))
            return h.hexdigest()

        split_hashes = {
            "train_hash": _hash_list(train_names),
            "val_hash": _hash_list([Path(p).name for p in val_images]),
            "test_hash": _hash_list([Path(p).name for p in test_images]),
        }

        # leakage checks by filename
        train_set = set(train_names)
        val_set = set([Path(p).name for p in val_images])
        test_set = set([Path(p).name for p in test_images])

        train_val_overlap = len(train_set & val_set)
        train_test_overlap = len(train_set & test_set)
        val_test_overlap = len(val_set & test_set)
        leakage_passed = (train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0)

        # split_before_augmentation check: require that for every augmented file, the base original exists
        split_before_aug = True

        # augmentation_train_only: true if val/test contain no augmented files
        aug_in_val = any(aug_pattern.search(Path(p).stem) for p in val_images)
        aug_in_test = any(aug_pattern.search(Path(p).stem) for p in test_images)
        if not aug_in_val and not aug_in_test:
            augmentation_train_only_val = {"value": True, "source": "filesystem:no_aug_in_val_test"}
        else:
            augmentation_train_only_val = {"value": False, "source": "filesystem:aug_in_val_or_test"}

        # class distribution: read label files in OUTPUT_DIR/labels/<split>
        def _class_counts_for_split(split: str) -> dict[str, int]:
            counts: dict[str, int] = {}
            lbl_dir = self.output_dir / "labels" / split
            if not lbl_dir.exists():
                return counts
            for p in lbl_dir.glob("*.txt"):
                try:
                    for line in p.read_text(encoding="utf-8").splitlines():
                        tok = line.strip().split()
                        if not tok:
                            continue
                        cls = tok[0]
                        counts[cls] = counts.get(cls, 0) + 1
                except Exception:
                    continue
            return counts

        class_dist = {
            "train": _class_counts_for_split("train"),
            "val": _class_counts_for_split("val"),
            "test": _class_counts_for_split("test"),
        }

        # attach enhanced fields to manifest
        manifest.update(
            {
                "split_source": split_source,
                "split_hashes": split_hashes,
                "original_train_count": orig_train_count,
                "augmented_train_count": aug_train_count,
                "n_aug_determined": {"value": n_aug_determined, "source": n_aug_source},
                "split_before_augmentation": split_before_aug,
                "augmentation_train_only": augmentation_train_only_val,
                "leakage_check": {
                    "passed": leakage_passed,
                    "train_val_overlap": train_val_overlap,
                    "train_test_overlap": train_test_overlap,
                    "val_test_overlap": val_test_overlap,
                },
                "class_distribution": class_dist,
            }
        )

        out_path = logs_dir / f"{run_id}_dataset_manifest.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def _write_runtime(
        self,
        run_id: str,
        logs_dir: Path,
        started_at: datetime,
        finished_at: datetime,
        epoch_timing: dict[str, Any],
    ) -> None:
        try:
            import torch
        except ImportError:
            torch = None

        try:
            import ultralytics
            ultralytics_version = getattr(ultralytics, "__version__", None)
        except ImportError:
            ultralytics_version = None

        cuda_version = None
        if torch is not None:
            cuda_version = getattr(torch.version, "cuda", None)

        runtime = {
            "run_id": run_id,
            "module": MODULE_EXPERIMENT_FOLDER,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": int((finished_at - started_at).total_seconds()),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "device": str(self.device),
            "cuda_available": bool(torch.cuda.is_available()) if torch is not None else False,
            "cuda_version": cuda_version,
            "torch_version": str(torch.__version__) if torch is not None else None,
            "ultralytics_version": ultralytics_version,
            "epoch_times": epoch_timing["epoch_times"],
            "epoch_timing_source": epoch_timing["source"],
            "epoch_timing_available": epoch_timing["available"],
        }

        if torch is not None and torch.cuda.is_available():
            runtime["gpu_name"] = torch.cuda.get_device_name(0)

        out_path = logs_dir / f"{run_id}_runtime.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(runtime, f, indent=2, ensure_ascii=False)

    def _register_epoch_timing_callbacks(self, model: Any) -> dict[str, Any]:
        """Register non-invasive Ultralytics callbacks that time completed epochs."""
        timing: dict[str, Any] = {
            "epoch_times": [],
            "started_epochs": {},
            "source": "ultralytics_callbacks:on_train_epoch_start,on_train_epoch_end",
            "available": False,
        }

        def _epoch_number(trainer: Any) -> int | None:
            epoch = getattr(trainer, "epoch", None)
            return epoch + 1 if isinstance(epoch, int) else None

        def _on_epoch_start(trainer: Any) -> None:
            epoch = _epoch_number(trainer)
            if epoch is None:
                return
            timing["started_epochs"][epoch] = (datetime.now(), perf_counter())

        def _on_epoch_end(trainer: Any) -> None:
            epoch = _epoch_number(trainer)
            if epoch is None:
                return
            started = timing["started_epochs"].pop(epoch, None)
            if started is None:
                timing["epoch_times"].append(
                    {
                        "epoch": epoch,
                        "started_at": None,
                        "finished_at": datetime.now().isoformat(),
                        "duration_seconds": None,
                    }
                )
                return
            started_at, started_tick = started
            finished_at = datetime.now()
            timing["epoch_times"].append(
                {
                    "epoch": epoch,
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_seconds": perf_counter() - started_tick,
                }
            )

        try:
            model.add_callback("on_train_epoch_start", _on_epoch_start)
            model.add_callback("on_train_epoch_end", _on_epoch_end)
            timing["available"] = True
        except (AttributeError, TypeError, ValueError):
            timing["source"] = "unavailable: Ultralytics epoch callbacks could not be registered"
        return timing

    def _finalize_epoch_timing(self, timing: dict[str, Any]) -> dict[str, Any]:
        """Retain partially observed epochs without inventing an end time or duration."""
        recorded_epochs = {item["epoch"] for item in timing["epoch_times"]}
        for epoch, (started_at, _) in timing["started_epochs"].items():
            if epoch not in recorded_epochs:
                timing["epoch_times"].append(
                    {
                        "epoch": epoch,
                        "started_at": started_at.isoformat(),
                        "finished_at": None,
                        "duration_seconds": None,
                    }
                )
        timing["epoch_times"].sort(key=lambda item: item["epoch"])
        timing.pop("started_epochs", None)
        return timing

    def _capture_training_config(self, run_id: str, logs_dir: Path, model: Any) -> None:
        """Capture minimal training config (learning_rate, optimizer, scheduler, class_agnostic).

        Priority: read resolved values from `model.trainer.args` after `model.train()`.
        If unavailable, fall back to project config (CLASS_AGNOSTIC) or mark as unavailable.
        """
        trainer = getattr(model, "trainer", None)
        trainer_args = getattr(trainer, "args", None) if trainer is not None else None

        def _get_arg(obj, *keys):
            if obj is None:
                return None
            for k in keys:
                # handle Namespace-like and dict-like
                try:
                    if hasattr(obj, k):
                        val = getattr(obj, k)
                        if val is not None:
                            return val
                except Exception:
                    pass
                try:
                    if isinstance(obj, dict) and k in obj:
                        return obj[k]
                except Exception:
                    pass
            return None

        lr_val = _get_arg(trainer_args, "lr0", "lr", "learning_rate")
        opt_val = _get_arg(trainer_args, "optimizer", "opt")
        sched_val = _get_arg(trainer_args, "scheduler", "lr_scheduler", "sched")

        # Class agnostic: prefer trainer args, else project config
        class_agnostic_val = _get_arg(trainer_args, "class_agnostic")
        if class_agnostic_val is None:
            try:
                class_agnostic_val = CLASS_AGNOSTIC
                class_agnostic_source = "project_config"
            except Exception:
                class_agnostic_val = None
                class_agnostic_source = "unavailable"
        else:
            class_agnostic_source = "trainer.args"

        def _format_source(val, source_name: str):
            if val is None:
                return {"value": None, "source": "unavailable"}
            return {"value": val, "source": source_name}

        cfg = {
            "run_id": run_id,
            "learning_rate": _format_source(lr_val, "trainer.args" if lr_val is not None else "unavailable"),
            "optimizer": _format_source(opt_val, "trainer.args" if opt_val is not None else "unavailable"),
            "scheduler": _format_source(sched_val, "trainer.args" if sched_val is not None else "unavailable"),
            "class_agnostic": {"value": class_agnostic_val, "source": class_agnostic_source},
        }

        # --- Capture augmentation_online from trainer args (do not guess missing values) ---
        online_keys = [
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
            "flipud",
            "fliplr",
            "mosaic",
            "mixup",
            "copy_paste",
        ]

        augmentation_online: dict[str, object] = {}
        for k in online_keys:
            val = _get_arg(trainer_args, k)
            augmentation_online[k] = val if val is not None else None

        # --- Capture augmentation_offline by introspecting augment_utils.get_augmentation_pipeline() ---
        augmentation_offline = None
        exact_offline: list[dict[str, object]] = []
        try:
            from src.pill_safety.cv.segmentation.transforms import augment_utils

            try:
                pipeline = augment_utils.get_augmentation_pipeline()
                # albumentations Compose has `.transforms`
                transforms = getattr(pipeline, "transforms", None)
                if transforms is None:
                    augmentation_offline = None
                else:
                    for t in transforms:
                        try:
                            t_name = t.__class__.__name__
                        except Exception:
                            t_name = str(type(t))
                        t_p = getattr(t, "p", None)
                        # capture parameters conservatively
                        try:
                            params = {k: v for k, v in vars(t).items() if k != "p"}
                        except Exception:
                            params = {}
                        item = {"name": t_name, "p": t_p, "params": params}
                        exact_offline.append(item)
                    augmentation_offline = exact_offline
            except Exception:
                augmentation_offline = None
        except Exception:
            augmentation_offline = None

        # exact_transforms contains both offline and online concrete descriptions
        exact_transforms = {"offline": augmentation_offline, "online": augmentation_online}

        # merge into cfg
        cfg.update(
            {
                "augmentation_online": augmentation_online,
                "augmentation_offline": augmentation_offline,
                "exact_transforms": exact_transforms,
            }
        )

        # --- Capture n_aug from dataset (count augmented files) when deterministically possible ---
        n_aug_val = None
        n_aug_source = "unavailable"
        try:
            from pathlib import Path
            import re

            train_dir = Path(self.output_dir) / "images" / "train"
            if train_dir.exists() and train_dir.is_dir():
                all_files = [p.stem for p in train_dir.glob("*.*")]
                aug_pattern = re.compile(r"_aug(\d+)$", re.IGNORECASE)
                original_stems = [s for s in all_files if not aug_pattern.search(s)]
                aug_stems = [s for s in all_files if aug_pattern.search(s)]
                num_original = len(set(original_stems))
                num_aug_files = len(aug_stems)
                if num_original > 0:
                    # if augmentation count divides evenly, determine n_aug per original
                    if num_aug_files % num_original == 0:
                        n_aug_val = num_aug_files // num_original
                        n_aug_source = f"filesystem:{train_dir}"
                    else:
                        # cannot determine exact per-image augmentation
                        n_aug_val = None
                        n_aug_source = f"indeterminate_filesystem:{train_dir}"
                else:
                    n_aug_val = None
                    n_aug_source = f"no_originals_in_filesystem:{train_dir}"
        except Exception:
            n_aug_val = None
            n_aug_source = "unavailable"

        # If still unavailable, fall back to project config N_AUG_PER_IMAGE (explicit source)
        if n_aug_val is None:
            try:
                from src.pill_safety.cv.segmentation.utils.config import N_AUG_PER_IMAGE

                # Only use config value if it is explicitly set (non-None)
                if N_AUG_PER_IMAGE is not None:
                    n_aug_val = N_AUG_PER_IMAGE
                    n_aug_source = "project_config:N_AUG_PER_IMAGE"
            except Exception:
                pass

        # --- Explicit copy_paste field (prefer trainer args, else project config if defined) ---
        copy_paste_val = _get_arg(trainer_args, "copy_paste")
        copy_paste_source = "trainer.args" if copy_paste_val is not None else "unavailable"
        if copy_paste_val is None:
            try:
                from src.pill_safety.cv.segmentation.utils.config import COPY_PASTE

                copy_paste_val = COPY_PASTE
                copy_paste_source = "project_config:COPY_PASTE"
            except Exception:
                copy_paste_val = None
                copy_paste_source = copy_paste_source if copy_paste_source != "unavailable" else "unavailable"

        # Attach n_aug and copy_paste metadata
        cfg.update(
            {
                "n_aug": {"value": n_aug_val, "source": n_aug_source},
                "copy_paste": {"value": copy_paste_val, "source": copy_paste_source},
            }
        )

        out_path = logs_dir / f"{run_id}_training_config.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _generate_training_plots(
        self,
        run_id: str,
        logs_dir: Path,
        plots_dir: Path,
        train_log_path: Path,
        metrics_dir: Path,
    ) -> None:
        """Create run-scoped plot artifacts from recorded training/evaluation data.

        Ultralytics records per-epoch values with names such as ``train/box_loss``
        and ``metrics/mAP50-95(M)``.  The mappings below are deliberately
        explicit so plots never manufacture values when a source column is absent.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir.mkdir(parents=True, exist_ok=True)

        def _save_unavailable(filename: str, reason: str) -> None:
            figure, axis = plt.subplots(figsize=(8, 4.5))
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                f"Unavailable\n{reason}",
                ha="center",
                va="center",
                wrap=True,
                fontsize=12,
            )
            figure.tight_layout()
            figure.savefig(plots_dir / filename, dpi=150)
            plt.close(figure)

        def _to_float(value: str | None) -> float | None:
            try:
                number = float(value) if value not in (None, "") else None
            except (TypeError, ValueError):
                return None
            return number if number is not None and math.isfinite(number) else None

        def _value(row: dict[str, str], columns: tuple[str, ...]) -> float | None:
            for column in columns:
                value = _to_float(row.get(column))
                if value is not None:
                    return value
            return None

        rows: list[dict[str, str]] = []
        if train_log_path.exists():
            with open(train_log_path, "r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows.append({(key or "").strip(): (value or "").strip() for key, value in row.items()})

        # Direct contract columns are preferred.  For unnormalised Ultralytics logs,
        # losses are the recorded sum of their available loss components.
        train_loss_columns = ("train_loss",)
        train_loss_components = ("train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss")
        val_loss_columns = ("val_loss",)
        val_loss_components = ("val/box_loss", "val/seg_loss", "val/cls_loss", "val/dfl_loss")
        metric_columns = (
            "best_metric",
            "metrics/mAP50-95(M)",
            "metrics/mAP50-95(B)",
            "metrics/mAP50(M)",
            "metrics/mAP50(B)",
        )
        learning_rate_columns = ("learning_rate", "lr/pg0", "lr")
        precision_columns = ("precision", "metrics/precision(M)", "metrics/precision(B)")
        recall_columns = ("recall", "metrics/recall(M)", "metrics/recall(B)")

        def _loss(row: dict[str, str], direct: tuple[str, ...], components: tuple[str, ...]) -> float | None:
            direct_value = _value(row, direct)
            if direct_value is not None:
                return direct_value
            values = [_value(row, (column,)) for column in components]
            available_values = [value for value in values if value is not None]
            return sum(available_values) if available_values else None

        all_epochs: list[float] = []
        epochs: list[float] = []
        train_losses: list[float] = []
        val_epochs: list[float] = []
        val_losses: list[float] = []
        metric_epochs: list[float] = []
        metrics: list[float] = []
        learning_rates: list[float] = []
        pr_pairs: list[tuple[float, float]] = []
        for row in rows:
            epoch = _value(row, ("epoch",))
            if epoch is None:
                continue
            all_epochs.append(epoch)
            train_loss = _loss(row, train_loss_columns, train_loss_components)
            if train_loss is not None:
                epochs.append(epoch)
                train_losses.append(train_loss)
            val_loss = _loss(row, val_loss_columns, val_loss_components)
            if val_loss is not None:
                val_epochs.append(epoch)
                val_losses.append(val_loss)
            metric = _value(row, metric_columns)
            if metric is not None:
                metric_epochs.append(epoch)
                metrics.append(metric)
            learning_rate = _value(row, learning_rate_columns)
            if learning_rate is not None:
                learning_rates.append(learning_rate)
            precision = _value(row, precision_columns)
            recall = _value(row, recall_columns)
            if precision is not None and recall is not None:
                pr_pairs.append((precision, recall))

        loss_filename = f"{run_id}_loss_curve.png"
        if train_losses:
            figure, axis = plt.subplots(figsize=(8, 4.5))
            axis.plot(epochs, train_losses, label="train_loss", color="tab:blue")
            if val_losses:
                axis.plot(val_epochs, val_losses, label="val_loss", color="tab:orange")
            axis.set(xlabel="epoch", ylabel="loss", title="Training and validation loss")
            axis.legend()
            axis.grid(alpha=0.3)
            figure.tight_layout()
            figure.savefig(plots_dir / loss_filename, dpi=150)
            plt.close(figure)
        else:
            _save_unavailable(loss_filename, "No epoch and train_loss data in train_log.csv.")

        metric_filename = f"{run_id}_metric_curve.png"
        if metrics:
            figure, axis = plt.subplots(figsize=(8, 4.5))
            axis.plot(metric_epochs, metrics, label="validation metric", color="tab:green")
            axis.set(xlabel="epoch", ylabel="metric", title="Validation metric by epoch")
            axis.legend()
            axis.grid(alpha=0.3)
            figure.tight_layout()
            figure.savefig(plots_dir / metric_filename, dpi=150)
            plt.close(figure)
        else:
            _save_unavailable(metric_filename, "No best_metric or mapped validation metric in train_log.csv.")

        summary_filename = f"{run_id}_summary.png"
        if rows:
            best_metric_index = max(range(len(metrics)), key=metrics.__getitem__) if metrics else None
            summary_lines = [f"Recorded epochs: {len(set(all_epochs))}"]
            if best_metric_index is None:
                summary_lines.extend(("Best epoch: unavailable", "Best metric: unavailable"))
            else:
                summary_lines.extend(
                    (
                        f"Best epoch: {metric_epochs[best_metric_index]:g}",
                        f"Best metric: {metrics[best_metric_index]:.6g}",
                    )
                )
            if learning_rates:
                summary_lines.append(f"Final learning rate: {learning_rates[-1]:.6g}")
            else:
                summary_lines.append("Final learning rate: unavailable")
            figure, axis = plt.subplots(figsize=(8, 4.5))
            axis.axis("off")
            axis.text(0.05, 0.9, "Training run summary", fontsize=16, weight="bold", va="top")
            axis.text(0.05, 0.72, "\n".join(summary_lines), fontsize=12, va="top")
            figure.tight_layout()
            figure.savefig(plots_dir / summary_filename, dpi=150)
            plt.close(figure)
        else:
            _save_unavailable(summary_filename, "train_log.csv is missing or contains no rows.")

        pr_filename = f"{run_id}_precision_recall_curve.png"
        if pr_pairs:
            precision, recall = zip(*pr_pairs)
            figure, axis = plt.subplots(figsize=(6, 6))
            axis.plot(recall, precision, marker="o", label="validation precision/recall")
            axis.set(xlabel="recall", ylabel="precision", title="Precision-recall observations")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.3)
            axis.legend()
            figure.tight_layout()
            figure.savefig(plots_dir / pr_filename, dpi=150)
            plt.close(figure)
        else:
            _save_unavailable(pr_filename, "No paired precision and recall values in train_log.csv.")

        threshold_filename = f"{run_id}_threshold_vs_recall.png"
        threshold_sweep_path = metrics_dir / f"{run_id}_threshold_sweep.json"
        threshold_pairs: list[tuple[float, float]] = []
        if threshold_sweep_path.exists():
            try:
                with open(threshold_sweep_path, "r", encoding="utf-8") as handle:
                    sweep = json.load(handle).get("sweep", [])
                for entry in sweep:
                    threshold = _to_float(str(entry.get("confidence", entry.get("threshold", ""))))
                    metric_values = entry.get("metrics", entry)
                    recall = _to_float(str(metric_values.get("mask_recall", metric_values.get("recall", ""))))
                    if threshold is not None and recall is not None:
                        threshold_pairs.append((threshold, recall))
            except (OSError, ValueError, AttributeError, TypeError):
                threshold_pairs = []
        if threshold_pairs:
            thresholds, recalls = zip(*sorted(threshold_pairs))
            figure, axis = plt.subplots(figsize=(8, 4.5))
            axis.plot(thresholds, recalls, marker="o", label="validation recall")
            axis.set(xlabel="threshold", ylabel="recall", title="Threshold versus recall")
            axis.grid(alpha=0.3)
            axis.legend()
            figure.tight_layout()
            figure.savefig(plots_dir / threshold_filename, dpi=150)
            plt.close(figure)
        elif threshold_sweep_path.exists():
            _save_unavailable(threshold_filename, "Run-scoped threshold sweep has no threshold/recall pairs.")
        else:
            _save_unavailable(threshold_filename, "No run-scoped threshold sweep artifact is available.")

    def _collect_training_artifacts(self, run_id: str, save_dir: Path, output_dirs: dict[str, Path]) -> None:
        checkpoint_dir = output_dirs["checkpoint_dir"]
        logs_dir = output_dirs["logs_dir"]
        metrics_dir = output_dirs["metrics_dir"]
        plots_dir = output_dirs["plots_dir"]

        weights_dir = save_dir / "weights"
        if weights_dir.exists():
            for checkpoint in weights_dir.glob("*.pt"):
                dest = checkpoint_dir / f"{run_id}_{checkpoint.name}"
                shutil.copy2(checkpoint, dest)

        results_csv = save_dir / "results.csv"
        train_log_path = logs_dir / f"{run_id}_train_log.csv"
        if results_csv.exists():
            shutil.copy2(results_csv, train_log_path)

        self._generate_training_plots(
            run_id,
            logs_dir,
            plots_dir,
            train_log_path,
            metrics_dir,
        )

        for image_path in save_dir.glob("*.png"):
            dest = plots_dir / f"{run_id}_{image_path.name}"
            shutil.move(str(image_path), dest)
        for image_path in save_dir.glob("*.jpg"):
            dest = plots_dir / f"{run_id}_{image_path.name}"
            shutil.move(str(image_path), dest)

    def _cleanup_temp(self, save_dir: Path) -> None:
        if save_dir.exists() and save_dir.is_dir():
            shutil.rmtree(save_dir)

    def train(self) -> None:
        if not self.data_yaml.exists():
            raise FileNotFoundError(
                f"Không thấy {self.data_yaml}. Chạy data_preparation/prepare_data.py trước."
            )

        train_images_dir = self.output_dir / "images" / "train"
        n_train = len(list(train_images_dir.glob("*.*")))
        print(f"[train.py] {n_train} ảnh train (gốc + augmented) tại {train_images_dir}")

        run_id = self._generate_run_id()
        output_dirs = self._prepare_experiment_dirs(run_id)
        save_dir = self.module_dir / ".tmp" / run_id
        save_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[train.py] Bắt đầu FULL fine-tune {self.model.base_weights} "
            f"({self.epochs} epochs, imgsz={self.imgsz}, freeze={self.freeze})"
        )

        model = self.model.build()
        epoch_timing = self._register_epoch_timing_callbacks(model)
        started_at = datetime.now()
        model.train(
            data=str(self.data_yaml),
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            patience=self.patience,
            device=self.device,
            freeze=self.freeze,
            seed=self.seed,
            save_dir=str(save_dir),
            exist_ok=True,
        )
        finished_at = datetime.now()
        epoch_timing = self._finalize_epoch_timing(epoch_timing)

        # Capture resolved training configuration (learning_rate, optimizer, scheduler, class_agnostic)
        try:
            self._capture_training_config(run_id, output_dirs["logs_dir"], model)
        except Exception:
            pass

        self._collect_training_artifacts(run_id, save_dir, output_dirs)
        self._write_config(run_id, output_dirs["logs_dir"])
        self._write_dataset_manifest(run_id, output_dirs["logs_dir"])
        self._write_runtime(run_id, output_dirs["logs_dir"], started_at, finished_at, epoch_timing)
        self._cleanup_temp(save_dir)

        best_checkpoint = output_dirs["checkpoint_dir"] / f"{run_id}_best.pt"
        print(f"\n[train.py] Xong. Checkpoint tốt nhất: {best_checkpoint}")
        print("Tiếp theo: chạy evaluation/evaluate.py để tính mask mAP trên tập test.")
