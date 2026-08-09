from __future__ import annotations

import json
import platform
import shutil
import socket
from datetime import datetime
from pathlib import Path
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
            "split_before_augmentation": True,
            "augmentation_train_only": True,
            "label_mapping_file": None,
        }

        out_path = logs_dir / f"{run_id}_dataset_manifest.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def _write_runtime(self, run_id: str, logs_dir: Path, started_at: datetime, finished_at: datetime) -> None:
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
        }

        if torch is not None and torch.cuda.is_available():
            runtime["gpu_name"] = torch.cuda.get_device_name(0)

        out_path = logs_dir / f"{run_id}_runtime.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for key, value in runtime.items():
                f.write(f"{key}: {value}\n")

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

    def _collect_training_artifacts(self, run_id: str, save_dir: Path, output_dirs: dict[str, Path]) -> None:
        checkpoint_dir = output_dirs["checkpoint_dir"]
        logs_dir = output_dirs["logs_dir"]
        plots_dir = output_dirs["plots_dir"]

        weights_dir = save_dir / "weights"
        if weights_dir.exists():
            for checkpoint in weights_dir.glob("*.pt"):
                dest = checkpoint_dir / f"{run_id}_{checkpoint.name}"
                shutil.copy2(checkpoint, dest)

        results_csv = save_dir / "results.csv"
        if results_csv.exists():
            dest = logs_dir / f"{run_id}_train_log.csv"
            shutil.copy2(results_csv, dest)

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

        # Capture resolved training configuration (learning_rate, optimizer, scheduler, class_agnostic)
        try:
            self._capture_training_config(run_id, output_dirs["logs_dir"], model)
        except Exception:
            pass

        self._collect_training_artifacts(run_id, save_dir, output_dirs)
        self._write_config(run_id, output_dirs["logs_dir"])
        self._write_dataset_manifest(run_id, output_dirs["logs_dir"])
        self._write_runtime(run_id, output_dirs["logs_dir"], started_at, finished_at)
        self._cleanup_temp(save_dir)

        best_checkpoint = output_dirs["checkpoint_dir"] / f"{run_id}_best.pt"
        print(f"\n[train.py] Xong. Checkpoint tốt nhất: {best_checkpoint}")
        print("Tiếp theo: chạy evaluation/evaluate.py để tính mask mAP trên tập test.")
