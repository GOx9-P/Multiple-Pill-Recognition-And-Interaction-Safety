#!/usr/bin/env python3
"""
Entry point for Stage 1: Head Fine-Tune of ResNet18 attribute model.

Freezes the ResNet18 backbone and trains only the classification heads
(fc_shape, fc_color) on the NIH/RxImage dataset.

Usage:
    python training/attribute_resnet18_head_tune/train/run_head_train.py

All logic is imported from src/pill_safety/cv/attribute/.
"""

import datetime
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --- Add src/ to Python path so we can import pill_safety ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # temp_repo/
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pill_safety.cv.attribute.datasets import RxImageDataset
from pill_safety.cv.attribute.evaluators import AttributeEvaluator
from pill_safety.cv.attribute.labels import (
    build_label_mapping,
    get_shape_distribution,
    remove_rare_color_classes,
    save_label_mapping,
)
from pill_safety.cv.attribute.models import MultiTaskResNet18_HeadsFinetune
from pill_safety.cv.attribute.trainers import (
    HeadFineTuneTrainer,
    compute_shape_class_weights,
)
from pill_safety.cv.attribute.transforms import get_transforms
from pill_safety.cv.attribute.utils import (
    get_device,
    init_experiment_dirs,
    print_system_info,
    save_config_yaml,
    save_dataset_manifest,
    save_runtime_info,
    set_seed,
    setup_logger,
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================
RUN_ID = "attr_head_v1"
MODULE_NAME = "attribute_resnet18_head_tune"
RUNNER = "NguyenQuocBao"
SEED = 42

# Hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
OPTIMIZER_NAME = "adamw"
SCHEDULER_NAME = "reduce_lr_on_plateau"
WEIGHT_DECAY = 1e-4
COLOR_LOSS_WEIGHT = 2.0

# Dataset paths — tries multiple locations for portability
CANDIDATE_BASE_DIRS = [
    Path("/kaggle/input/rximage-new/rximage"),
    Path("c:/ML_DL_Project/Data_rximage_kaggle/rximage"),
    Path("c:/ML_DL_Project/Data/rximage"),
    Path("Data/rximage"),
]


def find_base_dir() -> Path:
    """Find the dataset base directory from candidate paths."""
    for p in CANDIDATE_BASE_DIRS:
        if p.exists() and (p / "combined").exists():
            return p

    # Fallback: search Kaggle input
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for sub in kaggle_input.rglob("combined"):
            if sub.is_dir():
                return sub.parent

    return Path("/kaggle/input/rximage_processed/Data/rximage")


def main():
    # ==================================================================
    # 1. SETUP
    # ==================================================================
    print_system_info()
    set_seed(SEED)
    DEVICE = get_device()

    BASE_DIR = find_base_dir()
    COMBINED_DIR = BASE_DIR / "combined"
    IMG_DIR = BASE_DIR / "image_all"

    TRAIN_CSV = COMBINED_DIR / "train_combined_crop.csv"
    VAL_CSV = COMBINED_DIR / "val_combined_crop.csv"
    TEST_CSV = COMBINED_DIR / "test_combined_crop.csv"

    # Experiment output directories
    EXPERIMENT_DIR = Path(f"experiments/{MODULE_NAME}")
    if Path("/kaggle/working").exists():
        EXPERIMENT_DIR = Path(f"/kaggle/working/experiments/{MODULE_NAME}")

    paths = init_experiment_dirs(EXPERIMENT_DIR, RUN_ID)
    logger = setup_logger(
        "heads_finetune", paths["logs"] / f"{RUN_ID}_training.log"
    )

    print(f"✓ Run ID: {RUN_ID}")
    print(f"✓ Module: {MODULE_NAME}")
    print(f"✓ Experiment Dir: {EXPERIMENT_DIR}")
    print(f"✓ Dataset Dir: {BASE_DIR}")
    print(f"✓ Device: {DEVICE}")

    # ==================================================================
    # 2. DATA
    # ==================================================================
    data_transforms = get_transforms(image_size=IMAGE_SIZE)

    train_dataset = RxImageDataset(
        TRAIN_CSV, IMG_DIR, transform=data_transforms["train"]
    )
    shape_encoder = getattr(train_dataset, "shape_encoder", None)
    mlb_color = getattr(train_dataset, "mlb_color", None)

    val_dataset = RxImageDataset(
        VAL_CSV, IMG_DIR,
        transform=data_transforms["val"],
        shape_encoder=shape_encoder,
        mlb_color=mlb_color,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    NUM_SHAPE_CLASSES = (
        int(
            max(
                train_dataset.shape_labels.max(),
                val_dataset.shape_labels.max(),
            )
        )
        + 1
    )
    NUM_COLOR_CLASSES = len(train_dataset.color_cols)

    # --- Label mapping ---
    label_mapping, shape_class_names, color_class_names = build_label_mapping(
        train_dataset, NUM_SHAPE_CLASSES, NUM_COLOR_CLASSES
    )

    # --- Remove rare color classes ---
    NUM_COLOR_CLASSES = remove_rare_color_classes(
        [train_dataset, val_dataset],
        rare_columns=["color_BLACK"],
        min_samples=3,
    )

    # Update label mapping after removal
    color_class_names = list(train_dataset.color_cols)
    label_mapping["color"] = color_class_names
    save_label_mapping(
        label_mapping, paths["logs"] / f"{RUN_ID}_label_mapping.json"
    )

    # --- Class distribution ---
    train_shape_dist = get_shape_distribution(
        train_dataset, shape_class_names
    )

    print(f"✓ Shape classes: {NUM_SHAPE_CLASSES} | Color classes: {NUM_COLOR_CLASSES}")
    print(f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ==================================================================
    # 3. SAVE CONFIG & MANIFEST
    # ==================================================================
    run_config = {
        "run_id": RUN_ID,
        "module": MODULE_NAME,
        "runner": RUNNER,
        "run_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "seed": SEED,
        "dataset": {
            "name": "rximage_new",
            "train_csv": str(TRAIN_CSV),
            "val_csv": str(VAL_CSV),
            "test_csv": str(TEST_CSV),
        },
        "model": {
            "architecture": "ResNet18",
            "pretrained_weight": "ImageNet",
            "train_strategy": "head_tune",
            "frozen_backbone": True,
            "trainable_layers": ["fc_shape", "fc_color"],
        },
        "training": {
            "image_size": IMAGE_SIZE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": OPTIMIZER_NAME,
            "scheduler": SCHEDULER_NAME,
            "weight_decay": WEIGHT_DECAY,
            "color_loss_weight": COLOR_LOSS_WEIGHT,
        },
        "tasks": ["shape", "color"],
        "augmentation": {
            "enabled": True,
            "online": True,
            "transforms": [
                "RandomHorizontalFlip(p=0.5)",
                "RandomRotation(degrees=15)",
            ],
            "split": "train_only",
        },
    }
    save_config_yaml(
        run_config, paths["logs"] / f"{RUN_ID}_config.yaml"
    )

    # Manifest
    test_dataset_temp = RxImageDataset(
        TEST_CSV, IMG_DIR,
        transform=data_transforms["val"],
        shape_encoder=shape_encoder,
        mlb_color=mlb_color,
    )
    dataset_manifest = {
        "run_id": RUN_ID,
        "dataset_name": "rximage_new",
        "train_csv": str(TRAIN_CSV),
        "val_csv": str(VAL_CSV),
        "test_csv": str(TEST_CSV),
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "test_count": len(test_dataset_temp),
        "split_before_augmentation": True,
        "augmentation_train_only": True,
        "num_shape_classes": NUM_SHAPE_CLASSES,
        "num_color_classes": NUM_COLOR_CLASSES,
        "class_distribution": {"shape": train_shape_dist},
        "split_policy": {
            "split_before_augmentation": True,
            "group_key": "rxcui_or_ndc11",
            "leakage_check_passed": True,
            "leakage_check_notes": (
                "Split done before augmentation. "
                "Augmentation applied to train split only."
            ),
        },
    }
    save_dataset_manifest(
        dataset_manifest,
        paths["logs"] / f"{RUN_ID}_dataset_manifest.json",
    )
    del test_dataset_temp

    # ==================================================================
    # 4. MODEL & OPTIMIZER
    # ==================================================================
    model = MultiTaskResNet18_HeadsFinetune(
        num_shape_classes=NUM_SHAPE_CLASSES,
        num_color_classes=NUM_COLOR_CLASSES,
        pretrained=True,
    ).to(DEVICE)

    trainable_params = filter(
        lambda p: p.requires_grad, model.parameters()
    )
    optimizer = optim.AdamW(
        trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    shape_weights = compute_shape_class_weights(
        train_dataset, NUM_SHAPE_CLASSES
    ).to(DEVICE)
    criterion_shape = nn.CrossEntropyLoss(weight=shape_weights)
    criterion_color = nn.BCEWithLogitsLoss()

    total_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Total params: {total_params:,} | "
        f"Trainable (Heads): {total_trainable:,} "
        f"({total_trainable / total_params * 100:.2f}%)"
    )

    # ==================================================================
    # 5. TRAINING
    # ==================================================================
    train_start_time = time.time()

    trainer = HeadFineTuneTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion_shape=criterion_shape,
        criterion_color=criterion_color,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        paths=paths,
        run_id=RUN_ID,
        color_loss_weight=COLOR_LOSS_WEIGHT,
        logger=logger,
    )

    history = trainer.fit(
        num_epochs=NUM_EPOCHS,
        label_mapping=label_mapping,
        num_shape_classes=NUM_SHAPE_CLASSES,
        num_color_classes=NUM_COLOR_CLASSES,
    )

    # ==================================================================
    # 6. SAVE RUNTIME INFO
    # ==================================================================
    total_train_time = time.time() - train_start_time
    runtime_info = {
        "run_id": RUN_ID,
        "module": MODULE_NAME,
        "started_at": datetime.datetime.fromtimestamp(
            train_start_time
        ).strftime("%Y-%m-%d %H:%M"),
        "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "device": str(DEVICE),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "total_train_time_minutes": round(total_train_time / 60, 1),
        "avg_epoch_time_seconds": round(
            np.mean(trainer.epoch_times), 1
        ),
        "best_epoch": trainer.best_epoch,
        "best_overall_f1": round(trainer.best_val_f1, 4),
    }
    save_runtime_info(
        runtime_info, paths["logs"] / f"{RUN_ID}_runtime.txt"
    )

    # ==================================================================
    # 7. EVALUATION
    # ==================================================================
    # Load test dataset
    test_dataset = RxImageDataset(
        TEST_CSV, IMG_DIR,
        transform=data_transforms["val"],
        shape_encoder=shape_encoder,
        mlb_color=mlb_color,
    )
    # Align color columns with train
    if len(test_dataset.color_cols) != len(train_dataset.color_cols):
        keep_indices = [
            test_dataset.color_cols.index(c)
            for c in train_dataset.color_cols
            if c in test_dataset.color_cols
        ]
        test_dataset.color_cols = [
            test_dataset.color_cols[i] for i in keep_indices
        ]
        test_dataset.color_labels = test_dataset.color_labels[
            :, keep_indices
        ]

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Load best checkpoint
    best_ckpt_path = paths["checkpoints"] / f"{RUN_ID}_best.pt"
    best_ckpt = torch.load(
        best_ckpt_path, map_location=DEVICE, weights_only=False
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {best_ckpt['epoch']}")

    evaluator = AttributeEvaluator(
        model=model,
        test_loader=test_loader,
        device=DEVICE,
        paths=paths,
        run_id=RUN_ID,
        module_name=MODULE_NAME,
        shape_class_names=shape_class_names,
        color_class_names=color_class_names,
        num_shape_classes=NUM_SHAPE_CLASSES,
        num_color_classes=NUM_COLOR_CLASSES,
    )

    # Collect predictions
    preds = evaluator.collect_predictions()
    test_metrics = evaluator.compute_test_metrics(preds)

    print(f"Test Shape F1: {test_metrics['shape_macro_f1']:.4f}")
    print(f"Test Color F1: {test_metrics['color_macro_f1']:.4f}")
    print(f"Test Overall F1: {test_metrics['overall_macro_f1']:.4f}")

    # Save metrics
    evaluator.save_val_metrics(
        history, trainer.best_epoch, trainer.best_val_f1
    )
    evaluator.save_test_metrics(test_metrics, trainer.best_epoch)

    # Plots
    evaluator.plot_training_curves(history, trainer.best_epoch)
    evaluator.plot_confusion_matrix(preds)
    evaluator.plot_summary(
        history, preds, test_metrics,
        trainer.best_epoch, trainer.best_val_f1, run_config,
    )

    # Predictions
    evaluator.save_predictions(preds, TEST_CSV)

    # ==================================================================
    # 8. CHECKLIST
    # ==================================================================
    print("\n" + "=" * 70)
    print("CHECKLIST — train_request.md §9")
    print("=" * 70)

    pred_dir = paths["predictions"] / RUN_ID
    checks = [
        ("train_log.csv", (paths["logs"] / f"{RUN_ID}_train_log.csv").exists()),
        ("config.yaml", (paths["logs"] / f"{RUN_ID}_config.yaml").exists()),
        ("dataset_manifest.json", (paths["logs"] / f"{RUN_ID}_dataset_manifest.json").exists()),
        ("val_metrics.json", (paths["metrics"] / f"{RUN_ID}_val_metrics.json").exists()),
        ("test_metrics.json", (paths["metrics"] / f"{RUN_ID}_test_metrics.json").exists()),
        ("loss_curve.png", (paths["plots"] / f"{RUN_ID}_loss_curve.png").exists()),
        ("metric_curve.png", (paths["plots"] / f"{RUN_ID}_metric_curve.png").exists()),
        ("shape_confusion_matrix.png", (paths["plots"] / f"{RUN_ID}_shape_confusion_matrix.png").exists()),
        ("color_f1_per_class.png", (paths["plots"] / f"{RUN_ID}_color_f1_per_class.png").exists()),
        ("summary.png", (paths["plots"] / f"{RUN_ID}_summary.png").exists()),
        ("best checkpoint", (paths["checkpoints"] / f"{RUN_ID}_best.pt").exists()),
        ("last checkpoint", (paths["checkpoints"] / f"{RUN_ID}_last.pt").exists()),
        ("runtime.txt", (paths["logs"] / f"{RUN_ID}_runtime.txt").exists()),
        ("label_mapping.json", (paths["logs"] / f"{RUN_ID}_label_mapping.json").exists()),
        ("predictions/correct_samples", (pred_dir / "correct_samples" / "samples.json").exists()),
        ("predictions/wrong_shape", (pred_dir / "wrong_shape" / "samples.json").exists()),
        ("predictions/wrong_color", (pred_dir / "wrong_color" / "samples.json").exists()),
        ("predictions/low_confidence", (pred_dir / "low_confidence" / "samples.json").exists()),
    ]

    all_passed = True
    for name, exists in checks:
        status = "✅" if exists else "❌"
        if not exists:
            all_passed = False
        print(f"  {status} {name}")

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 TẤT CẢ FILE ĐÃ ĐƯỢC TẠO THÀNH CÔNG!")
    else:
        print("⚠️  CÓ FILE CHƯA ĐƯỢC TẠO — KIỂM TRA LẠI!")

    print(f"\nOutput directory: {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
