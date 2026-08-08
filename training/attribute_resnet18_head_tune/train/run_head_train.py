"""
Head-Tune Entrypoint (Stage 1) — Thin Wrapper.

Freezes the entire ResNet18 backbone and trains only the fc_shape
and fc_color classification heads.

Usage:
    python run_head_train.py --run_id attr_head_v2 --epochs 30 --batch_size 32
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --- Project root setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pill_safety.cv.attribute.models import MultiTaskResNet18
from pill_safety.cv.attribute.datasets.rximage import RxImageDataset
from pill_safety.cv.attribute.transforms.augmentations import get_attribute_transforms
from pill_safety.cv.attribute.labels.label_mapping import (
    build_label_mapping, save_label_mapping,
    get_shape_distribution, get_color_distribution,
)
from pill_safety.cv.attribute.trainers import BaseTrainer, compute_shape_class_weights
from pill_safety.cv.attribute.utils.leakage import check_split_leakage
from pill_safety.cv.attribute.utils.artifacts import (
    save_config_yaml, save_dataset_manifest, save_runtime_info,
)
from pill_safety.cv.attribute.utils.config import AttributeConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Head-Tune Training (Stage 1)")
    parser.add_argument("--run_id", type=str, required=True, help="Unique run identifier")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lambda_color", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    started_at = datetime.now()

    # --- Seed ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    DEVICE = AttributeConfig.DEVICE
    MODULE_NAME = "attribute_resnet18_head_tune"

    # --- Paths ---
    paths = AttributeConfig.get_experiment_paths(MODULE_NAME, args.run_id)
    AttributeConfig.setup_directories(paths)

    # --- Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(paths["logs"] / f"{args.run_id}_training.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"=== Head-Tune Training | Run ID: {args.run_id} ===")
    logger.info(f"Device: {DEVICE}")

    # --- Data ---
    train_csv = AttributeConfig.COMBINED_DIR / "train_clean.csv"
    val_csv = AttributeConfig.COMBINED_DIR / "val_clean.csv"
    test_csv = AttributeConfig.COMBINED_DIR / "test_clean.csv"

    # Fallback to old CSV names if clean ones don't exist yet
    if not train_csv.exists():
        logger.warning("train_clean.csv not found, falling back to train_combined_crop.csv")
        train_csv = AttributeConfig.COMBINED_DIR / "train_combined_crop.csv"
        val_csv = AttributeConfig.COMBINED_DIR / "val_combined_crop.csv"
        test_csv = AttributeConfig.COMBINED_DIR / "test_combined_crop.csv"

    transforms_dict = get_attribute_transforms()
    train_dataset = RxImageDataset(train_csv, AttributeConfig.IMG_DIR, transform=transforms_dict["train"])
    val_dataset = RxImageDataset(val_csv, AttributeConfig.IMG_DIR, transform=transforms_dict["val"])
    test_dataset = RxImageDataset(test_csv, AttributeConfig.IMG_DIR, transform=transforms_dict["val"])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- Label Mapping (Source of Truth for all stages) ---
    num_shape_classes = len(np.unique(train_dataset.shape_labels))
    num_color_classes = len(train_dataset.color_cols)

    label_mapping, shape_names, color_names = build_label_mapping(
        train_dataset, num_shape_classes, num_color_classes
    )
    mapping_path = paths["logs"] / f"{args.run_id}_label_mapping.json"
    mapping_hash = save_label_mapping(label_mapping, mapping_path)
    logger.info(f"Label mapping saved: {num_shape_classes} shapes, {num_color_classes} colors")

    # --- Leakage Check (ACTUAL, not hard-coded) ---
    try:
        leakage_passed, leakage_details = check_split_leakage(
            str(train_csv), str(val_csv), str(test_csv)
        )
    except KeyError:
        logger.warning("NDC11 column not found — skipping leakage check")
        leakage_passed, leakage_details = False, {"error": "NDC11 column not found"}

    logger.info(f"Leakage check: passed={leakage_passed}, details={leakage_details}")

    # --- Model ---
    model = MultiTaskResNet18(
        num_shape_classes=num_shape_classes,
        num_color_classes=num_color_classes,
        pretrained=True,
    ).to(DEVICE)
    model.freeze_backbone()
    logger.info("Model created. Backbone FROZEN (head-tune mode).")

    # --- Loss ---
    shape_weights = compute_shape_class_weights(train_dataset.shape_labels, num_shape_classes, DEVICE)
    criterion_shape = nn.CrossEntropyLoss(weight=shape_weights)

    color_targets = train_dataset.color_labels
    pos_counts = color_targets.sum(axis=0)
    neg_counts = len(color_targets) - pos_counts
    pos_weights = np.clip(neg_counts / (pos_counts + 1e-5), a_min=1.0, a_max=10.0)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)
    criterion_color = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(model.get_trainable_params(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # --- Save Config & Manifest ---
    save_config_yaml(
        path=paths["logs"] / f"{args.run_id}_config.yaml",
        run_id=args.run_id, module=MODULE_NAME, train_strategy="head_tune",
        seed=args.seed,
        model_config={"architecture": "ResNet18", "pretrained": True},
        training_config={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                         "optimizer": "AdamW", "weight_decay": 1e-4, "patience": args.patience},
        frozen_layers=["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
        trainable_layers=["fc_shape", "fc_color"],
        label_mapping_file=str(mapping_path.relative_to(PROJECT_ROOT)),
        augmentation={"mode": "online", "train_only": True,
                      "transforms": ["RandomHorizontalFlip", "RandomRotation(15)"]},
        scheduler_config={"name": "ReduceLROnPlateau", "mode": "min", "patience": 3, "factor": 0.5},
    )

    shape_dist = get_shape_distribution(train_dataset, shape_names)
    color_dist = get_color_distribution(train_dataset, color_names)

    save_dataset_manifest(
        path=paths["logs"] / f"{args.run_id}_dataset_manifest.json",
        run_id=args.run_id, dataset_name="rximage_nih",
        train_csv=str(train_csv), val_csv=str(val_csv), test_csv=str(test_csv),
        train_count=len(train_dataset), val_count=len(val_dataset), test_count=len(test_dataset),
        num_shape_classes=num_shape_classes, num_color_classes=num_color_classes,
        label_mapping_file=str(mapping_path.relative_to(PROJECT_ROOT)),
        class_distribution={"shape": shape_dist, "color": color_dist},
        leakage_check_passed=leakage_passed,
        leakage_check_details=leakage_details,
    )

    # --- Train ---
    trainer = BaseTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion_shape=criterion_shape, criterion_color=criterion_color,
        optimizer=optimizer, scheduler=scheduler, device=DEVICE,
        run_id=args.run_id, paths=paths,
        label_mapping=label_mapping, mapping_hash=mapping_hash,
        lambda_color=args.lambda_color, patience=args.patience,
    )
    trainer.fit(num_epochs=args.epochs)

    # --- Save Runtime ---
    finished_at = datetime.now()
    save_runtime_info(
        path=paths["logs"] / f"{args.run_id}_runtime.txt",
        run_id=args.run_id, module=MODULE_NAME, device=DEVICE,
        started_at=started_at, finished_at=finished_at,
        total_train_time_minutes=trainer.total_train_time_minutes,
        avg_epoch_time_seconds=trainer.avg_epoch_time_seconds,
        best_epoch=trainer.best_epoch, best_metric=trainer.best_metric,
        num_epochs_run=trainer.num_epochs_run,
    )

    logger.info(f"Training complete! Best epoch: {trainer.best_epoch}, Best F1: {trainer.best_metric:.4f}")
    # NOTE: Test evaluation is NOT run here. Use eval_head.py separately.


if __name__ == "__main__":
    main()
