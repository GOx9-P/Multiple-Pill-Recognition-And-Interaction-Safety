"""
Last-Blocks Fine-Tune Entrypoint (Stage 2) — Thin Wrapper.

Loads checkpoint and label mapping from head-tune (Stage 1),
unfreezes the last N backbone blocks, and trains with low LR.

FAIL-FAST: Crashes immediately if head-tune artifacts are missing.

Usage:
    python train_last_blocks.py --run_id attr_last_v2 --head_run_id attr_head_v2 --epochs 15
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
from pill_safety.cv.attribute.labels.label_mapping import load_label_mapping
from pill_safety.cv.attribute.trainers import BaseTrainer, compute_shape_class_weights
from pill_safety.cv.attribute.utils.checkpoint import load_checkpoint
from pill_safety.cv.attribute.utils.leakage import check_split_leakage
from pill_safety.cv.attribute.utils.artifacts import (
    save_config_yaml, save_dataset_manifest, save_runtime_info,
)
from pill_safety.cv.attribute.utils.config import AttributeConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Last-Blocks Fine-Tune (Stage 2)")
    parser.add_argument("--run_id", type=str, required=True, help="Unique run identifier")
    parser.add_argument("--head_run_id", type=str, required=True, help="Run ID of head-tune stage")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_heads", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_blocks", type=int, default=2, help="Number of backbone blocks to unfreeze")
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
    MODULE_NAME = "attribute_resnet18_last_blocks_finetune"
    HEAD_MODULE = "attribute_resnet18_head_tune"

    # --- Paths ---
    paths = AttributeConfig.get_experiment_paths(MODULE_NAME, args.run_id)
    AttributeConfig.setup_directories(paths)
    head_paths = AttributeConfig.get_experiment_paths(HEAD_MODULE, args.head_run_id)

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
    logger.info(f"=== Last-Blocks Fine-Tune | Run ID: {args.run_id} | Head: {args.head_run_id} ===")
    logger.info(f"Device: {DEVICE}")

    # =========================================================================
    # FAIL-FAST: Load label mapping from head-tune
    # =========================================================================
    head_mapping_path = head_paths["logs"] / f"{args.head_run_id}_label_mapping.json"
    logger.info(f"Loading label mapping from: {head_mapping_path}")
    label_mapping, num_shape_classes, num_color_classes, mapping_hash = load_label_mapping(head_mapping_path)
    logger.info(f"Label mapping loaded: {num_shape_classes} shapes, {num_color_classes} colors")

    # =========================================================================
    # FAIL-FAST: Load checkpoint from head-tune
    # =========================================================================
    head_ckpt_path = head_paths["checkpoints"] / f"{args.head_run_id}_best.pt"
    logger.info(f"Loading head-tune checkpoint from: {head_ckpt_path}")
    ckpt = load_checkpoint(head_ckpt_path, DEVICE, expected_mapping_hash=mapping_hash)
    logger.info(f"Head checkpoint loaded (epoch {ckpt.get('epoch', '?')})")

    # --- Data (same CSVs as head-tune) ---
    train_csv = AttributeConfig.COMBINED_DIR / "train_clean.csv"
    val_csv = AttributeConfig.COMBINED_DIR / "val_clean.csv"
    test_csv = AttributeConfig.COMBINED_DIR / "test_clean.csv"

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

    # --- Model (load from head checkpoint, then unfreeze) ---
    model = MultiTaskResNet18(
        num_shape_classes=num_shape_classes,
        num_color_classes=num_color_classes,
        pretrained=False,  # We load weights from checkpoint
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.unfreeze_last_blocks(num_blocks=args.num_blocks)
    logger.info(f"Model loaded from head checkpoint. Last {args.num_blocks} blocks UNFROZEN.")

    # --- Loss ---
    shape_weights = compute_shape_class_weights(train_dataset.shape_labels, num_shape_classes, DEVICE)
    criterion_shape = nn.CrossEntropyLoss(weight=shape_weights)

    color_targets = train_dataset.color_labels
    pos_counts = color_targets.sum(axis=0)
    neg_counts = len(color_targets) - pos_counts
    pos_weights = np.clip(neg_counts / (pos_counts + 1e-5), a_min=1.0, a_max=10.0)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)
    criterion_color = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # --- Optimizer (Differential LR) ---
    param_groups = [
        {"params": model.fc_shape.parameters(), "lr": args.lr_heads},
        {"params": model.fc_color.parameters(), "lr": args.lr_heads},
    ]
    # Add unfrozen backbone layers
    layers = [model.backbone.layer1, model.backbone.layer2,
              model.backbone.layer3, model.backbone.layer4]
    for layer in layers[-args.num_blocks:]:
        param_groups.append({"params": layer.parameters(), "lr": args.lr_backbone})

    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    # --- Leakage Check ---
    try:
        leakage_passed, leakage_details = check_split_leakage(str(train_csv), str(val_csv), str(test_csv))
    except KeyError:
        leakage_passed, leakage_details = False, {"error": "NDC11 column not found"}

    # --- Save Config & Manifest ---
    unfrozen_names = [f"layer{5 - i}" for i in range(args.num_blocks, 0, -1)]
    frozen_names = [f"layer{i}" for i in range(1, 5 - args.num_blocks + 1)]

    save_config_yaml(
        path=paths["logs"] / f"{args.run_id}_config.yaml",
        run_id=args.run_id, module=MODULE_NAME, train_strategy="last_blocks_finetune",
        seed=args.seed,
        model_config={"architecture": "ResNet18", "pretrained_from": str(head_ckpt_path)},
        training_config={"epochs": args.epochs, "batch_size": args.batch_size,
                         "lr_backbone": args.lr_backbone, "lr_heads": args.lr_heads,
                         "optimizer": "AdamW", "weight_decay": 1e-4, "patience": args.patience},
        frozen_layers=["conv1", "bn1"] + frozen_names,
        trainable_layers=unfrozen_names + ["fc_shape", "fc_color"],
        label_mapping_file=str(head_mapping_path.relative_to(PROJECT_ROOT)),
        augmentation={"mode": "online", "train_only": True,
                      "transforms": ["RandomHorizontalFlip", "RandomRotation(15)", "ColorJitter"]},
        scheduler_config={"name": "ReduceLROnPlateau", "mode": "min", "patience": 2, "factor": 0.5},
        extra={"head_run_id": args.head_run_id},
    )

    from pill_safety.cv.attribute.labels.label_mapping import get_shape_distribution, get_color_distribution
    shape_dist = get_shape_distribution(train_dataset, label_mapping["shape"])
    color_dist = get_color_distribution(train_dataset, label_mapping["color"])

    save_dataset_manifest(
        path=paths["logs"] / f"{args.run_id}_dataset_manifest.json",
        run_id=args.run_id, dataset_name="rximage_nih",
        train_csv=str(train_csv), val_csv=str(val_csv), test_csv=str(test_csv),
        train_count=len(train_dataset), val_count=len(val_dataset), test_count=len(test_dataset),
        num_shape_classes=num_shape_classes, num_color_classes=num_color_classes,
        label_mapping_file=str(head_mapping_path.relative_to(PROJECT_ROOT)),
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
    # NOTE: Test evaluation is NOT run here. Use eval_last_blocks.py separately.


if __name__ == "__main__":
    main()