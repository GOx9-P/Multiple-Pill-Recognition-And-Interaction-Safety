import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import yaml

# Khởi tạo đường dẫn dự án
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.pill_safety.cv.attribute.datasets.rximage_dataset import RxImageDataset
from src.pill_safety.cv.attribute.models.multitask_resnet import MultiTaskResNet18

# ==============================================================================
# 1. CẤU HÌNH & KHỞI TẠO HỆ THỐNG
# ==============================================================================
RUN_ID = "attr_last_v1"
MODULE_NAME = "attribute_resnet18_last_blocks_finetune"

BASE_DIR = Path("/kaggle/input/datasets/thuongnguoiquantu/dataset/Data/rximage")
COMBINED_DIR = BASE_DIR / "combined"
IMG_DIR = BASE_DIR / "image_all"
STAGE1_CHECKPOINT = Path("/kaggle/input/datasets/thuongnguoiquantu/head-1/best_heads_finetuned (1).pth")

EXP_DIR = PROJECT_ROOT / "experiments" / MODULE_NAME
PATHS = {
    "checkpoints": EXP_DIR / "checkpoints",
    "logs": EXP_DIR / "logs",
    "metrics": EXP_DIR / "metrics",
}

for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 15
LR_BACKBONE = 1e-5
LR_HEADS = 1e-4
PATIENCE = 4
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def setup_logger(log_file):
    logger = logging.getLogger(RUN_ID)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger

logger = setup_logger(PATHS["logs"] / f"{RUN_ID}_runtime.log")
logger.info(f"Khởi chạy Training Run ID: {RUN_ID} trên module {MODULE_NAME}")

# ==============================================================================
# 2. DATASET, TRANSFORMS & MANIFEST
# ==============================================================================
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

train_csv = COMBINED_DIR / "augmented_train_combined.csv"
val_csv = COMBINED_DIR / "val_combined_crop.csv"

train_dataset = RxImageDataset(train_csv, IMG_DIR, transform=data_transforms["train"])
val_dataset = RxImageDataset(val_csv, IMG_DIR, transform=data_transforms["val"])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

NUM_SHAPE_CLASSES = len(np.unique(train_dataset.shape_labels))
NUM_COLOR_CLASSES = len(train_dataset.color_cols)

# Lưu Metadata & Manifest
label_mapping = {
    "shape": {int(i): f"shape_class_{i}" for i in range(NUM_SHAPE_CLASSES)},
    "color": {i: col.replace("color_", "") for i, col in enumerate(train_dataset.color_cols)}
}
label_mapping_path = PATHS["logs"] / f"{RUN_ID}_label_mapping.json"
with open(label_mapping_path, "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=4)

dataset_manifest = {
    "run_id": RUN_ID,
    "dataset_name": "RxImage_NIH",
    "split_file": str(COMBINED_DIR),
    "train_count": len(train_dataset),
    "val_count": len(val_dataset),
    "num_shape_classes": NUM_SHAPE_CLASSES,
    "num_color_classes": NUM_COLOR_CLASSES,
    "split_policy": {
        "split_before_augmentation": True,
        "augmentation_train_only": True,
        "group_key": "rxnavImageFileName",
        "leakage_check_passed": True,
    },
    "label_mapping_file": str(label_mapping_path),
}
with open(PATHS["logs"] / f"{RUN_ID}_dataset_manifest.json", "w", encoding="utf-8") as f:
    json.dump(dataset_manifest, f, indent=4)

config_data = {
    "run_id": RUN_ID,
    "module": MODULE_NAME,
    "seed": SEED,
    "model": {"architecture": "ResNet18", "pretrained_from": str(STAGE1_CHECKPOINT)},
    "training": {
        "image_size": 224,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr_backbone": LR_BACKBONE,
        "lr_heads": LR_HEADS,
        "optimizer": "AdamW",
        "patience": PATIENCE
    },
    "trainable_layers": ["layer4", "fc_shape", "fc_color"],
}
with open(PATHS["logs"] / f"{RUN_ID}_config.yaml", "w", encoding="utf-8") as f:
    yaml.dump(config_data, f)

# ==============================================================================
# 3. KHỞI TẠO MÔ HÌNH VÀ TỐI ƯU HÓA
# ==============================================================================
model = MultiTaskResNet18(num_shape_classes=NUM_SHAPE_CLASSES, num_color_classes=NUM_COLOR_CLASSES).to(DEVICE)

weights_base = models.ResNet18_Weights.DEFAULT.get_state_dict(progress=True)
weights_base.pop("fc.weight", None)
weights_base.pop("fc.bias", None)
model.backbone.load_state_dict(weights_base, strict=False)

if STAGE1_CHECKPOINT.exists():
    heads_weights = torch.load(STAGE1_CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.fc_shape.load_state_dict(heads_weights["fc_shape"])
    model.fc_color.load_state_dict(heads_weights["fc_color"])
    logger.info("Đã load thành công trọng số Stage 1 cho FC Heads!")

model.unfreeze_last_blocks()

color_targets_all = train_dataset.color_labels
pos_counts = color_targets_all.sum(axis=0)
neg_counts = len(color_targets_all) - pos_counts
pos_weights = np.clip(neg_counts / (pos_counts + 1e-5), a_min=1.0, a_max=10.0)
pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)

optimizer = optim.AdamW([
    {"params": model.backbone.layer4.parameters(), "lr": LR_BACKBONE},
    {"params": model.fc_shape.parameters(), "lr": LR_HEADS},
    {"params": model.fc_color.parameters(), "lr": LR_HEADS},
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
criterion_shape = nn.CrossEntropyLoss()
criterion_color = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

# ==============================================================================
# 4. VÒNG LẶP HUẤN LUYỆN VÀ EARLY STOPPING
# ==============================================================================
csv_log_path = PATHS["logs"] / f"{RUN_ID}_train_log.csv"
best_val_loss = float("inf")
best_overall_macro_f1 = 0.0
patience_counter = 0
start_time = time.time()

train_history_rows = []

for epoch in range(NUM_EPOCHS):
    model.train()
    model.set_bn_eval()

    running_loss, running_shape_loss, running_color_loss = 0.0, 0.0, 0.0
    total_samples = 0
    train_s_preds, train_s_targets = [], []
    train_c_preds, train_c_targets = [], []

    for images, s_targets, c_targets, _ in train_loader:
        images, s_targets, c_targets = images.to(DEVICE), s_targets.to(DEVICE), c_targets.to(DEVICE)
        optimizer.zero_grad()

        s_outputs, c_outputs = model(images)
        loss_s = criterion_shape(s_outputs, s_targets)
        loss_c = criterion_color(c_outputs, c_targets)
        total_loss = loss_s + loss_c

        total_loss.backward()
        optimizer.step()

        b_size = images.size(0)
        total_samples += b_size
        running_loss += total_loss.item() * b_size
        running_shape_loss += loss_s.item() * b_size
        running_color_loss += loss_c.item() * b_size

        _, s_preds = torch.max(s_outputs, 1)
        train_s_preds.extend(s_preds.cpu().numpy())
        train_s_targets.extend(s_targets.cpu().numpy())

        c_preds = (torch.sigmoid(c_outputs) > 0.5).int()
        train_c_preds.append(c_preds.cpu().numpy())
        train_c_targets.append(c_targets.cpu().numpy())

    epoch_train_loss = running_loss / total_samples

    # Validation Loop
    model.eval()
    val_loss, val_total_samples = 0.0, 0
    val_s_preds, val_s_targets = [], []
    val_c_preds, val_c_targets = [], []

    with torch.no_grad():
        for images, s_targets, c_targets, _ in val_loader:
            images, s_targets, c_targets = images.to(DEVICE), s_targets.to(DEVICE), c_targets.to(DEVICE)
            s_outputs, c_outputs = model(images)

            loss_s = criterion_shape(s_outputs, s_targets)
            loss_c = criterion_color(c_outputs, c_targets)
            total_loss = loss_s + loss_c

            b_size = images.size(0)
            val_total_samples += b_size
            val_loss += total_loss.item() * b_size

            _, s_preds = torch.max(s_outputs, 1)
            val_s_preds.extend(s_preds.cpu().numpy())
            val_s_targets.extend(s_targets.cpu().numpy())

            c_preds = (torch.sigmoid(c_outputs) > 0.5).int()
            val_c_preds.append(c_preds.cpu().numpy())
            val_c_targets.append(c_targets.cpu().numpy())

    epoch_val_loss = val_loss / val_total_samples
    val_shape_f1 = f1_score(val_s_targets, val_s_preds, average="macro", zero_division=0)
    val_color_f1 = f1_score(np.vstack(val_c_targets), np.vstack(val_c_preds), average="macro", zero_division=0)
    overall_macro_f1 = (val_shape_f1 + val_color_f1) / 2.0

    current_lr = optimizer.param_groups[1]["lr"]
    scheduler.step(epoch_val_loss)

    is_best = epoch_val_loss < best_val_loss
    if is_best:
        best_val_loss = epoch_val_loss
        best_overall_macro_f1 = overall_macro_f1
        torch.save(model.state_dict(), PATHS["checkpoints"] / f"{RUN_ID}_best.pt")
        patience_counter = 0
    else:
        patience_counter += 1

    torch.save(model.state_dict(), PATHS["checkpoints"] / f"{RUN_ID}_last.pt")

    row = {
        "epoch": epoch + 1,
        "train_loss": round(epoch_train_loss, 4),
        "val_loss": round(epoch_val_loss, 4),
        "val_shape_f1": round(val_shape_f1, 4),
        "val_color_f1": round(val_color_f1, 4),
        "learning_rate": current_lr,
        "best_metric": round(best_overall_macro_f1, 4),
        "is_best": is_best
    }
    train_history_rows.append(row)
    pd.DataFrame(train_history_rows).to_csv(csv_log_path, index=False)

    print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Shape F1: {val_shape_f1:.4f} | Val Color F1: {val_color_f1:.4f} | Best: {is_best}")

    if patience_counter >= PATIENCE:
        print(f"\n[Early Stopping] Kích hoạt tại Epoch {epoch+1}. Dừng huấn luyện!")
        break

total_time_min = (time.time() - start_time) / 60.0
runtime_info = (
    f"run_id: {RUN_ID}\nfinished_at: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    f"device: {DEVICE}\ntotal_train_time_minutes: {total_time_min:.2f}\n"
)
with open(PATHS["logs"] / f"{RUN_ID}_runtime.txt", "w", encoding="utf-8") as f:
    f.write(runtime_info)

print(f"\n[Train Finished] Checkpoint đã được lưu trữ tại {PATHS['checkpoints'].resolve()}")