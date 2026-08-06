import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import models

from pill_safety.cv.attribute.utils.config import AttributeConfig
from pill_safety.cv.attribute.datasets.rximage_dataset import RxImageDataset
from pill_safety.cv.attribute.transforms.augmentations import get_attribute_transforms
from pill_safety.cv.attribute.models.resnet18_multitask import MultiTaskResNet18
from pill_safety.cv.attribute.evaluators.metrics import find_best_thresholds, save_evaluation_plots
from pill_safety.cv.attribute.labels.mapping import create_and_save_label_mapping
from pill_safety.cv.attribute.utils.logger import setup_logger

def train(epochs=None, batch_size=None):
    AttributeConfig.setup_directories()
    if epochs: AttributeConfig.NUM_EPOCHS = epochs
    if batch_size: AttributeConfig.BATCH_SIZE = batch_size

    logger = setup_logger(AttributeConfig.LOG_DIR / f"{AttributeConfig.RUN_ID}_runtime.log", AttributeConfig.RUN_ID)
    logger.info(f"Start Training Stage 2 - Run ID: {AttributeConfig.RUN_ID}")

    torch.manual_seed(AttributeConfig.SEED)
    np.random.seed(AttributeConfig.SEED)

    transforms_dict = get_attribute_transforms()
    train_dataset = RxImageDataset(AttributeConfig.COMBINED_DIR / "augmented_train_combined.csv", AttributeConfig.IMG_DIR, transform=transforms_dict["train"])
    val_dataset = RxImageDataset(AttributeConfig.COMBINED_DIR / "val_combined_crop.csv", AttributeConfig.IMG_DIR, transform=transforms_dict["val"])

    train_loader = DataLoader(train_dataset, batch_size=AttributeConfig.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=AttributeConfig.BATCH_SIZE, shuffle=False, num_workers=2)

    label_mapping = create_and_save_label_mapping(
        AttributeConfig.COMBINED_DIR / "augmented_train_combined.csv",
        train_dataset.color_cols,
        AttributeConfig.METRIC_DIR / "label_mapping.json"
    )

    num_shape_classes = len(label_mapping["shape"])
    num_color_classes = len(label_mapping["color"])

    model = MultiTaskResNet18(num_shape_classes, num_color_classes).to(AttributeConfig.DEVICE)
    weights_base = models.ResNet18_Weights.DEFAULT.get_state_dict(progress=True)
    weights_base.pop("fc.weight", None)
    weights_base.pop("fc.bias", None)
    model.backbone.load_state_dict(weights_base, strict=False)

    if AttributeConfig.STAGE1_CHECKPOINT.exists():
        logger.info(f"Loading Stage 1 Checkpoint: {AttributeConfig.STAGE1_CHECKPOINT}")
        heads_weights = torch.load(AttributeConfig.STAGE1_CHECKPOINT, map_location=AttributeConfig.DEVICE, weights_only=True)
        model.fc_shape.load_state_dict(heads_weights["fc_shape"])
        model.fc_color.load_state_dict(heads_weights["fc_color"])

    for param in model.parameters(): param.requires_grad = False
    for param in model.backbone.layer4.parameters(): param.requires_grad = True
    for param in model.fc_shape.parameters(): param.requires_grad = True
    for param in model.fc_color.parameters(): param.requires_grad = True

    color_targets_all = train_dataset.color_labels
    pos_counts = color_targets_all.sum(axis=0)
    neg_counts = len(color_targets_all) - pos_counts
    pos_weights = np.clip(neg_counts / (pos_counts + 1e-5), a_min=1.0, a_max=10.0)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(AttributeConfig.DEVICE)

    optimizer = optim.AdamW([
        {"params": model.backbone.layer4.parameters(), "lr": AttributeConfig.LR_BACKBONE},
        {"params": model.fc_shape.parameters(), "lr": AttributeConfig.LR_HEADS},
        {"params": model.fc_color.parameters(), "lr": AttributeConfig.LR_HEADS},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    criterion_shape = nn.CrossEntropyLoss()
    criterion_color = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    best_val_loss = float("inf")
    history = []

    for epoch in range(AttributeConfig.NUM_EPOCHS):
        model.train()
        running_loss, total_samples = 0.0, 0
        for images, s_targets, c_targets, _ in train_loader:
            images, s_targets, c_targets = images.to(AttributeConfig.DEVICE), s_targets.to(AttributeConfig.DEVICE), c_targets.to(AttributeConfig.DEVICE)
            
            optimizer.zero_grad()
            s_outputs, c_outputs = model(images)
            total_loss = criterion_shape(s_outputs, s_targets) + criterion_color(c_outputs, c_targets)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * images.size(0)
            total_samples += images.size(0)

        epoch_train_loss = running_loss / total_samples

        model.eval()
        val_loss, val_samples = 0.0, 0
        val_s_targets, val_s_preds, val_c_targets, val_c_probs = [], [], [], []

        with torch.no_grad():
            for images, s_targets, c_targets, _ in val_loader:
                images, s_targets, c_targets = images.to(AttributeConfig.DEVICE), s_targets.to(AttributeConfig.DEVICE), c_targets.to(AttributeConfig.DEVICE)
                s_outputs, c_outputs = model(images)
                
                loss = criterion_shape(s_outputs, s_targets) + criterion_color(c_outputs, c_targets)
                val_loss += loss.item() * images.size(0)
                val_samples += images.size(0)

                val_s_targets.extend(s_targets.cpu().numpy())
                val_s_preds.extend(torch.argmax(s_outputs, dim=1).cpu().numpy())
                val_c_targets.append(c_targets.cpu().numpy())
                val_c_probs.append(torch.sigmoid(c_outputs).cpu().numpy())

        epoch_val_loss = val_loss / val_samples
        scheduler.step(epoch_val_loss)

        history.append({"epoch": epoch + 1, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss})
        logger.info(f"Epoch {epoch+1:02d}/{AttributeConfig.NUM_EPOCHS:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), AttributeConfig.CHECKPOINT_DIR / f"{AttributeConfig.RUN_ID}_best.pt")

    val_c_targets, val_c_probs = np.vstack(val_c_targets), np.vstack(val_c_probs)
    best_thresholds = find_best_thresholds(val_c_targets, val_c_probs, num_color_classes)
    
    with open(AttributeConfig.METRIC_DIR / "optimal_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(best_thresholds, f, indent=2)

    save_evaluation_plots(pd.DataFrame(history), np.array(val_s_targets), np.array(val_s_preds), AttributeConfig)
    logger.info("Training complete and artifacts saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)