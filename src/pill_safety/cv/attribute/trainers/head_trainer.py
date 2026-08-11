"""
Head fine-tune trainer for multi-task pill attribute recognition.

Implements the training loop for Stage 1 (freeze backbone, train heads only),
with per-epoch logging, CSV export, and checkpoint saving.
"""

import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score


def compute_shape_class_weights(
    train_dataset, num_shape_classes: int
) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced shape labels.

    Args:
        train_dataset: A fitted ``RxImageDataset`` instance.
        num_shape_classes: Total number of shape classes.

    Returns:
        Tensor of shape ``(num_shape_classes,)`` with per-class weights.
    """
    shape_counts = Counter(train_dataset.shape_labels.tolist())
    total = sum(shape_counts.values())

    weights = torch.ones(num_shape_classes)
    for cls_idx, count in shape_counts.items():
        weights[int(cls_idx)] = total / (num_shape_classes * count)

    return weights


class HeadFineTuneTrainer:
    """Training loop for Stage 1: freeze backbone, train classification heads.

    Handles train/validate epochs, CSV logging, checkpoint saving, and
    learning rate scheduling.

    Args:
        model: The ``MultiTaskResNet18_HeadsFinetune`` model.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        criterion_shape: Loss function for shape (e.g., CrossEntropyLoss).
        criterion_color: Loss function for color (e.g., BCEWithLogitsLoss).
        optimizer: PyTorch optimizer.
        scheduler: Learning rate scheduler (ReduceLROnPlateau).
        device: Target device (cuda/cpu).
        paths: Dictionary of experiment output paths.
        run_id: Unique identifier for this training run.
        color_loss_weight: Multiplier for the color loss term.
        logger: Optional Python logger for file logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion_shape,
        criterion_color,
        optimizer,
        scheduler,
        device: torch.device,
        paths: Dict[str, Path],
        run_id: str,
        color_loss_weight: float = 2.0,
        logger=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion_shape = criterion_shape
        self.criterion_color = criterion_color
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.paths = paths
        self.run_id = run_id
        self.color_loss_weight = color_loss_weight
        self.logger = logger

        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "shape_loss_train": [],
            "color_loss_train": [],
            "shape_loss_val": [],
            "color_loss_val": [],
            "train_shape_acc": [],
            "val_shape_acc": [],
            "train_shape_f1": [],
            "val_shape_f1": [],
            "train_color_acc": [],
            "val_color_acc": [],
            "train_color_f1": [],
            "val_color_f1": [],
            "lr": [],
        }
        self.epoch_times: List[float] = []

    def _train_one_epoch(self) -> dict:
        """Run one training epoch. Returns dict of epoch metrics."""
        self.model.train()
        self.model.backbone.eval()  # Keep BN layers frozen

        running_loss = 0.0
        running_shape_loss = 0.0
        running_color_loss = 0.0
        total_samples = 0
        s_preds_list, s_targets_list = [], []
        c_preds_list, c_targets_list = [], []

        for images, s_targets, c_targets in self.train_loader:
            images = images.to(self.device)
            s_targets = s_targets.to(self.device)
            c_targets = c_targets.to(self.device)

            self.optimizer.zero_grad()
            s_outputs, c_outputs = self.model(images)
            loss_shape = self.criterion_shape(s_outputs, s_targets)
            loss_color = self.criterion_color(c_outputs, c_targets)
            total_loss = loss_shape + self.color_loss_weight * loss_color

            total_loss.backward()
            self.optimizer.step()

            bs = images.size(0)
            total_samples += bs
            running_loss += total_loss.item() * bs
            running_shape_loss += loss_shape.item() * bs
            running_color_loss += loss_color.item() * bs

            _, s_preds = torch.max(s_outputs, 1)
            s_preds_list.extend(s_preds.cpu().numpy())
            s_targets_list.extend(s_targets.cpu().numpy())
            c_preds = (torch.sigmoid(c_outputs) > 0.5).int()
            c_preds_list.append(c_preds.cpu().numpy())
            c_targets_list.append(c_targets.cpu().numpy())

        # Aggregate metrics
        tr_s = np.array(s_preds_list)
        tr_st = np.array(s_targets_list)
        tr_c = np.vstack(c_preds_list)
        tr_ct = np.vstack(c_targets_list)

        return {
            "loss": running_loss / total_samples,
            "shape_loss": running_shape_loss / total_samples,
            "color_loss": running_color_loss / total_samples,
            "shape_acc": float(np.mean(tr_s == tr_st)),
            "shape_f1": float(
                f1_score(tr_st, tr_s, average="macro", zero_division=0)
            ),
            "color_acc": float(np.mean(np.all(tr_c == tr_ct, axis=1))),
            "color_f1": float(
                f1_score(tr_ct, tr_c, average="macro", zero_division=0)
            ),
        }

    @torch.no_grad()
    def _validate_one_epoch(self) -> dict:
        """Run one validation epoch. Returns dict of epoch metrics."""
        self.model.eval()

        val_loss = 0.0
        val_shape_loss = 0.0
        val_color_loss = 0.0
        val_total = 0
        s_preds_list, s_targets_list = [], []
        c_preds_list, c_targets_list = [], []

        for images, s_targets, c_targets in self.val_loader:
            images = images.to(self.device)
            s_targets = s_targets.to(self.device)
            c_targets = c_targets.to(self.device)

            s_outputs, c_outputs = self.model(images)
            loss_shape = self.criterion_shape(s_outputs, s_targets)
            loss_color = self.criterion_color(c_outputs, c_targets)
            total_loss = loss_shape + self.color_loss_weight * loss_color

            bs = images.size(0)
            val_total += bs
            val_loss += total_loss.item() * bs
            val_shape_loss += loss_shape.item() * bs
            val_color_loss += loss_color.item() * bs

            _, s_preds = torch.max(s_outputs, 1)
            s_preds_list.extend(s_preds.cpu().numpy())
            s_targets_list.extend(s_targets.cpu().numpy())
            c_preds = (torch.sigmoid(c_outputs) > 0.5).int()
            c_preds_list.append(c_preds.cpu().numpy())
            c_targets_list.append(c_targets.cpu().numpy())

        vl_s = np.array(s_preds_list)
        vl_st = np.array(s_targets_list)
        vl_c = np.vstack(c_preds_list)
        vl_ct = np.vstack(c_targets_list)

        return {
            "loss": val_loss / val_total,
            "shape_loss": val_shape_loss / val_total,
            "color_loss": val_color_loss / val_total,
            "shape_acc": float(np.mean(vl_s == vl_st)),
            "shape_f1": float(
                f1_score(vl_st, vl_s, average="macro", zero_division=0)
            ),
            "color_acc": float(np.mean(np.all(vl_c == vl_ct, axis=1))),
            "color_f1": float(
                f1_score(vl_ct, vl_c, average="macro", zero_division=0)
            ),
        }

    def _save_checkpoint(
        self,
        epoch: int,
        val_metrics: dict,
        is_best: bool,
        label_mapping: Optional[dict] = None,
        num_shape_classes: int = 0,
        num_color_classes: int = 0,
    ) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch (1-indexed).
            val_metrics: Validation metrics for this epoch.
            is_best: Whether this is the best epoch so far.
            label_mapping: Optional label mapping dict to embed in checkpoint.
            num_shape_classes: Number of shape classes.
            num_color_classes: Number of color classes.
        """
        overall_f1 = (val_metrics["shape_f1"] + val_metrics["color_f1"]) / 2.0
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_val_f1,
            "shape_f1": val_metrics["shape_f1"],
            "color_f1": val_metrics["color_f1"],
            "overall_f1": overall_f1,
            "label_mapping": label_mapping,
            "num_shape_classes": num_shape_classes,
            "num_color_classes": num_color_classes,
        }

        # Always save last
        torch.save(
            ckpt_data,
            self.paths["checkpoints"] / f"{self.run_id}_last.pt",
        )

        # Save best
        if is_best:
            torch.save(
                ckpt_data,
                self.paths["checkpoints"] / f"{self.run_id}_best.pt",
            )

    def fit(
        self,
        num_epochs: int,
        label_mapping: Optional[dict] = None,
        num_shape_classes: int = 0,
        num_color_classes: int = 0,
    ) -> dict:
        """Run the full training loop.

        Args:
            num_epochs: Number of epochs to train.
            label_mapping: Label mapping dict to embed in saved checkpoints.
            num_shape_classes: Number of shape classes.
            num_color_classes: Number of color classes.

        Returns:
            The training history dictionary.
        """
        train_log_path = (
            self.paths["logs"] / f"{self.run_id}_train_log.csv"
        )
        train_start_time = time.time()

        print(
            f"Starting training: {num_epochs} epochs, "
            f"lr={self.optimizer.param_groups[0]['lr']}"
        )
        print("=" * 100)

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self._train_one_epoch()
            # Validate
            val_metrics = self._validate_one_epoch()

            # Overall F1
            overall_f1 = (
                val_metrics["shape_f1"] + val_metrics["color_f1"]
            ) / 2.0
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_metrics["loss"])

            # Is best?
            is_best = overall_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = overall_f1
                self.best_epoch = epoch + 1

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            # Update history
            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["shape_loss_train"].append(
                train_metrics["shape_loss"]
            )
            self.history["color_loss_train"].append(
                train_metrics["color_loss"]
            )
            self.history["shape_loss_val"].append(val_metrics["shape_loss"])
            self.history["color_loss_val"].append(val_metrics["color_loss"])
            self.history["train_shape_acc"].append(train_metrics["shape_acc"])
            self.history["val_shape_acc"].append(val_metrics["shape_acc"])
            self.history["train_shape_f1"].append(train_metrics["shape_f1"])
            self.history["val_shape_f1"].append(val_metrics["shape_f1"])
            self.history["train_color_acc"].append(train_metrics["color_acc"])
            self.history["val_color_acc"].append(val_metrics["color_acc"])
            self.history["train_color_f1"].append(train_metrics["color_f1"])
            self.history["val_color_f1"].append(val_metrics["color_f1"])
            self.history["lr"].append(current_lr)

            # Log
            log_msg = (
                f"Epoch {epoch+1:02d}/{num_epochs} | "
                f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                f"Shape F1: {val_metrics['shape_f1']:.4f} | "
                f"Color F1: {val_metrics['color_f1']:.4f} | "
                f"Overall F1: {overall_f1:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"{'★ BEST' if is_best else ''}"
            )
            print(log_msg)
            if self.logger:
                self.logger.info(log_msg)

            # Append to CSV log
            log_row = {
                "epoch": epoch + 1,
                "train_loss": round(train_metrics["loss"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "shape_loss_train": round(train_metrics["shape_loss"], 6),
                "color_loss_train": round(train_metrics["color_loss"], 6),
                "shape_loss_val": round(val_metrics["shape_loss"], 6),
                "color_loss_val": round(val_metrics["color_loss"], 6),
                "val_shape_f1": round(val_metrics["shape_f1"], 6),
                "val_color_f1": round(val_metrics["color_f1"], 6),
                "learning_rate": current_lr,
                "best_metric": round(self.best_val_f1, 6),
                "is_best": is_best,
            }
            pd.DataFrame([log_row]).to_csv(
                train_log_path,
                mode="a",
                header=not train_log_path.exists() or (epoch == 0),
                index=False,
            )

            # Save checkpoint
            self._save_checkpoint(
                epoch + 1,
                val_metrics,
                is_best,
                label_mapping=label_mapping,
                num_shape_classes=num_shape_classes,
                num_color_classes=num_color_classes,
            )
            if is_best:
                print(
                    f"  ★ Best checkpoint saved (Overall F1: {overall_f1:.4f})"
                )

        total_time = time.time() - train_start_time
        print("=" * 100)
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(
            f"Best epoch: {self.best_epoch} | "
            f"Best Overall F1: {self.best_val_f1:.4f}"
        )

        return self.history
