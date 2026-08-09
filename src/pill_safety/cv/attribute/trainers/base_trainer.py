"""
Unified BaseTrainer for both head-tune and last-blocks fine-tune.

Key design decisions:
    - Best checkpoint selected by overall_macro_f1 (NOT val_loss)
    - Saves both best.pt AND last.pt every epoch
    - CSV log written fresh each run (no append)
    - Calls model.set_bn_eval() after every model.train()
    - Full column set: epoch, train_loss, val_loss, shape_loss_train,
      color_loss_train, shape_loss_val, color_loss_val, train_shape_f1,
      val_shape_f1, train_color_f1, val_color_f1, learning_rate, is_best
"""

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from pill_safety.cv.attribute.utils.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Unified trainer for multi-task attribute recognition.

    Args:
        model: The MultiTaskResNet18 model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion_shape: Loss function for shape (CrossEntropyLoss).
        criterion_color: Loss function for color (BCEWithLogitsLoss).
        optimizer: Optimizer instance.
        scheduler: LR scheduler (ReduceLROnPlateau recommended).
        device: Training device (cuda/cpu).
        run_id: Unique run identifier.
        paths: Dict with keys "checkpoints", "logs".
        label_mapping: Label mapping dict for checkpoint metadata.
        mapping_hash: SHA-256 hash of the label mapping.
        lambda_shape: Weight for shape loss (default 1.0).
        lambda_color: Weight for color loss (default 1.0).
        patience: Early stopping patience (0 = disabled).
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
        run_id: str,
        paths: Dict[str, Path],
        label_mapping: Dict,
        mapping_hash: str = "",
        lambda_shape: float = 1.0,
        lambda_color: float = 1.0,
        patience: int = 0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion_shape = criterion_shape
        self.criterion_color = criterion_color
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.run_id = run_id
        self.paths = paths
        self.label_mapping = label_mapping
        self.mapping_hash = mapping_hash
        self.lambda_shape = lambda_shape
        self.lambda_color = lambda_color
        self.patience = patience

        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history: List[Dict] = []

    @property
    def history_dict(self) -> Dict[str, List]:
        """Return history as a dictionary of lists (columnar format) for plotting/evaluation."""
        if not self.history:
            return {}
        return {k: [d[k] for d in self.history] for k in self.history[0].keys()}

    def _train_one_epoch(self) -> Dict:
        """Run one training epoch. Returns dict of metrics."""
        self.model.train()
        self.model.set_bn_eval()  # Freeze BN stats

        running_loss = 0.0
        running_shape_loss = 0.0
        running_color_loss = 0.0
        total_samples = 0
        all_s_preds, all_s_targets = [], []
        all_c_preds, all_c_targets = [], []

        for batch in self.train_loader:
            images = batch[0].to(self.device)
            s_targets = batch[1].to(self.device)
            c_targets = batch[2].to(self.device)

            self.optimizer.zero_grad()
            s_outputs, c_outputs = self.model(images)

            loss_s = self.criterion_shape(s_outputs, s_targets)
            loss_c = self.criterion_color(c_outputs, c_targets)
            total_loss = self.lambda_shape * loss_s + self.lambda_color * loss_c

            total_loss.backward()
            self.optimizer.step()

            b_size = images.size(0)
            total_samples += b_size
            running_loss += total_loss.item() * b_size
            running_shape_loss += loss_s.item() * b_size
            running_color_loss += loss_c.item() * b_size

            _, s_preds = torch.max(s_outputs, 1)
            all_s_preds.extend(s_preds.cpu().numpy())
            all_s_targets.extend(s_targets.cpu().numpy())

            c_preds = (torch.sigmoid(c_outputs) > 0.5).int()
            all_c_preds.append(c_preds.cpu().numpy())
            all_c_targets.append(c_targets.cpu().numpy())

        all_c_preds = np.vstack(all_c_preds)
        all_c_targets = np.vstack(all_c_targets)

        train_shape_f1 = f1_score(all_s_targets, all_s_preds, average="macro", zero_division=0)
        train_color_f1 = f1_score(all_c_targets, all_c_preds, average="macro", zero_division=0)

        return {
            "train_loss": running_loss / total_samples,
            "shape_loss_train": running_shape_loss / total_samples,
            "color_loss_train": running_color_loss / total_samples,
            "train_shape_f1": train_shape_f1,
            "train_color_f1": train_color_f1,
        }

    @torch.no_grad()
    def _validate_one_epoch(self) -> Dict:
        """Run one validation epoch. Returns dict of metrics."""
        self.model.eval()

        val_loss = 0.0
        val_shape_loss = 0.0
        val_color_loss = 0.0
        total_samples = 0
        all_s_preds, all_s_targets = [], []
        all_c_preds, all_c_targets = [], []

        for batch in self.val_loader:
            images = batch[0].to(self.device)
            s_targets = batch[1].to(self.device)
            c_targets = batch[2].to(self.device)

            s_outputs, c_outputs = self.model(images)

            loss_s = self.criterion_shape(s_outputs, s_targets)
            loss_c = self.criterion_color(c_outputs, c_targets)
            total_loss = self.lambda_shape * loss_s + self.lambda_color * loss_c

            b_size = images.size(0)
            total_samples += b_size
            val_loss += total_loss.item() * b_size
            val_shape_loss += loss_s.item() * b_size
            val_color_loss += loss_c.item() * b_size

            _, s_preds = torch.max(s_outputs, 1)
            all_s_preds.extend(s_preds.cpu().numpy())
            all_s_targets.extend(s_targets.cpu().numpy())

            c_preds = (torch.sigmoid(c_outputs) > 0.5).int()
            all_c_preds.append(c_preds.cpu().numpy())
            all_c_targets.append(c_targets.cpu().numpy())

        all_c_preds = np.vstack(all_c_preds)
        all_c_targets = np.vstack(all_c_targets)

        val_shape_f1 = f1_score(all_s_targets, all_s_preds, average="macro", zero_division=0)
        val_color_f1 = f1_score(all_c_targets, all_c_preds, average="macro", zero_division=0)
        overall_macro_f1 = (val_shape_f1 + val_color_f1) / 2.0

        return {
            "val_loss": val_loss / total_samples,
            "shape_loss_val": val_shape_loss / total_samples,
            "color_loss_val": val_color_loss / total_samples,
            "val_shape_f1": val_shape_f1,
            "val_color_f1": val_color_f1,
            "overall_macro_f1": overall_macro_f1,
        }

    def fit(self, num_epochs: int) -> List[Dict]:
        """Run the full training loop.

        Args:
            num_epochs: Number of epochs to train.

        Returns:
            Training history (list of per-epoch dicts).
        """
        csv_log_path = self.paths["logs"] / f"{self.run_id}_train_log.csv"
        start_time = time.time()
        epoch_times = []

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self._train_one_epoch()

            # Validate
            val_metrics = self._validate_one_epoch()

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            # Get current LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step scheduler (uses val_loss for ReduceLROnPlateau)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Check if best (by overall_macro_f1, NOT val_loss)
            is_best = val_metrics["overall_macro_f1"] > self.best_metric
            if is_best:
                self.best_metric = val_metrics["overall_macro_f1"]
                self.best_epoch = epoch
                self.patience_counter = 0

                # Save best checkpoint
                save_checkpoint(
                    path=self.paths["checkpoints"] / f"{self.run_id}_best.pt",
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    best_metric=self.best_metric,
                    label_mapping=self.label_mapping,
                    mapping_hash=self.mapping_hash,
                    extra={
                        "num_shape_classes": len(self.label_mapping.get("shape", [])),
                        "num_color_classes": len(self.label_mapping.get("color", [])),
                    },
                )
            else:
                self.patience_counter += 1

            # Always save last checkpoint
            save_checkpoint(
                path=self.paths["checkpoints"] / f"{self.run_id}_last.pt",
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                best_metric=self.best_metric,
                label_mapping=self.label_mapping,
                mapping_hash=self.mapping_hash,
            )

            # Build history row
            row = {
                "epoch": epoch,
                "train_loss": round(train_metrics["train_loss"], 4),
                "val_loss": round(val_metrics["val_loss"], 4),
                "shape_loss_train": round(train_metrics["shape_loss_train"], 4),
                "color_loss_train": round(train_metrics["color_loss_train"], 4),
                "shape_loss_val": round(val_metrics["shape_loss_val"], 4),
                "color_loss_val": round(val_metrics["color_loss_val"], 4),
                "train_shape_f1": round(train_metrics["train_shape_f1"], 4),
                "val_shape_f1": round(val_metrics["val_shape_f1"], 4),
                "train_color_f1": round(train_metrics["train_color_f1"], 4),
                "val_color_f1": round(val_metrics["val_color_f1"], 4),
                "overall_macro_f1": round(val_metrics["overall_macro_f1"], 4),
                "learning_rate": current_lr,
                "is_best": is_best,
            }
            self.history.append(row)

            # Write CSV fresh (not append)
            pd.DataFrame(self.history).to_csv(csv_log_path, index=False)

            # Log
            log_msg = (
                f"Epoch {epoch:02d}/{num_epochs:02d} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Shape F1: {val_metrics['val_shape_f1']:.4f} | "
                f"Color F1: {val_metrics['val_color_f1']:.4f} | "
                f"Overall F1: {val_metrics['overall_macro_f1']:.4f} | "
                f"{'★ BEST' if is_best else ''}"
            )
            logger.info(log_msg)
            print(log_msg)

            # Early stopping
            if self.patience > 0 and self.patience_counter >= self.patience:
                logger.info(f"[Early Stopping] at Epoch {epoch}.")
                print(f"[Early Stopping] at Epoch {epoch}.")
                break

        total_time = time.time() - start_time
        self.total_train_time_minutes = total_time / 60.0
        self.avg_epoch_time_seconds = np.mean(epoch_times) if epoch_times else 0.0
        self.num_epochs_run = len(self.history)

        return self.history


def compute_shape_class_weights(
    shape_labels: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute inverse-frequency weights for shape classes.

    Args:
        shape_labels: Array of shape label indices.
        num_classes: Total number of shape classes.
        device: Target device.

    Returns:
        Weight tensor of shape (num_classes,).
    """
    counts = Counter(shape_labels.tolist())
    total = len(shape_labels)
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        weights.append(total / (num_classes * c))
    return torch.tensor(weights, dtype=torch.float32).to(device)
