"""
Evaluator for multi-task pill attribute recognition.

Handles test-set evaluation, metric computation, per-class reporting,
visualization (loss/metric curves, confusion matrix, summary plot),
and prediction error categorization.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class AttributeEvaluator:
    """Evaluator for test-set inference, metric computation, and visualization.

    Args:
        model: Trained model (already loaded with best weights).
        test_loader: DataLoader for the test split.
        device: Target device.
        paths: Dictionary of experiment output paths.
        run_id: Unique identifier for this training run.
        module_name: Module name string for report metadata.
        shape_class_names: Ordered list of shape class names.
        color_class_names: Ordered list of color class names.
        num_shape_classes: Total number of shape classes.
        num_color_classes: Total number of color classes.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader,
        device: torch.device,
        paths: Dict[str, Path],
        run_id: str,
        module_name: str,
        shape_class_names: List[str],
        color_class_names: List[str],
        num_shape_classes: int,
        num_color_classes: int,
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.paths = paths
        self.run_id = run_id
        self.module_name = module_name
        self.shape_class_names = shape_class_names
        self.color_class_names = color_class_names
        self.num_shape_classes = num_shape_classes
        self.num_color_classes = num_color_classes

    @torch.no_grad()
    def collect_predictions(self, dataloader=None) -> dict:
        """Run inference on the test set and collect all predictions.

        Returns:
            Dictionary containing numpy arrays for shape/color predictions,
            targets, and probabilities.
        """
        self.model.eval()

        all_shape_preds, all_shape_targets = [], []
        all_color_preds, all_color_targets = [], []
        all_shape_probs, all_color_probs = [], []

        for images, s_targets, c_targets in (dataloader or self.test_loader):
            images = images.to(self.device)
            s_outputs, c_outputs = self.model(images)

            s_probs = torch.softmax(s_outputs, dim=1)
            _, s_preds = torch.max(s_outputs, 1)
            all_shape_preds.extend(s_preds.cpu().numpy())
            all_shape_targets.extend(s_targets.numpy())
            all_shape_probs.append(s_probs.cpu().numpy())

            c_probs = torch.sigmoid(c_outputs)
            c_preds = (c_probs > 0.5).int()
            all_color_preds.append(c_preds.cpu().numpy())
            all_color_targets.append(c_targets.numpy())
            all_color_probs.append(c_probs.cpu().numpy())

        return {
            "shape_preds": np.array(all_shape_preds),
            "shape_targets": np.array(all_shape_targets),
            "shape_probs": np.vstack(all_shape_probs),
            "color_preds": np.vstack(all_color_preds),
            "color_targets": np.vstack(all_color_targets),
            "color_probs": np.vstack(all_color_probs),
        }

    def compute_test_metrics(self, preds: dict) -> dict:
        """Compute comprehensive test metrics including per-class breakdown.

        Args:
            preds: Output from ``collect_predictions()``.

        Returns:
            Dictionary of test metrics.
        """
        shape_preds = preds["shape_preds"]
        shape_targets = preds["shape_targets"]
        color_preds = preds["color_preds"]
        color_targets = preds["color_targets"]

        shape_f1 = float(
            f1_score(
                shape_targets, shape_preds, average="macro", zero_division=0
            )
        )
        color_f1 = float(
            f1_score(
                color_targets, color_preds, average="macro", zero_division=0
            )
        )
        overall_f1 = (shape_f1 + color_f1) / 2.0

        # Per-class shape report
        shape_report = classification_report(
            shape_targets,
            shape_preds,
            labels=range(self.num_shape_classes),
            target_names=self.shape_class_names,
            output_dict=True,
            zero_division=0,
        )

        # Per-class color metrics
        color_per_class = {}
        for i, name in enumerate(self.color_class_names):
            col_t = color_targets[:, i]
            col_p = color_preds[:, i]
            color_per_class[name] = {
                "precision": round(
                    float(precision_score(col_t, col_p, zero_division=0)), 4
                ),
                "recall": round(
                    float(recall_score(col_t, col_p, zero_division=0)), 4
                ),
                "f1-score": round(
                    float(f1_score(col_t, col_p, zero_division=0)), 4
                ),
                "support": int(col_t.sum()),
            }

        return {
            "shape_macro_f1": round(shape_f1, 4),
            "color_macro_f1": round(color_f1, 4),
            "overall_macro_f1": round(overall_f1, 4),
            "shape_accuracy": round(
                float(np.mean(shape_preds == shape_targets)), 4
            ),
            "shape_balanced_accuracy": round(
                float(balanced_accuracy_score(shape_targets, shape_preds)), 4
            ),
            "color_precision_macro": round(
                float(
                    precision_score(
                        color_targets,
                        color_preds,
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            ),
            "color_recall_macro": round(
                float(
                    recall_score(
                        color_targets,
                        color_preds,
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            ),
            "per_class_shape": {
                k: v
                for k, v in shape_report.items()
                if k not in ["accuracy", "macro avg", "weighted avg"]
            },
            "per_class_color": color_per_class,
        }

    def save_val_metrics(
        self, history: dict, best_epoch: int, best_val_f1: float,
        per_class_metrics: Optional[dict] = None,
        label_mapping_file: Optional[str] = None
    ) -> Path:
        """Save validation metrics JSON.

        Args:
            history: Training history dictionary.
            best_epoch: Best epoch number (1-indexed).
            best_val_f1: Best overall F1 on validation set.
            per_class_metrics: Optional per class metrics dictionary.
            label_mapping_file: Optional path to label mapping file.

        Returns:
            Path to the saved JSON file.
        """
        val_metrics = {
            "run_id": self.run_id,
            "module": self.module_name,
            "split": "val",
            "best_epoch": best_epoch,
            "best_checkpoint": str(
                self.paths["checkpoints"] / f"{self.run_id}_best.pt"
            ),
            "selection_metric": "overall_macro_f1",
            "metrics": {
                "shape_macro_f1": round(
                    history["val_shape_f1"][best_epoch - 1], 4
                ),
                "color_macro_f1": round(
                    history["val_color_f1"][best_epoch - 1], 4
                ),
                "dosage_form_macro_f1": None,
                "scoreline_macro_f1": None,
                "overall_macro_f1": round(best_val_f1, 4),
            },
        }

        if label_mapping_file:
            val_metrics["label_mapping_file"] = label_mapping_file
        if per_class_metrics:
            val_metrics["per_class_metrics"] = per_class_metrics

        path = self.paths["metrics"] / f"{self.run_id}_val_metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2, ensure_ascii=False)

        print(f"✓ Val metrics saved: {path}")
        return path

    def save_test_metrics(
        self, test_metrics: dict, best_epoch: int
    ) -> Path:
        """Save test metrics JSON.

        Args:
            test_metrics: Output from ``compute_test_metrics()``.
            best_epoch: Best epoch number.

        Returns:
            Path to the saved JSON file.
        """
        output = {
            "run_id": self.run_id,
            "module": self.module_name,
            "split": "test",
            "best_epoch": best_epoch,
            "best_checkpoint": str(
                self.paths["checkpoints"] / f"{self.run_id}_best.pt"
            ),
            "selection_metric": "overall_macro_f1",
            "metrics": {
                k: v
                for k, v in test_metrics.items()
                if not k.startswith("per_class")
            },
            "per_class_metrics": {
                "shape": test_metrics.get("per_class_shape", {}),
                "color": test_metrics.get("per_class_color", {}),
            },
        }

        path = self.paths["metrics"] / f"{self.run_id}_test_metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"✓ Test metrics saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_training_curves(
        self, history: dict, best_epoch: int
    ) -> None:
        """Plot loss and metric curves from training history.

        Saves ``_loss_curve.png`` and ``_metric_curve.png``.

        Args:
            history: Training history dictionary from the trainer.
            best_epoch: Best epoch number (1-indexed).
        """
        plots_dir = self.paths["plots"]

        # --- Loss Curve ---
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].plot(
            history["epoch"], history["train_loss"],
            "b-o", label="Train Loss", markersize=4,
        )
        axes[0].plot(
            history["epoch"], history["val_loss"],
            "r-o", label="Val Loss", markersize=4,
        )
        axes[0].axvline(
            x=best_epoch, color="g", linestyle="--", alpha=0.7,
            label=f"Best epoch ({best_epoch})",
        )
        axes[0].set_title("Total Weighted Loss", fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            history["epoch"], history["shape_loss_train"],
            "b-o", label="Train Shape Loss", markersize=4,
        )
        axes[1].plot(
            history["epoch"], history["shape_loss_val"],
            "r-o", label="Val Shape Loss", markersize=4,
        )
        axes[1].set_title("Shape Loss", fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(
            history["epoch"], history["color_loss_train"],
            "b-o", label="Train Color Loss", markersize=4,
        )
        axes[2].plot(
            history["epoch"], history["color_loss_val"],
            "r-o", label="Val Color Loss", markersize=4,
        )
        axes[2].set_title("Color Loss (BCE)", fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Loss")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            plots_dir / f"{self.run_id}_loss_curve.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()

        # --- Metric Curve ---
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        best_val_f1 = max(
            (s + c) / 2
            for s, c in zip(
                history["val_shape_f1"], history["val_color_f1"]
            )
        )

        axes[0].plot(
            history["epoch"], history["val_shape_f1"],
            "r-o", label="Val Shape F1", markersize=4,
        )
        axes[0].plot(
            history["epoch"], history["train_shape_f1"],
            "b-o", label="Train Shape F1", markersize=4, alpha=0.5,
        )
        axes[0].axvline(
            x=best_epoch, color="g", linestyle="--", alpha=0.7,
            label=f"Best ({best_epoch})",
        )
        axes[0].set_title("Shape Macro F1", fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("F1 Score")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            history["epoch"], history["val_color_f1"],
            "r-o", label="Val Color F1", markersize=4,
        )
        axes[1].plot(
            history["epoch"], history["train_color_f1"],
            "b-o", label="Train Color F1", markersize=4, alpha=0.5,
        )
        axes[1].axvline(
            x=best_epoch, color="g", linestyle="--", alpha=0.7,
            label=f"Best ({best_epoch})",
        )
        axes[1].set_title("Color Macro F1", fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1 Score")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        overall_f1_history = [
            (s + c) / 2
            for s, c in zip(
                history["val_shape_f1"], history["val_color_f1"]
            )
        ]
        axes[2].plot(
            history["epoch"], overall_f1_history,
            "r-o", label="Val Overall F1", markersize=4,
        )
        axes[2].axvline(
            x=best_epoch, color="g", linestyle="--", alpha=0.7,
            label=f"Best ({best_epoch})",
        )
        axes[2].axhline(
            y=best_val_f1, color="orange", linestyle=":", alpha=0.7,
            label=f"Best F1: {best_val_f1:.4f}",
        )
        axes[2].set_title("Overall Macro F1", fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("F1 Score")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            plots_dir / f"{self.run_id}_metric_curve.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()

        print(
            f"✓ Saved: {self.run_id}_loss_curve.png, "
            f"{self.run_id}_metric_curve.png"
        )

    def plot_confusion_matrix(self, preds: dict) -> None:
        """Plot shape confusion matrix and color per-class F1 bar chart.

        Args:
            preds: Output from ``collect_predictions()``.
        """
        plots_dir = self.paths["plots"]
        shape_preds = preds["shape_preds"]
        shape_targets = preds["shape_targets"]
        color_preds = preds["color_preds"]
        color_targets = preds["color_targets"]

        # 1. Shape Confusion Matrix
        cm = confusion_matrix(
            shape_targets, shape_preds,
            labels=range(self.num_shape_classes),
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.shape_class_names,
            yticklabels=self.shape_class_names,
        )
        plt.title("Shape Confusion Matrix (Test Set)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            plots_dir / f"{self.run_id}_shape_confusion_matrix.png", dpi=300
        )
        plt.close()

        # 2. Color F1 per class
        f1_per_class = []
        for i in range(self.num_color_classes):
            f1_per_class.append(
                float(
                    f1_score(
                        color_targets[:, i],
                        color_preds[:, i],
                        zero_division=0,
                    )
                )
            )

        plt.figure(figsize=(10, 5))
        sns.barplot(
            x=self.color_class_names, y=f1_per_class,
            hue=self.color_class_names, palette="viridis", legend=False,
        )
        plt.title("Color F1-Score per Class (Test Set)")
        plt.ylabel("Macro F1")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45, ha="right")
        for i, v in enumerate(f1_per_class):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.tight_layout()
        plt.savefig(
            plots_dir / f"{self.run_id}_color_f1_per_class.png", dpi=300
        )
        plt.close()

        print(
            f"✓ Saved: {self.run_id}_shape_confusion_matrix.png, "
            f"{self.run_id}_color_f1_per_class.png"
        )

    def plot_summary(
        self,
        history: dict,
        preds: dict,
        test_metrics: Optional[dict],
        best_epoch: int,
        best_val_f1: float,
        config: dict,
    ) -> None:
        """Create a 2x2 summary plot with loss, F1, confusion matrix, and results table.

        Args:
            history: Training history dictionary.
            preds: Output from ``collect_predictions()``.
            test_metrics: Output from ``compute_test_metrics()``.
            best_epoch: Best epoch number.
            best_val_f1: Best validation overall F1.
            config: Run configuration dict (for the results table).
        """
        plots_dir = self.paths["plots"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Training Summary — {self.run_id} ({self.module_name})",
            fontsize=16, fontweight="bold",
        )

        # Loss curve
        axes[0, 0].plot(
            history["epoch"], history["train_loss"],
            "b-o", label="Train", markersize=3,
        )
        axes[0, 0].plot(
            history["epoch"], history["val_loss"],
            "r-o", label="Val", markersize=3,
        )
        axes[0, 0].axvline(
            x=best_epoch, color="g", linestyle="--", alpha=0.5
        )
        axes[0, 0].set_title("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # F1 curves
        overall_f1_history = [
            (s + c) / 2
            for s, c in zip(
                history["val_shape_f1"], history["val_color_f1"]
            )
        ]
        axes[0, 1].plot(
            history["epoch"], history["val_shape_f1"],
            "r-o", label="Shape F1", markersize=3,
        )
        axes[0, 1].plot(
            history["epoch"], history["val_color_f1"],
            "b-o", label="Color F1", markersize=3,
        )
        axes[0, 1].plot(
            history["epoch"], overall_f1_history,
            "g-s", label="Overall F1", markersize=3,
        )
        axes[0, 1].axvline(
            x=best_epoch, color="g", linestyle="--", alpha=0.5
        )
        axes[0, 1].set_title("Macro F1 (Val)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Shape confusion matrix (normalized)
        cm = confusion_matrix(
            preds["shape_targets"], preds["shape_preds"],
            labels=range(self.num_shape_classes),
        )
        cm_normalized = cm.astype("float") / cm.sum(
            axis=1, keepdims=True
        )
        cm_normalized = np.nan_to_num(cm_normalized)
        sns.heatmap(
            cm_normalized, annot=False, cmap="Blues", ax=axes[1, 0],
            xticklabels=self.shape_class_names,
            yticklabels=self.shape_class_names,
            cbar_kws={"shrink": 0.8},
        )
        axes[1, 0].set_title("Shape Confusion (Normalized)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Results table
        axes[1, 1].axis("off")
        num_epochs = config.get("training", {}).get(
            "epochs", len(history["epoch"])
        )
        lr = config.get("training", {}).get("learning_rate", "N/A")
        batch_size = config.get("training", {}).get("batch_size", "N/A")
        table_data = [
            ["Run ID", self.run_id],
            ["Strategy", "head_tune"],
            ["Epochs", f"{num_epochs} (best: {best_epoch})"],
            ["LR", str(lr)],
            ["Batch Size", str(batch_size)],
            ["", ""],
            [
                "Val Shape F1",
                f"{history['val_shape_f1'][best_epoch-1]:.4f}",
            ],
            [
                "Val Color F1",
                f"{history['val_color_f1'][best_epoch-1]:.4f}",
            ],
            ["Val Overall F1", f"{best_val_f1:.4f}"],
        ]

        if test_metrics is not None:
            table_data.extend([
                ["", ""],
                ["Test Shape F1", f"{test_metrics['shape_macro_f1']:.4f}"],
                ["Test Color F1", f"{test_metrics['color_macro_f1']:.4f}"],
                ["Test Overall F1", f"{test_metrics['overall_macro_f1']:.4f}"],
            ])

        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        axes[1, 1].set_title("Results Summary")

        plt.tight_layout()
        plt.savefig(
            plots_dir / f"{self.run_id}_summary.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()
        print(f"✓ Saved: {self.run_id}_summary.png")

    # ------------------------------------------------------------------
    # Prediction categorization
    # ------------------------------------------------------------------

    def save_predictions(
        self, preds: dict, test_csv_path, filenames_col: str = "rxnavImageFileName"
    ) -> None:
        """Categorize predictions and save error analysis files.

        Saves per-category JSON samples and a detailed CSV of all predictions.

        Args:
            preds: Output from ``collect_predictions()``.
            test_csv_path: Path to the test CSV (for filenames).
            filenames_col: Column name for image filenames.
        """
        pred_dir = self.paths["predictions"]
        test_df = pd.read_csv(test_csv_path)

        if filenames_col not in test_df.columns:
            filenames_col = "filename"
        filenames = test_df[filenames_col].tolist()

        shape_preds = preds["shape_preds"]
        shape_targets = preds["shape_targets"]
        shape_probs = preds["shape_probs"]
        color_preds = preds["color_preds"]
        color_targets = preds["color_targets"]
        color_probs = preds["color_probs"]

        predictions = []
        for i in range(len(shape_preds)):
            shape_pred_idx = int(shape_preds[i])
            shape_true_idx = int(shape_targets[i])
            shape_conf = float(shape_probs[i, shape_pred_idx])

            color_pred_names = [
                self.color_class_names[j]
                for j in range(self.num_color_classes)
                if color_preds[i, j] == 1
            ]
            color_true_names = [
                self.color_class_names[j]
                for j in range(self.num_color_classes)
                if color_targets[i, j] == 1
            ]
            color_conf = (
                float(
                    np.mean(
                        [
                            color_probs[i, j]
                            for j in range(self.num_color_classes)
                            if color_preds[i, j] == 1
                        ]
                    )
                )
                if color_pred_names
                else 0.0
            )

            shape_correct = shape_pred_idx == shape_true_idx
            color_correct = np.array_equal(
                color_preds[i], color_targets[i]
            )

            # Categorize
            if shape_correct and color_correct:
                category = "correct_samples"
            elif not shape_correct:
                category = "wrong_shape"
            elif not color_correct:
                category = "wrong_color"
            else:
                category = "wrong_shape"

            if shape_conf < 0.5 or color_conf < 0.5:
                category = "low_confidence"

            record = {
                "image_path": (
                    filenames[i] if i < len(filenames) else f"sample_{i}"
                ),
                "category": category,
                "ground_truth": {
                    "shape": self.shape_class_names[shape_true_idx],
                    "color": color_true_names,
                },
                "prediction": {
                    "shape": {
                        "label": self.shape_class_names[shape_pred_idx],
                        "confidence": round(shape_conf, 4),
                    },
                    "color": {
                        "labels": color_pred_names,
                        "confidence": round(color_conf, 4),
                    },
                },
                "correct": {
                    "shape": shape_correct,
                    "color": color_correct,
                },
            }
            predictions.append(record)

        # Save per-category JSON (max 50 each)
        category_counts = Counter(p["category"] for p in predictions)
        for cat in [
            "correct_samples",
            "wrong_shape",
            "wrong_color",
            "low_confidence",
        ]:
            cat_preds = [
                p for p in predictions if p["category"] == cat
            ][:50]
            cat_path = pred_dir / cat / "samples.json"
            cat_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cat_path, "w", encoding="utf-8") as f:
                json.dump(cat_preds, f, indent=2, ensure_ascii=False)

        # Save all predictions as CSV
        pred_rows = []
        for p in predictions:
            pred_rows.append(
                {
                    "image_path": p["image_path"],
                    "category": p["category"],
                    "true_shape": p["ground_truth"]["shape"],
                    "pred_shape": p["prediction"]["shape"]["label"],
                    "shape_confidence": p["prediction"]["shape"][
                        "confidence"
                    ],
                    "true_colors": "|".join(p["ground_truth"]["color"]),
                    "pred_colors": "|".join(
                        p["prediction"]["color"]["labels"]
                    ),
                    "color_confidence": p["prediction"]["color"][
                        "confidence"
                    ],
                    "shape_correct": p["correct"]["shape"],
                    "color_correct": p["correct"]["color"],
                }
            )

        pd.DataFrame(pred_rows).to_csv(
            pred_dir / "test_predictions_detailed.csv", index=False
        )

        print(f"✓ Predictions saved to: {pred_dir}")
        print("  Category distribution:")
        for cat, count in category_counts.items():
            print(
                f"    {cat}: {count} "
                f"({count / len(predictions) * 100:.1f}%)"
            )
