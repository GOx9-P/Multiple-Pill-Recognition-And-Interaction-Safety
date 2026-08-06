import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

def find_best_thresholds(val_targets: np.ndarray, val_probs: np.ndarray, num_classes: int):
    best_thresholds = []
    for i in range(num_classes):
        b_thresh, b_f1 = 0.5, 0.0
        for thresh in np.arange(0.1, 0.9, 0.05):
            score = f1_score(val_targets[:, i], (val_probs[:, i] > thresh).astype(int), zero_division=0)
            if score > b_f1:
                b_f1 = score
                b_thresh = float(thresh)
        best_thresholds.append(b_thresh)
    return best_thresholds

def save_evaluation_plots(df_history, shape_targets, shape_preds, config):
    plt.figure(figsize=(8, 5))
    plt.plot(df_history["epoch"], df_history["train_loss"], label="Train Loss", color="blue", marker="o")
    plt.plot(df_history["epoch"], df_history["val_loss"], label="Val Loss", color="red", marker="o")
    plt.title(f"{config.RUN_ID} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.PLOT_DIR / f"{config.RUN_ID}_loss_curve.png", dpi=300)
    plt.close()

    cm = confusion_matrix(shape_targets, shape_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Shape Confusion Matrix ({config.RUN_ID})")
    plt.savefig(config.PLOT_DIR / f"{config.RUN_ID}_shape_confusion_matrix.png", dpi=300)
    plt.close()