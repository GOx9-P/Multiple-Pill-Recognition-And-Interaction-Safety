"""Workflow train, calibrate va evaluate cho attribute ResNet18."""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from pill_safety.cv.attribute.datasets.color_dataset import ColorDataset
from pill_safety.cv.attribute.datasets.shape_dataset import ShapeDataset
from pill_safety.cv.attribute.evaluators.metric_evaluator import evaluate
from pill_safety.cv.attribute.models.resnet18_multitask import MultiTaskResNet18
from pill_safety.cv.attribute.trainers.multitask_trainer import MultiTaskTrainer
from pill_safety.cv.attribute.utils.transforms import get_color_transforms, get_shape_transforms
from pill_safety.cv.attribute.training.data_contract import COLOR_COLUMNS, validate_attribute_data


def load_config(config_path: Path) -> dict:
    """Doc YAML config va bao loi ro rang neu file khong hop le."""
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid training config: {config_path}")
    return config


def resolve_paths(config: dict, data_root_override: str | None, output_root_override: str | None) -> tuple[dict[str, Path], Path]:
    """Resolve path tu data root Kaggle/local ma khong hard-code duong dan may ca nhan."""
    data_root = Path(data_root_override or config["data"].get("root", "data")).expanduser()
    output_root = Path(output_root_override or config.get("output_root", "experiments")).expanduser()
    paths = {}
    for task in ("shape", "color"):
        task_config = config["data"][task]
        paths[f"{task}_image_dir"] = data_root / task_config["image_dir"]
        for split in ("train", "val", "test"):
            paths[f"{task}_{split}_csv"] = data_root / task_config[f"{split}_csv"]
    paths["label_mapping"] = data_root / config["data"]["label_mapping"]
    return paths, output_root


def _load_mapping(path: Path) -> tuple[dict[str, int], list[str]]:
    """Nap mapping va xac dinh chinh xac so class cua hai head."""
    if not path.is_file():
        raise FileNotFoundError(f"Label mapping not found: {path}")
    mapping = json.loads(path.read_text(encoding="utf-8"))
    shape_mapping = mapping["shape_classification"]
    colors = mapping["color_multilabel"]["labels"]
    if colors != [column.removeprefix("label_") for column in COLOR_COLUMNS]:
        raise ValueError("Color label_mapping order does not match ColorDataset column order.")
    return shape_mapping, colors


def _seed_everything(seed: int) -> None:
    """Dat seed cho cac RNG duoc dung trong train."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit() -> str:
    """Lay commit hien tai neu repo co Git; khong lam training that bai neu khong co."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _artifact_paths(output_root: Path, module: str, run_id: str) -> dict[str, Path]:
    """Tao cac thu muc artifact dung contract train_request."""
    base = output_root / module
    directories = {name: base / name for name in ("checkpoints", "logs", "metrics", "plots", "predictions")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    directories["prediction_run"] = directories["predictions"] / run_id
    for category in ("correct_samples", "wrong_shape", "wrong_color", "low_confidence"):
        (directories["prediction_run"] / category).mkdir(parents=True, exist_ok=True)
    directories["base"] = base
    return directories


def _build_loaders(paths: dict[str, Path], config: dict, split: str, shuffle: bool) -> tuple[DataLoader, DataLoader]:
    """Khoi tao hai dataloader doc lap cho shape va color."""
    image_size = config["training"]["image_size"]
    workers = config["training"].get("num_workers", 2)
    pin_memory = torch.cuda.is_available()
    shape_set = ShapeDataset(paths[f"shape_{split}_csv"], paths["shape_image_dir"], get_shape_transforms(image_size))
    color_set = ColorDataset(paths[f"color_{split}_csv"], paths["color_image_dir"], get_color_transforms(image_size))
    return (
        DataLoader(shape_set, batch_size=config["sampling"]["shape_batch_size"], shuffle=shuffle, num_workers=workers, pin_memory=pin_memory),
        DataLoader(color_set, batch_size=config["sampling"]["color_batch_size"], shuffle=shuffle, num_workers=workers, pin_memory=pin_memory),
    )


def _losses(paths: dict[str, Path], num_shape: int, device: torch.device) -> tuple[nn.Module, nn.Module, dict]:
    """Tinh class/positive weights chi tren train split de xu ly mat can bang."""
    shape_frame = pd.read_csv(paths["shape_train_csv"])
    shape_counts = np.bincount(shape_frame["label_shape"].astype(int), minlength=num_shape)
    shape_weights = len(shape_frame) / (num_shape * np.clip(shape_counts, 1, None))
    color_frame = pd.read_csv(paths["color_train_csv"])
    color_matrix = color_frame[COLOR_COLUMNS].to_numpy(dtype=np.float32)
    positive = color_matrix.sum(axis=0)
    pos_weight = (len(color_matrix) - positive) / np.clip(positive, 1, None)
    return (
        nn.CrossEntropyLoss(weight=torch.tensor(shape_weights, dtype=torch.float32, device=device)),
        nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)),
        {"shape_counts": shape_counts.tolist(), "color_positive_counts": positive.tolist()},
    )


def _plot_training(log_frame: pd.DataFrame, plot_dir: Path, run_id: str) -> None:
    """Xuat loss, metric va summary co ca Overall Macro F1."""
    epochs = log_frame["epoch"]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, log_frame["train_loss"], label="Train loss")
    plt.plot(epochs, log_frame["val_loss"], label="Validation loss")
    plt.legend(); plt.grid(True); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.tight_layout()
    plt.savefig(plot_dir / f"{run_id}_loss_curve.png"); plt.close()
    plt.figure(figsize=(8, 5))
    for column, label in (("val_shape_f1", "Shape Macro F1"), ("val_color_f1", "Color Macro F1"), ("val_combined_f1", "Overall Macro F1")):
        plt.plot(epochs, log_frame[column], label=label)
    plt.legend(); plt.grid(True); plt.xlabel("Epoch"); plt.ylabel("F1"); plt.ylim(0, 1); plt.tight_layout()
    plt.savefig(plot_dir / f"{run_id}_metric_curve.png"); plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(epochs, log_frame["train_loss"], label="Train"); axes[0].plot(epochs, log_frame["val_loss"], label="Val"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs, log_frame["val_shape_f1"], label="Shape"); axes[1].plot(epochs, log_frame["val_color_f1"], label="Color"); axes[1].plot(epochs, log_frame["val_combined_f1"], label="Overall"); axes[1].legend(); axes[1].grid(True)
    fig.tight_layout(); fig.savefig(plot_dir / f"{run_id}_summary.png"); plt.close(fig)


def train(config: dict, data_root_override=None, output_root_override=None, run_id_override=None, pretrained_override=None) -> dict:
    """Chay mot phase head-tune hoac last-block va luu day du artifact."""
    paths, output_root = resolve_paths(config, data_root_override, output_root_override)
    run = config["run"].copy()
    run_id = run_id_override or run.get("run_id") or datetime.now().strftime("attr_%Y%m%d_%H%M%S")
    module = run["module"]
    strategy = config["training"]["strategy"]
    if strategy not in {"head_tune", "last_blocks_finetune"}:
        raise ValueError(f"Unsupported strategy: {strategy}")
    if config["sampling"]["shape_batches_per_step"] != 1:
        raise ValueError("This training contract requires exactly one shape batch per optimizer step.")
    _seed_everything(config["training"]["seed"])
    artifacts = _artifact_paths(output_root, module, run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = validate_attribute_data(paths, verify_images=config["data"].get("verify_images", True))
    shape_mapping, color_labels = _load_mapping(paths["label_mapping"])
    criterion_shape, criterion_color, distribution = _losses(paths, len(shape_mapping), device)
    manifest.update({
        "run_id": run_id,
        "label_mapping_file": str(paths["label_mapping"]),
        "class_distribution": distribution,
        "sampling_policy": {
            "shape_batches_per_optimizer_step": config["sampling"]["shape_batches_per_step"],
            "color_batches_per_optimizer_step": config["sampling"]["color_batches_per_step"],
            "shape_batch_size": config["sampling"]["shape_batch_size"],
            "color_batch_size": config["sampling"]["color_batch_size"],
            "gradient_accumulation": "shape_loss + lambda_color * mean(color_batch_losses), then one optimizer.step",
        },
    })
    (artifacts["logs"] / f"{run_id}_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    config_snapshot = {**config, "run": {**run, "run_id": run_id}, "resolved_paths": {key: str(value) for key, value in paths.items()}}
    (artifacts["logs"] / f"{run_id}_config.yaml").write_text(yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8")
    shape_train, color_train = _build_loaders(paths, config, "train", shuffle=True)
    shape_val, color_val = _build_loaders(paths, config, "val", shuffle=False)
    model = MultiTaskResNet18(len(shape_mapping), len(color_labels), pretrained=strategy == "head_tune").to(device)

    pretrained_from = pretrained_override or config["training"].get("pretrained_from")
    if strategy == "last_blocks_finetune":
        if not pretrained_from or not Path(pretrained_from).is_file():
            raise FileNotFoundError(f"Last-block fine-tune requires a valid head checkpoint: {pretrained_from}")
        model.load_state_dict(torch.load(pretrained_from, map_location=device, weights_only=True))

    placeholder_optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    trainer = MultiTaskTrainer(model, placeholder_optimizer, criterion_shape, criterion_color, device, config["training"].get("lambda_color", 1.0))
    trainable_names = trainer.set_training_strategy(strategy, config["training"].get("trainable_layers"))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0001),
    )
    trainer.optimizer = optimizer
    scheduler_config = config["training"].get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config.get("mode", "max"),
        factor=scheduler_config.get("factor", 0.5),
        patience=scheduler_config.get("patience", 2),
    )
    steps = config["sampling"].get("steps_per_epoch") or max(math.ceil(len(shape_train.dataset) / config["sampling"]["shape_batch_size"]), math.ceil(len(color_train.dataset) / (config["sampling"]["color_batch_size"] * config["sampling"]["color_batches_per_step"])))

    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.time(); best_score = -1.0; best_metrics = {}; best_epoch = 0; logs = []
    for epoch in range(1, config["training"]["epochs"] + 1):
        epoch_start = time.time()
        train_metrics = trainer.train_epoch(shape_train, color_train, steps, config["sampling"]["color_batches_per_step"])
        val_metrics = evaluate(model, shape_val, color_val, criterion_shape, criterion_color, device)
        scheduler.step(val_metrics["combined_f1"])
        is_best = val_metrics["combined_f1"] >= best_score
        if is_best:
            best_score, best_metrics, best_epoch = val_metrics["combined_f1"], val_metrics, epoch
            torch.save(model.state_dict(), artifacts["checkpoints"] / f"{run_id}_best.pt")
        logs.append({
            "epoch": epoch,
            **train_metrics,
            "val_loss": val_metrics["val_loss"],
            "val_shape_loss": val_metrics["shape_loss"],
            "val_color_loss": val_metrics["color_loss"],
            "val_shape_f1": val_metrics["shape_f1"],
            "val_color_f1": val_metrics["color_f1"],
            "val_combined_f1": val_metrics["combined_f1"],
            "val_shape_acc": val_metrics["shape_acc"],
            "val_color_acc": val_metrics["color_acc"],
            "val_combined_acc": val_metrics["combined_acc"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(time.time() - epoch_start, 2),
            "best_metric": best_score,
            "is_best": is_best,
        })
        print(
            f"Epoch [{epoch}/{config['training']['epochs']}] "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"shape_loss={train_metrics['shape_loss']:.4f} "
            f"color_loss={train_metrics['color_loss']:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"shape_f1={val_metrics['shape_f1']:.4f} "
            f"color_f1={val_metrics['color_f1']:.4f} "
            f"overall_f1={val_metrics['combined_f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6g} "
            f"time={time.time() - epoch_start:.1f}s "
            f"best={best_score:.4f}{' *' if is_best else ''}",
            flush=True,
        )

    torch.save(model.state_dict(), artifacts["checkpoints"] / f"{run_id}_last.pt")
    log_frame = pd.DataFrame(logs); log_frame.to_csv(artifacts["logs"] / f"{run_id}_train_log.csv", index=False)
    _plot_training(log_frame, artifacts["plots"], run_id)
    best_model, best_shape_mapping, best_color_labels = _load_model_for_evaluation(config, paths, artifacts["checkpoints"] / f"{run_id}_best.pt", device)
    best_shape_labels = [name for name, _ in sorted(best_shape_mapping.items(), key=lambda item: item[1])]
    per_class_metrics = _validation_per_class(best_model, paths, config, device, best_shape_labels, best_color_labels)
    val_payload = {"run_id": run_id, "module": module, "split": "val", "best_epoch": best_epoch, "best_checkpoint": str(artifacts["checkpoints"] / f"{run_id}_best.pt"), "selection_metric": "overall_macro_f1_at_0.5", "metrics": best_metrics, "per_class_metrics": per_class_metrics, "label_mapping_file": str(paths["label_mapping"])}
    (artifacts["metrics"] / f"{run_id}_val_metrics.json").write_text(json.dumps(val_payload, indent=2), encoding="utf-8")
    runtime = {"run_id": run_id, "module": module, "author": run.get("author", "unknown"), "started_at": started_at, "finished_at": datetime.now().isoformat(timespec="seconds"), "git_commit": _git_commit(), "device": str(device), "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "total_train_seconds": round(time.time() - start, 2), "python_version": sys.version.split()[0], "torch_version": torch.__version__, "trainable_parameters": trainable_names, "pretrained_from": str(pretrained_from) if pretrained_from else None}
    (artifacts["logs"] / f"{run_id}_runtime.txt").write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
    return {"run_id": run_id, "checkpoint": str(artifacts["checkpoints"] / f"{run_id}_best.pt"), "module": module}


def _load_model_for_evaluation(config: dict, paths: dict[str, Path], checkpoint_path: str | Path, device: torch.device) -> tuple[MultiTaskResNet18, dict[str, int], list[str]]:
    """Khoi tao dung model contract va chi nap checkpoint duoc chi dinh."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    shape_mapping, color_labels = _load_mapping(paths["label_mapping"])
    model = MultiTaskResNet18(len(shape_mapping), len(color_labels), pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model, shape_mapping, color_labels


def _collect_task_outputs(model: MultiTaskResNet18, loader: DataLoader, task_type: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Thu thap logits va label theo dung thu tu dataframe cua loader khong shuffle."""
    logits_parts, target_parts = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits_parts.append(model(images.to(device, non_blocking=True), task_type=task_type).cpu().numpy())
            target_parts.append(labels.cpu().numpy())
    return np.concatenate(logits_parts), np.concatenate(target_parts)


def _best_threshold(y_true: np.ndarray, probabilities: np.ndarray, candidates: list[float]) -> tuple[float, float]:
    """Chon nguong F1 tot nhat tren validation; neu hoa thi gan 0.5 hon."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        score = float(f1_score(y_true, probabilities >= threshold, zero_division=0))
        if score > best_f1 or (math.isclose(score, best_f1) and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_threshold, best_f1 = threshold, score
    return best_threshold, best_f1


def calibrate_color_thresholds(config: dict, checkpoint_path: str | Path, data_root_override=None, output_root_override=None, run_id_override=None) -> dict:
    """Tune tung color threshold tren validation va luu artifact rieng cho checkpoint."""
    paths, output_root = resolve_paths(config, data_root_override, output_root_override)
    run_id = run_id_override or config["run"].get("run_id")
    if not run_id:
        raise ValueError("run_id is required when calibrating color thresholds.")
    artifacts = _artifact_paths(output_root, config["run"]["module"], run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, color_labels = _load_model_for_evaluation(config, paths, checkpoint_path, device)
    _, color_val = _build_loaders(paths, config, "val", shuffle=False)
    logits, targets = _collect_task_outputs(model, color_val, "color", device)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    candidates = [round(float(value), 4) for value in config["calibration"].get("threshold_candidates", np.arange(0.05, 1.0, 0.05).tolist())]
    thresholds, class_f1 = {}, {}
    for index, color in enumerate(color_labels):
        threshold, score = _best_threshold(targets[:, index], probabilities[:, index], candidates)
        thresholds[color] = threshold
        class_f1[color] = round(score, 4)
    payload = {
        "run_id": run_id,
        "checkpoint": str(Path(checkpoint_path)),
        "split": "validation",
        "selection_rule": "per_color_best_f1_with_tie_break_nearest_0.5",
        "thresholds": thresholds,
        "validation_f1_at_selected_threshold": class_f1,
        "label_mapping_file": str(paths["label_mapping"]),
    }
    output_path = artifacts["checkpoints"] / f"{run_id}_optimal_thresholds.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"path": str(output_path), **payload}


def _load_thresholds(path: str | Path, color_labels: list[str]) -> np.ndarray:
    """Nap threshold da tune va kiem tra day du thu tu color theo label mapping."""
    threshold_path = Path(path)
    if not threshold_path.is_file():
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}")
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds", {})
    missing = [color for color in color_labels if color not in thresholds]
    if missing:
        raise ValueError(f"Threshold file is missing colors: {missing}")
    return np.array([float(thresholds[color]) for color in color_labels], dtype=np.float32)


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Tinh F1 tung class/thuoc tinh de bao cao va so sanh class hiem."""
    metric_labels = list(range(len(labels))) if y_true.ndim == 1 else None
    scores = f1_score(y_true, y_pred, average=None, labels=metric_labels, zero_division=0)
    return {label: round(float(score), 4) for label, score in zip(labels, scores)}


def _validation_per_class(model: MultiTaskResNet18, paths: dict[str, Path], config: dict, device: torch.device, shape_labels: list[str], color_labels: list[str]) -> dict[str, dict[str, float]]:
    """Tinh per-class F1 cua best checkpoint tren validation de chan giam class hiem."""
    shape_val, color_val = _build_loaders(paths, config, "val", shuffle=False)
    shape_logits, shape_true = _collect_task_outputs(model, shape_val, "shape", device)
    color_logits, color_true = _collect_task_outputs(model, color_val, "color", device)
    return {
        "shape": _per_class_f1(shape_true, shape_logits.argmax(axis=1), shape_labels),
        "color": _per_class_f1(color_true, (1.0 / (1.0 + np.exp(-color_logits)) >= 0.5).astype(int), color_labels),
    }


def _save_test_plots(shape_true: np.ndarray, shape_pred: np.ndarray, color_true: np.ndarray, color_pred: np.ndarray, shape_labels: list[str], color_labels: list[str], plot_dir: Path, run_id: str) -> None:
    """Ve confusion matrix shape va F1 tung color cho artifact bao cao."""
    matrix = confusion_matrix(shape_true, shape_pred, labels=list(range(len(shape_labels))))
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(shape_labels)), shape_labels, rotation=45, ha="right")
    axis.set_yticks(range(len(shape_labels)), shape_labels)
    axis.set_xlabel("Predicted"); axis.set_ylabel("True"); axis.set_title("Shape confusion matrix")
    for row in range(len(shape_labels)):
        for column in range(len(shape_labels)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis); figure.tight_layout()
    figure.savefig(plot_dir / f"{run_id}_shape_confusion_matrix.png"); plt.close(figure)
    color_scores = list(_per_class_f1(color_true, color_pred, color_labels).values())
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(color_labels, color_scores)
    axis.set_ylim(0, 1); axis.set_ylabel("F1"); axis.set_title("Color F1 per class")
    axis.tick_params(axis="x", rotation=45); figure.tight_layout()
    figure.savefig(plot_dir / f"{run_id}_color_f1_per_class.png"); plt.close(figure)


def _export_prediction_examples(dataset, task_name: str, predictions: np.ndarray, targets: np.ndarray, confidences: np.ndarray, output_dir: Path, max_items: int) -> None:
    """Luu mot tap vi du dung/sai/low-confidence, khong chep toan bo dataset lon."""
    buckets = {"correct_samples": [], "wrong_shape": [], "wrong_color": [], "low_confidence": []}
    for index, (prediction, target, confidence) in enumerate(zip(predictions, targets, confidences)):
        is_correct = bool(prediction == target) if task_name == "shape" else bool(np.array_equal(prediction, target))
        if is_correct:
            buckets["correct_samples"].append(index)
        else:
            buckets["wrong_shape" if task_name == "shape" else "wrong_color"].append(index)
        if confidence < 0.70:
            buckets["low_confidence"].append(index)
    frame = dataset.data_frame if task_name == "shape" else dataset.df
    image_dir = Path(dataset.img_dir)
    for category, indices in buckets.items():
        for index in indices[:max_items]:
            filename = str(frame.iloc[index]["rximageFileName"])
            source = image_dir / filename
            destination = output_dir / category / f"{task_name}_{index:06d}_{Path(filename).name}"
            if source.is_file():
                shutil.copy2(source, destination)
        metadata = [{"index": int(index), "filename": str(frame.iloc[index]["rximageFileName"])} for index in indices[:max_items]]
        (output_dir / category / f"{task_name}_examples.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def evaluate_test(config: dict, checkpoint_path: str | Path, thresholds_path: str | Path, data_root_override=None, output_root_override=None, run_id_override=None) -> dict:
    """Danh gia test mot lan bang checkpoint va threshold da chon tu validation."""
    paths, output_root = resolve_paths(config, data_root_override, output_root_override)
    run_id = run_id_override or config["run"].get("run_id")
    if not run_id:
        raise ValueError("run_id is required when evaluating test.")
    artifacts = _artifact_paths(output_root, config["run"]["module"], run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, shape_mapping, color_labels = _load_model_for_evaluation(config, paths, checkpoint_path, device)
    shape_labels = [name for name, _ in sorted(shape_mapping.items(), key=lambda item: item[1])]
    thresholds = _load_thresholds(thresholds_path, color_labels)
    shape_test, color_test = _build_loaders(paths, config, "test", shuffle=False)
    shape_logits, shape_true = _collect_task_outputs(model, shape_test, "shape", device)
    color_logits, color_true = _collect_task_outputs(model, color_test, "color", device)
    shape_pred = shape_logits.argmax(axis=1)
    color_probabilities = 1.0 / (1.0 + np.exp(-color_logits))
    color_pred = (color_probabilities >= thresholds).astype(int)
    shape_f1 = float(f1_score(shape_true, shape_pred, average="macro", zero_division=0))
    color_f1 = float(f1_score(color_true, color_pred, average="macro", zero_division=0))
    payload = {
        "run_id": run_id,
        "module": config["run"]["module"],
        "split": "test",
        "checkpoint": str(Path(checkpoint_path)),
        "thresholds_file": str(Path(thresholds_path)),
        "label_mapping_file": str(paths["label_mapping"]),
        "metrics": {
            "shape_macro_f1": round(shape_f1, 4),
            "color_macro_f1": round(color_f1, 4),
            "dosage_form_macro_f1": None,
            "scoreline_macro_f1": None,
            "overall_macro_f1": round((shape_f1 + color_f1) / 2.0, 4),
            "shape_accuracy": round(float(accuracy_score(shape_true, shape_pred)), 4),
            "color_subset_accuracy": round(float(accuracy_score(color_true, color_pred)), 4),
        },
        "per_class_metrics": {
            "shape": _per_class_f1(shape_true, shape_pred, shape_labels),
            "color": _per_class_f1(color_true, color_pred, color_labels),
        },
    }
    result_path = artifacts["metrics"] / f"{run_id}_test_metrics.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _save_test_plots(shape_true, shape_pred, color_true, color_pred, shape_labels, color_labels, artifacts["plots"], run_id)
    _export_prediction_examples(shape_test.dataset, "shape", shape_pred, shape_true, shape_logits.max(axis=1), artifacts["prediction_run"], config["reporting"].get("max_prediction_artifacts", 50))
    _export_prediction_examples(color_test.dataset, "color", color_pred, color_true, np.maximum(color_probabilities, 1.0 - color_probabilities).mean(axis=1), artifacts["prediction_run"], config["reporting"].get("max_prediction_artifacts", 50))
    return {"path": str(result_path), **payload}


def compare_validation_runs(head_val_metrics_path: str | Path, last_val_metrics_path: str | Path, output_path: str | Path, max_per_class_f1_drop: float = 0.05) -> dict:
    """So sanh head va last-block bang validation, khong dung test de chon model."""
    head_path, last_path, destination = Path(head_val_metrics_path), Path(last_val_metrics_path), Path(output_path)
    if not head_path.is_file() or not last_path.is_file():
        raise FileNotFoundError("Both head and last-block validation metrics files are required.")
    head = json.loads(head_path.read_text(encoding="utf-8"))
    last = json.loads(last_path.read_text(encoding="utf-8"))
    head_metrics, last_metrics = head["metrics"], last["metrics"]
    delta = round(float(last_metrics["combined_f1"] - head_metrics["combined_f1"]), 4)
    drops = {}
    for task_name in ("shape", "color"):
        for label, head_score in head.get("per_class_metrics", {}).get(task_name, {}).items():
            last_score = last.get("per_class_metrics", {}).get(task_name, {}).get(label)
            if last_score is not None:
                drops[f"{task_name}:{label}"] = round(float(last_score - head_score), 4)
    worst_drop = min(drops.values(), default=0.0)
    selected = bool(delta > 0.0 and worst_drop >= -max_per_class_f1_drop)
    payload = {
        "selection_split": "validation",
        "head_tune_run_id": head["run_id"],
        "last_blocks_run_id": last["run_id"],
        "head_metrics": head_metrics,
        "last_blocks_metrics": last_metrics,
        "overall_macro_f1_delta": delta,
        "worst_per_class_f1_delta": worst_drop,
        "selected_for_inference": "last_blocks" if selected else "head_tune",
        "selection_rule": f"higher validation overall_macro_f1 and no class F1 drop below {-max_per_class_f1_drop:.2f}; test is reporting-only",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    figure, axis = plt.subplots(figsize=(7, 4))
    metric_names = ["shape_f1", "color_f1", "combined_f1"]
    positions = np.arange(len(metric_names))
    width = 0.35
    axis.bar(positions - width / 2, [head_metrics[name] for name in metric_names], width, label="Head tune")
    axis.bar(positions + width / 2, [last_metrics[name] for name in metric_names], width, label="Last blocks")
    axis.set_xticks(positions, ["Shape F1", "Color F1", "Overall F1"])
    axis.set_ylim(0, 1); axis.legend(); axis.grid(axis="y")
    figure.tight_layout(); figure.savefig(destination.with_suffix(".png")); plt.close(figure)
    return {"path": str(destination), **payload}
