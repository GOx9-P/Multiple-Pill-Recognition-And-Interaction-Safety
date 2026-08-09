"""
Artifact generation utilities for standardized config, manifest, and runtime files.

Ensures both head-tune and last-blocks produce identical artifact formats
with all required fields.
"""

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml


def save_config_yaml(
    path: Path,
    run_id: str,
    module: str,
    train_strategy: str,
    seed: int,
    model_config: Dict,
    training_config: Dict,
    frozen_layers: List[str],
    trainable_layers: List[str],
    label_mapping_file: str,
    augmentation: Dict,
    scheduler_config: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> None:
    """Save training configuration as a YAML file.

    Args:
        path: Output file path.
        run_id: Unique run identifier.
        module: Module name (e.g. "attribute_resnet18_head_tune").
        train_strategy: "head_tune" or "last_blocks_finetune".
        seed: Training seed.
        model_config: Model architecture details.
        training_config: Hyperparameters (epochs, batch_size, lr, etc.).
        frozen_layers: List of frozen layer names.
        trainable_layers: List of trainable layer names.
        label_mapping_file: Relative path to label mapping JSON.
        augmentation: Augmentation policy details.
        scheduler_config: LR scheduler configuration.
        extra: Additional config entries.
    """
    config = {
        "run_id": run_id,
        "module": module,
        "train_strategy": train_strategy,
        "tasks": ["shape", "color"],
        "seed": seed,
        "model": model_config,
        "training": training_config,
        "frozen_layers": frozen_layers,
        "trainable_layers": trainable_layers,
        "label_mapping_file": label_mapping_file,
        "augmentation": augmentation,
    }

    if scheduler_config:
        config["scheduler"] = scheduler_config

    if extra:
        config.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def compute_file_hash(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        return "file_not_found"


def save_dataset_manifest(
    path: Path,
    run_id: str,
    dataset_name: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    train_count: int,
    val_count: int,
    test_count: int,
    num_shape_classes: int,
    num_color_classes: int,
    label_mapping_file: str,
    class_distribution: Dict[str, Dict[str, int]],
    leakage_check_passed: bool,
    leakage_check_details: Dict,
    transform_repr: str,
    group_key: str = "NDC11",
    extra: Optional[Dict] = None,
) -> None:
    """Save dataset manifest as JSON with all required fields.

    Args:
        path: Output file path.
        run_id: Unique run identifier.
        dataset_name: Name of the dataset.
        train_csv, val_csv, test_csv: Paths to CSV splits.
        train_count, val_count, test_count: Sample counts per split.
        num_shape_classes, num_color_classes: Number of classes.
        label_mapping_file: Path to label mapping JSON.
        class_distribution: {"shape": {...}, "color": {...}}.
        leakage_check_passed: ACTUAL result from check_split_leakage().
        leakage_check_details: Details dict from check_split_leakage().
        group_key: Column used for group splitting.
        extra: Additional manifest entries.
    """
    manifest = {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "num_shape_classes": num_shape_classes,
        "num_color_classes": num_color_classes,
        "label_mapping_file": str(label_mapping_file),
        "class_distribution": class_distribution,
        "split_policy": {
            "group_key": group_key,
            "leakage_check_passed": leakage_check_passed,
            "leakage_check_notes": leakage_check_details,
            "provenance": {
                "train_csv_sha256": compute_file_hash(train_csv),
                "val_csv_sha256": compute_file_hash(val_csv),
                "test_csv_sha256": compute_file_hash(test_csv),
                "augmentation_transform_repr": transform_repr,
            },
        },
    }

    if extra:
        manifest.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def save_runtime_info(
    path: Path,
    run_id: str,
    module: str,
    device: torch.device,
    started_at: datetime,
    finished_at: datetime,
    total_train_time_minutes: float,
    avg_epoch_time_seconds: float,
    best_epoch: int,
    best_metric: float,
    num_epochs_run: int,
) -> None:
    """Save runtime information as a text file.

    Includes system info (Python version, PyTorch version, GPU name, CUDA status).
    """
    gpu_name = "N/A"
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)

    info_lines = [
        f"run_id: {run_id}",
        f"module: {module}",
        f"started_at: {started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"finished_at: {finished_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"device: {device}",
        f"gpu_name: {gpu_name}",
        f"python_version: {sys.version.split()[0]}",
        f"torch_version: {torch.__version__}",
        f"cuda_available: {cuda_available}",
        f"platform: {platform.platform()}",
        f"total_train_time_minutes: {total_train_time_minutes:.2f}",
        f"avg_epoch_time_seconds: {avg_epoch_time_seconds:.2f}",
        f"num_epochs_run: {num_epochs_run}",
        f"best_epoch: {best_epoch}",
        f"best_overall_macro_f1: {best_metric:.4f}",
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(info_lines) + "\n")
