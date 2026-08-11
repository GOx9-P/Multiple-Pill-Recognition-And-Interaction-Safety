"""
Shared utilities for pill attribute recognition experiments.

Provides logger setup, experiment directory initialization, seed management,
and artifact saving (config YAML, dataset manifest, runtime info).
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: The random seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(name: str, log_file) -> logging.Logger:
    """Create a file-based logger.

    Args:
        name: Logger name (used for ``logging.getLogger``).
        log_file: Path to the log file.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.FileHandler(log_file)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(message)s")
    )
    logger.addHandler(handler)
    return logger


def init_experiment_dirs(
    experiment_dir: Path, run_id: str
) -> Dict[str, Path]:
    """Create standard experiment subdirectories.

    Creates: ``checkpoints/``, ``logs/``, ``metrics/``, ``plots/``,
    ``predictions/<run_id>/`` with error-category subfolders.

    Args:
        experiment_dir: Root directory for this experiment.
        run_id: Unique identifier for this training run.

    Returns:
        Dictionary mapping subdirectory names to their ``Path`` objects.
    """
    subdirs = ["checkpoints", "logs", "metrics", "plots", "predictions"]
    paths = {}
    for sub in subdirs:
        path = experiment_dir / sub
        path.mkdir(parents=True, exist_ok=True)
        paths[sub] = path

    # Prediction category subfolders
    pred_dir = paths["predictions"] / run_id
    for folder in [
        "correct_samples",
        "wrong_shape",
        "wrong_color",
        "low_confidence",
    ]:
        (pred_dir / folder).mkdir(parents=True, exist_ok=True)

    return paths


def save_config_yaml(config: dict, path: Path) -> None:
    """Save run configuration as a YAML file.

    Args:
        config: Configuration dictionary.
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def save_dataset_manifest(manifest: dict, path: Path) -> None:
    """Save dataset manifest as a JSON file.

    Args:
        manifest: Manifest dictionary containing split counts,
            class distribution, leakage check info, etc.
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def save_runtime_info(info: dict, path: Path) -> None:
    """Save runtime environment information as a text file.

    Args:
        info: Dictionary of runtime key-value pairs.
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")


def get_device() -> torch.device:
    """Return the best available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_system_info() -> None:
    """Print Python, PyTorch, and CUDA version information."""
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
