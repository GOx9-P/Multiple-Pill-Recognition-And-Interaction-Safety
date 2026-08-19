"""
Project path configuration for pill attribute recognition.

Note: RUN_ID and stage-specific paths should be passed from entrypoints,
NOT hard-coded here.
"""

import torch
from pathlib import Path


class AttributeConfig:
    """Base project paths. Stage-specific config comes from entrypoints."""

    # --- Base Paths (computed from project root) ---
    PROJECT_ROOT = Path(__file__).resolve().parents[5]

    BASE_DIR = PROJECT_ROOT / "data"
    COMBINED_DIR = BASE_DIR / "splits" / "nih_attribute"  # Fixed: was "data/data/splits"
    IMG_DIR = BASE_DIR / "image_all" / "nih_attribute"    # Fixed: was "data/data/image_all"

    # --- Common Settings ---
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    IMAGE_SIZE = 224

    @classmethod
    def get_experiment_paths(cls, module_name: str, run_id: str) -> dict:
        """Generate experiment directory paths for a given module and run.

        Args:
            module_name: e.g. "attribute_resnet18_head_tune"
            run_id: e.g. "attr_head_v2"

        Returns:
            Dict with keys: exp_dir, checkpoints, logs, metrics, plots, predictions.
        """
        exp_dir = cls.PROJECT_ROOT / "experiments" / module_name
        return {
            "exp_dir": exp_dir,
            "checkpoints": exp_dir / "checkpoints",
            "logs": exp_dir / "logs",
            "metrics": exp_dir / "metrics",
            "plots": exp_dir / "plots",
            "predictions": exp_dir / "predictions" / run_id,
        }

    @classmethod
    def setup_directories(cls, paths: dict) -> None:
        """Create all directories in paths dict."""
        for p in paths.values():
            if isinstance(p, Path):
                p.mkdir(parents=True, exist_ok=True)