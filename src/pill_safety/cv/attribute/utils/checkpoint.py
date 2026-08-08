"""
Checkpoint utilities for safe saving and loading.

Features:
    - Atomic save (temp file → os.replace)
    - Strict loading with DataParallel prefix stripping
    - Mapping hash verification on load
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import torch


def save_checkpoint(
    path: Path,
    epoch: int,
    model,
    optimizer,
    scheduler=None,
    best_metric: float = 0.0,
    label_mapping: Optional[Dict] = None,
    mapping_hash: str = "",
    extra: Optional[Dict] = None,
) -> None:
    """Save checkpoint atomically (write to temp, then os.replace).

    Args:
        path: Target checkpoint file path.
        epoch: Current epoch number.
        model: The model (nn.Module).
        optimizer: The optimizer.
        scheduler: Optional LR scheduler.
        best_metric: Best validation metric so far.
        label_mapping: Label mapping dict.
        mapping_hash: SHA-256 hash of the label mapping.
        extra: Any additional metadata to store.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "label_mapping": label_mapping,
        "mapping_hash": mapping_hash,
    }

    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    if extra:
        ckpt.update(extra)

    # Atomic save: write to temp file in same directory, then replace
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        os.close(fd)
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, str(path))
    except Exception:
        # Clean up temp file on failure; old checkpoint remains intact
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_checkpoint(
    path: Path,
    device: torch.device,
    expected_mapping_hash: str = "",
) -> Dict:
    """Load checkpoint with strict validation.

    Args:
        path: Checkpoint file path.
        device: Device to map tensors to.
        expected_mapping_hash: If non-empty, verify mapping hash matches.

    Returns:
        Checkpoint dictionary.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If mapping hash mismatch.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FATAL: Checkpoint not found: {path}\n"
            "Last-blocks fine-tune requires head-tune checkpoint."
        )

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Strip DataParallel 'module.' prefix if present
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        keys = list(state_dict.keys())
        if keys and all(k.startswith("module.") for k in keys):
            ckpt["model_state_dict"] = {
                k[len("module."):]: v for k, v in state_dict.items()
            }

    # Verify mapping hash if expected
    if expected_mapping_hash:
        ckpt_hash = ckpt.get("mapping_hash", "")
        if ckpt_hash != expected_mapping_hash:
            raise RuntimeError(
                f"FATAL: Label mapping hash mismatch!\n"
                f"  Expected: {expected_mapping_hash}\n"
                f"  Got:      {ckpt_hash}\n"
                "This means the checkpoint was trained with a different label mapping."
            )

    return ckpt
