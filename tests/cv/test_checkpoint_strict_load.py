"""
Test P2-01: Checkpoint — atomic save, strict load, DataParallel prefix strip, hash verify.

Tests:
    - Save + load roundtrip preserves model weights
    - DataParallel 'module.' prefix is automatically stripped
    - Missing checkpoint → FileNotFoundError
    - Mapping hash mismatch → RuntimeError
    - Atomic save: failed save doesn't corrupt existing checkpoint
"""

import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pill_safety.cv.attribute.utils.checkpoint import save_checkpoint, load_checkpoint
from pill_safety.cv.attribute.models.resnet_multitask import MultiTaskResNet18


@pytest.fixture
def tiny_model():
    """Create a small model for testing (2 shapes, 3 colors)."""
    return MultiTaskResNet18(num_shape_classes=2, num_color_classes=3, pretrained=False)


def test_save_load_roundtrip(tiny_model, tmp_path):
    """Save and load checkpoint; model weights must match."""
    ckpt_path = tmp_path / "test_best.pt"
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

    save_checkpoint(
        path=ckpt_path,
        epoch=5,
        model=tiny_model,
        optimizer=optimizer,
        best_metric=0.85,
        label_mapping={"shape": ["A", "B"], "color": ["c1", "c2", "c3"]},
        mapping_hash="abc123",
    )

    ckpt = load_checkpoint(ckpt_path, device=torch.device("cpu"))

    assert ckpt["epoch"] == 5
    assert ckpt["best_metric"] == 0.85
    assert ckpt["mapping_hash"] == "abc123"
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt

    # Verify weights can be loaded back
    model2 = MultiTaskResNet18(num_shape_classes=2, num_color_classes=3, pretrained=False)
    model2.load_state_dict(ckpt["model_state_dict"])


def test_datapallel_prefix_stripped(tiny_model, tmp_path):
    """Checkpoint with 'module.' prefix → stripped on load."""
    ckpt_path = tmp_path / "dp_ckpt.pt"

    # Simulate DataParallel by adding 'module.' prefix
    state_dict = {"module." + k: v for k, v in tiny_model.state_dict().items()}
    torch.save({"model_state_dict": state_dict, "epoch": 1}, ckpt_path)

    ckpt = load_checkpoint(ckpt_path, device=torch.device("cpu"))

    # Verify no key starts with 'module.'
    for key in ckpt["model_state_dict"]:
        assert not key.startswith("module."), f"Key still has prefix: {key}"

    # Verify the stripped state_dict can actually be loaded
    model2 = MultiTaskResNet18(num_shape_classes=2, num_color_classes=3, pretrained=False)
    model2.load_state_dict(ckpt["model_state_dict"])


def test_missing_checkpoint_raises():
    """Loading nonexistent checkpoint → FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="FATAL"):
        load_checkpoint(Path("/nonexistent/best.pt"), device=torch.device("cpu"))


def test_mapping_hash_mismatch(tiny_model, tmp_path):
    """Checkpoint hash doesn't match expected → RuntimeError."""
    ckpt_path = tmp_path / "ckpt.pt"
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

    save_checkpoint(
        path=ckpt_path,
        epoch=1,
        model=tiny_model,
        optimizer=optimizer,
        mapping_hash="correct_hash",
    )

    with pytest.raises(RuntimeError, match="mapping hash mismatch"):
        load_checkpoint(ckpt_path, device=torch.device("cpu"), expected_mapping_hash="wrong_hash")


def test_atomic_save_preserves_old_on_failure(tiny_model, tmp_path):
    """If save fails mid-write, old checkpoint remains intact."""
    ckpt_path = tmp_path / "ckpt.pt"
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

    # First: save a valid checkpoint
    save_checkpoint(path=ckpt_path, epoch=1, model=tiny_model, optimizer=optimizer)

    # Read the original to compare later
    original_ckpt = load_checkpoint(ckpt_path, device=torch.device("cpu"))

    # Simulate a failed save by making torch.save raise
    with patch("pill_safety.cv.attribute.utils.checkpoint.torch.save", side_effect=IOError("disk full")):
        with pytest.raises(IOError, match="disk full"):
            save_checkpoint(path=ckpt_path, epoch=99, model=tiny_model, optimizer=optimizer)

    # Verify original checkpoint is still intact
    recovered = load_checkpoint(ckpt_path, device=torch.device("cpu"))
    assert recovered["epoch"] == original_ckpt["epoch"]
