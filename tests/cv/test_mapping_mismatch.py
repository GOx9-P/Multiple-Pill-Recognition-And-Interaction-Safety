"""
Test P0-02: Label Mapping — SHA-256 hash, save/load round-trip, mismatch detection.

Tests:
    - build → save → load roundtrip preserves data
    - Same mapping always produces same hash (deterministic)
    - Missing file → FileNotFoundError
    - Malformed file → ValueError
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pill_safety.cv.attribute.labels.label_mapping import (
    save_label_mapping,
    load_label_mapping,
)


@pytest.fixture
def sample_mapping():
    return {
        "shape": ["CAPSULE", "OVAL", "ROUND", "DIAMOND"],
        "color": ["color_BLUE", "color_WHITE", "color_YELLOW"],
    }


def test_save_load_roundtrip(sample_mapping, tmp_path):
    """Save and load a mapping; data must be identical."""
    path = tmp_path / "mapping.json"
    hash1 = save_label_mapping(sample_mapping, path)

    loaded_mapping, n_shape, n_color, hash2 = load_label_mapping(path)

    assert loaded_mapping["shape"] == sample_mapping["shape"]
    assert loaded_mapping["color"] == sample_mapping["color"]
    assert n_shape == 4
    assert n_color == 3
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex digest


def test_deterministic_hash(sample_mapping, tmp_path):
    """Same mapping saved twice → same hash both times."""
    path1 = tmp_path / "m1.json"
    path2 = tmp_path / "m2.json"

    h1 = save_label_mapping(sample_mapping, path1)
    h2 = save_label_mapping(sample_mapping, path2)

    assert h1 == h2


def test_different_mapping_different_hash(tmp_path):
    """Different mappings → different hashes."""
    m1 = {"shape": ["CAPSULE", "ROUND"], "color": ["color_BLUE"]}
    m2 = {"shape": ["CAPSULE", "OVAL"], "color": ["color_BLUE"]}

    h1 = save_label_mapping(m1, tmp_path / "m1.json")
    h2 = save_label_mapping(m2, tmp_path / "m2.json")

    assert h1 != h2


def test_missing_file_raises():
    """Loading from nonexistent path → FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="FATAL"):
        load_label_mapping(Path("/nonexistent/mapping.json"))


def test_malformed_file_raises(tmp_path):
    """Loading a JSON without 'shape'/'color' keys → ValueError."""
    bad_path = tmp_path / "bad.json"
    bad_path.write_text('{"foo": "bar"}', encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed"):
        load_label_mapping(bad_path)


def test_no_fake_labels(sample_mapping, tmp_path):
    """Mapping must contain real names, not 'shape_class_0' placeholders."""
    path = tmp_path / "mapping.json"
    save_label_mapping(sample_mapping, path)

    loaded, _, _, _ = load_label_mapping(path)
    for name in loaded["shape"]:
        assert not name.startswith("shape_class_"), (
            f"Found fake label '{name}' — mapping must use real class names"
        )
