"""
Test P0-01: Data Split Contract — No group leakage between Train/Val/Test.

Tests:
    - Valid split with zero overlap → passes
    - Split with NDC overlap → fails
    - Missing group key column → raises KeyError
"""

import os
import tempfile

import pandas as pd
import pytest

import sys
from pathlib import Path

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pill_safety.cv.attribute.utils.leakage import check_split_leakage


@pytest.fixture
def clean_splits(tmp_path):
    """Create 3 CSVs with ZERO NDC overlap."""
    train_df = pd.DataFrame({
        "NDC11": ["001", "002", "003", "004"],
        "filename": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
    })
    val_df = pd.DataFrame({
        "NDC11": ["005", "006"],
        "filename": ["e.jpg", "f.jpg"],
    })
    test_df = pd.DataFrame({
        "NDC11": ["007", "008"],
        "filename": ["g.jpg", "h.jpg"],
    })

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return str(train_path), str(val_path), str(test_path)


@pytest.fixture
def leaky_splits(tmp_path):
    """Create 3 CSVs with NDC overlap between train and val."""
    train_df = pd.DataFrame({
        "NDC11": ["001", "002", "003", "004"],
        "filename": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
    })
    val_df = pd.DataFrame({
        "NDC11": ["003", "005"],  # "003" leaks from train
        "filename": ["e.jpg", "f.jpg"],
    })
    test_df = pd.DataFrame({
        "NDC11": ["006", "007"],
        "filename": ["g.jpg", "h.jpg"],
    })

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return str(train_path), str(val_path), str(test_path)


def test_clean_split_passes(clean_splits):
    """Zero overlap → leakage check passes."""
    train, val, test = clean_splits
    passed, details = check_split_leakage(train, val, test, group_key="NDC11")

    assert passed is True
    assert details["overlap_train_val"] == 0
    assert details["overlap_train_test"] == 0
    assert details["overlap_val_test"] == 0
    assert details["total_overlap"] == 0


def test_leaky_split_fails(leaky_splits):
    """Overlap exists → leakage check fails."""
    train, val, test = leaky_splits
    passed, details = check_split_leakage(train, val, test, group_key="NDC11")

    assert passed is False
    assert details["overlap_train_val"] == 1  # "003" overlaps
    assert details["total_overlap"] >= 1


def test_missing_group_key_raises(clean_splits):
    """Missing column → raises KeyError."""
    train, val, test = clean_splits
    with pytest.raises(KeyError, match="NONEXISTENT"):
        check_split_leakage(train, val, test, group_key="NONEXISTENT")


def test_leading_zeros_preserved(tmp_path):
    """Leading zeros in NDC codes must be preserved as distinct groups."""
    train_df = pd.DataFrame({"NDC11": ["001", "01", "1"], "f": ["a", "b", "c"]})
    val_df = pd.DataFrame({"NDC11": ["002"], "f": ["d"]})
    test_df = pd.DataFrame({"NDC11": ["003"], "f": ["e"]})

    train_df.to_csv(tmp_path / "train.csv", index=False)
    val_df.to_csv(tmp_path / "val.csv", index=False)
    test_df.to_csv(tmp_path / "test.csv", index=False)

    passed, details = check_split_leakage(
        str(tmp_path / "train.csv"),
        str(tmp_path / "val.csv"),
        str(tmp_path / "test.csv"),
    )
    assert passed is True
    assert details["train_groups"] == 3  # "001", "01", "1" are distinct
