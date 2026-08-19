"""
Data leakage detection utilities.

Provides functions to check that Train/Val/Test splits have zero
group overlap (by NDC11 or RXCUI), preventing data leakage.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


def check_split_leakage(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    group_key: str = "NDC11",
) -> Tuple[bool, Dict]:
    """Check whether Train/Val/Test CSVs have overlapping group keys.

    Args:
        train_csv: Path to training CSV.
        val_csv: Path to validation CSV.
        test_csv: Path to test CSV.
        group_key: Column name to use as group identifier.

    Returns:
        Tuple of (passed: bool, details: dict).
        ``passed`` is True only if all three pairwise overlaps are zero.
    """
    train_df = pd.read_csv(train_csv, dtype={group_key: str})
    val_df = pd.read_csv(val_csv, dtype={group_key: str})
    test_df = pd.read_csv(test_csv, dtype={group_key: str})

    # Canonicalize: convert to string, strip whitespace, preserve leading zeros
    def get_groups(df, key):
        if key not in df.columns:
            raise KeyError(
                f"Column '{key}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
        return set(df[key].dropna().astype(str).str.strip())

    train_groups = get_groups(train_df, group_key)
    val_groups = get_groups(val_df, group_key)
    test_groups = get_groups(test_df, group_key)

    overlap_train_val = train_groups & val_groups
    overlap_train_test = train_groups & test_groups
    overlap_val_test = val_groups & test_groups

    total_overlap = len(overlap_train_val) + len(overlap_train_test) + len(overlap_val_test)

    details = {
        "group_key": group_key,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "overlap_train_val": len(overlap_train_val),
        "overlap_train_test": len(overlap_train_test),
        "overlap_val_test": len(overlap_val_test),
        "total_overlap": total_overlap,
    }

    passed = total_overlap == 0
    return passed, details
