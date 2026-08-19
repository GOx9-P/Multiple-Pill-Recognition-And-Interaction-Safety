"""Validate schema, split va lineage cua data attribute offline augmentation."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


COLOR_COLUMNS = [
    "label_BLACK", "label_BLUE", "label_BROWN", "label_GRAY", "label_GREEN", "label_ORANGE",
    "label_PINK", "label_PURPLE", "label_RED", "label_TURQUOISE", "label_WHITE", "label_YELLOW",
]
_AUGMENTATION_SUFFIX = re.compile(r"_(?:AUG|ADV)_\d+$", re.IGNORECASE)
_SIDE_SUFFIX = re.compile(r"_[12]$")


def source_image_group(filename: str) -> str:
    """Tra source group cua anh goc, gom ca hai mat va cac bien the AUG/ADV."""
    stem = Path(str(filename)).stem
    stem = _AUGMENTATION_SUFFIX.sub("", stem)
    return _SIDE_SUFFIX.sub("", stem)


def _read_csv(path: Path) -> pd.DataFrame:
    """Doc CSV va bao loi som khi path khong ton tai."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV split not found: {path}")
    return pd.read_csv(path)


def _validate_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    """Dam bao CSV co du cot can thiet cho dataset loader."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _split_stats(frames: dict[str, pd.DataFrame], is_color: bool) -> tuple[dict[str, dict], dict[str, set[str]]]:
    """Tinh so luong anh, synthetic va source group cho manifest."""
    stats, groups = {}, {}
    for split_name, frame in frames.items():
        source_groups = set(frame["rximageFileName"].map(source_image_group))
        if is_color:
            synthetic_mask = frame["is_synthetic"].astype(int).eq(1)
        else:
            synthetic_mask = frame["rximageFileName"].astype(str).str.contains(r"_(?:AUG|ADV)_\d+\.(?:jpg|jpeg|png)$", case=False, regex=True)
        stats[split_name] = {
            "samples": int(len(frame)),
            "original_samples": int((~synthetic_mask).sum()),
            "synthetic_samples": int(synthetic_mask.sum()),
            "source_image_groups": int(len(source_groups)),
        }
        groups[split_name] = source_groups
        if split_name != "train" and synthetic_mask.any():
            raise ValueError(f"{split_name} contains synthetic samples for {'color' if is_color else 'shape'}.")
    return stats, groups


def _assert_disjoint(groups: dict[str, set[str]], task_name: str) -> None:
    """Chan image goc hoac variant cua no xuat hien o nhieu split."""
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[first] & groups[second]
        if overlap:
            preview = sorted(overlap)[:5]
            raise ValueError(f"{task_name} source-image leakage between {first}/{second}: {preview}")


def validate_attribute_data(data_paths: dict[str, Path], verify_images: bool = True) -> dict:
    """Kiem tra hai task va tra manifest tai lap dung data da train."""
    shape_frames = {split: _read_csv(data_paths[f"shape_{split}_csv"]) for split in ("train", "val", "test")}
    color_frames = {split: _read_csv(data_paths[f"color_{split}_csv"]) for split in ("train", "val", "test")}

    for split, frame in shape_frames.items():
        _validate_columns(frame, ["rximageFileName", "shape", "label_shape"], f"shape/{split}")
    for split, frame in color_frames.items():
        _validate_columns(frame, ["rximageFileName", "is_synthetic", *COLOR_COLUMNS], f"color/{split}")

    shape_stats, shape_groups = _split_stats(shape_frames, is_color=False)
    color_stats, color_groups = _split_stats(color_frames, is_color=True)
    _assert_disjoint(shape_groups, "shape")
    _assert_disjoint(color_groups, "color")

    if verify_images:
        for task_name, frames, image_dir in (
            ("shape", shape_frames, data_paths["shape_image_dir"]),
            ("color", color_frames, data_paths["color_image_dir"]),
        ):
            if not image_dir.is_dir():
                raise FileNotFoundError(f"{task_name} image directory not found: {image_dir}")
            for split, frame in frames.items():
                missing = [name for name in frame["rximageFileName"] if not (image_dir / str(name)).is_file()]
                if missing:
                    raise FileNotFoundError(f"{task_name}/{split} has {len(missing)} missing images, e.g. {missing[:3]}")

    return {
        "split_before_augmentation": True,
        "augmentation_train_only": True,
        "split_policy": {
            "group_key": "source_image_group",
            "source_group_rule": "remove file extension, optional _AUG_n/_ADV_n, then side suffix _1/_2",
            "leakage_check_passed": True,
        },
        "shape": shape_stats,
        "color": color_stats,
    }
