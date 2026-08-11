"""
RxImage Dataset loader for NIH/RxImage pill attribute recognition.

Reads CSV files with shape labels and multi-label color columns,
loads corresponding images, and returns (image, shape_target, color_target) tuples.
"""

from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


# 12 known missing images in the NIH/RxImage dataset.
# These are automatically filtered out when loading CSVs.
MISSING_FILES: Set[str] = {
    "63459-0502-30_RXNAVIMAGE10_8641C37E_1.jpg",
    "63459-0502-30_RXNAVIMAGE10_8641C37E_2.jpg",
    "63459-0504-30_RXNAVIMAGE10_8841C43E_1.jpg",
    "63459-0504-30_RXNAVIMAGE10_8841C43E_2.jpg",
    "63459-0506-30_RXNAVIMAGE10_8941C4CE_1.jpg",
    "63459-0506-30_RXNAVIMAGE10_8941C4CE_2.jpg",
    "63459-0508-30_RXNAVIMAGE10_8941C4FE_1.jpg",
    "63459-0508-30_RXNAVIMAGE10_8941C4FE_2.jpg",
    "63459-0512-30_RXNAVIMAGE10_8A41C55E_1.jpg",
    "63459-0512-30_RXNAVIMAGE10_8A41C55E_2.jpg",
    "63459-0516-30_RXNAVIMAGE10_8B41C5BE_1.jpg",
    "63459-0516-30_RXNAVIMAGE10_8B41C5BE_2.jpg",
}


class RxImageDataset(Dataset):
    """Multi-task dataset for pill shape (single-label) and color (multi-label) recognition.

    Supports two CSV formats:
    - Pre-encoded: columns ``shape_label`` (int) and ``color_*`` (binary).
    - Raw string: columns ``shape`` (str) and ``color`` (str with delimiters).

    Args:
        csv_file: Path to the CSV file containing labels.
        img_dir: Directory containing pill images.
        transform: Optional torchvision transform to apply to images.
        shape_encoder: Optional sklearn LabelEncoder fitted on the training set.
            Pass ``None`` for the training split (will be fitted automatically).
        mlb_color: Optional sklearn MultiLabelBinarizer fitted on training set.
            Pass ``None`` for the training split (will be fitted automatically).
    """

    def __init__(
        self,
        csv_file,
        img_dir,
        transform=None,
        shape_encoder=None,
        mlb_color=None,
    ):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

        # ------------------------------------------------------------------
        # Filter out known missing images
        # ------------------------------------------------------------------
        filename_col = (
            "rxnavImageFileName"
            if "rxnavImageFileName" in self.df.columns
            else "filename"
        )
        if filename_col in self.df.columns:
            before_len = len(self.df)
            self.df = self.df[
                ~self.df[filename_col].isin(MISSING_FILES)
            ].reset_index(drop=True)
            if len(self.df) < before_len:
                print(
                    f"  [INFO] Auto-filtered {before_len - len(self.df)} "
                    f"missing image rows from {Path(csv_file).name}"
                )

        # ------------------------------------------------------------------
        # 1. Shape encoding (single-label classification)
        # ------------------------------------------------------------------
        has_shape_label = "shape_label" in self.df.columns
        has_shape_name = "shape" in self.df.columns

        if has_shape_label:
            self.shape_labels = self.df["shape_label"].values
            self.shape_encoder = shape_encoder

            # Build label -> name dict if both columns exist
            if has_shape_name and shape_encoder is None:
                mapping = self.df.dropna(
                    subset=["shape", "shape_label"]
                ).drop_duplicates(subset=["shape_label"])
                self.shape_encoder_dict = dict(
                    zip(mapping["shape_label"], mapping["shape"])
                )
            else:
                self.shape_encoder_dict = (
                    getattr(shape_encoder, "shape_encoder_dict", None)
                    if shape_encoder
                    else None
                )
        elif has_shape_name:
            from sklearn.preprocessing import LabelEncoder

            if shape_encoder is None:
                self.shape_encoder = LabelEncoder()
                self.shape_labels = self.shape_encoder.fit_transform(
                    self.df["shape"].fillna("UNKNOWN").astype(str)
                )
                self.shape_encoder_dict = {
                    i: name
                    for i, name in enumerate(self.shape_encoder.classes_)
                }
            else:
                self.shape_encoder = shape_encoder
                self.shape_labels = self.shape_encoder.transform(
                    self.df["shape"].fillna("UNKNOWN").astype(str)
                )
                self.shape_encoder_dict = getattr(
                    shape_encoder, "shape_encoder_dict", None
                )
        else:
            raise KeyError(
                "CSV missing both 'shape_label' and 'shape' columns."
            )

        # ------------------------------------------------------------------
        # 2. Color encoding (multi-label classification)
        # ------------------------------------------------------------------
        self.color_cols = [
            c for c in self.df.columns if c.startswith("color_")
        ]
        if len(self.color_cols) > 0:
            self.color_labels = self.df[self.color_cols].values.astype(
                np.float32
            )
        elif "color" in self.df.columns:
            from sklearn.preprocessing import MultiLabelBinarizer

            def _parse_colors(color_str):
                if not color_str or pd.isna(color_str):
                    return ["unknown"]
                colors = (
                    str(color_str)
                    .replace(";", " ")
                    .replace("/", " ")
                    .replace(",", " ")
                    .split()
                )
                return [c.strip().lower() for c in colors if c.strip()]

            color_series = self.df["color"].apply(_parse_colors)
            if mlb_color is None:
                self.mlb_color = MultiLabelBinarizer()
                color_bin = self.mlb_color.fit_transform(color_series)
            else:
                self.mlb_color = mlb_color
                color_bin = self.mlb_color.transform(color_series)
            self.color_labels = color_bin.astype(np.float32)
            self.color_cols = [
                f"color_{c}"
                for c in getattr(self.mlb_color, "classes_", [])
            ]
        else:
            raise KeyError(
                "CSV missing both 'color_*' columns and 'color' column."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = str(
            row.get("rxnavImageFileName", row.get("filename", ""))
        ).strip()
        img_path = self.img_dir / filename

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        shape_target = torch.tensor(
            self.shape_labels[idx], dtype=torch.long
        )
        color_target = torch.tensor(
            self.color_labels[idx], dtype=torch.float32
        )

        return image, shape_target, color_target
