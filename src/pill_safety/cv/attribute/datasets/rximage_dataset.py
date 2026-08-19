from pathlib import Path
from PIL import Image
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class RxImageDataset(Dataset):
    def __init__(self, csv_file: str | Path, img_dir: str | Path, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

        self.shape_labels = self.df["shape_label"].values
        self.color_cols = [c for c in self.df.columns if c.startswith("color_")]
        self.color_labels = self.df[self.color_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row["rxnavImageFileName"]).strip()
        img_path = self.img_dir / img_name

        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        shape_target = torch.tensor(self.shape_labels[idx], dtype=torch.long)
        color_target = torch.tensor(self.color_labels[idx], dtype=torch.float32)

        return image, shape_target, color_target, img_name