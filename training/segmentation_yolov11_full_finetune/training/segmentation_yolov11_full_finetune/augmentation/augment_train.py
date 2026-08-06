from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.segmentation.config import N_AUG_PER_IMAGE
from src.segmentation.augment_utils import augment_train_split


def main():
    augment_train_split(n_aug=N_AUG_PER_IMAGE)


if __name__ == "__main__":
    main()
