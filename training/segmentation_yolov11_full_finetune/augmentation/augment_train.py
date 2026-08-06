from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.pill_safety.cv.segmentation.utils.config import N_AUG_PER_IMAGE
from src.pill_safety.cv.segmentation.transforms.augment_utils import augment_train_split


def main():
    augment_train_split(n_aug=N_AUG_PER_IMAGE)


if __name__ == "__main__":
    main()
