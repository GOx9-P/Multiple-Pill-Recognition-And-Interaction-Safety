from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.segmentation.config import RAW_ANN_PATH, OUTPUT_DIR, SPLIT_RATIOS, RANDOM_SEED
from src.segmentation.coco_utils import (
    load_coco,
    drop_empty_annotation_images,
    split_dataset,
    build_splits,
    write_data_yaml,
)


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    images, anns_by_image, categories = load_coco(RAW_ANN_PATH)
    images, anns_by_image = drop_empty_annotation_images(images, anns_by_image)
    splits = split_dataset(images, anns_by_image, ratios=SPLIT_RATIOS, seed=RANDOM_SEED)
    class_map = build_splits(images, anns_by_image, categories, splits)
    write_data_yaml(categories, class_map)

    for split in ["train", "val", "test"]:
        n = len(list((OUTPUT_DIR / "images" / split).glob("*.*")))
        print(f"  {split}: {n} images")

    print("\nTiếp theo: chạy augmentation/augment_train.py (chỉ augment split train).")


if __name__ == "__main__":
    main()
