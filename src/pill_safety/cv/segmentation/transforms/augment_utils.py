from __future__ import annotations

import cv2
import numpy as np
import re
from pathlib import Path

from src.pill_safety.cv.segmentation.utils.config import OUTPUT_DIR, N_AUG_PER_IMAGE


def get_augmentation_pipeline():
    import albumentations as A

    try:
        gauss_noise = A.GaussNoise(var_limit=(0.01, 0.03), p=0.1)
    except TypeError:
        gauss_noise = A.GaussNoise(std_range=(0.01, 0.03), p=0.1)

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            gauss_noise,
            A.MotionBlur(blur_limit=3, p=0.15),
        ]
    )


def yolo_seg_lines_to_masks(lines, img_w, img_h):
    masks, class_ids = [], []
    for line in lines:
        parts = line.split()
        class_ids.append(int(parts[0]))
        coords = list(map(float, parts[1:]))
        pts = np.array(
            [[coords[i] * img_w, coords[i + 1] * img_h] for i in range(0, len(coords), 2)],
            dtype=np.int32,
        )
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        masks.append(mask)
    return masks, class_ids


def masks_to_yolo_lines(masks, class_ids, img_w, img_h):
    lines = []
    for mask, cid in zip(masks, class_ids):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)  # bỏ mảnh vụn nhỏ do augment
        if cv2.contourArea(contour) < 15:
            continue
        pts = contour.reshape(-1, 2)
        norm = []
        for x, y in pts:
            norm.extend([f"{x / img_w:.6f}", f"{y / img_h:.6f}"])
        if len(norm) >= 6:
            lines.append(f"{cid} " + " ".join(norm))
    return lines


def augment_train_split(n_aug=N_AUG_PER_IMAGE):
    pipeline = get_augmentation_pipeline()
    img_dir = OUTPUT_DIR / "images" / "train"
    lbl_dir = OUTPUT_DIR / "labels" / "train"

    all_images = list(img_dir.glob("*.*"))
    aug_pattern = re.compile(r"_aug\d+$", re.IGNORECASE)
    original_images = [p for p in all_images if not aug_pattern.search(p.stem)]
    num_aug_files = len(all_images) - len(original_images)
    print(
        f"Augmenting {len(original_images)} original train images x{n_aug} copies each... "
        f"({num_aug_files} _aug* files excluded from input)"
    )

    for img_path in original_images:
        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        lines = [l for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            continue

        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        masks, class_ids = yolo_seg_lines_to_masks(lines, w, h)

        for aug_idx in range(n_aug):
            transformed = pipeline(image=image, masks=masks)
            aug_image, aug_masks = transformed["image"], transformed["masks"]

            new_lines = masks_to_yolo_lines(aug_masks, class_ids, w, h)
            if not new_lines:
                continue  # augment làm mất hết instance (vd rotate crop hết) -> bỏ

            out_stem = f"{img_path.stem}_aug{aug_idx+1}"
            cv2.imwrite(str(img_dir / f"{out_stem}{img_path.suffix}"), aug_image)
            (lbl_dir / f"{out_stem}.txt").write_text("\n".join(new_lines), encoding="utf-8")

    print("Augmentation done. Train folder now contains original + augmented images.")
