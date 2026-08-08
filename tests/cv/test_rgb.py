"""
Test P0-04: RGB Input Contract — All images must be 3-channel RGB.

Tests:
    - Grayscale (1-channel) image → converted to 3-channel
    - RGBA (4-channel) image → converted to 3-channel
    - Normal RGB image → stays 3-channel
    - Output tensor shape is [3, H, W]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pill_safety.cv.attribute.datasets.rximage import RxImageDataset
from pill_safety.cv.attribute.transforms.augmentations import get_transforms


def _create_test_csv(tmp_path, filenames, shapes, colors):
    """Helper to create a mini CSV + dummy images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(exist_ok=True)

    rows = []
    for fname, shape, color in zip(filenames, shapes, colors):
        rows.append({"filename": fname, "shape": shape, "color": color})

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, img_dir


def test_grayscale_converted_to_rgb(tmp_path):
    """A 1-channel grayscale image must become [3, H, W] tensor."""
    csv_path, img_dir = _create_test_csv(
        tmp_path,
        filenames=["gray.jpg"],
        shapes=["ROUND"],
        colors=["WHITE"],
    )

    # Create a grayscale image (mode="L")
    gray_img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8), mode="L")
    gray_img.save(img_dir / "gray.jpg")

    transforms_dict = get_transforms(image_size=64)
    dataset = RxImageDataset(csv_path, img_dir, transform=transforms_dict["val"])

    image, shape_target, color_target = dataset[0]
    assert image.shape == (3, 64, 64), f"Expected (3,64,64), got {image.shape}"


def test_rgba_converted_to_rgb(tmp_path):
    """A 4-channel RGBA image must become [3, H, W] tensor."""
    csv_path, img_dir = _create_test_csv(
        tmp_path,
        filenames=["rgba.png"],
        shapes=["CAPSULE"],
        colors=["BLUE"],
    )

    # Create an RGBA image (mode="RGBA")
    rgba_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8), mode="RGBA")
    rgba_img.save(img_dir / "rgba.png")

    transforms_dict = get_transforms(image_size=64)
    dataset = RxImageDataset(csv_path, img_dir, transform=transforms_dict["val"])

    image, shape_target, color_target = dataset[0]
    assert image.shape == (3, 64, 64), f"Expected (3,64,64), got {image.shape}"


def test_rgb_stays_rgb(tmp_path):
    """A normal 3-channel RGB image stays [3, H, W]."""
    csv_path, img_dir = _create_test_csv(
        tmp_path,
        filenames=["rgb.jpg"],
        shapes=["OVAL"],
        colors=["YELLOW"],
    )

    rgb_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), mode="RGB")
    rgb_img.save(img_dir / "rgb.jpg")

    transforms_dict = get_transforms(image_size=64)
    dataset = RxImageDataset(csv_path, img_dir, transform=transforms_dict["val"])

    image, shape_target, color_target = dataset[0]
    assert image.shape == (3, 64, 64)
    assert shape_target.dtype == torch.long
    assert color_target.dtype == torch.float32


# Need torch for assertions
import torch
