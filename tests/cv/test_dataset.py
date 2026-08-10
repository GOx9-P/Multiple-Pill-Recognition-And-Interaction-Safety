import os
import pytest
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from pill_safety.cv.attribute.datasets.rximage import RxImageDataset

@pytest.fixture
def dummy_dataset_dir(tmp_path):
    """Fixture to create a temporary dataset with images and a split CSV."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    # Create dummy images
    img1_path = img_dir / "img1.jpg"
    img2_path = img_dir / "img2.jpg"
    
    Image.new("RGB", (300, 300), color="red").save(img1_path)
    Image.new("RGB", (300, 300), color="blue").save(img2_path)
    
    # create a RGBA image to test conversion
    img3_path = img_dir / "img3.png"
    Image.new("RGBA", (300, 300), color=(255, 0, 0, 128)).save(img3_path)
    
    # Create split csv
    csv_path = tmp_path / "split.csv"
    pd.DataFrame({
        "filename": ["img1.jpg", "img2.jpg", "img3.png"],
        "shape": ["ROUND", "CAPSULE", "OVAL"],
        "color": ["red", "blue", "red;blue"] # Using standard delimiter if needed
    }).to_csv(csv_path, index=False)
    
    return img_dir, csv_path


def test_rximage_dataset_getitem(dummy_dataset_dir):
    img_dir, csv_path = dummy_dataset_dir
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = RxImageDataset(
        csv_file=csv_path,
        img_dir=img_dir,
        transform=transform
    )
    
    assert len(dataset) == 3
    
    img, s_target, c_target = dataset[0]
    
    # Check types and shapes
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224), "Image shape mismatch. Did resize or RGB conversion fail?"
    
    assert isinstance(s_target, torch.Tensor)
    assert s_target.dtype == torch.long, "Shape target must be long for CrossEntropyLoss"
    
    assert isinstance(c_target, torch.Tensor)
    assert c_target.dtype == torch.float32, "Color target must be float32 for BCEWithLogitsLoss"

    # test RGBA image conversion (img3.png is index 2)
    img_rgba, _, _ = dataset[2]
    assert img_rgba.shape == (3, 224, 224), "RGBA image failed to convert to 3 channels (RGB)"


def test_rximage_dataloader_batching(dummy_dataset_dir):
    img_dir, csv_path = dummy_dataset_dir
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = RxImageDataset(
        csv_file=csv_path,
        img_dir=img_dir,
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    batch = next(iter(dataloader))
    imgs, s_targets, c_targets = batch
    
    assert imgs.shape == (2, 3, 224, 224)
    assert s_targets.shape == (2,)
    # The second dimension of c_targets depends on the number of unique colors found during fit
    assert len(c_targets.shape) == 2


def test_missing_image_handling(dummy_dataset_dir):
    img_dir, csv_path = dummy_dataset_dir
    
    # Corrupt the CSV with a missing image
    df = pd.read_csv(csv_path)
    df.loc[0, "filename"] = "does_not_exist.jpg"
    df.to_csv(csv_path, index=False)
    
    dataset = RxImageDataset(
        csv_file=csv_path,
        img_dir=img_dir,
    )
    
    # the dataset tries to open the image dynamically in __getitem__
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]
