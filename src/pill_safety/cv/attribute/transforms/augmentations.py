"""
Data augmentation and normalization transforms for pill attribute recognition.

Uses ImageNet normalization statistics since the backbone is a pretrained ResNet18.
"""

from torchvision import transforms


# ImageNet channel-wise mean and std
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224) -> dict:
    """Return a dictionary of transforms for training and validation.

    Args:
        image_size: Target spatial size (H=W) for input images.

    Returns:
        dict with keys ``"train"`` and ``"val"``, each mapping to a
        ``torchvision.transforms.Compose`` pipeline.
    """
    return {
        "train": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
    }
