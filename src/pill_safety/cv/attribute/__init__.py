# Unified pill attribute recognition module
from pill_safety.cv.attribute.models.resnet_multitask import MultiTaskResNet18
from pill_safety.cv.attribute.datasets.rximage import RxImageDataset

__all__ = [
    "MultiTaskResNet18",
    "RxImageDataset",
]