# Head-finetune strategy (NguyenQuocBao)
from pill_safety.cv.attribute.models.resnet_multitask import MultiTaskResNet18_HeadsFinetune
from pill_safety.cv.attribute.datasets.rximage import RxImageDataset

# Last-blocks-finetune strategy (NguyenGiaBao)
from pill_safety.cv.attribute.models.resnet18_multitask import MultiTaskResNet18
from pill_safety.cv.attribute.datasets.rximage_dataset import RxImageDataset as RxImageDatasetV2
from pill_safety.cv.attribute.predictors.attribute_predictor import AttributePredictor
from pill_safety.cv.attribute.utils.config import AttributeConfig as Config

__all__ = [
    "MultiTaskResNet18_HeadsFinetune",
    "MultiTaskResNet18",
    "RxImageDataset",
    "RxImageDatasetV2",
    "AttributePredictor",
    "Config",
]