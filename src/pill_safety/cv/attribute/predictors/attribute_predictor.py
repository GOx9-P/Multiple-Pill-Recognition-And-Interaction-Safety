"""Inference helper cho checkpoint multi-task ResNet18 shape/color."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from pill_safety.cv.attribute.models.resnet18_multitask import MultiTaskResNet18
from pill_safety.cv.attribute.postprocessing.formatter import AttributeFormatter


class AttributePredictor:
    """Nap checkpoint cung label mapping va optional calibrated color thresholds."""

    def __init__(self, checkpoint_path: str | Path, label_mapping_path: str | Path, thresholds_path: str | Path | None = None, device: str | None = None):
        """Khoi tao model voi so class lay tu mapping, khong hard-code shape class."""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.formatter = AttributeFormatter(label_mapping_path)
        self.model = MultiTaskResNet18(len(self.formatter.shape_labels), len(self.formatter.color_labels), pretrained=False)
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()
        self.color_thresholds = 0.5
        if thresholds_path is not None:
            threshold_payload = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))
            self.color_thresholds = [threshold_payload["thresholds"][color] for color in self.formatter.color_labels]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict_image(self, image_path: str | Path) -> dict:
        """Du doan shape/color cho mot pill crop RGB da ton tai tren disk."""
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            shape_logits = self.model(tensor, task_type="shape").cpu().numpy()[0]
            color_logits = self.model(tensor, task_type="color").cpu().numpy()[0]
        return self.formatter.format_output(shape_logits, color_logits, self.color_thresholds)
