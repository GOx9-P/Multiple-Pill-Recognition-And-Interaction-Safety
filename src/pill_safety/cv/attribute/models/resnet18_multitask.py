"""ResNet18 dung chung backbone va hai head cho shape/color."""

from collections import OrderedDict

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class MultiTaskResNet18(nn.Module):
    """ResNet18 giu nguyen ten layer de fine-tune layer3/layer4 ro rang."""

    _LEGACY_PREFIXES = {
        "backbone.0.": "backbone.conv1.",
        "backbone.1.": "backbone.bn1.",
        "backbone.4.": "backbone.layer1.",
        "backbone.5.": "backbone.layer2.",
        "backbone.6.": "backbone.layer3.",
        "backbone.7.": "backbone.layer4.",
    }

    def __init__(self, num_shape_classes, num_color_classes=12, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.shape_head = nn.Linear(512, num_shape_classes)
        self.color_head = nn.Linear(512, num_color_classes)

    def forward(self, x, task_type="shape"):
        """Tra logits cua head duoc chi dinh; color dung BCEWithLogitsLoss."""
        features = self.backbone(x)
        if task_type == "shape":
            return self.shape_head(features)
        if task_type == "color":
            return self.color_head(features)
        raise ValueError(f"Unknown task_type: {task_type}")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Nap ca checkpoint moi va checkpoint cu dung backbone Sequential."""
        remapped = OrderedDict()
        for key, value in state_dict.items():
            mapped_key = key
            for legacy_prefix, current_prefix in self._LEGACY_PREFIXES.items():
                if key.startswith(legacy_prefix):
                    mapped_key = current_prefix + key[len(legacy_prefix):]
                    break
            remapped[mapped_key] = value
        return super().load_state_dict(remapped, strict=strict, assign=assign)
