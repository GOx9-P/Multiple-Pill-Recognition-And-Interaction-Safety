"""
Multi-task ResNet18 model for pill attribute recognition.

Architecture:
    - Backbone: ResNet18 (pretrained on ImageNet, frozen)
    - fc_shape: Dropout → Linear (single-label shape classification)
    - fc_color: Linear → BN → ReLU → Dropout → Linear (multi-label color classification)
"""

import torch.nn as nn
from torchvision import models


class MultiTaskResNet18_HeadsFinetune(nn.Module):
    """ResNet18 with frozen backbone and trainable classification heads.

    This model freezes the entire ResNet18 backbone and only trains
    two classification heads for shape and color prediction.

    Args:
        num_shape_classes: Number of shape categories (single-label).
        num_color_classes: Number of color categories (multi-label).
        pretrained: Whether to load ImageNet pretrained weights for the backbone.
    """

    def __init__(
        self,
        num_shape_classes: int,
        num_color_classes: int,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Shape classification head
        self.fc_shape = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_shape_classes),
        )

        # Color classification head (enhanced with hidden layer)
        self.fc_color = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_color_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        shape_out = self.fc_shape(features)
        color_out = self.fc_color(features)
        return shape_out, color_out
