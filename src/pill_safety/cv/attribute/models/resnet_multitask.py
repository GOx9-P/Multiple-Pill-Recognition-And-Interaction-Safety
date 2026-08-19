"""
Unified Multi-task ResNet18 model for pill attribute recognition.

Architecture:
    - Backbone: ResNet18 (pretrained on ImageNet)
    - fc_shape: Dropout → Linear (single-label shape classification)
    - fc_color: Linear → BN → ReLU → Dropout → Linear (multi-label color classification)

Supports two training strategies via explicit method calls:
    - Head-tune (Stage 1): freeze_backbone() → train only fc_shape + fc_color
    - Last-blocks fine-tune (Stage 2): unfreeze_last_blocks() → train layer3/4 + heads
"""

import torch.nn as nn
from torchvision import models


class MultiTaskResNet18(nn.Module):
    """ResNet18 with two classification heads for shape and color prediction.

    The model does NOT auto-freeze any layers. Callers MUST explicitly invoke
    ``freeze_backbone()`` or ``unfreeze_last_blocks()`` after construction.

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

        # Shape classification head (single-label)
        self.fc_shape = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_shape_classes),
        )

        # Color classification head (multi-label, enhanced with hidden layer)
        self.fc_color = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_color_classes),
        )

        # Preflight assertions
        assert self.fc_shape[-1].out_features == num_shape_classes, (
            f"fc_shape output {self.fc_shape[-1].out_features} != {num_shape_classes}"
        )
        assert self.fc_color[-1].out_features == num_color_classes, (
            f"fc_color output {self.fc_color[-1].out_features} != {num_color_classes}"
        )

        # Apply weight initialization to prevent loss explosion
        self.fc_shape.apply(self._init_weights)
        self.fc_color.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for newly added layers to prevent loss explosion."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    # Freeze / Unfreeze API
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze the entire backbone. Only fc_shape and fc_color remain trainable.

        Used for **head-tune** (Stage 1).
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_blocks(self, num_blocks: int = 2) -> None:
        """Unfreeze the last N residual blocks of the backbone.

        ``num_blocks=2`` unfreezes ``layer3`` and ``layer4``.
        ``num_blocks=1`` unfreezes only ``layer4``.

        All earlier layers remain frozen.  Used for **last-blocks fine-tune** (Stage 2).
        """
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Then selectively unfreeze last N blocks
        layers = [self.backbone.layer1, self.backbone.layer2,
                  self.backbone.layer3, self.backbone.layer4]
        for layer in layers[-num_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

    def set_bn_eval(self) -> None:
        """Set all BatchNorm layers in the backbone to eval mode.

        MUST be called after ``model.train()`` in every training step to prevent
        batch statistics corruption with small batch sizes.
        """
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()

    def get_trainable_params(self) -> list:
        """Return only parameters that have requires_grad=True."""
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        features = self.backbone(x)
        shape_out = self.fc_shape(features)
        color_out = self.fc_color(features)
        return shape_out, color_out
