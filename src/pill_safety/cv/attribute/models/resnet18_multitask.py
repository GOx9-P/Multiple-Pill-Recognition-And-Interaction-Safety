import torch.nn as nn
from torchvision import models

class MultiTaskResNet18(nn.Module):
    def __init__(self, num_shape_classes: int, num_color_classes: int):
        super(MultiTaskResNet18, self).__init__()
        self.backbone = models.resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc_shape = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_shape_classes)
        )
        self.fc_color = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_color_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        shape_out = self.fc_shape(features)
        color_out = self.fc_color(features)
        return shape_out, color_out