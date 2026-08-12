import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class MultiTaskResNet18(nn.Module):
    def __init__(self, num_shape_classes, num_color_classes=12, pretrained=True, freeze_backbone=False):
        super(MultiTaskResNet18, self).__init__()
        
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        
        # Cắt bỏ tầng Linear cuối cùng của ResNet18 để lấy phần feature extractor dùng chung
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Hai head riêng biệt
        self.shape_head = nn.Linear(512, num_shape_classes)
        self.color_head = nn.Linear(512, num_color_classes) # Multi-label dùng BCEWithLogitsLoss nên xuất ra logits

    def forward(self, x, task_type='shape'):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        
        if task_type == 'shape':
            return self.shape_head(features)
        elif task_type == 'color':
            return self.color_head(features)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")