import torch
import pytest
from pill_safety.cv.attribute.models.resnet18_multitask import MultiTaskResNet18

def test_model_forward():
    """Kiểm tra xem mô hình MultiTaskResNet18 có khởi tạo và forward pass đúng kích thước output hay không."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Khởi tạo model với 5 class shape và 12 class color
    model = MultiTaskResNet18(num_shape_classes=5, num_color_classes=12, pretrained=False).to(device)
    model.eval()

    # Tạo dummy input (batch_size=2, channels=3, height=224, width=224)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        # Test nhánh Shape
        shape_logits = model(dummy_input, task_type='shape')
        assert shape_logits.shape == (2, 5), f"Shape output shape không khớp: {shape_logits.shape}, kỳ vọng (2, 5)"

        # Test nhánh Color
        color_logits = model(dummy_input, task_type='color')
        assert color_logits.shape == (2, 12), f"Color output shape không khớp: {color_logits.shape}, kỳ vọng (2, 12)"

def test_device_placement():
    """Kiểm tra model chuyển đổi thiết bị thành công"""
    device = torch.device('cpu')
    model = MultiTaskResNet18(num_shape_classes=5, num_color_classes=12, pretrained=False).to(device)
    
    param = next(model.parameters())
    assert param.device.type == 'cpu'