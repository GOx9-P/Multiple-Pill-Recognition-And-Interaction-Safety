import torch
import pytest
from pill_safety.cv.attribute.models.resnet_multitask import MultiTaskResNet18

def test_multitask_resnet18_forward_pass():
    """
    Dry-run forward pass to ensure the model accepts correctly sized inputs
    and returns a Tuple of (shape_out, color_out) with no extra dimensions.
    """
    batch_size = 4
    num_shape_classes = 10
    num_color_classes = 15
    
    # Khởi tạo mô hình (pretrained=False để chạy test nhanh hơn)
    model = MultiTaskResNet18(
        num_shape_classes=num_shape_classes,
        num_color_classes=num_color_classes,
        pretrained=False
    )
    model.eval()
    
    # Tạo dummy tensor đóng vai trò là "ảnh giả"
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Chạy Forward Pass
    with torch.no_grad():
        outputs = model(dummy_input)
    
    # Kiểm tra kiểu trả về (Phải là Tuple cho thân thiện với ONNX)
    assert isinstance(outputs, tuple), "Hàm forward phải trả về một Tuple"
    assert len(outputs) == 2, "Hàm forward phải trả về chính xác 2 tensors (shape_out, color_out)"
    
    shape_out, color_out = outputs
    
    # Kiểm tra kích thước (Shape) của tensor đầu ra
    # Tránh trường hợp vướng các chiều không gian dư thừa như [B, num_classes, 1, 1]
    assert shape_out.shape == (batch_size, num_shape_classes), \
        f"Kích thước shape_out sai. Kỳ vọng: {(batch_size, num_shape_classes)}, Thực tế: {shape_out.shape}"
        
    assert color_out.shape == (batch_size, num_color_classes), \
        f"Kích thước color_out sai. Kỳ vọng: {(batch_size, num_color_classes)}, Thực tế: {color_out.shape}"

def test_multitask_resnet18_freeze_backbone():
    """
    Kiểm tra chức năng freeze_backbone() cho chiến thuật Transfer Learning (Head-tune).
    Đảm bảo backbone bị đóng băng, nhưng 2 heads vẫn học bình thường.
    """
    model = MultiTaskResNet18(num_shape_classes=2, num_color_classes=2, pretrained=False)
    
    model.freeze_backbone()
    
    # Kiểm tra backbone đã bị đóng băng (requires_grad = False)
    for param in model.backbone.parameters():
        assert not param.requires_grad, "Các tham số của backbone phải được đóng băng (requires_grad=False)"
        
    # Kiểm tra 2 heads vẫn được cập nhật trọng số (requires_grad = True)
    for param in model.fc_shape.parameters():
        assert param.requires_grad, "Nhánh Shape phải được unfreeze"
    for param in model.fc_color.parameters():
        assert param.requires_grad, "Nhánh Color phải được unfreeze"

def test_multitask_resnet18_unfreeze_last_blocks():
    """
    Kiểm tra chức năng unfreeze_last_blocks() cho chiến thuật Fine-tune chặng cuối.
    Đảm bảo layer1, layer2 bị đóng băng, nhưng layer3, layer4 và 2 heads được mở khóa.
    """
    model = MultiTaskResNet18(num_shape_classes=2, num_color_classes=2, pretrained=False)
    
    # Giả lập mở khóa 2 blocks cuối (layer3 và layer4)
    model.unfreeze_last_blocks(num_blocks=2)
    
    # layer1 và layer2 phải bị đóng băng
    for param in model.backbone.layer1.parameters():
        assert not param.requires_grad, "layer1 phải bị đóng băng"
    for param in model.backbone.layer2.parameters():
        assert not param.requires_grad, "layer2 phải bị đóng băng"
        
    # layer3, layer4 phải được mở khóa
    for param in model.backbone.layer3.parameters():
        assert param.requires_grad, "layer3 phải được mở khóa"
    for param in model.backbone.layer4.parameters():
        assert param.requires_grad, "layer4 phải được mở khóa"
