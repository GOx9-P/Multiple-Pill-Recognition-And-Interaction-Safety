import torch
from PIL import Image
from torchvision import transforms
import os
import sys

# Đảm bảo import được các module trong project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from pill_safety.cv.attribute.models.resnet18_multitask import MultiTaskResNet18
from pill_safety.cv.attribute.postprocessing.formatter import AttributeFormatter

class AttributePredictor:
    def __init__(self, checkpoint_path, label_mapping_path="src/pill_safety/cv/attribute/labels/label_mapping.json", device='cpu'):
        self.device = device
        self.model = MultiTaskResNet18(num_shape_classes=4, num_color_classes=12, pretrained=False)
        
        # Load trọng số đã train
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print(f"[Predictor] Đã tải thành công checkpoint từ: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {checkpoint_path}")
            
        self.model.to(self.device)
        self.model.eval()

        # Pipeline transform chuẩn hóa cho inference (khớp với quá trình train offline)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Khởi tạo bộ postprocessing format kết quả
        self.formatter = AttributeFormatter(label_mapping_path=label_mapping_path)

    def predict_image(self, image_path, color_threshold=0.5):
        """
        Dự đoán thuộc tính (shape và color) cho một ảnh crop viên thuốc độc lập.
        Trả về dictionary chuẩn JSON Contract.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image_path}")

        # 1. Đọc và xử lý ảnh
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 2. Forward qua mô hình Multi-Task
        with torch.no_grad():
            shape_logits = self.model(input_tensor, task_type='shape').cpu().numpy()[0]
            color_logits = self.model(input_tensor, task_type='color').cpu().numpy()[0]

        # 3. Format kết quả thông qua module postprocessing
        result = self.formatter.format_output(shape_logits, color_logits, color_threshold=color_threshold)
        return result