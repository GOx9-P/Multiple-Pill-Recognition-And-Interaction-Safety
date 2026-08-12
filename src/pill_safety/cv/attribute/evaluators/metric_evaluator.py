import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report

class AttributeEvaluator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def evaluate_predictions(self, model, dataloader, device):
        """
        Đánh giá toàn bộ tập dữ liệu (validation/test) cho cả 2 nhiệm vụ shape và color
        """
        model.eval()
        
        all_shape_preds = []
        all_shape_targets = []
        
        all_color_preds = []
        all_color_targets = []

        with torch.no_grad():
            # Duyệt qua các batch (đại diện ở đây là shape_loader hoặc kết hợp)
            for images, shape_labels, color_labels in dataloader: # Giả định loader trả về cả 2 hoặc tách riêng
                pass 
        
        # Phiên bản xử lý mảng đầu ra trực tiếp từ logits và targets:
        pass

    @staticmethod
    def evaluate_shape(logits, targets):
        """
        Tính Macro F1 cho Shape (Multi-class classification)
        - logits: numpy array shape (N, num_shape_classes)
        - targets: numpy array shape (N,)
        """
        preds = np.argmax(logits, axis=1)
        macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
        return macro_f1

    @staticmethod
    def evaluate_color(logits, targets, threshold=0.5):
        """
        Tính Macro F1 cho Color (Multi-label classification)
        - logits: numpy array shape (N, num_color_classes)
        - targets: numpy array shape (N, num_color_classes) dạng one-hot/binary
        """
        probs = 1 / (1 + np.exp(-logits)) # Sigmoid function
        binary_preds = (probs >= threshold).astype(int)
        macro_f1 = f1_score(targets, binary_preds, average='macro', zero_division=0)
        return macro_f1

    @classmethod
    def evaluate_multitask(cls, shape_logits, shape_targets, color_logits, color_targets):
        """
        Hàm tổng hợp tính toán metrics cho cả mô hình multi-task để ghi nhận vào val_metrics.json
        """
        shape_f1 = cls.evaluate_shape(shape_logits, shape_targets)
        color_f1 = cls.evaluate_color(color_logits, color_targets)
        
        # Overall Macro F1 trung bình cộng của 2 tác vụ
        overall_f1 = (shape_f1 + color_f1) / 2.0
        
        return {
            "shape_macro_f1": round(float(shape_f1), 4),
            "color_macro_f1": round(float(color_f1), 4),
            "overall_macro_f1": round(float(overall_f1), 4)
        }