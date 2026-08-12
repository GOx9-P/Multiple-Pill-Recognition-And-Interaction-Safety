import numpy as np
import json
import os

class AttributeFormatter:
    def __init__(self, label_mapping_path="src/pill_safety/cv/attribute/labels/label_mapping.json"):
        if not os.path.exists(label_mapping_path):
            raise FileNotFoundError(f"Không tìm thấy file mapping nhãn tại: {label_mapping_path}")
            
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            self.mapping = json.load(f)
            
        self.shape_map = self.mapping["shape"]
        self.color_list = self.mapping["color"]

    def format_output(self, shape_logits, color_logits, color_threshold=0.5):
        """
        Chuẩn hóa logits thô từ mô hình thành kết quả JSON có ý nghĩa ngôn ngữ tự nhiên.
        - shape_logits: numpy array hoặc tensor 1D cho nhánh shape
        - color_logits: numpy array hoặc tensor 1D cho nhánh color (12 chiều)
        - color_threshold: Ngưỡng xác suất để nhận diện màu (mặc định 0.5)
        """
        # 1. Xử lý Shape (Multi-class sử dụng Softmax)
        shape_probs = np.exp(shape_logits - np.max(shape_logits)) # Tránh tràn số
        shape_probs = shape_probs / np.sum(shape_probs)
        shape_idx = int(np.argmax(shape_probs))
        shape_label = self.shape_map.get(str(shape_idx), "UNKNOWN")
        shape_conf = float(shape_probs[shape_idx])
        
        # 2. Xử lý Color (Multi-label sử dụng Sigmoid)
        color_probs = 1 / (1 + np.exp(-color_logits))
        color_labels = []
        
        for idx, prob in enumerate(color_probs):
            if prob >= color_threshold:
                color_labels.append({
                    "label": self.color_list[idx],
                    "confidence": float(prob)
                })
                
        # Sắp xếp các màu theo độ tự tin giảm dần
        color_labels = sorted(color_labels, key=lambda x: x["confidence"], reverse=True)

        # 3. Trả về cấu trúc JSON Contract
        return {
            "shape": {
                "label": shape_label,
                "confidence": shape_conf
            },
            "color": {
                "labels": color_labels
            }
        }