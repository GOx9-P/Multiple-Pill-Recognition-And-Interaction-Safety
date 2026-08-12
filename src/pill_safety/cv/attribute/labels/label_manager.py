import json
import os

class LabelManager:
    def __init__(self, mapping_path="src/pill_safety/cv/attribute/labels/label_mapping.json"):
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Không tìm thấy file mapping nhãn tại: {mapping_path}")
            
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.mapping = json.load(f)
            
        self.shape_map = self.mapping["shape"]
        self.color_list = self.mapping["color"]

    def get_shape_name(self, idx):
        """Lấy tên hình dạng từ index"""
        return self.shape_map.get(str(idx), "UNKNOWN")

    def get_color_names(self, binary_vector):
        """Lấy danh sách tên màu từ vector multi-label (threshold >= 0.5)"""
        colors = []
        for idx, val in enumerate(binary_vector):
            if val >= 0.5:
                colors.append(self.color_list[idx])
        return colors

    @property
    def num_shapes(self):
        return len(self.shape_map)

    @property
    def num_colors(self):
        return len(self.color_list)