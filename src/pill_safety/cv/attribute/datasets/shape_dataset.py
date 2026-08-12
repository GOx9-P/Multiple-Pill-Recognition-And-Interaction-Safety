import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class ShapeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Dataset cho nhánh Shape (Multi-class Classification).
        
        Args:
            csv_file (str): Đường dẫn tới file CSV phân tách (ví dụ: train_combined_crop.csv).
            img_dir (str): Thư mục chứa toàn bộ ảnh của nhánh shape.
            transform (callable, optional): Các phép biến đổi (transforms) áp dụng trên ảnh.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Tự động nhận diện tên cột dựa trên cấu trúc file CSV thông thường
        # Bạn có thể thay thế trực tiếp tên cột nếu file CSV của bạn dùng tên khác (ví dụ: 'filename', 'label',...)
        self.image_col = 'image_path' if 'image_path' in self.data_frame.columns else self.data_frame.columns[0]
        self.label_col = 'label_shape' if 'label_shape' in self.data_frame.columns else self.data_frame.columns[1]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Chuyển đổi tensor index sang list nếu cần thiết
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Lấy tên file hoặc đường dẫn ảnh từ DataFrame
        img_name = str(self.data_frame.iloc[idx][self.image_col])
        img_path = os.path.join(self.img_dir, os.path.basename(img_name))

        # Mở ảnh bằng PIL và chuyển sang hệ màu RGB
        image = Image.open(img_path).convert('RGB')

        # Lấy nhãn dạng số nguyên (Multi-class)
        label = int(self.data_frame.iloc[idx][self.label_col])
    

        # Áp dụng các phép biến đổi (transforms) nếu được truyền vào
        if self.transform:
            image = self.transform(image)

        return image, label # Trả về label dạng scalar int chuẩn