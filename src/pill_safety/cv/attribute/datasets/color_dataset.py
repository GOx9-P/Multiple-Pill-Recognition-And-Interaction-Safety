import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class ColorDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Các cột one-hot color theo schema đã đọc
        self.color_cols = [
            'label_BLACK', 'label_BLUE', 'label_BROWN', 'label_GRAY', 
            'label_GREEN', 'label_ORANGE', 'label_PINK', 'label_PURPLE', 
            'label_RED', 'label_TURQUOISE', 'label_WHITE', 'label_YELLOW'
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['rximageFileName'])
        if not img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        else:
            img_path = os.path.join(self.img_dir, img_name)
            
        image = Image.open(img_path).convert('RGB')
        
        # Multi-label vector cho color
        labels = torch.tensor([float(row[col]) for col in self.color_cols], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels