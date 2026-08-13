import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from pill_safety.cv.attribute.models.resnet18_multitask import MultiTaskResNet18
from pill_safety.cv.attribute.datasets.color_dataset import ColorDataset
from pill_safety.cv.attribute.utils.transforms import get_color_transforms

def find_optimal_thresholds():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = "experiments/attribute_resnet18_last_blocks_finetune/checkpoints/attr_last_blocks_v1_best.pt"
    
    print(f"[Info] Đang tải mô hình từ: {ckpt_path} trên thiết bị: {device}")
    
    # 1. Load model tốt nhất
    model = MultiTaskResNet18(num_shape_classes=5, num_color_classes=12, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 2. Load tập Validation của Color
    color_val_loader = DataLoader(
        ColorDataset(
            csv_file="data/splits/nih_attribute/color/val_multilabel.csv", 
            img_dir="data/image_all/nih_attribute/color", 
            transform=get_color_transforms()
        ), 
        batch_size=64, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 3. Thu thập toàn bộ xác suất (probabilities) và nhãn thật (targets)
    all_probs = []
    all_targets = []

    print("[Info] Đang trích xuất dự đoán trên tập Validation...")
    with torch.no_grad():
        for c_imgs, c_labels in color_val_loader:
            c_imgs = c_imgs.to(device)
            c_logits = model(c_imgs, task_type='color')
            probs = torch.sigmoid(c_logits).cpu().numpy()
            
            all_probs.append(probs)
            all_targets.append(c_labels.numpy())

    all_probs = np.vstack(all_probs)     # Shape: (N_samples, num_classes)
    all_targets = np.vstack(all_targets) # Shape: (N_samples, num_classes)

    # 4. Quét tìm ngưỡng tối ưu cho từng class (từ 0.1 đến 0.9)
    num_classes = all_probs.shape[1]
    optimal_thresholds = {}
    threshold_range = np.arange(0.1, 0.95, 0.05)

    print("[Info] Đang tiến hành tìm ngưỡng tối ưu cho từng nhãn màu...")
    for i in range(num_classes):
        best_thresh = 0.5
        best_f1 = 0.0
        
        y_true_col = all_targets[:, i]
        y_prob_col = all_probs[:, i]

        for th in threshold_range:
            y_pred_col = (y_prob_col >= th).astype(int)
            score = f1_score(y_true_col, y_pred_col, zero_division=0)
            
            if score > best_f1:
                best_f1 = score
                best_thresh = float(th)

        optimal_thresholds[f"color_{i}"] = round(best_thresh, 2)
        print(f" - Color Class {i}: Ngưỡng tốt nhất = {best_thresh:.2f} (F1: {best_f1:.4f})")

    # 5. Lưu kết quả thành file optimal_thresholds.json
    output_path = "experiments/attribute_resnet18_last_blocks_finetune/checkpoints/optimal_thresholds.json"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(optimal_thresholds, f, indent=4, ensure_ascii=False)
    
    print(f"[Done] Đã lưu file tại: {output_path}")

if __name__ == "__main__":
    find_optimal_thresholds()