import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from pill_safety.cv.attribute.models.resnet18_multitask import MultiTaskResNet18
from pill_safety.cv.attribute.datasets.shape_dataset import ShapeDataset
from pill_safety.cv.attribute.datasets.color_dataset import ColorDataset
from pill_safety.cv.attribute.utils.transforms import get_shape_transforms, get_color_transforms

def evaluate_test():
    run_id = "attr_head_v1"
    module_name = "attribute_resnet18_head_tune"
    base_exp_dir = os.path.join("experiments", module_name)
    
    ckpt_path = os.path.join(base_exp_dir, "checkpoints", f"{run_id}_best.pt")
    threshold_path = os.path.join(base_exp_dir, "checkpoints", "optimal_thresholds.json")
    metric_dir = os.path.join(base_exp_dir, "metrics")
    pred_dir = os.path.join(base_exp_dir, "predictions")
    error_dir = pred_dir
    plot_dir = os.path.join(base_exp_dir, "plots")
    
    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Đang đánh giá tập [test] cho run: {run_id}")

    # --- Load file optimal_thresholds.json (nếu có) ---
    if os.path.exists(threshold_path):
        with open(threshold_path, "r", encoding="utf-8") as f:
            thresh_dict = json.load(f)
        # Chuyển đổi thành mảng numpy với kích thước 12 lớp màu tương ứng
        color_thresholds = np.array([thresh_dict[f"color_{i}"] for i in range(12)])
        print(f"[Info] Đã load thành công optimal_thresholds từ: {threshold_path}")
    else:
        color_thresholds = 0.5
        print("[Info] Không tìm thấy file optimal_thresholds.json. Sử dụng ngưỡng mặc định: 0.5")

    shape_loader = DataLoader(
        ShapeDataset(csv_file="data/splits/nih_attribute/shape/test_combined_crop.csv", img_dir="data/image_all/nih_attribute/shape", transform=get_shape_transforms()), 
        batch_size=32, shuffle=False
    )
    color_loader = DataLoader(
        ColorDataset(csv_file="data/splits/nih_attribute/color/test_multilabel.csv", img_dir="data/image_all/nih_attribute/color", transform=get_color_transforms()), 
        batch_size=64, shuffle=False
    )

    model = MultiTaskResNet18(num_shape_classes=5, num_color_classes=12, pretrained=False).to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"[Warning] Không tìm thấy checkpoint tại {ckpt_path}. Model chạy với trạng thái ngẫu nhiên.")
    model.eval()

    all_shape_preds, all_shape_targets = [], []
    all_color_preds, all_color_targets = [], []

    # Duyệt qua tập test đúng 1 lần duy nhất để lấy predictions và targets
    with torch.no_grad():
        for images, labels in shape_loader:
            s_logits = model(images.to(device), task_type='shape')
            all_shape_preds.extend(torch.argmax(s_logits, dim=1).cpu().numpy())
            
            if labels.dim() > 1:
                labels = labels.squeeze(1)
            all_shape_targets.extend(labels.long().numpy())
            
        for images, labels in color_loader:
            c_logits = model(images.to(device), task_type='color')
            c_probs = torch.sigmoid(c_logits).cpu().numpy()
            
            # Áp dụng threshold (có thể là mảng tối ưu hoặc số 0.5 mặc định)
            preds_bin = (c_probs >= color_thresholds).astype(int)
            
            all_color_preds.extend(preds_bin)
            all_color_targets.extend(labels.float().numpy())

    all_shape_preds = np.array(all_shape_preds)
    all_shape_targets = np.array(all_shape_targets)
    all_color_preds = np.array(all_color_preds)
    all_color_targets = np.array(all_color_targets)

    # --- Tính toán Metrics F1-Score ---
    shape_macro_f1 = f1_score(all_shape_targets, all_shape_preds, average='macro', zero_division=0)
    color_macro_f1 = f1_score(all_color_targets, all_color_preds, average='macro', zero_division=0)
    overall_macro_f1 = (shape_macro_f1 + color_macro_f1) / 2.0

    # --- Tính toán Metrics Accuracy ---
    shape_acc = accuracy_score(all_shape_targets, all_shape_preds)
    color_acc = accuracy_score(all_color_targets, all_color_preds)  # Subset accuracy cho multi-label
    overall_acc = (shape_acc + color_acc) / 2.0

    # --- Tính toán per-class metrics chi tiết để lấp đầy báo cáo ---
    shape_f1_per_class = f1_score(all_shape_targets, all_shape_preds, average=None, zero_division=0)
    shape_acc_per_class = []
    # Tính accuracy từng class cho shape (multiclass)
    for c in range(len(shape_f1_per_class)):
        mask = (all_shape_targets == c)
        if mask.sum() > 0:
            shape_acc_per_class.append(float((all_shape_preds[mask] == c).mean()))
        else:
            shape_acc_per_class.append(0.0)

    color_f1_per_class = f1_score(all_color_targets, all_color_preds, average=None, zero_division=0)
    color_acc_per_class = accuracy_score(all_color_targets, all_color_preds, normalize=False) # Hoặc per-label accuracy

    metrics_content = {
        "run_id": run_id,
        "module": module_name,
        "split": "test",
        "best_checkpoint": ckpt_path,
        "selection_metric": "overall_macro_f1",
        "label_mapping_file": "data/processed/nih_attribute/label_mapping.json",
        "metrics": {
            # F1-Score
            "shape_macro_f1": round(float(shape_macro_f1), 4),
            "color_macro_f1": round(float(color_macro_f1), 4),
            "overall_macro_f1": round(float(overall_macro_f1), 4),
            # Accuracy
            "shape_acc": round(float(shape_acc), 4),
            "color_acc": round(float(color_acc), 4),
            "overall_acc": round(float(overall_acc), 4)
        },
        "per_class_metrics": {
            "shape": {
                f"class_{i}": {
                    "f1_score": round(float(shape_f1_per_class[i]), 4),
                    "accuracy": round(float(shape_acc_per_class[i]), 4)
                } for i in range(len(shape_f1_per_class))
            },
            "color": {
                f"color_{i}": {
                    "f1_score": round(float(color_f1_per_class[i]), 4)
                } for i in range(len(color_f1_per_class))
            }
        }
    }

    # Lưu file test_metrics.json
    output_metric_path = os.path.join(metric_dir, f"{run_id}_test_metrics.json")
    with open(output_metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics_content, f, indent=4, ensure_ascii=False)

    # Lưu bảng predictions CSV
    shape_pred_df = pd.DataFrame({"shape_true": all_shape_targets, "shape_pred": all_shape_preds})
    color_true_df = pd.DataFrame(all_color_targets, columns=[f"color_true_{i}" for i in range(all_color_targets.shape[1])])
    color_pred_df = pd.DataFrame(all_color_preds, columns=[f"color_pred_{i}" for i in range(all_color_preds.shape[1])])
    
    preds_df = pd.concat([shape_pred_df, color_true_df, color_pred_df], axis=1)
    output_pred_path = os.path.join(pred_dir, f"{run_id}_test_predictions.csv")
    preds_df.to_csv(output_pred_path, index=False, encoding="utf-8")

    # ==========================================
    # XUẤT BIỂU ĐỒ (PLOTS) CHO TỪNG TASK RIÊNG BIỆT
    # ==========================================
    
    # ------------------------------------------
    # 1. TASK: SHAPE (Multi-class Classification)
    # ------------------------------------------
    
    # A. Confusion Matrix cho Shape (Đã chuẩn hóa tỷ lệ %)
    plt.figure(figsize=(7, 6))
    cm_shape = confusion_matrix(all_shape_targets, all_shape_preds, normalize='true') # Thêm normalize='true'
    
    # Dùng fmt='.2f' để hiển thị dạng số thập phân (ví dụ: 0.85 thay vì số đếm)
    sns.heatmap(cm_shape, annot=True, fmt='.2f', cmap='Blues', cbar=False) 
    
    plt.title(f"Shape - Normalized Confusion Matrix ({run_id})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{run_id}_shape_confusion_matrix.png"))
    plt.close()

    # ------------------------------------------
    # 2. TASK: COLOR (Multi-label Classification)
    # ------------------------------------------

    # A. F1-Score Per Class cho Color
    color_f1s = f1_score(all_color_targets, all_color_preds, average=None, zero_division=0)
    plt.figure(figsize=(10, 5))
    plt.bar([f"Color {i}" for i in range(len(color_f1s))], color_f1s, color='purple')
    plt.title(f"Color - F1-Score Per Class ({run_id})")
    plt.xlabel("Color Labels")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{run_id}_color_f1_per_class.png"))
    plt.close()

    print(f"[Done] Đã xuất toàn bộ biểu đồ phân tích vào: {plot_dir}")
    print(f"[Done] Đã lưu metrics vào: {output_metric_path}")
    print(f"[Done] Đã lưu bảng predictions chi tiết vào: {output_pred_path}")

    # ==========================================
    # TỔNG HỢP CÁC MẪU LỖI RA FILE CSV (SHAPE & COLOR)
    # ==========================================
    
    # 1. Chuẩn bị dữ liệu từ các dataset
    shape_test_df = shape_loader.dataset.data_frame.reset_index(drop=True)
    color_test_df = color_loader.dataset.df.reset_index(drop=True) # Lưu ý Color dùng .df
    
    # 2. Xác định các mẫu lỗi
    shape_error_indices = np.where(all_shape_targets != all_shape_preds)[0]
    # Lỗi Color: ít nhất 1 nhãn trong vector dự đoán khác với nhãn thật
    color_error_indices = np.where(~np.all(all_color_targets == all_color_preds, axis=1))[0]
    
    print(f"[Info] Số lượng mẫu sai Shape: {len(shape_error_indices)}")
    print(f"[Info] Số lượng mẫu sai Color: {len(color_error_indices)}")
    
    # 3. Tổng hợp Shape Errors
    shape_error_records = []
    for idx in shape_error_indices:
        row = shape_test_df.iloc[idx].to_dict()
        row.update({'true_shape': int(all_shape_targets[idx]), 'pred_shape': int(all_shape_preds[idx]), 'sample_index': int(idx)})
        shape_error_records.append(row)
        
    # 4. Tổng hợp Color Errors
    color_error_records = []
    for idx in color_error_indices:
        row = color_test_df.iloc[idx].to_dict()
        # Thêm thông tin nhãn vào records
        row.update({'true_color': all_color_targets[idx].tolist(), 'pred_color': all_color_preds[idx].tolist(), 'sample_index': int(idx)})
        color_error_records.append(row)
        
    # 5. Lưu ra file CSV
    pd.DataFrame(shape_error_records).to_csv(os.path.join(error_dir, "shape_error_cases.csv"), index=False)
    pd.DataFrame(color_error_records).to_csv(os.path.join(error_dir, "color_error_cases.csv"), index=False)
    
    # 6. Lưu summary JSON
    error_summary = {
        "total_shape_errors": int(len(shape_error_indices)),
        "total_color_errors": int(len(color_error_indices)),
        "shape_error_indices": shape_error_indices.tolist(),
        "color_error_indices": color_error_indices.tolist()
    }
    with open(os.path.join(error_dir, "error_cases_summary.json"), "w", encoding="utf-8") as f:
        json.dump(error_summary, f, indent=4, ensure_ascii=False)
        
    print(f"[Done] Đã lưu báo cáo lỗi cho cả Shape và Color tại: {error_dir}")

if __name__ == "__main__":
    evaluate_test()