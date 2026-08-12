import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

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
    metric_dir = os.path.join(base_exp_dir, "metrics")
    pred_dir = os.path.join(base_exp_dir, "predictions", run_id)
    
    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Đang đánh giá tập [test] cho run: {run_id}")

    shape_loader = DataLoader(
        ShapeDataset(csv_file="data/splits/nih_attribute/shape/test_combined_crop.csv", img_dir="data/image_all/nih_attribute/shape", transform=get_shape_transforms()), 
        batch_size=32, shuffle=False
    )
    color_loader = DataLoader(
        ColorDataset(csv_file="data/splits/nih_attribute/color/test_multilabel.csv", img_dir="data/image_all/nih_attribute/color", transform=get_color_transforms()), 
        batch_size=32, shuffle=False
    )

    model = MultiTaskResNet18(num_shape_classes=5, num_color_classes=12, pretrained=False).to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"[Warning] Không tìm thấy checkpoint tại {ckpt_path}. Model chạy với trạng thái ngẫu nhiên.")
    model.eval()

    all_shape_preds, all_shape_targets = [], []
    all_color_preds, all_color_targets = [], []

    with torch.no_grad():
        for images, labels in shape_loader:
            s_logits = model(images.to(device), task_type='shape')
            all_shape_preds.extend(torch.argmax(s_logits, dim=1).cpu().numpy())
            
            if labels.dim() > 1:
                labels = labels.squeeze(1)
            all_shape_targets.extend(labels.long().numpy())
            
        for images, labels in color_loader:
            c_logits = model(images.to(device), task_type='color')
            all_color_preds.extend((torch.sigmoid(c_logits) >= 0.5).int().cpu().numpy())
            all_color_targets.extend(labels.float().numpy())

    all_shape_preds = np.array(all_shape_preds)
    all_shape_targets = np.array(all_shape_targets)
    all_color_preds = np.array(all_color_preds)
    all_color_targets = np.array(all_color_targets)

    # Tính toán Metrics chi tiết
    shape_macro_f1 = f1_score(all_shape_targets, all_shape_preds, average='macro', zero_division=0)
    color_macro_f1 = f1_score(all_color_targets, all_color_preds, average='macro', zero_division=0)
    overall_macro_f1 = (shape_macro_f1 + color_macro_f1) / 2.0

    metrics_content = {
        "run_id": run_id,
        "module": module_name,
        "split": "test",
        "best_checkpoint": ckpt_path,
        "selection_metric": "overall_macro_f1",
        "label_mapping_file": "data/processed/nih_attribute/label_mapping.json",
        "metrics": {
            "shape_macro_f1": round(float(shape_macro_f1), 4),
            "color_macro_f1": round(float(color_macro_f1), 4),
            "dosage_form_macro_f1": None,
            "scoreline_macro_f1": None,
            "overall_macro_f1": round(float(overall_macro_f1), 4)
        },
        "per_class_metrics": {
            "shape": {},
            "color": {}
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
    output_pred_path = os.path.join(pred_dir, "test_predictions.csv")
    preds_df.to_csv(output_pred_path, index=False, encoding="utf-8")

    # 3. Xuất biểu đồ (Plots)
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    plot_dir = os.path.join(base_exp_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # A. <run_id>_shape_confusion_matrix.png
    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(all_shape_targets, all_shape_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Test Shape Confusion Matrix - {run_id}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{run_id}_test_shape_confusion_matrix.png"))
    plt.close()

    # B. <run_id>_color_f1_per_class.png
    color_f1s = f1_score(all_color_targets, all_color_preds, average=None, zero_division=0)
    plt.figure(figsize=(10, 5))
    plt.bar([f"Color {i}" for i in range(len(color_f1s))], color_f1s, color='teal')
    plt.title(f"Test Color F1-Score Per Class - {run_id}")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plot_dir, f"{run_id}_test_color_f1_per_class.png"))
    plt.close()

    # C. <run_id>_summary.png (Tóm tắt nhanh)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
    axes[0].set_title("Shape Confusion Matrix")
    
    # Right: F1 Score Per Class
    axes[1].bar([f"C{i}" for i in range(len(color_f1s))], color_f1s, color='teal')
    axes[1].set_title("Color F1 Per Class")
    axes[1].set_ylim(0, 1.0)
    
    plt.suptitle(f"Test Summary - {run_id}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{run_id}_test_summary.png"))
    plt.close()

    print(f"[Done] Đã xuất toàn bộ biểu đồ phân tích vào: {plot_dir}")

    print(f"[Done] Đã lưu metrics vào: {output_metric_path}")
    print(f"[Done] Đã lưu bảng predictions chi tiết vào: {output_pred_path}")

if __name__ == "__main__":
    evaluate_test()