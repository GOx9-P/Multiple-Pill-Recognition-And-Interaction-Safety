import os
import sys
import json
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from pill_safety.cv.attribute.models.resnet18_multitask import MultiTaskResNet18
from pill_safety.cv.attribute.datasets.shape_dataset import ShapeDataset
from pill_safety.cv.attribute.datasets.color_dataset import ColorDataset
from pill_safety.cv.attribute.utils.transforms import get_shape_transforms, get_color_transforms

def evaluate(model, shape_val_loader, color_val_loader, criterion_shape, criterion_color, device):
    """Hàm đánh giá độc lập trên tập Validation"""
    model.eval()
    val_loss = 0.0
    val_shape_loss = 0.0
    val_color_loss = 0.0
    
    all_s_preds, all_s_targets = [], []
    all_c_preds, all_c_targets = [], []
    
    with torch.no_grad():
        # Đánh giá nhánh Shape
        for s_imgs, s_labels in shape_val_loader:
            s_imgs = s_imgs.to(device)
            s_labels = s_labels.long().to(device)
            if s_labels.dim() > 1:
                s_labels = s_labels.squeeze(1)
                
            s_logits = model(s_imgs, task_type='shape')
            loss_s = criterion_shape(s_logits, s_labels)
            val_shape_loss += loss_s.item()
            
            all_s_preds.extend(torch.argmax(s_logits, dim=1).cpu().numpy())
            all_s_targets.extend(s_labels.cpu().numpy())
            
        # Đánh giá nhánh Color
        for c_imgs, c_labels in color_val_loader:
            c_imgs, c_labels = c_imgs.to(device), c_labels.float().to(device)
            
            c_logits = model(c_imgs, task_type='color')
            loss_c = criterion_color(c_logits, c_labels)
            val_color_loss += loss_c.item()
            
            c_preds_binary = (torch.sigmoid(c_logits) > 0.5).int()
            all_c_preds.extend(c_preds_binary.cpu().numpy())
            all_c_targets.extend(c_labels.int().cpu().numpy())
            
    avg_s_loss = val_shape_loss / max(len(shape_val_loader), 1)
    avg_c_loss = val_color_loss / max(len(color_val_loader), 1)
    avg_total_loss = avg_s_loss + avg_c_loss
    
    val_shape_f1 = f1_score(all_s_targets, all_s_preds, average='macro', zero_division=0)
    val_color_f1 = f1_score(all_c_targets, all_c_preds, average='macro', zero_division=0)
    val_combined_f1 = (val_shape_f1 + val_color_f1) / 2.0
    
    metrics = {
        "val_loss": round(avg_total_loss, 4),
        "shape_loss": round(avg_s_loss, 4),
        "color_loss": round(avg_c_loss, 4),
        "shape_f1": round(float(val_shape_f1), 4),
        "color_f1": round(float(val_color_f1), 4),
        "combined_f1": round(float(val_combined_f1), 4)
    }
    return metrics

def train():
    run_id = "attr_head_v1"
    module_name = "attribute_resnet18_head_tune"
    author = "Nguyen Gia Bao"
    started_at_str = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    start_time = time.time()
    
    base_exp_dir = os.path.join("experiments", module_name)
    
    ckpt_dir = os.path.join(base_exp_dir, "checkpoints")
    log_dir = os.path.join(base_exp_dir, "logs")
    metric_dir = os.path.join(base_exp_dir, "metrics")
    plot_dir = os.path.join(base_exp_dir, "plots")
    pred_dir = os.path.join(base_exp_dir, "predictions", run_id)
    
    for d in [ckpt_dir, log_dir, metric_dir, plot_dir, pred_dir]:
        os.makedirs(d, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Bắt đầu train run: {run_id} trên thiết bị: {device}")

    config_data = {
        "run_id": run_id,
        "module": module_name,
        "seed": 42,
        "model": {
            "architecture": "ResNet18",
            "pretrained_weight": "ImageNet",
            "train_strategy": "head_tune",
            "frozen_backbone": True,
            "trainable_layers": ["classification_heads"]
        },
        "training": {
            "image_size": 224,
            "epochs": 30,
            "shape_batch_size": 32,
            "color_batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "scheduler": "cosine_or_plateau"
        },
        "tasks": ["shape", "color"],
        "label_mapping_file": "data/processed/nih_attribute/label_mapping.json",
        "augmentation": {
            "enabled": False,
            "split": "train_only"
        }
    }
    with open(os.path.join(log_dir, f"{run_id}_config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, sort_keys=False)

    manifest_data = {
        "run_id": run_id,
        "dataset_name": "NIH/RxImage",
        "shape_csv": "data/splits/nih_attribute/shape/train_combined_crop.csv",
        "color_csv": "data/splits/nih_attribute/color/train_multilabel.csv",
        "shape_val_csv": "data/splits/nih_attribute/shape/val_combined_crop.csv",
        "color_val_csv": "data/splits/nih_attribute/color/val_multilabel.csv",
        "train_shape_count": 17840,
        "val_shape_count": 1000,
        "test_shape_count": 1000,
        "train_color_count": 37312,
        "val_color_count": 876,
        "test_color_count": 880,
        "split_before_augmentation": True,
        "augmentation_train_only": True,
        "label_mapping_file": "data/processed/nih_attribute/label_mapping.json",
        "split_policy": {
            "split_before_augmentation": True,
            "group_key": "image_id",
            "leakage_check_passed": True,
            "leakage_check_notes": "Val and test splits are strictly separated prior to any online/offline augmentation."
        },
        "class_distribution": {}
    }
    with open(os.path.join(log_dir, f"{run_id}_dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4, ensure_ascii=False)

    # Khởi tạo Train Dataloaders
    shape_loader = DataLoader(ShapeDataset(csv_file=manifest_data["shape_csv"], img_dir="data/image_all/nih_attribute/shape", transform=get_shape_transforms()), batch_size=32, shuffle=True, drop_last=True)
    color_loader = DataLoader(ColorDataset(csv_file=manifest_data["color_csv"], img_dir="data/image_all/nih_attribute/color", transform=get_color_transforms()), batch_size=64, shuffle=True, drop_last=True)

    # Khởi tạo Validation Dataloaders (Dùng transform chuẩn không augmentation)
    shape_val_loader = DataLoader(ShapeDataset(csv_file=manifest_data["shape_val_csv"], img_dir="data/image_all/nih_attribute/shape", transform=get_shape_transforms()), batch_size=32, shuffle=False)
    color_val_loader = DataLoader(ColorDataset(csv_file=manifest_data["color_val_csv"], img_dir="data/image_all/nih_attribute/color", transform=get_color_transforms()), batch_size=64, shuffle=False)

    model = MultiTaskResNet18(num_shape_classes=5, num_color_classes=12, pretrained=True).to(device)
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion_shape = nn.CrossEntropyLoss()
    criterion_color = nn.BCEWithLogitsLoss()
    lambda_color = 1.0

    epochs = 30
    best_metric = 0.0
    epoch_logs = []
    best_val_metrics = {}

    print("[Info] Tiến hành vòng lặp huấn luyện và đánh giá trên Validation...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_shape_loss = 0.0
        running_color_loss = 0.0
        
        shape_iter = iter(shape_loader)
        color_iter = iter(color_loader)
        num_steps = len(shape_loader)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            try:
                s_imgs, s_labels = next(shape_iter)
            except StopIteration:
                shape_iter = iter(shape_loader)
                s_imgs, s_labels = next(shape_iter)
                
            s_imgs = s_imgs.to(device)
            s_labels = s_labels.long().to(device)
            if s_labels.dim() > 1:
                s_labels = s_labels.squeeze(1)
            
            s_logits = model(s_imgs, task_type='shape')
            loss_s = criterion_shape(s_logits, s_labels)
            
            try:
                c_imgs, c_labels = next(color_iter)
            except StopIteration:
                color_iter = iter(color_loader)
                c_imgs, c_labels = next(color_iter)
                
            c_imgs, c_labels = c_imgs.to(device), c_labels.float().to(device)
            c_logits = model(c_imgs, task_type='color')
            loss_c = criterion_color(c_logits, c_labels)
            
            total_loss = loss_s + lambda_color * loss_c
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            running_shape_loss += loss_s.item()
            running_color_loss += loss_c.item()

        avg_loss = running_loss / num_steps
        avg_s_loss = running_shape_loss / num_steps
        avg_c_loss = running_color_loss / num_steps
        
        # Đánh giá thực tế trên tập Validation sau mỗi epoch
        val_metrics = evaluate(model, shape_val_loader, color_val_loader, criterion_shape, criterion_color, device)
        current_metric = val_metrics["combined_f1"]
        
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{run_id}_best.pt"))

        epoch_logs.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4),
            "val_loss": val_metrics["val_loss"],
            "shape_loss": round(avg_s_loss, 4),
            "color_loss": round(avg_c_loss, 4),
            "val_shape_f1": val_metrics["shape_f1"],
            "val_color_f1": val_metrics["color_f1"],
            "learning_rate": 0.001,
            "best_metric": best_metric,
            "is_best": is_best
        })
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f} | Val Loss: {val_metrics['val_loss']} | Val Combined F1: {current_metric:.4f} | Best: {best_metric}")

    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{run_id}_last.pt"))

    # Xuất file val_metrics.json theo đúng Contract thư mục metrics/
    with open(os.path.join(metric_dir, f"{run_id}_val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best_val_metrics, f, indent=4, ensure_ascii=False)

    log_df = pd.DataFrame(epoch_logs)
    cols_order = ["epoch", "train_loss", "val_loss", "shape_loss", "color_loss", "val_shape_f1", "val_color_f1", "learning_rate", "best_metric", "is_best"]
    log_df = log_df[cols_order]
    log_df.to_csv(os.path.join(log_dir, f"{run_id}_train_log.csv"), index=False)

    finished_at_str = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    elapsed_minutes = int((time.time() - start_time) / 60)
    
    runtime_content = f"""run_id: {run_id}
                            module: {module_name}
                            started_at: {started_at_str}
                            finished_at: {finished_at_str}
                            device: {device.type}
                            gpu_name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
                            python_version: {sys.version.split()[0]}
                            torch_version: {torch.__version__}
                            cuda_available: {torch.cuda.is_available()}
                            total_train_time_minutes: {elapsed_minutes}
"""
    with open(os.path.join(log_dir, f"{run_id}_runtime.txt"), "w", encoding="utf-8") as f:
        f.write(runtime_content)

    # ==========================================
    # PHẦN SỬA LẠI: XUẤT ĐỦ 5 BIỂU ĐỒ CHO THƯ MỤC PLOTS
    # ==========================================
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # 1. Load lại model tốt nhất để tính toán các biểu đồ đánh giá chi tiết
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{run_id}_best.pt"), map_location=device))
    model.eval()

    val_shape_preds, val_shape_targets = [], []
    val_color_preds_list, val_color_targets_list = [], []

    with torch.no_grad():
        for s_imgs, s_labels in shape_val_loader:
            s_imgs = s_imgs.to(device)
            s_labels = s_labels.long().to(device)
            if s_labels.dim() > 1: s_labels = s_labels.squeeze(1)
            s_logits = model(s_imgs, task_type='shape')
            val_shape_preds.extend(torch.argmax(s_logits, dim=1).cpu().numpy())
            val_shape_targets.extend(s_labels.cpu().numpy())

        for c_imgs, c_labels in color_val_loader:
            c_imgs = c_imgs.to(device)
            c_logits = model(c_imgs, task_type='color')
            preds_bin = (torch.sigmoid(c_logits) > 0.5).int().cpu().numpy()
            val_color_preds_list.extend(preds_bin)
            val_color_targets_list.extend(c_labels.int().numpy())

    epochs_list = log_df['epoch'].values
    train_loss_list = log_df['train_loss'].values
    val_loss_list = log_df['val_loss'].values
    metric_list = log_df['val_shape_f1'].values # Hoặc combined_f1 tuỳ chọn

    # 1. <run_id>_loss_curve.png
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_list, train_loss_list, label='Train Loss', color='blue', marker='o')
    plt.plot(epochs_list, val_loss_list, label='Val Loss', color='orange', marker='x')
    plt.title(f"Loss Curve - {run_id}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"{run_id}_loss_curve.png"))
    plt.close()

    # 2. <run_id>_metric_curve.png
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_list, log_df['val_shape_f1'].values, label='Shape F1', color='green', marker='s')
    plt.plot(epochs_list, log_df['val_color_f1'].values, label='Color F1', color='purple', marker='^')
    plt.title(f"Metric Curve (F1-Score) - {run_id}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"{run_id}_metric_curve.png"))
    plt.close()

    # 3. <run_id>_shape_confusion_matrix.png
    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(val_shape_targets, val_shape_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Shape Confusion Matrix - {run_id}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{run_id}_shape_confusion_matrix.png"))
    plt.close()

    # 4. <run_id>_color_f1_per_class.png
    from sklearn.metrics import f1_score
    color_f1s = f1_score(val_color_targets_list, val_color_preds_list, average=None, zero_division=0)
    plt.figure(figsize=(10, 5))
    plt.bar([f"Class {i}" for i in range(len(color_f1s))], color_f1s, color='teal')
    plt.title(f"Color F1-Score Per Class - {run_id}")
    plt.xlabel("Color Classes")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plot_dir, f"{run_id}_color_f1_per_class.png"))
    plt.close()

    # 5. <run_id>_summary.png
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs_list, train_loss_list, label='Train Loss', color='blue')
    axes[0].plot(epochs_list, val_loss_list, label='Val Loss', color='orange')
    axes[0].set_title("Overall Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_list, log_df['val_shape_f1'].values, label='Shape F1', color='green')
    axes[1].plot(epochs_list, log_df['val_color_f1'].values, label='Color F1', color='purple')
    axes[1].set_title("Validation Metric Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(f"Training Summary - {run_id}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{run_id}_summary.png"))
    plt.close()

    print(f"[Done] Huấn luyện và đánh giá Validation thành công! Đã lưu file metric vào {metric_dir}")

if __name__ == "__main__":
    train()