import json
import logging
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# ==============================================================================
# 0. KHỞI TẠO ĐƯỜNG DẪN DỰ ÁN (PROJECT ROOT)
# ==============================================================================
# File: .../training/attribute_resnet18_last_blocks_finetune/eval/eval_last_blocks.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.pill_safety.cv.attribute.datasets.rximage_dataset import RxImageDataset
from src.pill_safety.cv.attribute.models.multitask_resnet import MultiTaskResNet18

# ==============================================================================
# 1. CẤU HÌNH & CHUẨN HÓA ĐƯỜNG DẪN DỮ LIỆU / EXPERIMENTS
# ==============================================================================
RUN_ID = "attr_last_v1"
MODULE_NAME = "attribute_resnet18_last_blocks_finetune"

# Đường dẫn Dữ liệu
DATA_DIR = PROJECT_ROOT / "data"
COMBINED_DIR = DATA_DIR / "splits" / "nih_attribute"
IMG_DIR = DATA_DIR / "image_all" / "nih_attribute"

# Đường dẫn Artifacts Thí nghiệm (Experiments)
EXP_DIR = PROJECT_ROOT / "experiments" / MODULE_NAME
PATHS = {
    "checkpoints": EXP_DIR / "checkpoints",
    "logs": EXP_DIR / "logs",
    "metrics": EXP_DIR / "metrics",
    "plots": EXP_DIR / "plots",
    "predictions": EXP_DIR / "predictions" / RUN_ID,
}

# Khởi tạo toàn bộ cây thư mục artifact
for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)

# Tạo các thư mục con phân loại dự đoán
PRED_SUBDIRS = ["correct_samples", "wrong_shape", "wrong_color", "low_confidence"]
for sub_pred in PRED_SUBDIRS:
    (PATHS["predictions"] / sub_pred).mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.5  # Ngưỡng độ tin cậy cho Shape

def setup_logger(log_file):
    logger = logging.getLogger(f"{RUN_ID}_eval")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)
    return logger

logger = setup_logger(PATHS["logs"] / f"{RUN_ID}_eval_runtime.log")
logger.info(f"Khởi chạy Evaluation cho Run ID: {RUN_ID}")
logger.info(f"Device đang sử dụng: {DEVICE}")

# Nạp Dataset Manifest
manifest_path = PATHS["logs"] / f"{RUN_ID}_dataset_manifest.json"
if not manifest_path.exists():
    logger.error(f"Không tìm thấy dataset manifest tại: {manifest_path}")
    raise FileNotFoundError(
        f"Không tìm thấy dataset manifest tại: {manifest_path}\n"
        f"Vui lòng kiểm tra lại quá trình huấn luyện (training phase)."
    )

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

NUM_SHAPE_CLASSES = manifest["num_shape_classes"]
NUM_COLOR_CLASSES = manifest["num_color_classes"]

# ==============================================================================
# 2. DATASET & DATALOADER
# ==============================================================================
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_csv = COMBINED_DIR / "val_combined_crop.csv"
test_csv = COMBINED_DIR / "test_combined_crop.csv"

# Kiểm tra tồn tại CSV
for csv_f in [val_csv, test_csv]:
    if not csv_f.exists():
        logger.error(f"Không tìm thấy file dataset: {csv_f}")
        raise FileNotFoundError(f"Không tìm thấy file tập dữ liệu tại: {csv_f}")

val_dataset = RxImageDataset(val_csv, IMG_DIR, transform=data_transform)
test_dataset = RxImageDataset(test_csv, IMG_DIR, transform=data_transform)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)

# ==============================================================================
# 3. LOAD MODEL & CHECKPOINT
# ==============================================================================
model = MultiTaskResNet18(
    num_shape_classes=NUM_SHAPE_CLASSES, 
    num_color_classes=NUM_COLOR_CLASSES
).to(DEVICE)

checkpoint_path = PATHS["checkpoints"] / f"{RUN_ID}_best.pt"
if not checkpoint_path.exists():
    logger.error(f"Không tìm thấy checkpoint: {checkpoint_path}")
    raise FileNotFoundError(f"Không tìm thấy file checkpoint trọng số tại: {checkpoint_path}")

model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
model.eval()
logger.info(f"Đã load checkpoint trọng số từ: {checkpoint_path}")

# ==============================================================================
# 4. TỐI ƯU THRESHOLD MÀU SẮC TRÊN VAL SET
# ==============================================================================
logger.info("Bắt đầu tìm Threshold tối ưu cho thuộc tính Màu sắc trên Validation set...")
val_color_probs, val_color_targets = [], []
with torch.no_grad():
    for images, _, c_targets, _ in val_loader:
        images = images.to(DEVICE)
        _, c_outputs = model(images)
        val_color_probs.append(torch.sigmoid(c_outputs).cpu().numpy())
        val_color_targets.append(c_targets.cpu().numpy())

val_color_probs = np.vstack(val_color_probs)
val_color_targets = np.vstack(val_color_targets)

best_thresholds = []
for i in range(NUM_COLOR_CLASSES):
    b_thresh, b_f1 = 0.5, 0.0
    for thresh in np.arange(0.1, 0.9, 0.05):
        score = f1_score(
            val_color_targets[:, i], 
            (val_color_probs[:, i] > thresh).astype(int), 
            zero_division=0
        )
        if score > b_f1:
            b_f1 = score
            b_thresh = float(thresh)
    best_thresholds.append(b_thresh)

best_thresholds = np.array(best_thresholds)

# Lưu Optimal Thresholds
thresh_out_file = PATHS["metrics"] / f"{RUN_ID}_optimal_thresholds.json"
with open(thresh_out_file, "w", encoding="utf-8") as f:
    json.dump(best_thresholds.tolist(), f, indent=4)
logger.info(f"Đã lưu Optimal Thresholds tại: {thresh_out_file}")

# ==============================================================================
# 5. DỰ ĐOÁN & ĐÁNH GIÁ TRÊN TEST SET
# ==============================================================================
logger.info("Bắt đầu đánh giá trên Test set...")
img_filenames, test_shape_preds, test_shape_targets = [], [], []
test_shape_confs = []
test_color_probs_list, test_color_targets_list = [], []

with torch.no_grad():
    for images, s_targets, c_targets, fnames in test_loader:
        images = images.to(DEVICE)
        s_outputs, c_outputs = model(images)
        
        # Softmax & Confidence cho Shape
        s_probs = torch.softmax(s_outputs, dim=1)
        s_conf, s_preds = torch.max(s_probs, dim=1)

        test_shape_preds.extend(s_preds.cpu().numpy())
        test_shape_targets.extend(s_targets.cpu().numpy())
        test_shape_confs.extend(s_conf.cpu().numpy())
        
        test_color_probs_list.append(torch.sigmoid(c_outputs).cpu().numpy())
        test_color_targets_list.append(c_targets.cpu().numpy())
        img_filenames.extend(fnames)

test_color_probs = np.vstack(test_color_probs_list)
test_color_targets = np.vstack(test_color_targets_list)
test_color_preds = (test_color_probs > best_thresholds).astype(int)

test_shape_targets = np.array(test_shape_targets)
test_shape_preds = np.array(test_shape_preds)

test_shape_f1 = float(f1_score(test_shape_targets, test_shape_preds, average="macro", zero_division=0))
test_color_f1 = float(f1_score(test_color_targets, test_color_preds, average="macro", zero_division=0))

test_metrics_contract = {
    "run_id": RUN_ID,
    "module": MODULE_NAME,
    "split": "test",
    "metrics": {
        "shape_macro_f1": round(test_shape_f1, 4),
        "color_macro_f1": round(test_color_f1, 4),
        "overall_macro_f1": round((test_shape_f1 + test_color_f1) / 2.0, 4)
    }
}

metrics_out_file = PATHS["metrics"] / f"{RUN_ID}_test_metrics.json"
with open(metrics_out_file, "w", encoding="utf-8") as f:
    json.dump(test_metrics_contract, f, indent=4)

logger.info(f"Test Shape Macro F1: {test_shape_f1:.4f}")
logger.info(f"Test Color Macro F1: {test_color_f1:.4f}")

# ==============================================================================
# 6. PHÂN LOẠI VÀ LƯU SAMPLES DỰ ĐOÁN
# ==============================================================================
logger.info("Phân loại các ảnh Test vào các thư mục predictions...")
for idx, img_name in enumerate(img_filenames):
    src_img_path = IMG_DIR / img_name
    if not src_img_path.exists():
        continue
    
    shape_correct = (test_shape_preds[idx] == test_shape_targets[idx])
    color_correct = np.array_equal(test_color_preds[idx], test_color_targets[idx])
    conf_score = test_shape_confs[idx]

    # Phân loại ưu tiên: Low Confidence -> Wrong Shape -> Wrong Color -> Correct
    if conf_score < CONFIDENCE_THRESHOLD:
        dest_folder = PATHS["predictions"] / "low_confidence"
    elif not shape_correct:
        dest_folder = PATHS["predictions"] / "wrong_shape"
    elif not color_correct:
        dest_folder = PATHS["predictions"] / "wrong_color"
    else:
        dest_folder = PATHS["predictions"] / "correct_samples"

    shutil.copy(src_img_path, dest_folder / img_name)

# ==============================================================================
# 7. XUẤT ĐỒ THỊ VÀ ARTIFACTS
# ==============================================================================
logger.info("Vẽ đồ thị Confusion Matrix & So sánh F1 Score...")

# Shape Confusion Matrix
cm = confusion_matrix(test_shape_targets, test_shape_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Shape Confusion Matrix ({RUN_ID})")
plt.savefig(PATHS["plots"] / f"{RUN_ID}_shape_confusion_matrix.png", dpi=300)
plt.close()

# Head-tune vs Last-blocks Comparison Plot
head_tune_baseline = {"Shape F1": 0.86, "Color F1": 0.81, "Overall F1": 0.835}
last_blocks_result = {
    "Shape F1": test_shape_f1, 
    "Color F1": test_color_f1, 
    "Overall F1": (test_shape_f1 + test_color_f1) / 2.0
}

labels = list(head_tune_baseline.keys())
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, list(head_tune_baseline.values()), width, label="Head-tune (Baseline)", color="gray")
plt.bar(x + width/2, list(last_blocks_result.values()), width, label="Last-blocks (Fine-tuned)", color="navy")
plt.ylabel("F1 Score")
plt.title("Comparison: Head-tune vs Last-blocks Fine-tune")
plt.xticks(x, labels)
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig(PATHS["plots"] / f"{RUN_ID}_head_vs_last_blocks_comparison.png", dpi=300)
plt.close()

logger.info(f"[Evaluation Finished] Đã hoàn thành đánh giá Test set.")
logger.info(f" -> Artifacts lưu tại: {EXP_DIR.resolve()}")