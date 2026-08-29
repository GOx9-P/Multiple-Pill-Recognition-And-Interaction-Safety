# Train Request — Logging and Evidence Contract

Tài liệu này quy định mỗi lần train phải log và lưu những gì để phục vụ:

- Báo cáo kết quả.
- So sánh giữa các lần train.
- Chứng minh split/augmentation không bị leakage.
- Chọn best checkpoint.
- Validate và đánh giá test sau này.

Hiện tại chỉ áp dụng cho 3 training job:

1. `segmentation_yolov11_full_finetune`
2. `attribute_resnet18_head_tune`
3. `attribute_resnet18_last_blocks_finetune`

---

## 1. Cấu Trúc Lưu Kết Quả Mỗi Run

Mỗi lần train phải có một `run_id` riêng.

```text
experiments/<module_name>/
├── checkpoints/
│   ├── <run_id>_best.pt
│   └── <run_id>_last.pt
├── logs/
│   ├── <run_id>_config.yaml
│   ├── <run_id>_dataset_manifest.json
│   ├── <run_id>_train_log.csv
│   └── <run_id>_runtime.txt
├── metrics/
│   ├── <run_id>_val_metrics.json
│   └── <run_id>_test_metrics.json
├── plots/
│   ├── <run_id>_loss_curve.png
│   ├── <run_id>_metric_curve.png
│   └── <run_id>_summary.png
└── predictions/
    └── <run_id>/
```

Ví dụ:

```text
experiments/segmentation_yolov11_full_finetune/
├── checkpoints/seg_v1_best.pt
├── logs/seg_v1_config.yaml
├── logs/seg_v1_dataset_manifest.json
├── logs/seg_v1_train_log.csv
├── metrics/seg_v1_val_metrics.json
├── plots/seg_v1_loss_curve.png
├── plots/seg_v1_metric_curve.png
└── predictions/seg_v1/
```

---

## 2. Thông Tin Bắt Buộc Cho Mọi Training Job

Mỗi run phải lưu đủ các nhóm thông tin sau.

| Nhóm | Cần log | Mục đích |
|---|---|---|
| Run info | `run_id`, module, người chạy, ngày chạy, git commit nếu có | Truy vết kết quả |
| Dataset info | dataset, số mẫu train/val/test, split file, anti-leakage rule | Chứng minh chia data đúng |
| Dataset manifest | danh sách file/split hoặc hash manifest, label mapping, class distribution | Tái lập đúng dữ liệu đã train |
| Augmentation | augmentation có dùng, online/offline, exact transform, số ảnh sinh thêm nếu có | Giải thích thay đổi dữ liệu train |
| Model info | architecture, pretrained weight, train strategy | Giải thích phương pháp transfer learning |
| Hyperparameters | image size, epochs, batch size, learning rate, optimizer, scheduler, seed | Tái lập thí nghiệm |
| Loss | train loss, val loss theo epoch | Chứng minh model có học |
| Metrics | metric validation tốt nhất và metric test nếu đã chạy | Đưa vào báo cáo |
| Plots | loss curve, metric curve, confusion matrix nếu có | Minh họa quá trình train trong báo cáo |
| Best checkpoint | epoch tốt nhất, metric dùng để chọn best, path checkpoint | Biết model nào được chọn |
| Runtime | device, GPU/CPU, thời gian train, thời gian mỗi epoch | Báo cáo chi phí tính toán |
| Error cases | sample dự đoán sai hoặc case khó | Phân tích hạn chế |

---

## 3. File Log Bắt Buộc

### 3.1. Run Config YAML

File:

```text
experiments/<module_name>/logs/<run_id>_config.yaml
```

Nội dung tối thiểu:

```yaml
run_id: seg_v1
module: segmentation_yolov11_full_finetune
seed: 42
dataset:
  name: MEDISEG
  split_file: data/splits/mediseg/split_v1.json
model:
  architecture: YOLOv11-Seg
  pretrained_weight: yolo11n-seg.pt
training:
  image_size: 640
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  optimizer: auto
  scheduler: default
augmentation:
  enabled: true
  split: train_only
```

### 3.2. Dataset Manifest JSON

File:

```text
experiments/<module_name>/logs/<run_id>_dataset_manifest.json
```

Nội dung tối thiểu:

```json
{
  "run_id": "seg_v1",
  "dataset_name": "MEDISEG",
  "split_file": "data/splits/mediseg/split_v1.json",
  "train_count": 5781,
  "val_count": 1239,
  "test_count": 1239,
  "split_before_augmentation": true,
  "augmentation_train_only": true,
  "label_mapping_file": null,
  "class_distribution": {}
}
```

Với attribute model, `label_mapping_file` bắt buộc phải có để inference decode đúng class.

### 3.3. Train Log CSV

File:

```text
experiments/<module_name>/logs/<run_id>_train_log.csv
```

Format tối thiểu:

```csv
epoch,train_loss,val_loss,learning_rate,best_metric,is_best
1,0.842,0.791,0.001,0.421,false
2,0.701,0.682,0.001,0.489,true
```

Nếu module có nhiều loss, thêm cột riêng.

Ví dụ attribute multi-head:

```csv
epoch,train_loss,val_loss,shape_loss,color_loss,learning_rate,best_metric,is_best
```

Nếu một head chưa train, cột loss của head đó có thể bỏ hoặc để `null`.

### 3.4. Runtime TXT

File:

```text
experiments/<module_name>/logs/<run_id>_runtime.txt
```

Nội dung tối thiểu:

```text
run_id: seg_v1
module: segmentation_yolov11_full_finetune
started_at: 2026-08-01 20:00
finished_at: 2026-08-01 23:10
device: cuda
gpu_name: NVIDIA T4
python_version: 3.11.9
torch_version: 2.6.0
cuda_available: true
total_train_time_minutes: 190
```

### 3.5. Validation Metrics JSON

File:

```text
experiments/<module_name>/metrics/<run_id>_val_metrics.json
```

Nội dung:

```json
{
  "run_id": "seg_v1",
  "module": "segmentation_yolov11_full_finetune",
  "split": "val",
  "best_epoch": 37,
  "best_checkpoint": "experiments/segmentation_yolov11_full_finetune/checkpoints/seg_v1_best.pt",
  "selection_metric": "instance_recall",
  "metrics": {}
}
```

### 3.6. Test Metrics JSON

File này chỉ tạo sau khi đã chốt model và chạy trên test set.

```text
experiments/<module_name>/metrics/<run_id>_test_metrics.json
```

Không dùng test set để tune hyperparameter.

### 3.7. Training Plots

Folder:

```text
experiments/<module_name>/plots/
```

Mỗi run cần lưu ít nhất:

```text
<run_id>_loss_curve.png
<run_id>_metric_curve.png
<run_id>_summary.png
```

Trong đó:

| Plot | Nội dung |
|---|---|
| `loss_curve.png` | Train loss và val loss theo epoch. |
| `metric_curve.png` | Metric chính trên validation theo epoch. |
| `summary.png` | Một hình tổng hợp ngắn để đưa vào slide/report. |

Với attribute model, cần thêm nếu có thể:

```text
<run_id>_shape_confusion_matrix.png
<run_id>_color_f1_per_class.png
```

Với segmentation model, cần thêm nếu có thể:

```text
<run_id>_precision_recall_curve.png
<run_id>_threshold_vs_recall.png
```

Plot phải được sinh từ `train_log.csv` hoặc `val_metrics.json`, không vẽ tay.

---

## 4. Job 1 — Segmentation YOLOv11-Seg Full Fine-tune

Module:

```text
segmentation_yolov11_full_finetune
```

Dataset:

```text
MEDISEG + augmentation/copy-paste nếu có
```

Train strategy:

```text
YOLOv11-Seg pretrained
-> full fine-tune trên MEDISEG
-> augmentation chỉ dùng train split
-> tune threshold trên validation split
```

### 4.1. Hyperparameters Cần Log

```json
{
  "model": "YOLOv11-Seg",
  "pretrained_weight": "yolo11n-seg.pt",
  "image_size": 640,
  "epochs": 50,
  "batch_size": 16,
  "learning_rate": 0.001,
  "optimizer": "auto_or_sgd_or_adamw",
  "scheduler": "default_or_custom",
  "seed": 42,
  "class_agnostic": true,
  "augmentation": {
    "enabled": true,
    "n_aug_per_image": 3,
    "copy_paste": false,
    "split": "train_only"
  }
}
```

### 4.2. Metrics Cần Log

Validation:

```json
{
  "metrics": {
    "instance_recall": 0.95,
    "merge_error_rate": 0.04,
    "mask_map50": 0.91,
    "mask_map50_95": 0.72,
    "precision": 0.93,
    "false_positive_rate": 0.03
  },
  "thresholds": {
    "detection_confidence_threshold": 0.25,
    "iou_threshold": 0.5,
    "mask_threshold": 0.5
  }
}
```

Ưu tiên báo cáo:

```text
Instance Recall
Merge Error Rate
Mask mAP@50
Mask mAP@50-95
```

`thresholds` phải được chọn trên validation split, không chọn trên test split.

### 4.3. Predictions Cần Lưu

```text
experiments/segmentation_yolov11_full_finetune/predictions/<run_id>/
├── easy/
├── touching/
├── overlap/
├── glare/
└── failed_cases/
```

Mỗi folder nên có ảnh overlay bbox/mask để đưa vào báo cáo.

### 4.4. Plots Cần Lưu

```text
experiments/segmentation_yolov11_full_finetune/plots/
├── <run_id>_loss_curve.png
├── <run_id>_metric_curve.png
├── <run_id>_precision_recall_curve.png
├── <run_id>_threshold_vs_recall.png
└── <run_id>_summary.png
```

Metric curve nên ưu tiên:

```text
Instance Recall theo epoch
Mask mAP@50 theo epoch
Mask mAP@50-95 theo epoch
```

---

## 5. Job 2 — Attribute ResNet18 Head-tune

Module:

```text
attribute_resnet18_head_tune
```

Dataset:

```text
NIH/RxImage + label normalization + multi-color handling
```

Label hiện tại đã rõ nhất cho:

```text
shape
color
```

Các head sau chỉ train nếu đã chuẩn hóa được label đủ tin cậy:

```text
dosage_form
scoreline
```

Train strategy:

```text
ResNet18 pretrained ImageNet
-> freeze backbone
-> train classification heads
```

### 5.1. Hyperparameters Cần Log

```json
{
  "model": "ResNet18",
  "pretrained_weight": "ImageNet",
  "train_strategy": "head_tune",
  "image_size": 224,
  "epochs": 30,
  "batch_size": 32,
  "learning_rate": 0.001,
  "optimizer": "adamw",
  "scheduler": "cosine_or_plateau",
  "seed": 42,
  "frozen_backbone": true,
  "trainable_layers": ["classification_heads"],
  "tasks": ["shape", "color"],
  "label_mapping_file": "data/processed/nih_attribute/label_mapping.json",
  "augmentation": {
    "enabled": true,
    "online": false,
    "sim2real": true
  }
}
```

### 5.2. Metrics Cần Log

```json
{
  "metrics": {
    "shape_macro_f1": 0.86,
    "color_macro_f1": 0.81,
    "overall_macro_f1": 0.835
  },
  "per_class_metrics": {
    "shape": {},
    "color": {}
  },
  "label_mapping_file": "data/processed/nih_attribute/label_mapping.json"
}
```

Ưu tiên báo cáo:

```text
Macro F1 từng attribute
Confusion matrix cho shape
Color multi-label F1
Per-class recall cho class ít dữ liệu
```

### 5.3. Predictions Cần Lưu

```text
experiments/attribute_resnet18_head_tune/predictions/<run_id>/
├── correct_samples/
├── wrong_shape/
├── wrong_color/
└── low_confidence/
```

Mỗi sample nên lưu:

```json
{
  "image_path": "outputs/example.jpg",
  "ground_truth": {
    "shape": "oval",
    "color": ["white"]
  },
  "prediction": {
    "shape": {"label": "polygon", "confidence": 0.62},
    "color": {"labels": ["white", "gray"], "confidence": 0.71}
  }
}
```

### 5.4. Plots Cần Lưu

```text
experiments/attribute_resnet18_head_tune/plots/
├── <run_id>_loss_curve.png
├── <run_id>_metric_curve.png
├── <run_id>_shape_confusion_matrix.png
├── <run_id>_color_f1_per_class.png
└── <run_id>_summary.png
```

Metric curve nên ưu tiên:

```text
Shape Macro F1 theo epoch
Color Macro F1 theo epoch
Overall Macro F1 theo epoch
```

---

## 6. Job 3 — Attribute ResNet18 Last-blocks Fine-tune

Module:

```text
attribute_resnet18_last_blocks_finetune
```

Dataset:

```text
NIH/RxImage giống head-tune, dùng cùng split để so sánh công bằng
```

Last-blocks fine-tune phải dùng cùng `label_mapping_file` và cùng split với head-tune.

Train strategy:

```text
Load checkpoint tốt nhất từ head-tune
-> unfreeze last ResNet blocks
-> train với learning rate nhỏ hơn
```

### 6.1. Hyperparameters Cần Log

```json
{
  "model": "ResNet18",
  "pretrained_from": "experiments/attribute_resnet18_head_tune/checkpoints/attr_head_v1_best.pt",
  "train_strategy": "last_blocks_finetune",
  "image_size": 224,
  "epochs": 20,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "optimizer": "adamw",
  "scheduler": "cosine_or_plateau",
  "seed": 42,
  "frozen_backbone": false,
  "trainable_layers": ["layer3", "layer4", "classification_heads"],
  "label_mapping_file": "data/processed/nih_attribute/label_mapping.json",
  "augmentation": {
    "enabled": true,
    "online": false,
    "sim2real": true
  }
}
```

### 6.2. Metrics Cần Log

Dùng cùng metric với head-tune:

```json
{
  "metrics": {
    "shape_macro_f1": 0.88,
    "color_macro_f1": 0.84,
    "overall_macro_f1": 0.855
  },
  "comparison_to_head_tune": {
    "head_tune_run_id": "attr_head_v1",
    "overall_macro_f1_delta": 0.02,
    "selected_for_inference": true
  },
  "per_class_metrics": {
    "shape": {},
    "color": {}
  }
}
```

### 6.3. Điều Kiện Chọn Last-blocks Model

Chỉ chọn last-blocks fine-tune làm model chính nếu:

```text
overall_macro_f1 tăng rõ trên validation
và không làm giảm mạnh class hiếm trên validation
```

Nếu cải thiện rất nhỏ hoặc overfit, giữ head-tune làm baseline chính.

### 6.4. Plots Cần Lưu

```text
experiments/attribute_resnet18_last_blocks_finetune/plots/
├── <run_id>_loss_curve.png
├── <run_id>_metric_curve.png
├── <run_id>_shape_confusion_matrix.png
├── <run_id>_color_f1_per_class.png
├── <run_id>_head_vs_last_blocks_comparison.png
└── <run_id>_summary.png
```

Plot so sánh head-tune vs last-blocks nên thể hiện:

```text
Shape F1 thay đổi bao nhiêu
Color F1 thay đổi bao nhiêu
Overall Macro F1 thay đổi bao nhiêu
Class hiếm có bị giảm mạnh không
```

---

## 7. Anti-leakage Yêu Cầu Khi Log

Mọi job phải ghi rõ split được tạo trước augmentation.

```json
{
  "split_policy": {
    "split_before_augmentation": true,
    "train_count": 0,
    "val_count": 0,
    "test_count": 0,
    "group_key": "source_image_group",
    "leakage_check_passed": true,
    "leakage_check_notes": ""
  }
}
```

Yêu cầu:

- Augmentation chỉ áp dụng train split.
- Val/test không chứa ảnh augment từ cùng ảnh gốc train.
- Attribute head-tune và last-blocks fine-tune phải dùng cùng split.
- Test set chỉ dùng sau khi đã chọn cấu hình cuối.

---

## 8. Bảng Báo Cáo Cuối Cần Có

### 8.1. Segmentation

| Run | Model | Image size | Epoch | Batch | Augmentation | Instance Recall | Merge Error Rate | Mask mAP@50 |
|---|---|---:|---:|---:|---|---:|---:|---:|
| seg_v1 | YOLOv11-Seg | 640 | 50 | 16 | yes | 0.95 | 0.04 | 0.91 |

Ảnh cần đưa vào báo cáo:

```text
loss_curve
metric_curve
mask overlay examples
failed cases
```

### 8.2. Attribute

| Run | Strategy | Image size | Epoch | Batch | LR | Shape F1 | Color F1 | Form F1 | Scoreline F1 | Overall F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attr_head_v1 | head-tune | 224 | 30 | 32 | 0.001 | 0.86 | 0.81 | N/A | N/A | 0.835 |
| attr_last_v1 | last-blocks | 224 | 20 | 32 | 0.0001 | 0.88 | 0.84 | N/A | N/A | 0.855 |

Ảnh cần đưa vào báo cáo:

```text
loss_curve
metric_curve
confusion_matrix
per-class F1 chart
head_vs_last_blocks_comparison
```

---

## 9. Checklist Trước Khi Kết Thúc Một Run

| Câu hỏi | Bắt buộc |
|---|---:|
| Có `train_log.csv` chưa? | yes |
| Có `config.yaml` chưa? | yes |
| Có `dataset_manifest.json` chưa? | yes |
| Có `val_metrics.json` chưa? | yes |
| Có `loss_curve.png` chưa? | yes |
| Có `metric_curve.png` chưa? | yes |
| Có plot phụ phù hợp với module chưa? | yes |
| Có lưu best/last checkpoint chưa? | yes |
| Có ghi hyperparameters chưa? | yes |
| Có ghi dataset split và anti-leakage chưa? | yes |
| Attribute model có lưu label mapping chưa? | yes |
| Có prediction/error cases để xem lỗi chưa? | yes |
| Nếu chạy test, có `test_metrics.json` riêng chưa? | yes |
