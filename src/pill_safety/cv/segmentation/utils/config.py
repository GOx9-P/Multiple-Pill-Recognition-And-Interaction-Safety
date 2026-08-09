from pathlib import Path

# ----------------------------------------------------------------------------
# Từ mediseg_split_augment.py — GIỮ NGUYÊN
# ----------------------------------------------------------------------------
RAW_IMAGES_DIR = Path(r"d:\PRJ_MLIoT\MEDISEG\32pills\images")   # <-- EDIT nếu cần
RAW_ANN_PATH = Path(r"d:\PRJ_MLIoT\MEDISEG\32pills\annotations.json")  # <-- EDIT nếu cần
OUTPUT_DIR = Path(r"d:\PRJ_MLIoT\MEDISEG\32pills\mediseg_yolo")  # <-- EDIT nếu cần

SPLIT_RATIOS = (0.70, 0.15, 0.15)          # train / val / test
RANDOM_SEED = 42

CLASS_AGNOSTIC = True                       # True = Module 1 (binary pill-vs-background)
                                             # False = giữ nguyên 32 class gốc

N_AUG_PER_IMAGE = 3                         # số bản augment sinh thêm mỗi ảnh train

BASE_WEIGHTS = "yolo11m-seg.pt"    # pretrained COCO-seg -> full fine-tune từ đây
                                    # nhẹ hơn: yolo11s-seg.pt | nặng hơn: yolo11l-seg.pt
IMGSZ = 640

EPOCHS = 50
BATCH = 9
PATIENCE = 10       # early stopping
DEVICE = 0            # "cpu" nếu không có GPU
FREEZE = None          # None = full fine-tune, KHÔNG freeze layer nào

EXPERIMENT_NAME = "module1_full_finetune_v1"
EXPERIMENTS_ROOT = Path("experiments")   # experiments/<EXPERIMENT_NAME>/{weights,metrics}

# Evaluation
EVAL_WEIGHTS_NAME = "best.pt"
EVAL_SPLIT = "val"
EVAL_CONF_THRESHOLD = 0.25
EVAL_IOU_THRESHOLD = 0.6
EVAL_CONF_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
EVAL_IOU_THRESHOLDS = [0.4, 0.5, 0.6, 0.7]
EVAL_MASK_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
EVAL_SELECTION_METRIC = "mask_mAP50_95"
TARGET_MASK_MAP50_95 = 0.85   # chỉ tiêu Module 1 theo Report.pdf
