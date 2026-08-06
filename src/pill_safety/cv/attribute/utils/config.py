import torch
from pathlib import Path

class AttributeConfig:
    # --- Base Paths (Tính động từ gốc dự án) ---
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    
    BASE_DIR = PROJECT_ROOT / "data" 
    COMBINED_DIR = BASE_DIR / "data" / "splits" / "nih_attribute"
    IMG_DIR = BASE_DIR / "data" / "image_all" / "nih_attribute"
    
    STAGE1_CHECKPOINT = PROJECT_ROOT / "experiments" / "attribute_resnet18_head_tune" / "checkpoints" / "best_heads_finetuned.pth"
    EXP_DIR = PROJECT_ROOT / "experiments" / "attribute_resnet18_last_blocks_finetune"
    RUN_ID = "attr_last_v1"
    
    CHECKPOINT_DIR = EXP_DIR / "checkpoints"
    LOG_DIR = EXP_DIR / "logs"
    METRIC_DIR = EXP_DIR / "metrics"
    PLOT_DIR = EXP_DIR / "plots"
    PRED_DIR = EXP_DIR / "predictions" / RUN_ID

    # --- Hyperparameters ---
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LR_BACKBONE = 1e-5
    LR_HEADS = 1e-4

    @classmethod
    def setup_directories(cls):
        for path in [cls.CHECKPOINT_DIR, cls.LOG_DIR, cls.METRIC_DIR, cls.PLOT_DIR, cls.PRED_DIR]:
            path.mkdir(parents=True, exist_ok=True)
        for sub in ["correct_samples", "wrong_shape", "wrong_color", "low_confidence"]:
            (cls.PRED_DIR / sub).mkdir(parents=True, exist_ok=True)