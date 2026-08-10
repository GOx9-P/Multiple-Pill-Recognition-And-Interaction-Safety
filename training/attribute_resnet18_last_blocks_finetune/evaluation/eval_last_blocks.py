"""
Evaluation Script for Last-Blocks Fine-Tune (Stage 2).

Loads the best checkpoint from a given run, runs inference on the test set,
computes metrics (Shape/Color F1), and generates visualization artifacts.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# --- Project root setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pill_safety.cv.attribute.models import MultiTaskResNet18
from pill_safety.cv.attribute.datasets.rximage import RxImageDataset
from pill_safety.cv.attribute.transforms.augmentations import get_attribute_transforms
from pill_safety.cv.attribute.labels.label_mapping import load_label_mapping
from pill_safety.cv.attribute.evaluators.attribute_evaluator import AttributeEvaluator
from pill_safety.cv.attribute.utils.checkpoint import load_checkpoint
from pill_safety.cv.attribute.utils.config import AttributeConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Last-Blocks Fine-Tune")
    parser.add_argument("--run_id", type=str, required=True, help="Unique run identifier (e.g. attr_last_v2)")
    parser.add_argument("--head_run_id", type=str, required=True, help="Run ID of the corresponding head-tune")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    DEVICE = AttributeConfig.DEVICE
    MODULE_NAME = "attribute_resnet18_last_blocks_finetune"
    HEAD_MODULE = "attribute_resnet18_head_tune"

    # --- Paths ---
    paths = AttributeConfig.get_experiment_paths(MODULE_NAME, args.run_id)
    head_paths = AttributeConfig.get_experiment_paths(HEAD_MODULE, args.head_run_id)
    AttributeConfig.setup_directories(paths)

    # --- Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(paths["logs"] / f"{args.run_id}_evaluation.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"=== Evaluating Last-Blocks | Run ID: {args.run_id} ===")

    # --- Load Label Mapping (from head-tune) ---
    head_mapping_path = head_paths["logs"] / f"{args.head_run_id}_label_mapping.json"
    if not head_mapping_path.exists():
        logger.error(f"Missing label mapping at {head_mapping_path}")
        sys.exit(1)
        
    label_mapping, num_shape_classes, num_color_classes, mapping_hash = load_label_mapping(head_mapping_path)
    shape_names = label_mapping["shape"]
    color_names = label_mapping["color"]

    # --- Load Checkpoint ---
    ckpt_path = paths["checkpoints"] / f"{args.run_id}_best.pt"
    if not ckpt_path.exists():
        logger.error(f"Missing checkpoint at {ckpt_path}")
        sys.exit(1)

    ckpt = load_checkpoint(ckpt_path, DEVICE, expected_mapping_hash=mapping_hash)
    best_epoch = ckpt.get("epoch", 0)

    # --- Data ---
    test_csv = AttributeConfig.COMBINED_DIR / "test_clean.csv"
    if not test_csv.exists():
        test_csv = AttributeConfig.COMBINED_DIR / "test_combined_crop.csv"
        
    transforms_dict = get_attribute_transforms()
    test_dataset = RxImageDataset(test_csv, AttributeConfig.IMG_DIR, transform=transforms_dict["val"])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- Model ---
    model = MultiTaskResNet18(
        num_shape_classes=num_shape_classes,
        num_color_classes=num_color_classes,
        pretrained=False,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- Evaluation ---
    evaluator = AttributeEvaluator(
        model=model,
        test_loader=test_loader,
        device=DEVICE,
        paths=paths,
        run_id=args.run_id,
        module_name=MODULE_NAME,
        shape_class_names=shape_names,
        color_class_names=color_names,
        num_shape_classes=num_shape_classes,
        num_color_classes=num_color_classes,
    )

    logger.info("Running inference on test set...")
    preds = evaluator.collect_predictions()
    
    logger.info("Computing metrics...")
    test_metrics = evaluator.compute_test_metrics(preds)
    
    logger.info(f"Test Shape Macro F1: {test_metrics['shape_macro_f1']:.4f}")
    logger.info(f"Test Color Macro F1: {test_metrics['color_macro_f1']:.4f}")
    logger.info(f"Test Overall F1: {test_metrics['overall_macro_f1']:.4f}")

    evaluator.save_test_metrics(test_metrics, best_epoch, label_mapping_file=str(head_mapping_path.relative_to(PROJECT_ROOT)))
    evaluator.plot_confusion_matrix(preds)
    evaluator.save_predictions(preds, test_csv, filenames_col="rxnavImageFileName")
    
    logger.info("Evaluation complete! Artifacts saved successfully.")


if __name__ == "__main__":
    main()