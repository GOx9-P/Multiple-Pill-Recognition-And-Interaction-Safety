"""
Attribute Predictor for inference pipeline.

Provides a unified interface for loading the attribute model and predicting
shapes and colors for single images.
"""

import json
import argparse
from pathlib import Path
from typing import Union, Dict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pill_safety.cv.attribute.utils.config import AttributeConfig
from pill_safety.cv.attribute.models import MultiTaskResNet18
from pill_safety.cv.attribute.postprocessing.formatter import format_attribute_predictions
from pill_safety.cv.attribute.labels.label_mapping import load_label_mapping
from pill_safety.cv.attribute.utils.checkpoint import load_checkpoint


class AttributePredictor:
    """Inference wrapper for the MultiTaskResNet18 attribute model."""
    
    def __init__(self, run_id: str, module_name: str, device: Union[str, torch.device] = None):
        """
        Initialize the predictor from a specific training run.
        
        Args:
            run_id: The run ID (e.g. 'attr_last_v2' or 'attr_head_v2')
            module_name: The module name ('attribute_resnet18_last_blocks_finetune' 
                         or 'attribute_resnet18_head_tune')
            device: Target device. If None, uses AttributeConfig.DEVICE.
        """
        self.device = device or AttributeConfig.DEVICE
        self.run_id = run_id
        self.module_name = module_name
        
        paths = AttributeConfig.get_experiment_paths(module_name, run_id)
        
        ckpt_path = paths["checkpoints"] / f"{run_id}_best.pt"
        mapping_path = paths["logs"] / f"{run_id}_label_mapping.json"
        threshold_path = paths["metrics"] / f"{run_id}_optimal_thresholds.json"
        
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        # Try to find mapping file. If it's a last-blocks run, it might not have its own
        # mapping file, but we should find it from the head run. Wait, last-blocks run
        # doesn't save a mapping file directly, it reads from head.
        # But wait, in train_last_blocks, we load from head. 
        # For simplicity, if mapping is not found in the run's logs, we check the dataset manifest
        # to find the original mapping file path.
        if not mapping_path.exists():
            manifest_path = paths["logs"] / f"{run_id}_dataset_manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                    mapping_path = AttributeConfig.BASE_DIR / manifest.get("label_mapping_file", "")
                    
        if not Path(mapping_path).exists():
            raise FileNotFoundError(f"Label mapping not found at {mapping_path}")

        # Load mapping
        self.label_mapping, num_shape_classes, num_color_classes, mapping_hash = load_label_mapping(mapping_path)
        
        # Load thresholds
        if threshold_path.exists():
            with open(threshold_path, "r", encoding="utf-8") as f:
                self.thresholds = np.array(json.load(f))
        else:
            self.thresholds = np.full(num_color_classes, 0.5)

        # Load model
        self.model = MultiTaskResNet18(
            num_shape_classes=num_shape_classes, 
            num_color_classes=num_color_classes,
            pretrained=False
        ).to(self.device)
        
        ckpt = load_checkpoint(ckpt_path, self.device, expected_mapping_hash=mapping_hash)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, image_path: Union[str, Path]) -> Dict:
        """Run inference on a single image.
        
        Args:
            image_path: Path to the image.
            
        Returns:
            Dictionary with formatted predictions.
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        s_out, c_out = self.model(tensor)
        
        # Shape prediction
        s_prob = torch.softmax(s_out, dim=1)
        s_pred_idx = torch.argmax(s_prob, dim=1).item()
        
        # Color prediction
        c_probs = torch.sigmoid(c_out).cpu().numpy()[0]
        c_preds = (c_probs > self.thresholds).astype(int)
        
        # Format names
        shape_names = self.label_mapping["shape"]
        color_names = self.label_mapping["color"]
        
        s_label = shape_names[s_pred_idx] if s_pred_idx < len(shape_names) else "UNKNOWN"
        c_labels = [color_names[i] for i, val in enumerate(c_preds) if val == 1]
        
        raw_color_probs = {color_names[i]: float(p) for i, p in enumerate(c_probs)}
        
        return format_attribute_predictions(s_label, float(s_prob[0][s_pred_idx]), c_labels, raw_color_probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attribute inference")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID of the model")
    parser.add_argument("--module", type=str, required=True, help="Module name (e.g. attribute_resnet18_head_tune)")
    args = parser.parse_args()

    predictor = AttributePredictor(run_id=args.run_id, module_name=args.module)
    result = predictor.predict(args.image)
    print(json.dumps(result, indent=2, ensure_ascii=False))