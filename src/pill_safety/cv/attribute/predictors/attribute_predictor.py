import json
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

from pill_safety.cv.attribute.utils.config import AttributeConfig
from pill_safety.cv.attribute.models.resnet18_multitask import MultiTaskResNet18
from pill_safety.cv.attribute.postprocessing.formatter import format_attribute_predictions

class AttributePredictor:
    def __init__(self, checkpoint_path=None, mapping_path=None, threshold_path=None):
        self.device = AttributeConfig.DEVICE
        
        checkpoint_path = checkpoint_path or (AttributeConfig.CHECKPOINT_DIR / f"{AttributeConfig.RUN_ID}_best.pt")
        mapping_path = mapping_path or (AttributeConfig.METRIC_DIR / "label_mapping.json")
        threshold_path = threshold_path or (AttributeConfig.METRIC_DIR / "optimal_thresholds.json")

        with open(mapping_path, "r", encoding="utf-8") as f:
            self.label_mapping = json.load(f)

        if Path(threshold_path).exists():
            with open(threshold_path, "r", encoding="utf-8") as f:
                self.thresholds = np.array(json.load(f))
        else:
            self.thresholds = np.full(len(self.label_mapping["color"]), 0.5)

        num_shape_classes = len(self.label_mapping["shape"])
        num_color_classes = len(self.label_mapping["color"])

        self.model = MultiTaskResNet18(num_shape_classes, num_color_classes).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path: str | Path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            s_out, c_out = self.model(tensor)
            
            s_prob = torch.softmax(s_out, dim=1)
            s_pred_idx = torch.argmax(s_prob, dim=1).item()
            s_label = self.label_mapping["shape"].get(str(s_pred_idx), "UNKNOWN")

            c_probs = torch.sigmoid(c_out).cpu().numpy()[0]
            c_preds = (c_probs > self.thresholds).astype(int)
            c_labels = [self.label_mapping["color"][i] for i, val in enumerate(c_preds) if val == 1]

        raw_color_probs = {self.label_mapping["color"][i]: float(p) for i, p in enumerate(c_probs)}
        return format_attribute_predictions(s_label, float(s_prob[0][s_pred_idx]), c_labels, raw_color_probs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    predictor = AttributePredictor()
    result = predictor.predict(args.image)
    print(json.dumps(result, indent=2, ensure_ascii=False))