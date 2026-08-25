"""Predictor inference cho ResNet18 shape/color của Module 2."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torchvision import transforms

from pill_safety.schemas import AttributeInferenceOutput, AttributeInferenceRequest

from ..config import AttributeInferenceConfig
from ..labels.label_mapping import load_color_threshold_values, load_label_mapping
from ..models.resnet_multitask import MultiTaskResNet18
from ..postprocessing import (
    build_attribute_output,
    format_attribute_predictions,
)


@dataclass(frozen=True)
class AttributeArtifacts:
    """Tập hợp output schema và JSON artifact được sinh bởi Module 2."""

    output: AttributeInferenceOutput
    schema_json_path: Path


def _safe_directory_name(value: str) -> str:
    """Chuẩn hóa ID thành tên thư mục an toàn để tránh ghi đè artifact."""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "request"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Ghi output JSON UTF-8 để các module phía sau đọc lại nguyên vẹn."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _resolve_device(device_name: str) -> torch.device:
    """Chọn CUDA khi cấu hình auto và runtime có GPU, ngược lại dùng CPU."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Attribute inference requested CUDA but torch.cuda.is_available() is false."
        )
    return torch.device(device_name)


def _load_color_thresholds(
    path: Path,
    color_names: list[str],
) -> torch.Tensor:
    """Nạp threshold multi-label theo list, key ``thresholds`` hoặc tên màu."""

    values = load_color_threshold_values(path, color_names)
    return torch.tensor(values, dtype=torch.float32)


def _validate_model_config(path: Path, image_size: int) -> None:
    """Kiểm tra artifact config để không nạp nhầm weight khác kiến trúc hoặc resize."""

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    model = raw.get("model", raw)
    architecture = str(model.get("architecture", "ResNet18"))
    if architecture.lower() != "resnet18":
        raise ValueError(
            "Attribute inference only supports a ResNet18 artifact, got "
            f"{architecture!r} from {path}."
        )

    training = raw.get("training", {})
    trained_image_size = training.get("image_size", raw.get("image_size"))
    if trained_image_size is not None and int(trained_image_size) != image_size:
        raise ValueError(
            "Inference image_size does not match model_config.yaml: "
            f"{image_size} != {trained_image_size}."
        )

    tasks = raw.get("tasks") or raw.get("optional_tasks") or []
    if tasks and not {"shape", "color"}.issubset(set(tasks)):
        raise ValueError(
            "model_config.yaml must declare both shape and color tasks."
        )


def _load_model_state_dict(
    path: Path,
    device: torch.device,
    num_shape_classes: int,
    num_color_classes: int,
    expected_mapping_hash: str,
) -> dict[str, Any]:
    """Nạp raw state dict của run mới hoặc checkpoint có metadata của run cũ."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Attribute checkpoint must contain a state dictionary.")

    if "model_state_dict" in payload:
        mapping_hash = payload.get("mapping_hash", "")
        if expected_mapping_hash and mapping_hash and mapping_hash != expected_mapping_hash:
            raise RuntimeError("Checkpoint label mapping hash differs from label_mapping.json.")
        state_dict = payload["model_state_dict"]
    else:
        # Workflow của run attr_*_v1 lưu trực tiếp model.state_dict().
        state_dict = payload

    if not isinstance(state_dict, dict):
        raise RuntimeError("Attribute checkpoint model_state_dict is invalid.")
    state_dict = {
        str(key).removeprefix("module."): value
        for key, value in state_dict.items()
    }

    shape_weight = None
    for k in ("fc_shape.1.weight", "shape_head.weight", "fc_shape.weight"):
        if k in state_dict:
            shape_weight = state_dict[k]
            break

    color_weight = None
    for k in ("fc_color.4.weight", "color_head.weight", "fc_color.weight"):
        if k in state_dict:
            color_weight = state_dict[k]
            break

    if shape_weight is None or color_weight is None:
        raise RuntimeError(
            "Checkpoint does not match the trained multi-task model: "
            "missing shape_head or color_head weights."
        )
    if shape_weight.shape[0] != num_shape_classes:
        raise RuntimeError("Checkpoint shape class count differs from label_mapping.json.")
    if color_weight.shape[0] != num_color_classes:
        raise RuntimeError("Checkpoint color class count differs from label_mapping.json.")
    return state_dict


class AttributePredictor:
    """Chạy ResNet18 đã chọn và xuất output Module 2 đúng schema."""

    def __init__(self, config: AttributeInferenceConfig | None = None):
        """Nạp artifact chính thức trong ``models/`` và chuẩn bị transform inference."""

        self.config = config or AttributeInferenceConfig()
        required_artifacts = {
            "checkpoint": self.config.weights_path,
            "label mapping": self.config.label_mapping_path,
            "color thresholds": self.config.color_thresholds_path,
            "model config": self.config.model_config_path,
        }
        missing = [
            f"{name}: {path}"
            for name, path in required_artifacts.items()
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing promoted Attribute inference artifacts in models/:\n- "
                + "\n- ".join(missing)
            )

        _validate_model_config(self.config.model_config_path, self.config.image_size)
        self.device = _resolve_device(self.config.device)
        (
            self.label_mapping,
            num_shape_classes,
            num_color_classes,
            mapping_hash,
        ) = load_label_mapping(self.config.label_mapping_path)
        self.color_thresholds = _load_color_thresholds(
            self.config.color_thresholds_path,
            self.label_mapping["color"],
        ).to(self.device)

        self.model = MultiTaskResNet18(
            num_shape_classes=num_shape_classes,
            num_color_classes=num_color_classes,
            pretrained=False,
        ).to(self.device)
        state_dict = _load_model_state_dict(
            self.config.weights_path,
            self.device,
            num_shape_classes,
            num_color_classes,
            mapping_hash,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    self.config.normalization_mean,
                    self.config.normalization_std,
                ),
            ]
        )

    def _predict_crop(self, crop_path: Path) -> dict[str, Any]:
        """Chạy shape head và color head trên một crop đã được Module 1 mask sẵn."""

        with Image.open(crop_path) as source:
            image = source.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(tensor)
            if isinstance(outputs, tuple):
                shape_logits, color_logits = outputs
            else:
                shape_logits = self.model(tensor, task_type="shape")
                color_logits = self.model(tensor, task_type="color")
            shape_probabilities = torch.softmax(shape_logits, dim=1)[0]
            color_probabilities = torch.sigmoid(color_logits)[0]

        shape_names = self.label_mapping["shape"]
        color_names = self.label_mapping["color"]
        top_k = min(self.config.shape_top_k, len(shape_names))
        top_probabilities, top_indices = torch.topk(shape_probabilities, k=top_k)
        selected_index = int(top_indices[0].item())
        shape_alternatives = [
            (shape_names[int(index.item())], float(probability.item()))
            for index, probability in zip(top_indices[1:], top_probabilities[1:])
        ]

        color_values = color_probabilities.detach().cpu().tolist()
        color_labels = [
            color_names[index]
            for index, probability in enumerate(color_values)
            if probability > float(self.color_thresholds[index].item())
        ]
        color_scores = {
            color_names[index]: float(probability)
            for index, probability in enumerate(color_values)
        }
        return format_attribute_predictions(
            shape_label=shape_names[selected_index],
            shape_conf=float(shape_probabilities[selected_index].item()),
            color_labels=color_labels,
            color_probs=color_scores,
            shape_alternatives=shape_alternatives,
        )

    def _predict_task_crops(
        self,
        shape_crop_path: Path,
        color_crop_path: Path,
    ) -> dict[str, Any]:
        """Run shape and color heads on their respective Module 1 crops."""

        if shape_crop_path == color_crop_path:
            return self._predict_crop(shape_crop_path)

        with Image.open(shape_crop_path) as source:
            shape_image = source.convert("RGB")
        with Image.open(color_crop_path) as source:
            color_image = source.convert("RGB")
        shape_tensor = self.transform(shape_image).unsqueeze(0).to(self.device)
        color_tensor = self.transform(color_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            shape_out = self.model(shape_tensor)
            shape_logits = shape_out[0] if isinstance(shape_out, tuple) else shape_out
            color_out = self.model(color_tensor)
            color_logits = color_out[1] if isinstance(color_out, tuple) else color_out
            shape_probabilities = torch.softmax(shape_logits, dim=1)[0]
            color_probabilities = torch.sigmoid(color_logits)[0]

        shape_names = self.label_mapping["shape"]
        color_names = self.label_mapping["color"]
        top_k = min(self.config.shape_top_k, len(shape_names))
        top_probabilities, top_indices = torch.topk(shape_probabilities, k=top_k)
        selected_index = int(top_indices[0].item())
        shape_alternatives = [
            (shape_names[int(index.item())], float(probability.item()))
            for index, probability in zip(top_indices[1:], top_probabilities[1:])
        ]
        color_values = color_probabilities.detach().cpu().tolist()
        color_labels = [
            color_names[index]
            for index, probability in enumerate(color_values)
            if probability > float(self.color_thresholds[index].item())
        ]
        color_scores = {
            color_names[index]: float(probability)
            for index, probability in enumerate(color_values)
        }
        return format_attribute_predictions(
            shape_label=shape_names[selected_index],
            shape_conf=float(shape_probabilities[selected_index].item()),
            color_labels=color_labels,
            color_probs=color_scores,
            shape_alternatives=shape_alternatives,
        )

    def predict(
        self,
        request: AttributeInferenceRequest | dict[str, Any],
    ) -> AttributeInferenceOutput:
        """Chạy inference và chỉ trả payload Module 2 cho pipeline CV."""

        return self.predict_with_artifacts(request).output

    def predict_with_artifacts(
        self,
        request: AttributeInferenceRequest | dict[str, Any],
    ) -> AttributeArtifacts:
        """Chạy inference, giữ ID nguồn và lưu JSON theo cây artifact của README."""

        request = AttributeInferenceRequest.model_validate(request)
        shape_crop_path = Path(request.shape_crop_path or request.crop_path)
        color_crop_path = Path(request.color_crop_path or request.crop_path)
        mask_path = Path(request.mask_path)
        if not shape_crop_path.is_file():
            raise FileNotFoundError(
                f"Attribute shape crop does not exist: {shape_crop_path}"
            )
        if not color_crop_path.is_file():
            raise FileNotFoundError(
                f"Attribute color crop does not exist: {color_crop_path}"
            )
        if not mask_path.is_file():
            raise FileNotFoundError(f"Attribute mask does not exist: {mask_path}")

        prediction = self._predict_task_crops(shape_crop_path, color_crop_path)
        output = build_attribute_output(request, prediction)
        artifact_directory = (
            self.config.output_dir
            / "predictions"
            / "attribute"
            / _safe_directory_name(request.request_id)
            / _safe_directory_name(request.image_id)
            / _safe_directory_name(request.instance_id)
        )
        schema_json_path = artifact_directory / "attribute_output.json"
        _write_json(schema_json_path, output.model_dump(mode="json"))
        return AttributeArtifacts(output=output, schema_json_path=schema_json_path)
