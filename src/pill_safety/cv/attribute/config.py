"""Cấu hình runtime dành riêng cho inference Module 2 attribute recognition."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AttributeInferenceConfig:
    """Tập hợp artifact model, tiền xử lý và nơi lưu output của Module 2."""

    selected_model: str = "attribute_resnet18_last_blocks_finetune"
    weights_path: Path = Path(
        "models/attribute_resnet18_last_blocks_finetune/best.pt"
    )
    label_mapping_path: Path = Path(
        "models/attribute_resnet18_last_blocks_finetune/label_mapping.json"
    )
    color_thresholds_path: Path = Path(
        "models/attribute_resnet18_last_blocks_finetune/"
        "optimal_thresholds.json"
    )
    model_config_path: Path = Path(
        "models/attribute_resnet18_last_blocks_finetune/model_config.yaml"
    )
    device: str = "auto"
    image_size: int = 224
    shape_top_k: int = 3
    normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        """Kiểm tra giá trị cấu hình độc lập với sự tồn tại artifact trên đĩa."""

        if not self.selected_model.strip():
            raise ValueError("selected_model must not be empty.")
        is_cuda_index = self.device.startswith("cuda:") and self.device[5:].isdigit()
        if self.device not in {"auto", "cpu", "cuda"} and not is_cuda_index:
            raise ValueError(
                "device must be auto, cpu, cuda, or cuda:<index>."
            )
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.shape_top_k <= 0:
            raise ValueError("shape_top_k must be positive.")
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("normalization_mean and normalization_std need 3 values.")
        if any(value <= 0.0 for value in self.normalization_std):
            raise ValueError("normalization_std values must be positive.")

    def with_output_dir(self, output_dir: str | Path) -> "AttributeInferenceConfig":
        """Trả về config mới với thư mục artifact runtime được thay thế."""

        return replace(self, output_dir=Path(output_dir))

    def resolve_paths(self, project_root: str | Path) -> "AttributeInferenceConfig":
        """Chuyển mọi đường dẫn tương đối sang đường dẫn tuyệt đối của project."""

        root = Path(project_root)

        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else root / path

        return replace(
            self,
            weights_path=resolve(self.weights_path),
            label_mapping_path=resolve(self.label_mapping_path),
            color_thresholds_path=resolve(self.color_thresholds_path),
            model_config_path=resolve(self.model_config_path),
            output_dir=resolve(self.output_dir),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AttributeInferenceConfig":
        """Đọc YAML inference và giữ default last-block khi trường bị thiếu."""

        with Path(path).open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}

        defaults = cls()
        model = raw.get("model", {})
        preprocessing = raw.get("preprocessing", {})
        artifacts = raw.get("artifacts", {})
        mean = preprocessing.get("normalization_mean", defaults.normalization_mean)
        std = preprocessing.get("normalization_std", defaults.normalization_std)

        values: dict[str, Any] = {
            "selected_model": model.get("selected_model", defaults.selected_model),
            "weights_path": Path(model.get("weights_path", defaults.weights_path)),
            "label_mapping_path": Path(
                model.get("label_mapping_path", defaults.label_mapping_path)
            ),
            "color_thresholds_path": Path(
                model.get("color_thresholds_path", defaults.color_thresholds_path)
            ),
            "model_config_path": Path(
                model.get("model_config_path", defaults.model_config_path)
            ),
            "device": model.get("device", defaults.device),
            "image_size": model.get("image_size", defaults.image_size),
            "shape_top_k": model.get("shape_top_k", defaults.shape_top_k),
            "normalization_mean": tuple(float(value) for value in mean),
            "normalization_std": tuple(float(value) for value in std),
            "output_dir": Path(artifacts.get("output_dir", defaults.output_dir)),
        }
        return cls(**values)
