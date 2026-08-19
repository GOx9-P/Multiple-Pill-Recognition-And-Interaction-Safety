"""Điều phối toàn bộ luồng suy luận và xuất artifact của module phân đoạn."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from ....schemas import (
    SegmentationInferenceOutput,
    SegmentationInferenceRequest,
)
from ..config import SegmentationConfig
from ..models import (
    RawSegmentationPrediction,
    SegmentationModel,
)
from ..postprocessing import (
    ProcessedInstance,
    build_segmentation_instance,
    build_segmentation_output,
    process_prediction,
    sort_instances_reading_order,
)
from ...pipeline.quality import assess_image_quality
from ..utils import draw_segmentation_overlay


class SegmentationEngine(Protocol):
    """Khai báo giao diện model để có thể thay bằng model giả khi kiểm thử."""

    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        image_size: int,
        confidence_threshold: float,
        iou_threshold: float,
        mask_threshold: float,
        device: str | int | None,
    ) -> list[RawSegmentationPrediction]:
        """Nhận ảnh và trả về các dự đoán mask đã chuẩn hóa tọa độ."""

        ...


@dataclass(frozen=True)
class SegmentationArtifacts:
    """Tập hợp output schema và đường dẫn các artifact đã được xuất."""

    output: SegmentationInferenceOutput
    schema_json_path: Path
    overlay_path: Path | None


def _safe_directory_name(value: str) -> str:
    """Chuẩn hóa mã định danh thành tên thư mục an toàn cho artifact."""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "request"


def _stable_instance_token(
    request: SegmentationInferenceRequest,
    instance_id: str,
) -> str:
    """Sinh token ổn định để Module 2 và Module 3 cùng tham chiếu một viên."""

    identity = f"{request.request_id}|{request.image_id}|{instance_id}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"pill_token_{digest}"


def _write_image(path: Path, image: np.ndarray) -> None:
    """Ghi ảnh artifact và báo lỗi rõ ràng nếu OpenCV ghi thất bại."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image artifact: {path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Ghi payload JSON theo UTF-8 và giữ nguyên ký tự tiếng Việt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


class SegmentationPredictor:
    """Chạy Module 1 từ ảnh đầu vào đến output và artifact đúng schema."""

    def __init__(
        self,
        config: SegmentationConfig | None = None,
        engine: SegmentationEngine | None = None,
    ):
        """Khởi tạo predictor bằng cấu hình và model thật hoặc model thay thế."""

        self.config = config or SegmentationConfig()
        if engine is None:
            if not self.config.weights_path.is_file():
                raise FileNotFoundError(
                    "Segmentation checkpoint does not exist: "
                    f"{self.config.weights_path}. Pass --weights or promote a "
                    "validated checkpoint into models/."
                )
            engine = SegmentationModel(str(self.config.weights_path))
        self.engine = engine

    def predict(
        self,
        request: SegmentationInferenceRequest | dict[str, Any],
    ) -> SegmentationInferenceOutput:
        """Chạy suy luận và chỉ trả về output có cấu trúc của Module 1."""

        return self.predict_with_artifacts(request).output

    def predict_with_artifacts(
        self,
        request: SegmentationInferenceRequest | dict[str, Any],
    ) -> SegmentationArtifacts:
        """Chạy toàn bộ pipeline rồi lưu mask, crop, overlay và JSON schema."""

        request = SegmentationInferenceRequest.model_validate(request)
        image_path = Path(request.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {image_path}")

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Input file is not a decodable image: {image_path}")

        image_quality = assess_image_quality(image_bgr, self.config)
        device: str | int | None = (
            None if self.config.device == "auto" else self.config.device
        )
        raw_predictions = self.engine.predict(
            image_bgr,
            image_size=self.config.image_size,
            confidence_threshold=self.config.confidence_threshold,
            iou_threshold=self.config.iou_threshold,
            mask_threshold=self.config.mask_threshold,
            device=device,
        )

        processed = [
            item
            for prediction in raw_predictions
            if (item := process_prediction(image_bgr, prediction, self.config))
            is not None
        ]
        ordered_instances = sort_instances_reading_order(processed)

        request_directory = _safe_directory_name(request.request_id)
        image_directory = _safe_directory_name(request.image_id)
        # Lưu artifact đúng các thư mục quy định trong README và tách theo từng
        # request để các mã viên trùng nhau không ghi đè kết quả cũ.
        mask_directory = (
            self.config.output_dir / "masks" / request_directory / image_directory
        )
        crop_directory = (
            self.config.output_dir / "crops" / request_directory / image_directory
        )
        prediction_directory = (
            self.config.output_dir
            / "predictions"
            / "segmentation"
            / request_directory
            / image_directory
        )

        schema_instances = []
        overlay_instances: list[tuple[str, ProcessedInstance]] = []
        for index, instance in enumerate(ordered_instances, start=1):
            instance_id = f"pill_{index:03d}"
            instance_token = _stable_instance_token(request, instance_id)
            mask_path = mask_directory / f"{instance_id}_clean_mask.png"
            color_crop_path = crop_directory / f"{instance_id}_color_crop.png"
            shape_crop_path = crop_directory / f"{instance_id}_shape_crop.png"
            ocr_crop_path = crop_directory / f"{instance_id}_ocr_crop.png"
            # mask_path la clean mask, dong bo pixel voi color_crop_path va ocr_crop_path.
            _write_image(mask_path, instance.crop_mask * 255)
            _write_image(color_crop_path, instance.crop)
            _write_image(shape_crop_path, instance.shape_crop)
            _write_image(ocr_crop_path, instance.ocr_crop)

            flags = list(
                dict.fromkeys(
                    (*image_quality.quality_flags, *instance.quality_flags)
                )
            )
            schema_instances.append(
                build_segmentation_instance(
                    instance_id=instance_id,
                    instance_token=instance_token,
                    mask_path=str(mask_path),
                    color_crop_path=str(color_crop_path),
                    shape_crop_path=str(shape_crop_path),
                    ocr_crop_path=str(ocr_crop_path),
                    # crop_path duoc giu lai de tuong thich artifact cu; no tro toi color crop.
                    crop_path=str(color_crop_path),
                    processed=instance,
                    quality_flags=flags,
                )
            )
            overlay_instances.append((instance_id, instance))

        output = build_segmentation_output(
            request,
            image_quality.output,
            schema_instances,
        )
        schema_json_path = prediction_directory / "segmentation_output.json"
        _write_json(schema_json_path, output.model_dump(mode="json"))

        overlay_path: Path | None = None
        if self.config.save_overlay:
            overlay_path = prediction_directory / "segmentation_overlay.jpg"
            overlay = draw_segmentation_overlay(image_bgr, overlay_instances)
            _write_image(overlay_path, overlay)

        return SegmentationArtifacts(
            output=output,
            schema_json_path=schema_json_path,
            overlay_path=overlay_path,
        )
