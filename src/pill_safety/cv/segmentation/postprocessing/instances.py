"""Hậu xử lý mask, quality gate hình học và chuẩn bị crop từng viên thuốc."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, sqrt

import cv2
import numpy as np

from ..config import SegmentationConfig
from ..models import RawSegmentationPrediction


@dataclass(frozen=True)
class ProcessedInstance:
    """Instance đã kiểm tra hình học nhưng chưa gán ID và đường dẫn artifact."""

    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    mask: np.ndarray
    crop: np.ndarray
    crop_mask: np.ndarray
    occlusion_estimate: float
    possible_merged_instance: bool
    possible_non_pill: bool
    quality_flags: tuple[str, ...]


def _clip_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Giới hạn bbox XYXY trong ảnh và giữ tối thiểu một pixel mỗi chiều."""

    x1, y1, x2, y2 = bbox_xyxy
    left = int(np.clip(np.floor(x1), 0, max(image_width - 1, 0)))
    top = int(np.clip(np.floor(y1), 0, max(image_height - 1, 0)))
    right = int(np.clip(np.ceil(x2), left + 1, image_width))
    bottom = int(np.clip(np.ceil(y2), top + 1, image_height))
    return left, top, right, bottom


def _clean_mask(
    mask: np.ndarray,
    min_component_area_ratio: float,
) -> tuple[np.ndarray, int]:
    """Loại component nhiễu nhỏ và giữ lại các phần mask có diện tích đáng kể."""

    binary = (mask > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if component_count <= 1:
        return binary, 0

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(component_areas.max(initial=0))
    if largest_area <= 0:
        return np.zeros_like(binary), 0

    minimum_area = largest_area * min_component_area_ratio
    retained_labels = [
        index + 1
        for index, area in enumerate(component_areas)
        if area >= minimum_area
    ]
    cleaned = np.isin(labels, retained_labels).astype(np.uint8)
    return cleaned, len(retained_labels)


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    """Lấy contour ngoài có diện tích lớn nhất để đánh giá hình dạng mask."""

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return max(contours, key=cv2.contourArea) if contours else None


def _rotate_bound(
    image: np.ndarray,
    angle_degrees: float,
    interpolation: int,
    border_value: int | tuple[int, int, int],
) -> np.ndarray:
    """Xoay ảnh không cắt góc và tô phần biên mới bằng màu được chỉ định."""

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = max(1, int(height * sine + width * cosine))
    new_height = max(1, int(height * cosine + width * sine))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _principal_axis_angle(mask: np.ndarray) -> tuple[float, float] | None:
    """Tính góc trục dài PCA và tỷ lệ trục đại diện cho độ thuôn của mask."""

    points = np.column_stack(np.nonzero(mask > 0))[:, ::-1].astype(np.float32)
    if len(points) < 3:
        return None
    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major = max(float(eigenvalues[order[0]]), 1e-6)
    minor = max(float(eigenvalues[order[1]]), 1e-6)
    axis = eigenvectors[:, order[0]]
    angle = degrees(atan2(float(axis[1]), float(axis[0])))
    return angle, sqrt(major / minor)


def _prepare_crop(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    config: SegmentationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Tạo crop chuẩn đã áp mask, padding, căn trục và resize về hình vuông."""

    image_height, image_width = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    padding = int(round(max(x2 - x1, y2 - y1) * config.bbox_padding_ratio))
    x1p, y1p = max(0, x1 - padding), max(0, y1 - padding)
    x2p, y2p = min(image_width, x2 + padding), min(image_height, y2 + padding)

    crop = image_bgr[y1p:y2p, x1p:x2p].copy()
    crop_mask = mask[y1p:y2p, x1p:x2p].astype(np.uint8)
    foreground = crop[crop_mask > 0]
    background_color = (
        tuple(int(value) for value in np.median(foreground, axis=0))
        if foreground.size
        else (0, 0, 0)
    )
    # Nền xám trung tính giữ được contour của viên sáng hoặc viên màu trắng.
    # Màu median chỉ dùng cho padding ngoài khi xoay và tạo canvas vuông cho OCR.
    masked_background = (config.crop_background_value,) * 3
    crop[crop_mask == 0] = masked_background

    if config.align_long_axis:
        axis = _principal_axis_angle(crop_mask)
        if axis is not None and axis[1] >= config.min_alignment_aspect_ratio:
            crop = _rotate_bound(
                crop,
                -axis[0],
                cv2.INTER_LINEAR,
                background_color,
            )
            crop_mask = _rotate_bound(
                crop_mask,
                -axis[0],
                cv2.INTER_NEAREST,
                0,
            )

    # Cắt padding thừa do phép xoay rồi đặt viên vào giữa canvas vuông ổn định.
    nonzero = cv2.findNonZero((crop_mask > 0).astype(np.uint8))
    if nonzero is not None:
        tx, ty, tw, th = cv2.boundingRect(nonzero)
        crop = crop[ty : ty + th, tx : tx + tw]
        crop_mask = crop_mask[ty : ty + th, tx : tx + tw]
    side = max(crop.shape[:2])
    square = np.full((side, side, 3), background_color, dtype=np.uint8)
    square_mask = np.zeros((side, side), dtype=np.uint8)
    offset_y = (side - crop.shape[0]) // 2
    offset_x = (side - crop.shape[1]) // 2
    square[
        offset_y : offset_y + crop.shape[0],
        offset_x : offset_x + crop.shape[1],
    ] = crop
    square_mask[
        offset_y : offset_y + crop_mask.shape[0],
        offset_x : offset_x + crop_mask.shape[1],
    ] = crop_mask
    resized_crop = cv2.resize(
        square,
        (config.crop_size, config.crop_size),
        interpolation=cv2.INTER_AREA if side > config.crop_size else cv2.INTER_CUBIC,
    )
    resized_mask = cv2.resize(
        square_mask,
        (config.crop_size, config.crop_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized_crop, (resized_mask > 0).astype(np.uint8)


def process_prediction(
    image_bgr: np.ndarray,
    prediction: RawSegmentationPrediction,
    config: SegmentationConfig,
) -> ProcessedInstance | None:
    """Làm sạch mask, chạy quality gate hình học và tạo crop cho một prediction."""

    image_height, image_width = image_bgr.shape[:2]
    if prediction.mask.shape != (image_height, image_width):
        raise ValueError("Prediction mask must use original image coordinates.")

    cleaned_mask, retained_components = _clean_mask(
        prediction.mask,
        config.min_component_area_ratio,
    )
    area = int(cleaned_mask.sum())
    if area == 0:
        return None

    bbox = _clip_bbox(prediction.bbox_xyxy, image_width, image_height)
    image_area = max(image_width * image_height, 1)
    area_ratio = area / image_area
    contour = _largest_contour(cleaned_mask)

    flags: list[str] = []
    if area_ratio < config.min_mask_area_ratio:
        flags.append("mask_too_small")
    if area_ratio > config.max_mask_area_ratio:
        flags.append("mask_too_large")
    if retained_components > 1:
        flags.append("fragmented_mask")

    solidity = 1.0
    if contour is not None:
        contour_area = float(cv2.contourArea(contour))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = contour_area / hull_area if hull_area > 0 else 0.0
    possible_merged = retained_components > 1 or solidity < config.merged_solidity_threshold
    if possible_merged:
        flags.append("possible_merged_instance")

    border_sides = sum(
        (
            bool(cleaned_mask[0, :].any()),
            bool(cleaned_mask[-1, :].any()),
            bool(cleaned_mask[:, 0].any()),
            bool(cleaned_mask[:, -1].any()),
        )
    )
    if border_sides:
        flags.append("touches_image_border")
    occlusion_estimate = min(
        1.0,
        # Chạm biên và component rời là các proxy quan sát được; giá trị này
        # không được xem là xác suất che khuất đã calibration.
        0.25 * border_sides + 0.15 * max(retained_components - 1, 0),
    )

    possible_non_pill = (
        prediction.confidence < config.non_pill_confidence_threshold
        or area_ratio < config.min_mask_area_ratio
        or area_ratio > config.max_mask_area_ratio
    )
    if possible_non_pill:
        flags.append("possible_non_pill")

    crop, crop_mask = _prepare_crop(image_bgr, cleaned_mask, bbox, config)
    return ProcessedInstance(
        bbox_xyxy=bbox,
        confidence=float(np.clip(prediction.confidence, 0.0, 1.0)),
        mask=cleaned_mask,
        crop=crop,
        crop_mask=crop_mask,
        occlusion_estimate=round(float(occlusion_estimate), 4),
        possible_merged_instance=possible_merged,
        possible_non_pill=possible_non_pill,
        quality_flags=tuple(dict.fromkeys(flags)),
    )


def sort_instances_reading_order(
    instances: list[ProcessedInstance],
) -> list[ProcessedInstance]:
    """Sắp xếp các hàng từ trên xuống và từng hàng từ trái sang phải."""

    if len(instances) < 2:
        return list(instances)

    heights = [item.bbox_xyxy[3] - item.bbox_xyxy[1] for item in instances]
    row_tolerance = max(float(np.median(heights)) * 0.5, 1.0)
    by_y = sorted(
        instances,
        key=lambda item: (
            (item.bbox_xyxy[1] + item.bbox_xyxy[3]) / 2.0,
            (item.bbox_xyxy[0] + item.bbox_xyxy[2]) / 2.0,
        ),
    )

    rows: list[list[ProcessedInstance]] = []
    row_centers: list[float] = []
    for item in by_y:
        center_y = (item.bbox_xyxy[1] + item.bbox_xyxy[3]) / 2.0
        if not rows or abs(center_y - row_centers[-1]) > row_tolerance:
            rows.append([item])
            row_centers.append(center_y)
        else:
            rows[-1].append(item)
            row_centers[-1] = float(
                np.mean(
                    [
                        (member.bbox_xyxy[1] + member.bbox_xyxy[3]) / 2.0
                        for member in rows[-1]
                    ]
                )
            )

    return [
        item
        for row in rows
        for item in sorted(row, key=lambda member: member.bbox_xyxy[0])
    ]
