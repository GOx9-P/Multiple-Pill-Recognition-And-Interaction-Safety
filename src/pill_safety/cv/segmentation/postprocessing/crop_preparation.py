"""Create task-specific color, shape, and OCR crops from one clean pill mask."""

from __future__ import annotations

from math import atan2, degrees, sqrt

import cv2
import numpy as np

from ..config import SegmentationConfig


def _rotate_bound(
    image: np.ndarray,
    angle_degrees: float,
    interpolation: int,
    border_value: int | tuple[int, int, int],
) -> np.ndarray:
    """Rotate an image without clipping its corners."""

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
    """Return the PCA major-axis angle and elongation ratio for a clean mask."""

    points = np.column_stack(np.nonzero(mask > 0))[:, ::-1].astype(np.float32)
    if len(points) < 3:
        return None
    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major = max(float(eigenvalues[order[0]]), 1e-6)
    minor = max(float(eigenvalues[order[1]]), 1e-6)
    axis = eigenvectors[:, order[0]]
    return degrees(atan2(float(axis[1]), float(axis[0]))), sqrt(major / minor)


def _dilate_region(mask: np.ndarray, ratio: float) -> np.ndarray:
    """Dilate a region only to expand the shape crop ROI around under-segmented edges."""

    binary_mask = (mask > 0).astype(np.uint8)
    if ratio <= 0:
        return binary_mask
    nonzero = cv2.findNonZero(binary_mask)
    if nonzero is None:
        return binary_mask
    _, _, width, height = cv2.boundingRect(nonzero)
    radius = int(round(max(width, height) * ratio))
    if radius <= 0:
        return binary_mask
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(binary_mask, kernel, iterations=1)


def _erode_region(mask: np.ndarray, ratio: float) -> np.ndarray:
    """Keep an interior color mask so boundary pixels from the scene cannot affect color."""

    binary_mask = (mask > 0).astype(np.uint8)
    if ratio <= 0:
        return binary_mask
    nonzero = cv2.findNonZero(binary_mask)
    if nonzero is None:
        return binary_mask
    _, _, width, height = cv2.boundingRect(nonzero)
    radius = int(round(max(width, height) * ratio))
    if radius <= 0:
        return binary_mask
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    return eroded if np.any(eroded) else binary_mask


def _crop_around_region(
    image: np.ndarray,
    clean_mask: np.ndarray,
    region_mask: np.ndarray,
    padding_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop around a region while retaining the matching clean-mask pixels."""

    nonzero = cv2.findNonZero((region_mask > 0).astype(np.uint8))
    if nonzero is None:
        return image.copy(), clean_mask.copy()
    x, y, width, height = cv2.boundingRect(nonzero)
    padding = int(round(max(width, height) * padding_ratio))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + width + padding)
    y2 = min(image.shape[0], y + height + padding)
    return image[y1:y2, x1:x2].copy(), clean_mask[y1:y2, x1:x2].copy()


def _render_square_crop(
    image: np.ndarray,
    clean_mask: np.ndarray,
    config: SegmentationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one crop on a square fixed-gray canvas using the supplied foreground mask."""

    background = (config.crop_background_value,) * 3
    binary_mask = (clean_mask > 0).astype(np.uint8)
    rendered = image.copy()
    rendered[binary_mask == 0] = background

    side = max(rendered.shape[:2])
    square = np.full((side, side, 3), background, dtype=np.uint8)
    square_mask = np.zeros((side, side), dtype=np.uint8)
    offset_y = (side - rendered.shape[0]) // 2
    offset_x = (side - rendered.shape[1]) // 2
    square[
        offset_y : offset_y + rendered.shape[0],
        offset_x : offset_x + rendered.shape[1],
    ] = rendered
    square_mask[
        offset_y : offset_y + binary_mask.shape[0],
        offset_x : offset_x + binary_mask.shape[1],
    ] = binary_mask

    interpolation = cv2.INTER_AREA if side > config.crop_size else cv2.INTER_CUBIC
    crop = cv2.resize(square, (config.crop_size, config.crop_size), interpolation=interpolation)
    resized_mask = cv2.resize(
        square_mask,
        (config.crop_size, config.crop_size),
        interpolation=cv2.INTER_NEAREST,
    )
    resized_mask = (resized_mask > 0).astype(np.uint8)
    crop[resized_mask == 0] = background
    return crop, resized_mask


def _render_square_rgb_crop(
    image: np.ndarray,
    config: SegmentationConfig,
) -> np.ndarray:
    """Place an unmasked RGB shape ROI on a square canvas without distorting the pill."""

    background = (config.crop_background_value,) * 3
    side = max(image.shape[:2])
    square = np.full((side, side, 3), background, dtype=np.uint8)
    offset_y = (side - image.shape[0]) // 2
    offset_x = (side - image.shape[1]) // 2
    square[
        offset_y : offset_y + image.shape[0],
        offset_x : offset_x + image.shape[1],
    ] = image
    interpolation = cv2.INTER_AREA if side > config.crop_size else cv2.INTER_CUBIC
    return cv2.resize(square, (config.crop_size, config.crop_size), interpolation=interpolation)


def prepare_task_crops(
    image_bgr: np.ndarray,
    clean_mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    config: SegmentationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build color, shape, OCR crops and an OCR-aligned clean mask for one pill."""

    image_height, image_width = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    padding = int(round(max(x2 - x1, y2 - y1) * config.bbox_padding_ratio))
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)

    base_image = image_bgr[y1:y2, x1:x2].copy()
    base_clean_mask = clean_mask[y1:y2, x1:x2].astype(np.uint8)
    
    # LƯU LẠI BẢN CHƯA XOAY DÀNH RIÊNG CHO SHAPE CROP
    original_base_image = base_image.copy()
    original_base_clean_mask = base_clean_mask.copy()

    background = (config.crop_background_value,) * 3
    if config.align_long_axis:
        axis = _principal_axis_angle(base_clean_mask)
        if axis is not None and axis[1] >= config.min_alignment_aspect_ratio:
            base_image = _rotate_bound(base_image, -axis[0], cv2.INTER_LINEAR, background)
            base_clean_mask = _rotate_bound(
                base_clean_mask,
                -axis[0],
                cv2.INTER_NEAREST,
                0,
            )

    # Color uses only the eroded interior. This removes scene pixels that leak through
    # an imperfect YOLO boundary while retaining the original clean mask for OCR.
    color_interior_mask = _erode_region(
        base_clean_mask,
        config.color_mask_erosion_ratio,
    )
    color_source, color_mask = _crop_around_region(
        base_image,
        color_interior_mask,
        base_clean_mask,
        config.bbox_padding_ratio,
    )
    color_crop, _ = _render_square_crop(color_source, color_mask, config)

    ocr_source, ocr_mask = _crop_around_region(
        base_image,
        base_clean_mask,
        base_clean_mask,
        config.bbox_padding_ratio,
    )
    ocr_crop, output_clean_mask = _render_square_crop(ocr_source, ocr_mask, config)

    # Shape uses original unrotated images to prevent polygon-like border artifacts
    shape_region = _dilate_region(original_base_clean_mask, config.crop_mask_dilation_ratio)
    shape_source, _ = _crop_around_region(
        original_base_image,
        original_base_clean_mask,
        shape_region,
        config.bbox_padding_ratio,
    )
    shape_crop = _render_square_rgb_crop(shape_source, config)
    return color_crop, shape_crop, ocr_crop, output_clean_mask
