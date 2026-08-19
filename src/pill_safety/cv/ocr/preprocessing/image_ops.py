from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PreparedImage:
    bgr: np.ndarray
    original_height: int
    original_width: int
    pad_px: int


def prepare_image_bgr(path: str | Path) -> PreparedImage:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read {path}")
    original_height, original_width = image.shape[:2]
    pad_px = max(10, int(min(image.shape[:2]) * 0.05))
    border_color = [
        int(value)
        for value in np.median(image.reshape(-1, image.shape[2]), axis=0)
    ]
    return PreparedImage(
        bgr=cv2.copyMakeBorder(
            image,
            pad_px,
            pad_px,
            pad_px,
            pad_px,
            cv2.BORDER_CONSTANT,
            value=border_color,
        ),
        original_height=original_height,
        original_width=original_width,
        pad_px=pad_px,
    )


def prepare_foreground_mask(
    path: str | Path,
    prepared_image: PreparedImage,
) -> np.ndarray | None:
    """Doc mask Module 1, pad dong bo voi crop OCR va tra ve mask nhi phan."""

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape != (
        prepared_image.original_height,
        prepared_image.original_width,
    ):
        raise ValueError("Segmentation mask must have the same size as OCR crop.")
    binary = (mask > 0).astype(np.uint8)
    return cv2.copyMakeBorder(
        binary,
        prepared_image.pad_px,
        prepared_image.pad_px,
        prepared_image.pad_px,
        prepared_image.pad_px,
        cv2.BORDER_CONSTANT,
        value=0,
    )


def rotate_foreground_mask(mask: np.ndarray, degrees: int) -> np.ndarray:
    """Xoay mask nhi phan cung goc voi crop de kiem tra line nam trong vien."""

    if degrees == 0:
        return mask.copy()
    if degrees == 90:
        return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(mask, cv2.ROTATE_180)
    if degrees in (270, -90):
        return cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

    matrix, new_width, new_height = _oblique_rotation_matrix(mask.shape, degrees)
    return cv2.warpAffine(
        mask,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def read_image_bgr(path: str | Path) -> np.ndarray:
    return prepare_image_bgr(path).bgr


def _oblique_rotation_matrix(
    image_shape: tuple[int, ...], degrees: int
) -> tuple[np.ndarray, int, int]:
    height, width = image_shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    cos_value, sin_value = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int((height * sin_value) + (width * cos_value))
    new_height = int((height * cos_value) + (width * sin_value))
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]
    return matrix, new_width, new_height


def rotate_bgr(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return image.copy()
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees in (270, -90):
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    matrix, new_width, new_height = _oblique_rotation_matrix(
        image.shape, degrees
    )
    border = [
        int(value)
        for value in np.median(image.reshape(-1, image.shape[2]), axis=0)
    ]
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def map_polygon_to_original(
    polygon: list[list[float]] | None,
    padded_shape: tuple[int, ...],
    rotation_degrees: int,
    pad_px: int,
    original_height: int,
    original_width: int,
) -> list[list[float]] | None:
    if not polygon:
        return None
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        return None
    points = points[:, :2]
    padded_height, padded_width = padded_shape[:2]

    if rotation_degrees == 0:
        restored = points.copy()
    elif rotation_degrees == 90:
        restored = np.column_stack(
            (points[:, 1], padded_height - 1.0 - points[:, 0])
        )
    elif rotation_degrees == 180:
        restored = np.column_stack(
            (
                padded_width - 1.0 - points[:, 0],
                padded_height - 1.0 - points[:, 1],
            )
        )
    elif rotation_degrees in (270, -90):
        restored = np.column_stack(
            (padded_width - 1.0 - points[:, 1], points[:, 0])
        )
    else:
        matrix, _, _ = _oblique_rotation_matrix(
            padded_shape, rotation_degrees
        )
        inverse = cv2.invertAffineTransform(matrix)
        homogeneous = np.column_stack(
            (points, np.ones(len(points), dtype=np.float32))
        )
        restored = homogeneous @ inverse.T

    restored[:, 0] -= pad_px
    restored[:, 1] -= pad_px
    restored[:, 0] = np.clip(restored[:, 0], 0, original_width - 1)
    restored[:, 1] = np.clip(restored[:, 1], 0, original_height - 1)
    return restored.round(2).tolist()


def attach_original_polygons(
    items: list[dict],
    padded_shape: tuple[int, ...],
    rotation_degrees: int,
    prepared_image: PreparedImage,
) -> None:
    for item in items:
        item["polygon_original"] = map_polygon_to_original(
            item.get("polygon"),
            padded_shape=padded_shape,
            rotation_degrees=rotation_degrees,
            pad_px=prepared_image.pad_px,
            original_height=prepared_image.original_height,
            original_width=prepared_image.original_width,
        )


def apply_preprocessing(image: np.ndarray, name: str) -> np.ndarray:
    if name == "original":
        return image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if name == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    if name == "blackhat":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(15, image.shape[1] // 20), max(15, image.shape[0] // 20)),
        )
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        inverted = cv2.bitwise_not(blackhat)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        return cv2.cvtColor(clahe.apply(inverted), cv2.COLOR_GRAY2BGR)
    if name == "blackhat_bold":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(15, image.shape[1] // 20), max(15, image.shape[0] // 20)),
        )
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        inverted = cv2.bitwise_not(blackhat)
        bold_inverted = cv2.erode(
            inverted, np.ones((3, 3), np.uint8), iterations=1
        )
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        return cv2.cvtColor(clahe.apply(bold_inverted), cv2.COLOR_GRAY2BGR)
    return image
