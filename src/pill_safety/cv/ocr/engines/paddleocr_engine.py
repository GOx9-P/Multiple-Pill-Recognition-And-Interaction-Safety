from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pill_safety.cv.ocr.config import OCRConfig

LOGGER = logging.getLogger(__name__)


class OCREngine(Protocol):
    def predict(
        self, image_path: Path, output_json_dir: Path, step_id: str
    ) -> list[dict[str, Any]]: ...


def to_polygon(poly: Any) -> list[list[float]] | None:
    if poly is None:
        return None
    array = np.asarray(poly).astype(float)
    if array.ndim == 1 and array.size >= 4:
        x1, y1, x2, y2 = array[:4]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if array.ndim == 2 and array.shape[1] >= 2:
        return array[:, :2].round(2).tolist()
    return None


def extract_items_from_dict(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("res"), dict):
        data = data["res"]

    texts = data.get("rec_texts") or data.get("texts")
    if isinstance(texts, list):
        scores = data.get("rec_scores") or data.get("scores") or [None] * len(texts)
        polygons = (
            data.get("rec_polys")
            or data.get("dt_polys")
            or data.get("polys")
            or data.get("boxes")
            or [None] * len(texts)
        )
        items = []
        for index, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                continue
            score = scores[index] if index < len(scores) else None
            polygon = polygons[index] if index < len(polygons) else None
            items.append(
                {
                    "text": text,
                    "confidence": float(score or 0.0),
                    "polygon": to_polygon(polygon),
                }
            )
        return items

    items = []
    for value in data.values():
        if isinstance(value, dict):
            items.extend(extract_items_from_dict(value))
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, dict):
                    items.extend(extract_items_from_dict(child))
    return items


def parse_prediction_result(result: Any) -> list[dict[str, Any]]:
    pages = result if isinstance(result, (list, tuple)) else [result]
    items = []
    for page in pages:
        if isinstance(page, dict):
            data = page
        else:
            json_value = getattr(page, "json", None)
            # PaddleOCR v3 exposes JSON as a dict property. Keep support for
            # older adapters that expose it as a callable method.
            data = json_value() if callable(json_value) else json_value
            if data is None:
                data = getattr(page, "res", None)
        items.extend(extract_items_from_dict(data))
    return items


class PaddleOCREngine:
    def __init__(self, config: OCRConfig):
        try:
            import paddle
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR inference requires paddlepaddle and paddleocr. "
                "Install the project requirements first."
            ) from exc

        try:
            import paddleocr._common_args as ca
            if hasattr(ca, "PaddlePredictorOption"):
                _orig_opt_init = ca.PaddlePredictorOption.__init__
                def _patched_opt_init(self, *args, **kwargs):
                    return _orig_opt_init(self, **kwargs)
                ca.PaddlePredictorOption.__init__ = _patched_opt_init
        except Exception:
            pass

        if config.device == "auto":
            self.device = (
                "gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu"
            )
        else:
            self.device = config.device
        paddle.set_device(self.device)
        self._ocr = PaddleOCR(
            ocr_version=config.ocr_version,
            lang=config.language,
            device=self.device,
            det_db_thresh=config.det_db_thresh,
            det_db_unclip_ratio=config.det_db_unclip_ratio,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def predict(
        self, image_path: Path, output_json_dir: Path, step_id: str
    ) -> list[dict[str, Any]]:
        try:
            (output_json_dir / step_id).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        try:
            result = self._ocr.predict(input=str(image_path))
        except Exception as exc:
            LOGGER.warning(
                "OCR failed for %s: %s: %s",
                step_id,
                type(exc).__name__,
                exc,
            )
            return []
        return parse_prediction_result(result)
