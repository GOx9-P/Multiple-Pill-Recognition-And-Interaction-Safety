from __future__ import annotations

from typing import Any

import numpy as np

from pill_safety.cv.ocr.config import OCRConfig
from pill_safety.cv.ocr.postprocessing.candidates import normalize_candidate_text
from pill_safety.schemas import OCRInferenceOutput, OCRInferenceRequest


def _text_regions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regions = []
    for index, item in enumerate(items or [], start=1):
        regions.append(
            {
                "region_id": f"region_{index:02d}",
                "polygon": item.get("polygon_original")
                or item.get("polygon")
                or [],
                "detection_confidence": round(
                    float(item.get("confidence", 0.0)), 4
                ),
            }
        )
    return regions


def _polygon_center(item: dict[str, Any]) -> tuple[float, float] | None:
    polygon = item.get("polygon_original") or item.get("polygon")
    if not polygon:
        return None
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        return None
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def _canonical_region_id(
    item: dict[str, Any],
    item_index: int,
    canonical_items: list[dict[str, Any]],
) -> str:
    if not canonical_items:
        return "region_01"
    item_center = _polygon_center(item)
    canonical_centers = [_polygon_center(candidate) for candidate in canonical_items]
    if item_center is not None and any(center is not None for center in canonical_centers):
        distances = [
            (
                float("inf")
                if center is None
                else (item_center[0] - center[0]) ** 2
                + (item_center[1] - center[1]) ** 2
            )
            for center in canonical_centers
        ]
        return f"region_{int(np.argmin(distances)) + 1:02d}"
    canonical_index = min(item_index, len(canonical_items))
    return f"region_{canonical_index:02d}"


def _ocr_observations(
    observations: list[dict[str, Any]],
    canonical_items: list[dict[str, Any]],
    config: OCRConfig,
) -> list[dict[str, Any]]:
    if not canonical_items:
        return []
    ordered = sorted(
        observations or [],
        key=lambda observation: (
            observation.get("priority", 0),
            observation.get("best_confidence", 0.0),
        ),
        reverse=True,
    )
    result = []
    for observation in ordered:
        items = observation.get("ordered_items") or []
        if not items:
            items = [
                {
                    "text": observation.get("detected_text", ""),
                    "confidence": observation.get("best_confidence", 0.0),
                }
            ]
        for region_index, item in enumerate(items, start=1):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            result.append(
                {
                    "region_id": _canonical_region_id(
                        item, region_index, canonical_items
                    ),
                    "rotation_degrees": int(
                        observation.get("rotation_degrees", 0)
                    ),
                    "preprocessing": str(
                        observation.get("preprocessing", "original")
                    ),
                    "text": text,
                    "confidence": round(
                        float(
                            item.get(
                                "confidence",
                                observation.get("best_confidence", 0.0),
                            )
                        ),
                        4,
                    ),
                }
            )
            if len(result) >= config.max_schema_ocr_observations:
                return result
    return result


def _candidate_evidence(candidate: dict[str, Any]) -> list[str]:
    evidence = []
    evidence.extend(f"mode={mode}" for mode in candidate.get("modes", []))
    evidence.extend(
        f"rot={rotation}" for rotation in candidate.get("rotations", [])
    )
    evidence.extend(
        f"preprocessing={name}"
        for name in candidate.get("preprocessings", [])
    )
    return evidence or ["raw_ocr"]


def _normalized_candidates(
    final_candidate: dict[str, Any],
    best_observation: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
    config: OCRConfig,
) -> list[dict[str, Any]]:
    final_key = normalize_candidate_text(final_candidate.get("text", ""))
    candidates = [
        {
            "text": final_candidate.get("text", ""),
            "score": round(float(final_candidate.get("score", 0.0)), 4),
            "source": "raw_ocr",
            "evidence": [
                "legacy_priority_confidence",
                f"mode={best_observation.get('mode', 'unknown')}",
                f"rot={best_observation.get('rotation_degrees', 0)}",
                f"preprocessing={best_observation.get('preprocessing', 'unknown')}",
            ],
        }
    ]
    seen = {final_key}
    for candidate in ranked_candidates or []:
        key = candidate.get("normalized_text") or normalize_candidate_text(
            candidate.get("text", "")
        )
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "text": candidate.get("text", ""),
                "score": round(float(candidate.get("score", 0.0)), 4),
                "source": "multi_angle_consensus",
                "evidence": _candidate_evidence(candidate),
            }
        )
        if len(candidates) >= config.max_schema_normalized_candidates:
            break
    return candidates


def _scoreline_result(scoreline: dict[str, Any] | None) -> dict[str, Any]:
    """Chuẩn hóa quyết định scoreline nội bộ sang public schema của Module 3."""

    value = scoreline or {}
    line = value.get("line_xyxy")
    return {
        "visible": bool(value.get("visible", False)),
        "confidence": round(float(value.get("confidence", 0.0)), 4),
        "angle_degrees": (
            round(float(value["angle_degrees"]), 4)
            if value.get("angle_degrees") is not None
            else None
        ),
        "orientation": str(value.get("orientation", "unknown")),
        "line_xyxy": [round(float(point), 4) for point in line] if line else None,
        "support_count": int(value.get("support_count", 0)),
        "rotation_degrees": (
            int(value["rotation_degrees"])
            if value.get("rotation_degrees") is not None
            else None
        ),
        "preprocessing": (
            str(value["preprocessing"])
            if value.get("preprocessing") is not None
            else None
        ),
        "source": str(value.get("source", "ocr_hough_consensus")),
    }


def build_ocr_output(
    request: OCRInferenceRequest,
    config: OCRConfig,
    final_candidate: dict[str, Any] | None = None,
    best_observation: dict[str, Any] | None = None,
    best_items: list[dict[str, Any]] | None = None,
    valid_observations: list[dict[str, Any]] | None = None,
    ranked_candidates: list[dict[str, Any]] | None = None,
    scoreline: dict[str, Any] | None = None,
) -> OCRInferenceOutput:
    """Ánh xạ kết quả OCR và scoreline sang contract công khai của Module 3."""

    visible = final_candidate is not None
    confidence = (
        round(float(final_candidate.get("score", 0.0)), 4) if visible else 0.0
    )
    return OCRInferenceOutput.model_validate(
        {
            "request_id": request.request_id,
            "session_id": request.session_id,
            "image_id": request.image_id,
            "instance_id": request.instance_id,
            "instance_token": request.instance_token,
            "scoreline": _scoreline_result(scoreline),
            "imprint_visibility": {
                "visible": visible,
                "confidence": confidence,
            },
            "imprint": {
                "visible": visible,
                "raw": final_candidate.get("text", "") if visible else "",
                "confidence": confidence,
                "text_regions": _text_regions(best_items if visible else []),
                "ocr_observations": _ocr_observations(
                    valid_observations if visible else [],
                    best_items if visible else [],
                    config,
                ),
                "normalized_candidates": (
                    _normalized_candidates(
                        final_candidate,
                        best_observation,
                        ranked_candidates or [],
                        config,
                    )
                    if visible
                    else []
                ),
            },
        }
    )
