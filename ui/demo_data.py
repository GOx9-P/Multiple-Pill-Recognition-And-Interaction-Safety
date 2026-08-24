"""Demo scenarios and pharmaceutical test presets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = PROJECT_ROOT / "tests" / "rag" / "fakeoutputCV"

PRESET_SCENARIOS: dict[str, dict[str, Any]] = {
    "critical": {
        "schema_version": "cv_output_v0",
        "request_id": "req_preset_critical_01",
        "session_id": "sess_preset_critical_01",
        "image_quality": {"status": "good", "blur_score": 0.02, "glare_detected": False, "lighting_warning": False},
        "pills": [
            {
                "instance_id": "pill_001_clopidogrel",
                "bbox_xyxy": [80.0, 100.0, 300.0, 320.0],
                "shape": {"label": "ROUND", "confidence": 0.98},
                "color": {"primary": "WHITE", "confidence": 0.96},
                "scoreline": {"visible": False},
                "imprint": {"raw": "84A", "confidence": 0.98, "normalized_candidates": [{"text": "84A", "score": 0.98}]},
            },
            {
                "instance_id": "pill_002_omeprazole",
                "bbox_xyxy": [380.0, 100.0, 620.0, 320.0],
                "shape": {"label": "OVAL", "confidence": 0.96},
                "color": {"primary": "ORANGE", "confidence": 0.94},
                "scoreline": {"visible": True},
                "imprint": {"raw": "8335BARR", "confidence": 0.97, "normalized_candidates": [{"text": "8335BARR", "score": 0.97}]},
            },
        ],
    },
    "moderate": {
        "schema_version": "cv_output_v0",
        "request_id": "req_preset_moderate_02",
        "session_id": "sess_preset_moderate_02",
        "image_quality": {"status": "good", "blur_score": 0.04, "glare_detected": False, "lighting_warning": False},
        "pills": [
            {
                "instance_id": "pill_001_aspirin",
                "bbox_xyxy": [100.0, 100.0, 320.0, 320.0],
                "shape": {"label": "ROUND", "confidence": 0.97},
                "color": {"primary": "YELLOW", "confidence": 0.95},
                "scoreline": {"visible": False},
                "imprint": {"raw": "TV5056", "confidence": 0.96, "normalized_candidates": [{"text": "TV5056", "score": 0.96}]},
            },
            {
                "instance_id": "pill_002_lisinopril",
                "bbox_xyxy": [400.0, 100.0, 620.0, 320.0],
                "shape": {"label": "ROUND", "confidence": 0.96},
                "color": {"primary": "PINK", "confidence": 0.93},
                "scoreline": {"visible": True},
                "imprint": {"raw": "LUPIN10", "confidence": 0.95, "normalized_candidates": [{"text": "LUPIN10", "score": 0.95}]},
            },
        ],
    },
    "unresolved": {
        "schema_version": "cv_output_v0",
        "request_id": "req_preset_unresolved_03",
        "session_id": "sess_preset_unresolved_03",
        "image_quality": {"status": "warning", "blur_score": 0.35, "glare_detected": True, "lighting_warning": True},
        "pills": [
            {
                "instance_id": "pill_001_identified",
                "bbox_xyxy": [90.0, 100.0, 300.0, 310.0],
                "shape": {"label": "ROUND", "confidence": 0.97},
                "color": {"primary": "WHITE", "confidence": 0.95},
                "scoreline": {"visible": False},
                "imprint": {"raw": "84A", "confidence": 0.97, "normalized_candidates": [{"text": "84A", "score": 0.97}]},
            },
            {
                "instance_id": "pill_002_blurred",
                "bbox_xyxy": [380.0, 110.0, 590.0, 320.0],
                "shape": {"label": "UNKNOWN", "confidence": 0.35},
                "color": {"primary": "WHITE", "confidence": 0.40},
                "scoreline": {"visible": False},
                "imprint": {"raw": "?", "confidence": 0.10, "normalized_candidates": []},
            },
        ],
    },
    "safe": {
        "schema_version": "cv_output_v0",
        "request_id": "req_preset_safe_04",
        "session_id": "sess_preset_safe_04",
        "image_quality": {"status": "good", "blur_score": 0.01, "glare_detected": False, "lighting_warning": False},
        "pills": [
            {
                "instance_id": "pill_001_aspirin_only",
                "bbox_xyxy": [250.0, 100.0, 500.0, 350.0],
                "shape": {"label": "ROUND", "confidence": 0.98},
                "color": {"primary": "YELLOW", "confidence": 0.97},
                "scoreline": {"visible": False},
                "imprint": {"raw": "TV5056", "confidence": 0.98, "normalized_candidates": [{"text": "TV5056", "score": 0.98}]},
            }
        ],
    },
}


def get_preset_scenario(scenario_name: str) -> dict[str, Any]:
    """Retrieve demo preset scenario dictionary by key or JSON filename."""
    if scenario_name in PRESET_SCENARIOS:
        return PRESET_SCENARIOS[scenario_name]

    # Try loading from tests/rag/fakeoutputCV/
    filepath = SCENARIOS_DIR / scenario_name
    if filepath.is_file():
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    # Fallback to critical preset
    return PRESET_SCENARIOS["critical"]
