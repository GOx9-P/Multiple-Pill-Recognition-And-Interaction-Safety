from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RotationTier:
    tier: str
    rotations: tuple[int, ...]


@dataclass(frozen=True)
class OCRConfig:
    ocr_version: str = "PP-OCRv5"
    language: str = "en"
    device: str = "auto"
    det_db_thresh: float = 0.2
    det_db_unclip_ratio: float = 2.0
    preprocessing_steps: tuple[str, ...] = (
        "original",
        "clahe",
        "blackhat",
        "blackhat_bold",
    )
    rotation_tiers: tuple[RotationTier, ...] = (
        RotationTier("tier1_0_180", (0, 180)),
        RotationTier("tier2_90_270", (90, 270)),
        RotationTier("tier3_oblique", (-45, -30, -15, 15, 30, 45)),
    )
    min_usable_confidence: float = 0.50
    min_usable_text_length: int = 1
    min_text_region_foreground_coverage: float = 0.70
    max_text_region_area_ratio: float = 0.75
    text_region_edge_margin_ratio: float = 0.02
    force_run_all_rotation_tiers: bool = True
    enable_scoreline_side_split: bool = True
    min_scoreline_detection_confidence: float = 0.45
    min_scoreline_support: int = 2
    scoreline_angle_consensus_tolerance_degrees: float = 12.0
    scoreline_consensus_distance_ratio: float = 0.08
    scoreline_center_max_distance_ratio: float = 0.30
    scoreline_use_center_roi: bool = True
    scoreline_min_foreground_coverage: float = 0.75
    split_margin_ratio: float = 0.03
    min_side_confidence: float = 0.60
    enable_circular_text_order: bool = True
    min_circular_boxes: int = 3
    max_final_candidates: int = 10
    max_schema_ocr_observations: int = 8
    max_schema_normalized_candidates: int = 5
    output_dir: Path = Path("outputs/predictions/ocr")

    def with_output_dir(self, output_dir: str | Path) -> "OCRConfig":
        return replace(self, output_dir=Path(output_dir))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OCRConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}

        model = raw.get("model", {})
        pipeline = raw.get("pipeline", {})
        scoreline = raw.get("scoreline", {})
        candidates = raw.get("candidates", {})
        schema = raw.get("schema", {})
        artifacts = raw.get("artifacts", {})

        tiers = tuple(
            RotationTier(str(item["tier"]), tuple(int(v) for v in item["rotations"]))
            for item in pipeline.get("rotation_tiers", [])
        )
        defaults = cls()
        values: dict[str, Any] = {
            "ocr_version": model.get("ocr_version", defaults.ocr_version),
            "language": model.get("language", defaults.language),
            "device": model.get("device", defaults.device),
            "det_db_thresh": model.get("det_db_thresh", defaults.det_db_thresh),
            "det_db_unclip_ratio": model.get(
                "det_db_unclip_ratio", defaults.det_db_unclip_ratio
            ),
            "preprocessing_steps": tuple(
                pipeline.get("preprocessing_steps", defaults.preprocessing_steps)
            ),
            "rotation_tiers": tiers or defaults.rotation_tiers,
            "min_usable_confidence": pipeline.get(
                "min_usable_confidence", defaults.min_usable_confidence
            ),
            "min_usable_text_length": pipeline.get(
                "min_usable_text_length", defaults.min_usable_text_length
            ),
            "min_text_region_foreground_coverage": pipeline.get(
                "min_text_region_foreground_coverage",
                defaults.min_text_region_foreground_coverage,
            ),
            "max_text_region_area_ratio": pipeline.get(
                "max_text_region_area_ratio",
                defaults.max_text_region_area_ratio,
            ),
            "text_region_edge_margin_ratio": pipeline.get(
                "text_region_edge_margin_ratio",
                defaults.text_region_edge_margin_ratio,
            ),
            "force_run_all_rotation_tiers": pipeline.get(
                "force_run_all_rotation_tiers",
                defaults.force_run_all_rotation_tiers,
            ),
            "enable_scoreline_side_split": scoreline.get(
                "enable_side_split", defaults.enable_scoreline_side_split
            ),
            "min_scoreline_detection_confidence": scoreline.get(
                "min_detection_confidence",
                defaults.min_scoreline_detection_confidence,
            ),
            "min_scoreline_support": scoreline.get(
                "min_support", defaults.min_scoreline_support
            ),
            "scoreline_angle_consensus_tolerance_degrees": scoreline.get(
                "angle_consensus_tolerance_degrees",
                defaults.scoreline_angle_consensus_tolerance_degrees,
            ),
            "scoreline_consensus_distance_ratio": scoreline.get(
                "consensus_distance_ratio",
                defaults.scoreline_consensus_distance_ratio,
            ),
            "scoreline_center_max_distance_ratio": scoreline.get(
                "center_max_distance_ratio",
                defaults.scoreline_center_max_distance_ratio,
            ),
            "scoreline_use_center_roi": scoreline.get(
                "use_center_roi", defaults.scoreline_use_center_roi
            ),
            "scoreline_min_foreground_coverage": scoreline.get(
                "min_foreground_coverage",
                defaults.scoreline_min_foreground_coverage,
            ),
            "split_margin_ratio": scoreline.get(
                "split_margin_ratio", defaults.split_margin_ratio
            ),
            "min_side_confidence": scoreline.get(
                "min_side_confidence", defaults.min_side_confidence
            ),
            "enable_circular_text_order": candidates.get(
                "enable_circular_text_order",
                defaults.enable_circular_text_order,
            ),
            "min_circular_boxes": candidates.get(
                "min_circular_boxes", defaults.min_circular_boxes
            ),
            "max_final_candidates": candidates.get(
                "max_final_candidates", defaults.max_final_candidates
            ),
            "max_schema_ocr_observations": schema.get(
                "max_ocr_observations", defaults.max_schema_ocr_observations
            ),
            "max_schema_normalized_candidates": schema.get(
                "max_normalized_candidates",
                defaults.max_schema_normalized_candidates,
            ),
            "output_dir": Path(artifacts.get("output_dir", defaults.output_dir)),
        }
        return cls(**values)
