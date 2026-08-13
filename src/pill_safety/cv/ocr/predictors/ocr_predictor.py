from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from pill_safety.cv.ocr.config import OCRConfig
from pill_safety.cv.ocr.engines import OCREngine, PaddleOCREngine
from pill_safety.cv.ocr.postprocessing import (
    build_final_candidate,
    build_ocr_output,
    build_order_candidates,
    candidate_text,
    detect_scoreline_for_split,
    filter_text_regions,
    finalize_scoreline,
    is_usable_observation,
    item_center,
    map_scoreline_to_original,
    rank_text_candidates,
    run_scoreline_side_split,
)
from pill_safety.cv.ocr.postprocessing.candidates import public_candidate
from pill_safety.cv.ocr.preprocessing import (
    apply_preprocessing,
    attach_original_polygons,
    prepare_image_bgr,
    prepare_foreground_mask,
    rotate_bgr,
    rotate_foreground_mask,
)
from pill_safety.cv.ocr.utils import draw_items
from pill_safety.schemas import OCRInferenceOutput, OCRInferenceRequest


@dataclass(frozen=True)
class OCRArtifacts:
    output: OCRInferenceOutput
    schema_json_path: Path
    debug_json_path: Path
    overlay_path: Path | None


def _safe_directory_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "pill"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


class OCRPredictor:
    def __init__(
        self,
        config: OCRConfig | None = None,
        engine: OCREngine | None = None,
    ):
        self.config = config or OCRConfig()
        self.engine = engine or PaddleOCREngine(self.config)

    def predict(
        self, request: OCRInferenceRequest | dict[str, Any]
    ) -> OCRInferenceOutput:
        return self.predict_with_artifacts(request).output

    def predict_with_artifacts(
        self, request: OCRInferenceRequest | dict[str, Any]
    ) -> OCRArtifacts:
        request = OCRInferenceRequest.model_validate(request)
        image_path = Path(request.crop_path)
        prepared_image = prepare_image_bgr(image_path)
        base_image = prepared_image.bgr
        foreground_mask = prepare_foreground_mask(request.mask_path, prepared_image)

        instance_directory = _safe_directory_name(request.instance_id)
        output_directory = (
            self.config.output_dir
            / _safe_directory_name(request.request_id)
            / _safe_directory_name(request.image_id)
            / instance_directory
        )
        variant_directory = output_directory / "variants"
        split_directory = output_directory / "side_split"
        paddle_json_directory = output_directory / "paddleocr_json"
        for directory in (
            output_directory,
            variant_directory,
            split_directory,
            paddle_json_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        observations: list[dict[str, Any]] = []
        performed_steps: list[dict[str, Any]] = []
        scoreline_observations: list[dict[str, Any]] = []

        for tier_config in self.config.rotation_tiers:
            tier_observation_start = len(observations)
            for rotation_degrees in tier_config.rotations:
                for preprocessing in self.config.preprocessing_steps:
                    rotated = rotate_bgr(base_image, rotation_degrees)
                    variant = apply_preprocessing(rotated, preprocessing)
                    rotated_foreground_mask = (
                        rotate_foreground_mask(foreground_mask, rotation_degrees)
                        if foreground_mask is not None
                        else None
                    )
                    step_id = (
                        f"{tier_config.tier}_rot{rotation_degrees}_{preprocessing}"
                    )
                    variant_path = variant_directory / f"{step_id}.jpg"
                    cv2.imwrite(str(variant_path), variant)

                    # Oblique warp borders can look like scorelines to Hough.
                    if int(rotation_degrees) % 90 == 0:
                        scoreline_variant = detect_scoreline_for_split(
                            variant,
                            self.config,
                            rotated_foreground_mask,
                        )
                    else:
                        scoreline_variant = {
                            "visible": False,
                            "confidence": 0.0,
                            "line_xyxy": None,
                            "angle_degrees": None,
                            "orientation": "unknown",
                            "reason": "oblique_rotation_border_guard",
                        }
                    scoreline_variant.update(
                        {
                            "tier": tier_config.tier,
                            "rotation_degrees": rotation_degrees,
                            "preprocessing": preprocessing,
                            "variant_path": str(variant_path),
                        }
                    )
                    # Public scoreline data always uses original crop coordinates.
                    # Variant coordinates remain internal for split and overlay work.
                    scoreline = map_scoreline_to_original(
                        scoreline_variant,
                        padded_shape=base_image.shape,
                        rotation_degrees=rotation_degrees,
                        prepared_image=prepared_image,
                    )
                    scoreline_observations.append(scoreline)
                    raw_items = self.engine.predict(
                        variant_path, paddle_json_directory, step_id
                    )
                    raw_items = [item for item in raw_items if item.get("text")]
                    items, rejected_regions = filter_text_regions(
                        raw_items,
                        rotated_foreground_mask,
                        self.config,
                    )
                    performed_steps.append(
                        {
                            "step_id": step_id,
                            "rotation_degrees": rotation_degrees,
                            "preprocessing": preprocessing,
                            "scoreline_visible": scoreline.get("visible", False),
                            "text_regions_kept": len(items),
                            "text_regions_rejected": len(rejected_regions),
                        }
                    )
                    attach_original_polygons(
                        items,
                        padded_shape=base_image.shape,
                        rotation_degrees=rotation_degrees,
                        prepared_image=prepared_image,
                    )
                    for item in items:
                        item["center_x"], item["center_y"] = item_center(item)
                    order_candidates = build_order_candidates(
                        items, variant.shape, rotation_degrees, self.config
                    )
                    ordered_items = (
                        order_candidates[0]["items"] if order_candidates else []
                    )
                    detected_text = (
                        order_candidates[0]["text"] if order_candidates else ""
                    )
                    best_confidence = max(
                        [item["confidence"] for item in items], default=0.0
                    )
                    full_priority = 2 if len(ordered_items) >= 2 else 1
                    if items:
                        observations.append(
                            {
                                "mode": "full_image",
                                "tier": tier_config.tier,
                                "priority": full_priority,
                                "rotation_degrees": rotation_degrees,
                                "preprocessing": preprocessing,
                                "variant_path": str(variant_path),
                                "items": items,
                                "ordered_items": ordered_items,
                                "text_candidates": order_candidates,
                                "best_confidence": best_confidence,
                                "detected_text": detected_text,
                                "scoreline": scoreline,
                                "scoreline_variant": scoreline_variant,
                            }
                        )

                    if (
                        self.config.enable_scoreline_side_split
                        and scoreline_variant.get("visible")
                    ):
                        split_items, split_info = run_scoreline_side_split(
                            variant,
                            rotation_degrees,
                            step_id,
                            split_directory,
                            paddle_json_directory,
                            scoreline_variant,
                            self.engine,
                            self.config,
                        )
                        attach_original_polygons(
                            split_items,
                            padded_shape=base_image.shape,
                            rotation_degrees=rotation_degrees,
                            prepared_image=prepared_image,
                        )
                        split_text = candidate_text(split_items)
                        split_confidence = max(
                            [item["confidence"] for item in split_items],
                            default=0.0,
                        )
                        split_priority = 3 if split_info.get("reliable") else 0
                        if split_items:
                            observations.append(
                                {
                                    "mode": "scoreline_side_split",
                                    "tier": tier_config.tier,
                                    "priority": split_priority,
                                    "rotation_degrees": rotation_degrees,
                                    "preprocessing": f"{preprocessing}_side_split",
                                    "variant_path": str(variant_path),
                                    "items": split_items,
                                    "ordered_items": split_items,
                                    "text_candidates": [
                                        {
                                            "ordering": "scoreline_normal",
                                            "text": split_text,
                                            "items": split_items,
                                        }
                                    ],
                                    "best_confidence": split_confidence,
                                    "detected_text": split_text,
                                    "scoreline": scoreline,
                                    "scoreline_variant": scoreline_variant,
                                    "split_info": split_info,
                                }
                            )

            tier_observations = observations[tier_observation_start:]
            if (
                any(
                    is_usable_observation(observation, self.config)
                    for observation in tier_observations
                )
                and not self.config.force_run_all_rotation_tiers
            ):
                break

        final_scoreline = finalize_scoreline(
            scoreline_observations, self.config
        )
        valid_observations = [
            observation
            for observation in observations
            if is_usable_observation(observation, self.config)
            and (
                observation.get("mode") != "scoreline_side_split"
                or final_scoreline.get("visible")
            )
        ]
        ranked_candidates = rank_text_candidates(
            valid_observations, self.config
        )

        debug_json_path = output_directory / f"{instance_directory}_final_result.json"
        schema_json_path = output_directory / f"{instance_directory}_ocr_schema.json"
        overlay_path: Path | None = None

        if valid_observations:
            if not ranked_candidates:
                raise RuntimeError("Valid OCR observations produced no ranked candidate.")
            best = ranked_candidates[0]["_best_observation"]
            best_items = ranked_candidates[0]["_best_items"]
            final_candidate = build_final_candidate(best, ranked_candidates)
            best_image = cv2.imread(best["variant_path"])
            if best_image is None:
                raise FileNotFoundError(
                    f"Cannot read selected OCR variant: {best['variant_path']}"
                )
            overlay_scoreline = (
                best.get("scoreline_variant")
                if best.get("mode") == "scoreline_side_split"
                and best.get("split_info", {}).get("reliable", False)
                else None
            )
            overlay = draw_items(best_image, best_items, overlay_scoreline)
            overlay_path = output_directory / f"{instance_directory}_final_overlay.jpg"
            cv2.imwrite(str(overlay_path), overlay)

            debug_payload = {
                "image_name": image_path.name,
                "final_answer": final_candidate,
                "scoreline": final_scoreline,
                "candidates": [
                    public_candidate(candidate) for candidate in ranked_candidates
                ],
                "selected_observation": {
                    "mode": best.get("mode"),
                    "rotation_degrees": best.get("rotation_degrees"),
                    "preprocessing": best.get("preprocessing"),
                    "variant_path": best.get("variant_path"),
                },
                "performed_steps": performed_steps,
                "overlay_path": str(overlay_path),
            }
            output = build_ocr_output(
                request=request,
                config=self.config,
                final_candidate=final_candidate,
                best_observation=best,
                best_items=best_items,
                valid_observations=valid_observations,
                ranked_candidates=ranked_candidates,
                scoreline=final_scoreline,
            )
        else:
            debug_payload = {
                "image_name": image_path.name,
                "final_answer": None,
                "scoreline": final_scoreline,
                "candidates": [],
                "performed_steps": performed_steps,
                "overlay_path": None,
            }
            output = build_ocr_output(
                request=request,
                config=self.config,
                scoreline=final_scoreline,
            )

        _write_json(debug_json_path, debug_payload)
        _write_json(schema_json_path, output.model_dump(mode="json"))
        return OCRArtifacts(
            output=output,
            schema_json_path=schema_json_path,
            debug_json_path=debug_json_path,
            overlay_path=overlay_path,
        )
