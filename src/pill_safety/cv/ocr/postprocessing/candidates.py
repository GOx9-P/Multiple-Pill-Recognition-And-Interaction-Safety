from __future__ import annotations

import re
from typing import Any

import numpy as np

from pill_safety.cv.ocr.config import OCRConfig
from pill_safety.cv.ocr.postprocessing.ordering import sequence_confidence


def is_usable_observation(observation: dict[str, Any], config: OCRConfig) -> bool:
    text = observation.get("detected_text", "")
    if "?" in text or re.search(r"[^\x00-\x7F]+", str(text)):
        return False
    text_without_spaces = str(text).replace(" ", "")
    if not text_without_spaces:
        return False
    cleaned = "".join(character for character in text_without_spaces if character.isalnum())
    if not cleaned:
        return False
    if len(cleaned) / len(text_without_spaces) < 0.5:
        return False
    return (
        observation["best_confidence"] >= config.min_usable_confidence
        and len(cleaned) >= config.min_usable_text_length
    )


def normalize_candidate_text(text: str) -> str:
    return "".join(character for character in str(text).upper() if character.isalnum())


def rank_text_candidates(
    observations: list[dict[str, Any]], config: OCRConfig
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for observation_index, observation in enumerate(observations):
        if observation.get("mode") == "scoreline_side_split" and not observation.get(
            "split_info", {}
        ).get("reliable", False):
            continue
        candidates = observation.get("text_candidates") or [
            {
                "ordering": "linear",
                "text": observation.get("detected_text", ""),
                "items": observation.get("ordered_items", []),
            }
        ]
        evidence_id = (
            f"{observation.get('tier')}|{observation.get('rotation_degrees')}|"
            f"{observation.get('preprocessing')}|{observation.get('mode')}"
        )
        for candidate in candidates:
            text = str(candidate.get("text", "")).strip()
            key = normalize_candidate_text(text)
            items = candidate.get("items", [])
            confidence = sequence_confidence(items)
            if not key or confidence < config.min_usable_confidence:
                continue
            group = groups.setdefault(key, {"evidence": {}})
            current = group["evidence"].get(evidence_id)
            evidence = {
                "text": text,
                "confidence": confidence,
                "ordering": candidate.get("ordering", "linear"),
                "observation": observation,
                "items": items,
                "obs_index": observation_index,
            }
            if current is None or confidence > current["confidence"]:
                group["evidence"][evidence_id] = evidence

    ranked = []
    for key, group in groups.items():
        evidence = list(group["evidence"].values())
        best_evidence = max(evidence, key=lambda item: item["confidence"])
        mean_confidence = float(np.mean([item["confidence"] for item in evidence]))
        support_count = len(evidence)
        modes = sorted(
            set(item["observation"].get("mode", "unknown") for item in evidence)
        )
        support_score = min(support_count / 3.0, 1.0)
        mode_score = min(len(modes) / 2.0, 1.0)
        score = float(
            np.clip(
                0.65 * mean_confidence
                + 0.25 * support_score
                + 0.10 * mode_score,
                0.0,
                1.0,
            )
        )
        ranked.append(
            {
                "text": best_evidence["text"],
                "normalized_text": key,
                "score": round(score, 4),
                "mean_ocr_confidence": round(mean_confidence, 4),
                "support_count": support_count,
                "modes": modes,
                "rotations": sorted(
                    set(
                        int(item["observation"].get("rotation_degrees", 0))
                        for item in evidence
                    )
                ),
                "preprocessings": sorted(
                    set(
                        item["observation"].get("preprocessing", "unknown")
                        for item in evidence
                    )
                ),
                "_best_observation": best_evidence["observation"],
                "_best_items": best_evidence["items"],
            }
        )
    return sorted(
        ranked,
        key=lambda candidate: (
            candidate["score"],
            candidate["support_count"],
            candidate["mean_ocr_confidence"],
        ),
        reverse=True,
    )[: config.max_final_candidates]


def public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in candidate.items() if not key.startswith("_")
    }


def finalize_scoreline(
    scoreline_observations: list[dict[str, Any]], config: OCRConfig
) -> dict[str, Any]:
    visible = [
        observation
        for observation in scoreline_observations
        if observation.get("visible")
        and observation.get("confidence", 0.0)
        >= config.min_scoreline_detection_confidence
    ]
    if len(visible) < config.min_scoreline_support:
        return {
            "visible": False,
            "confidence": 0.0,
            "angle_degrees": None,
            "orientation": "unknown",
            "line_xyxy": None,
            "support_count": len(visible),
            "source": "ocr_hough_consensus",
        }
    visible = sorted(
        visible, key=lambda observation: observation.get("confidence", 0.0), reverse=True
    )
    best = visible[0]
    top_confidences = [float(item["confidence"]) for item in visible[:3]]
    support_score = min(len(visible) / 3.0, 1.0)
    confidence = float(
        np.clip(0.75 * np.mean(top_confidences) + 0.25 * support_score, 0.0, 1.0)
    )
    return {
        "visible": True,
        "confidence": round(confidence, 4),
        "angle_degrees": best.get("angle_degrees"),
        "orientation": best.get("orientation", "unknown"),
        "line_xyxy": best.get("line_xyxy"),
        "support_count": len(visible),
        "rotation_degrees": best.get("rotation_degrees"),
        "preprocessing": best.get("preprocessing"),
        "source": "ocr_hough_consensus",
    }


def select_baseline_observation(
    observations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not observations:
        return None
    return sorted(
        observations,
        key=lambda observation: (
            observation.get("priority", 0),
            observation.get("best_confidence", 0.0),
        ),
        reverse=True,
    )[0]


def build_final_candidate(
    best_observation: dict[str, Any], ranked_candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    best_key = normalize_candidate_text(best_observation["detected_text"])
    final_sequence_confidence = sequence_confidence(
        best_observation.get("ordered_items", [])
    )
    matched_candidate = next(
        (
            candidate
            for candidate in ranked_candidates
            if candidate["normalized_text"] == best_key
        ),
        None,
    )
    return {
        "text": best_observation["detected_text"],
        "normalized_text": best_key,
        "score": round(final_sequence_confidence, 4),
        "mean_ocr_confidence": (
            round(float(matched_candidate["mean_ocr_confidence"]), 4)
            if matched_candidate
            else round(final_sequence_confidence, 4)
        ),
        "support_count": (
            int(matched_candidate["support_count"]) if matched_candidate else 1
        ),
        "modes": (
            matched_candidate["modes"]
            if matched_candidate
            else [best_observation.get("mode", "unknown")]
        ),
        "rotations": (
            matched_candidate["rotations"]
            if matched_candidate
            else [best_observation.get("rotation_degrees", 0)]
        ),
        "preprocessings": (
            matched_candidate["preprocessings"]
            if matched_candidate
            else [best_observation.get("preprocessing", "unknown")]
        ),
        "selection_method": "legacy_priority_confidence",
    }
