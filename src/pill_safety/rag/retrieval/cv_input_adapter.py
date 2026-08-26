from __future__ import annotations

import re
from typing import Any

from pill_safety.rag.retrieval.normalization import (
    expand_imprint_variants,
    normalize_color,
    normalize_dosage_form,
    normalize_imprint,
    normalize_scoreline_label,
    normalize_shape,
    normalize_token,
)
from pill_safety.rag.retrieval.types import (
    ColorEvidence,
    ImageQualityEvidence,
    ImprintCandidate,
    LabelEvidence,
    LogoEvidence,
    RecognitionInput,
    ScorelineEvidence,
    SegmentationEvidence,
)


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _adapt_imprints(pill: dict[str, Any]) -> list[ImprintCandidate]:
    imprint = _dict(pill.get("imprint"))
    candidates: dict[str, ImprintCandidate] = {}
    for row in _list(imprint.get("normalized_candidates")):
        if not isinstance(row, dict):
            continue
        raw_text = str(row.get("text") or "")
        base_score = _float(row.get("score"), 0.0)
        if not raw_text or base_score <= 0:
            continue
        for text, multiplier, evidence in expand_imprint_variants(raw_text):
            score = base_score * multiplier
            previous = candidates.get(text)
            if previous is None or score > previous.score:
                candidates[text] = ImprintCandidate(
                    text=text,
                    raw_text=raw_text,
                    score=score,
                    source=row.get("source"),
                    evidence=[*_list(row.get("evidence")), *evidence],
                )

    # Bổ sung các observation OCR trực tiếp (nhiều góc quay/tiền xử lý) nếu có
    for obs in _list(imprint.get("ocr_observations")):
        if isinstance(obs, dict):
            obs_text = str(obs.get("text") or "").strip()
            obs_conf = _float(obs.get("confidence"), 0.5)
            if obs_text and obs_conf > 0.25:
                for text, multiplier, evidence in expand_imprint_variants(obs_text):
                    score = obs_conf * multiplier * 0.95
                    previous = candidates.get(text)
                    if previous is None or score > previous.score:
                        candidates[text] = ImprintCandidate(
                            text=text,
                            raw_text=obs_text,
                            score=score,
                            source="ocr_observation",
                            evidence=["observation", *evidence],
                        )

    raw = imprint.get("raw")
    if raw:
        normalized = normalize_imprint(str(raw))
        if normalized and normalized not in candidates:
            candidates[normalized] = ImprintCandidate(
                text=normalized,
                raw_text=str(raw),
                score=_float(imprint.get("confidence"), 0.5),
                source="raw_ocr",
                evidence=["raw_ocr"],
            )
        # Bổ sung các token con nếu raw có ký tự phân cách (ví dụ "13;T", "13/T", "13 - T")
        raw_tokens = [normalize_imprint(t) for t in re.split(r"[;\s/\\|,-]+", str(raw)) if t.strip()]
        for tok in raw_tokens:
            if tok and tok not in candidates and len(tok) >= 1:
                candidates[tok] = ImprintCandidate(
                    text=tok,
                    raw_text=str(raw),
                    score=_float(imprint.get("confidence"), 0.5) * 0.95,
                    source="raw_token",
                    evidence=["raw_token"],
                )
    return sorted(candidates.values(), key=lambda item: item.score, reverse=True)[:8]


def adapt_cv_pill_to_recognition_input(
    *,
    rag_request: dict[str, Any],
    cv_output: dict[str, Any],
    pill: dict[str, Any],
) -> RecognitionInput:
    segmentation = _dict(pill.get("segmentation"))
    image_quality = _dict(cv_output.get("image_quality"))
    imprint = _dict(pill.get("imprint"))
    imprint_visibility = _dict(pill.get("imprint_visibility"))
    shape = _dict(pill.get("shape"))
    color = _dict(pill.get("color"))
    dosage_form = _dict(pill.get("dosage_form"))
    scoreline = _dict(pill.get("scoreline"))
    logo = _dict(pill.get("logo_or_symbol"))

    color_distribution = {
        normalized: _float(score)
        for label, score in _dict(color.get("distribution")).items()
        if (normalized := normalize_color(str(label))) is not None
    }

    raw_cv_status = str(pill.get("cv_status") or "").lower()
    if raw_cv_status in {"detected", "ready", "features_ready"}:
        cv_status = "features_ready"
    else:
        cv_status = raw_cv_status

    return RecognitionInput(
        instance_id=str(pill.get("instance_id") or ""),
        instance_token=pill.get("instance_token"),
        market=normalize_token(rag_request.get("market") or "US"),
        cv_status=cv_status,

        segmentation=SegmentationEvidence(
            confidence=_float(segmentation.get("confidence")),
            occlusion_estimate=segmentation.get("occlusion_estimate"),
            possible_merged_instance=_bool(segmentation.get("possible_merged_instance")),
            possible_non_pill=_bool(segmentation.get("possible_non_pill")),
        ),
        image_quality=ImageQualityEvidence(
            status=image_quality.get("status"),
            blur_score=image_quality.get("blur_score"),
            glare_detected=_bool(image_quality.get("glare_detected")),
            lighting_warning=_bool(image_quality.get("lighting_warning")),
        ),
        imprint_visible=_bool(imprint.get("visible"), _bool(imprint_visibility.get("visible"))),
        imprint_visibility_confidence=_float(imprint_visibility.get("confidence"), 1.0),
        imprint_confidence=_float(imprint.get("confidence"), 0.0),
        imprint_candidates=_adapt_imprints(pill),
        shape=LabelEvidence(
            label=normalize_shape(shape.get("label")),
            confidence=_float(shape.get("confidence")),
            alternatives=_list(shape.get("alternatives")),
        )
        if shape
        else None,
        color=ColorEvidence(
            primary=normalize_color(color.get("primary")),
            secondary=normalize_color(color.get("secondary")),
            distribution=color_distribution,
            confidence=_float(color.get("confidence")),
            lighting_warning=_bool(color.get("lighting_warning")),
        )
        if color
        else None,
        dosage_form=LabelEvidence(
            label=normalize_dosage_form(dosage_form.get("label")),
            confidence=_float(dosage_form.get("confidence")),
        )
        if dosage_form
        else None,
        scoreline=ScorelineEvidence(
            label=normalize_scoreline_label(scoreline.get("label"), scoreline.get("visible")),
            visible=scoreline.get("visible") if isinstance(scoreline.get("visible"), bool) else None,
            confidence=_float(scoreline.get("confidence")),
        )
        if scoreline
        else None,
        logo_or_symbol=LogoEvidence(
            visible=logo.get("visible") if isinstance(logo.get("visible"), bool) else None,
            confidence=_float(logo.get("confidence")),
        )
        if logo
        else None,
        quality_flags=[str(item) for item in _list(pill.get("quality_flags"))],
    )

