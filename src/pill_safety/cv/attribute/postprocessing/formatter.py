from __future__ import annotations


def _normalize_public_label(label: str) -> str:
    """Chuẩn hóa nhãn train như ``color_WHITE`` về nhãn public dễ dùng cho RAG."""

    normalized = " ".join(str(label).strip().lower().replace("_", " ").split())
    if normalized.startswith("color "):
        normalized = normalized.removeprefix("color ")
    if not normalized or normalized.startswith("unknown"):
        return "unknown"
    return normalized


def _normalized_distribution(color_probs: dict[str, float]) -> dict[str, float]:
    """Gộp score nếu nhiều nhãn train cùng ánh xạ về một nhãn public."""

    distribution: dict[str, float] = {}
    for raw_label, probability in color_probs.items():
        label = _normalize_public_label(raw_label)
        distribution[label] = max(distribution.get(label, 0.0), float(probability))
    return distribution


def format_attribute_predictions(
    shape_label: str,
    shape_conf: float,
    color_labels: list[str],
    color_probs: dict[str, float],
    shape_alternatives: list[tuple[str, float]] | None = None,
    lighting_warning: bool = False,
) -> dict:
    """Chuẩn hóa shape/color và đánh dấu rõ các thuộc tính chưa được train."""

    distribution = _normalized_distribution(color_probs)
    selected_labels = {
        _normalize_public_label(label) for label in color_labels
    }
    ranked_colors = sorted(
        distribution.items(), key=lambda item: item[1], reverse=True
    )
    selected_colors = [
        (label, probability)
        for label, probability in ranked_colors
        if label in selected_labels
    ]
    primary = selected_colors[0][0] if selected_colors else "unknown"
    secondary = selected_colors[1][0] if len(selected_colors) > 1 else None
    confidence = selected_colors[0][1] if selected_colors else 0.0
    alternatives = []
    seen_labels = {_normalize_public_label(shape_label)}
    for raw_label, probability in shape_alternatives or []:
        label = _normalize_public_label(raw_label)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        alternatives.append(
            {"label": label, "confidence": round(float(probability), 4)}
        )

    return {
        "shape": {
            "label": _normalize_public_label(shape_label),
            "confidence": round(shape_conf, 4),
            "alternatives": alternatives,
        },
        "color": {
            "primary": primary,
            "secondary": secondary,
            "distribution": {
                label: round(probability, 4)
                for label, probability in ranked_colors
            },
            "confidence": round(confidence, 4),
            "lighting_warning": lighting_warning,
        },
        "dosage_form": {
            "label": "unknown",
            "confidence": None,
            "source": "not_predicted_by_attribute",
        },
        "scoreline": {
            "label": "unknown",
            "visible": None,
            "confidence": None,
            "source": "not_predicted_by_attribute",
        },
        "logo_or_symbol": {
            "visible": None,
            "confidence": None,
            "source": "not_predicted_by_attribute",
        },
        "damage_or_occlusion": {
            "visible": None,
            "confidence": None,
            "source": "not_predicted_by_attribute",
        },
    }
