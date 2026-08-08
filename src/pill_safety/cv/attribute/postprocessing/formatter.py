def format_attribute_predictions(shape_label: str, shape_conf: float, color_labels: list, color_probs: dict) -> dict:
    return {
        "shape": {
            "class": shape_label,
            "confidence": round(shape_conf, 4)
        },
        "color": {
            "classes": color_labels,
            "raw_probabilities": {k: round(v, 4) for k, v in color_probs.items()}
        }
    }