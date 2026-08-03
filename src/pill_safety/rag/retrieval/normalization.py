from __future__ import annotations

import re


_IMPRINT_RE = re.compile(r"[^A-Z0-9]")
_SPACE_RE = re.compile(r"\s+")


def normalize_imprint(value: str | None) -> str:
    if not value:
        return ""
    return _IMPRINT_RE.sub("", value.upper())


def normalize_token(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _SPACE_RE.sub("_", value.strip().upper())
    return normalized or None


def normalize_shape(value: str | None) -> str | None:
    token = normalize_token(value)
    if token in {"OBLONG", "CAPLET", "ELLIPTICAL"}:
        return "OVAL"
    return token


def normalize_color(value: str | None) -> str | None:
    return normalize_token(value)


def normalize_dosage_form(value: str | None) -> str | None:
    token = normalize_token(value)
    if not token:
        return None
    if "CAPSULE" in token:
        return "CAPSULE"
    if "TABLET" in token or token in {"CAPLET", "PILL"}:
        return "TABLET"
    if "SOFTGEL" in token:
        return "SOFTGEL"
    return token


def normalize_scoreline_label(value: str | None, visible: bool | None = None) -> str | None:
    if visible is False:
        return "NONE"
    token = normalize_token(value)
    if token in {None, "UNKNOWN"}:
        return "UNKNOWN" if visible is None else ("SINGLE" if visible else "NONE")
    if token in {"NO", "NONE", "FALSE"}:
        return "NONE"
    if token in {"YES", "TRUE", "SCORED", "SCORE"}:
        return "SINGLE"
    return token


_CONFUSION_GROUPS = (
    {"0", "O", "Q"},
    {"1", "I", "L"},
    {"5", "S"},
    {"8", "B"},
    {"2", "Z"},
    {"6", "G"},
)

_CONFUSION_MAP: dict[str, set[str]] = {}
for group in _CONFUSION_GROUPS:
    for char in group:
        _CONFUSION_MAP[char] = group - {char}


def expand_imprint_variants(text: str, *, max_variants: int = 5) -> list[tuple[str, float, list[str]]]:
    normalized = normalize_imprint(text)
    if not normalized:
        return []

    variants: dict[str, tuple[float, list[str]]] = {normalized: (1.0, ["raw"])}
    for index, char in enumerate(normalized):
        for replacement in sorted(_CONFUSION_MAP.get(char, set())):
            candidate = normalized[:index] + replacement + normalized[index + 1 :]
            variants.setdefault(candidate, (0.85, [f"{char}->{replacement}"]))

    if max_variants > len(variants):
        one_edit = list(variants.items())
        for variant, (_, evidence) in one_edit:
            for index, char in enumerate(variant):
                for replacement in sorted(_CONFUSION_MAP.get(char, set())):
                    candidate = variant[:index] + replacement + variant[index + 1 :]
                    if candidate not in variants:
                        variants[candidate] = (0.65, [*evidence, f"{char}->{replacement}"])
                    if len(variants) >= max_variants:
                        break
                if len(variants) >= max_variants:
                    break
            if len(variants) >= max_variants:
                break

    ordered = sorted(variants.items(), key=lambda item: (-item[1][0], item[0]))
    return [(variant, penalty, evidence) for variant, (penalty, evidence) in ordered[:max_variants]]

