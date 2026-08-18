from __future__ import annotations

from pill_safety.rag.retrieval.normalization import normalize_imprint


_CONFUSABLE_PAIRS = {
    frozenset(("0", "O")),
    frozenset(("0", "Q")),
    frozenset(("O", "Q")),
    frozenset(("1", "I")),
    frozenset(("1", "L")),
    frozenset(("I", "L")),
    frozenset(("5", "S")),
    frozenset(("8", "B")),
    frozenset(("2", "Z")),
    frozenset(("6", "G")),
}


def weighted_edit_similarity(left: str | None, right: str | None) -> float:
    a = normalize_imprint(left)
    b = normalize_imprint(right)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    rows = len(a) + 1
    cols = len(b) + 1
    dp = [[0.0] * cols for _ in range(rows)]

    for i in range(1, rows):
        dp[i][0] = dp[i - 1][0] + 0.9
    for j in range(1, cols):
        dp[0][j] = dp[0][j - 1] + 0.9

    for i, char_a in enumerate(a, start=1):
        for j, char_b in enumerate(b, start=1):
            if char_a == char_b:
                substitution = 0.0
            elif frozenset((char_a, char_b)) in _CONFUSABLE_PAIRS:
                substitution = 0.25
            else:
                substitution = 1.0

            dp[i][j] = min(
                dp[i - 1][j] + 0.9,
                dp[i][j - 1] + 0.9,
                dp[i - 1][j - 1] + substitution,
            )

    distance = dp[-1][-1]
    similarity = 1.0 - distance / max(len(a), len(b))
    return max(0.0, min(1.0, similarity))

