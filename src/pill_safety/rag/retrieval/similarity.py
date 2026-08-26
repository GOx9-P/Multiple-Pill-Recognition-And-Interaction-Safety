from __future__ import annotations

import re
from pill_safety.rag.retrieval.normalization import normalize_imprint


_CONFUSABLE_PAIRS = {
    frozenset(("0", "O")),
    frozenset(("0", "Q")),
    frozenset(("O", "Q")),
    frozenset(("0", "D")),
    frozenset(("1", "I")),
    frozenset(("1", "L")),
    frozenset(("1", "T")),
    frozenset(("I", "L")),
    frozenset(("5", "S")),
    frozenset(("8", "B")),
    frozenset(("2", "Z")),
    frozenset(("6", "G")),
    frozenset(("8", "0")),
    frozenset(("V", "U")),
    frozenset(("M", "W")),
}


def weighted_edit_similarity(left: str | None, right: str | None) -> float:
    a = normalize_imprint(left)
    b = normalize_imprint(right)
    if a == b and a:
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
                substitution = 0.20
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


def multi_aspect_imprint_similarity(
    query: str | None,
    imprint_normalized: str | None,
    imprint_raw: str | None = None,
    imprint_side_a: str | None = None,
    imprint_side_b: str | None = None,
) -> float:
    """So khớp đa khía cạnh bao gồm cả toàn bộ chuỗi lẫn từng mặt của viên thuốc."""
    q_norm = normalize_imprint(query)
    if not q_norm:
        return 0.0

    target = normalize_imprint(imprint_normalized)
    best_sim = weighted_edit_similarity(q_norm, target)
    if best_sim >= 0.99:
        return 1.0

    # 1. Cơ chế so khớp 1 mặt: CHỈ áp dụng nếu thuốc trong DB có CẢ imprint_side_a và imprint_side_b khác null / không rỗng
    side_a_clean = normalize_imprint(imprint_side_a)
    side_b_clean = normalize_imprint(imprint_side_b)
    if bool(imprint_side_a and imprint_side_b and side_a_clean and side_b_clean):
        sim_a = weighted_edit_similarity(q_norm, side_a_clean)
        sim_b = weighted_edit_similarity(q_norm, side_b_clean)
        best_side_sim = max(sim_a, sim_b)
        if best_side_sim >= 0.99:
            best_sim = max(best_sim, 1.0)
        elif best_side_sim >= 0.70:
            best_sim = max(best_sim, best_side_sim * 0.95)

    # 2. Kiểm tra so khớp với các token phân tách trong imprint_raw (ví dụ "TV;5056", "44;438", "8335;BARR", "13/T")
    if imprint_raw:
        tokens = [normalize_imprint(t) for t in re.split(r"[;\s/\\|,-]+", str(imprint_raw)) if t.strip()]
        for tok in tokens:
            if tok and len(tok) >= 2:
                tok_sim = weighted_edit_similarity(q_norm, tok)
                if tok_sim >= 0.99:
                    best_sim = max(best_sim, 0.95)
                elif tok_sim >= 0.70:
                    best_sim = max(best_sim, tok_sim * 0.88)

    # 3. Kiểm tra khớp prefix/suffix khi chuỗi con đủ dài (tối thiểu 2 ký tự)
    if target and len(q_norm) >= 2 and len(target) > len(q_norm):
        if target.startswith(q_norm) or target.endswith(q_norm):
            sub_score = 0.82 + 0.13 * (len(q_norm) / len(target))
            best_sim = max(best_sim, sub_score)

    return max(0.0, min(1.0, best_sim))
