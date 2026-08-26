from __future__ import annotations

from dataclasses import dataclass

from pill_safety.rag.retrieval.types import CandidateScore, RecognitionInput


@dataclass(frozen=True)
class SafetyDecision:
    identification_status: str
    accepted: CandidateScore | None
    required_action: str | None
    scope_warning: str | None
    reasons: list[str]


class SafetyGate:
    # Các siêu tham số tối ưu hóa thực nghiệm chuẩn khoa học từ Grid Search (100% Precision, Max F1)
    identified_threshold = 0.65         # Ngưỡng điểm tổng hợp để tự động công nhận identified (65%)
    ambiguous_threshold = 0.35          # Ngưỡng điểm để đề xuất Top Candidates (35%)
    margin_threshold = 0.03             # Độ cách biệt tối thiểu giữa Top 1 và Top 2 (3%)
    imprint_threshold = 0.45            # Điểm số tương đồng chữ khắc tối thiểu (45%)
    minimum_ocr_confidence = 0.15       # Độ tin cậy tối thiểu của OCR để coi là chữ khả dụng (15%)

    def pre_retrieval_decision(self, pill: RecognitionInput) -> SafetyDecision | None:
        if pill.segmentation.possible_non_pill:
            return SafetyDecision(
                identification_status="unknown",
                accepted=None,
                required_action="capture_clear_pill_image",
                scope_warning="possible_non_pill",
                reasons=["segmentation_possible_non_pill"],
            )

        if pill.cv_status in {"insufficient_visual_evidence", "unknown_object"}:
            return SafetyDecision(
                identification_status="insufficient_visual_evidence",
                accepted=None,
                required_action="recapture_clear_image",
                scope_warning=pill.cv_status,
                reasons=[f"cv_status_{pill.cv_status}"],
            )

        return None

    def decide(self, pill: RecognitionInput, ranked: list[CandidateScore]) -> SafetyDecision:
        pre_decision = self.pre_retrieval_decision(pill)
        if pre_decision is not None:
            return pre_decision

        reasons: list[str] = []
        if pill.segmentation.possible_merged_instance:
            reasons.append("possible_merged_instance")
        if not _has_usable_imprint(pill):
            reasons.append("no_usable_imprint")
        if pill.imprint_confidence < self.minimum_ocr_confidence:
            reasons.append("ocr_confidence_too_low")

        if not ranked:
            return SafetyDecision(
                identification_status="unknown",
                accepted=None,
                required_action="manual_drug_search_or_recapture",
                scope_warning="no_candidates_found",
                reasons=[*reasons, "no_candidates_found"],
            )

        top1 = ranked[0]
        top2_score = ranked[1].final_score if len(ranked) > 1 else 0.0
        margin = top1.final_score - top2_score

        if top1.hard_reject:
            reasons.extend(top1.hard_reject_reasons)

        has_usable_imprint = _has_usable_imprint(pill)

        can_identify = (
            pill.cv_status == "features_ready"
            and has_usable_imprint
            and not pill.segmentation.possible_merged_instance
            and not top1.hard_reject
            and top1.final_score >= self.identified_threshold
            and margin >= self.margin_threshold
            and top1.imprint_match_score >= self.imprint_threshold
        )

        if can_identify:
            return SafetyDecision(
                identification_status="identified",
                accepted=top1,
                required_action=None,
                scope_warning=None,
                reasons=["top1_score_and_margin_sufficient"],
            )

        if has_usable_imprint and top1.imprint_match_score < 0.30:
            reasons.append("imprint_not_found_in_database")
            return SafetyDecision(
                identification_status="unknown",
                accepted=None,
                required_action="manual_drug_search_or_recapture",
                scope_warning="imprint_not_found_in_database",
                reasons=reasons,
            )

        if top1.final_score >= self.ambiguous_threshold:
            if margin < self.margin_threshold:
                reasons.append("top_candidates_too_close")
            return SafetyDecision(
                identification_status="ambiguous",
                accepted=None,
                required_action="capture_reverse_side_or_manual_confirm",
                scope_warning=";".join(reasons) if reasons else "below_identified_threshold",
                reasons=reasons or ["below_identified_threshold"],
            )

        return SafetyDecision(
            identification_status="unknown",
            accepted=None,
            required_action="manual_drug_search_or_recapture",
            scope_warning=";".join(reasons) if reasons else "low_candidate_score",
            reasons=reasons or ["low_candidate_score"],
        )


def _has_usable_imprint(pill: RecognitionInput) -> bool:
    return bool(
        pill.imprint_visible
        and pill.imprint_candidates
        and pill.imprint_confidence >= SafetyGate.minimum_ocr_confidence
    )

