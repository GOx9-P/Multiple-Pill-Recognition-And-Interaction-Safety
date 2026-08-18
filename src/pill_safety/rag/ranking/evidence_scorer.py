from __future__ import annotations

from pill_safety.rag.retrieval.idf_statistics import IdfStatistics
from pill_safety.rag.retrieval.normalization import normalize_color, normalize_dosage_form, normalize_shape
from pill_safety.rag.retrieval.similarity import weighted_edit_similarity
from pill_safety.rag.retrieval.types import CandidateRecord, CandidateScore, FieldScore, RecognitionInput


class EvidenceScorer:
    dosage_form_reject_threshold = 0.95

    def __init__(self, idf_statistics: IdfStatistics) -> None:
        self.idf_statistics = idf_statistics

    def score(self, pill: RecognitionInput, candidate: CandidateRecord) -> CandidateScore:
        field_scores: dict[str, FieldScore] = {}
        hard_reject_reasons: list[str] = []

        imprint_score = self._score_imprint(pill, candidate)
        if imprint_score:
            field_scores["imprint"] = imprint_score

        shape_score = self._score_shape(pill, candidate)
        if shape_score:
            field_scores["shape"] = shape_score

        color_score = self._score_color(pill, candidate)
        if color_score:
            field_scores["color"] = color_score

        dosage_form_score = self._score_dosage_form(pill, candidate)
        if dosage_form_score:
            field_scores["dosage_form"] = dosage_form_score
            if dosage_form_score.match_score == 0.0 and dosage_form_score.confidence >= self.dosage_form_reject_threshold:
                hard_reject_reasons.append("dosage_form_contradiction")

        scoreline_score = self._score_scoreline(pill, candidate)
        if scoreline_score:
            field_scores["scoreline"] = scoreline_score

        logo_score = self._score_logo(pill, candidate)
        if logo_score:
            field_scores["logo_or_symbol"] = logo_score

        max_possible = sum(item.max_score for item in field_scores.values())
        raw_score = sum(item.evidence_score for item in field_scores.values())
        final_score = raw_score / max_possible if max_possible > 0 else 0.0

        best_imprint_candidate = None
        if imprint_score:
            best_imprint_candidate = str(imprint_score.cv_value)

        return CandidateScore(
            candidate=candidate,
            final_score=max(0.0, min(1.0, final_score)),
            field_scores=field_scores,
            best_imprint_candidate=best_imprint_candidate,
            imprint_match_score=imprint_score.match_score if imprint_score else 0.0,
            hard_reject=bool(hard_reject_reasons),
            hard_reject_reasons=hard_reject_reasons,
        )

    def _score_imprint(self, pill: RecognitionInput, candidate: CandidateRecord) -> FieldScore | None:
        if not pill.imprint_candidates or not candidate.imprint_normalized:
            return None
        best = max(
            (
                (
                    item,
                    item.score * weighted_edit_similarity(item.text, candidate.imprint_normalized),
                )
                for item in pill.imprint_candidates
            ),
            key=lambda item: item[1],
        )
        imprint_candidate, match_score = best
        confidence = pill.imprint_confidence * pill.imprint_visibility_confidence
        idf_weight = self.idf_statistics.get_weight("imprint_normalized", candidate.imprint_normalized, 1.0)
        return self._field_score(
            field="imprint",
            cv_value=imprint_candidate.text,
            db_value=candidate.imprint_normalized,
            match_score=match_score,
            idf_weight=idf_weight,
            confidence=confidence,
            quality_multiplier=1.0,
            explanation="Imprint is the primary retrieval evidence.",
        )

    def _score_shape(self, pill: RecognitionInput, candidate: CandidateRecord) -> FieldScore | None:
        if not pill.shape or not pill.shape.label or not candidate.shape:
            return None
        match = _shape_consistency(pill.shape.label, candidate.shape)
        best_label = pill.shape.label
        best_confidence = pill.shape.confidence
        for alternative in pill.shape.alternatives:
            alt_label = normalize_shape(str(alternative.get("label") or ""))
            alt_confidence = _as_float(alternative.get("confidence"))
            alt_match = _shape_consistency(alt_label, candidate.shape)
            if alt_match * alt_confidence > match * best_confidence:
                match = alt_match
                best_label = alt_label
                best_confidence = alt_confidence
        idf_weight = self.idf_statistics.get_weight("shape", candidate.shape)
        return self._field_score(
            field="shape",
            cv_value=best_label,
            db_value=candidate.shape,
            match_score=match,
            idf_weight=idf_weight,
            confidence=best_confidence,
            quality_multiplier=1.0,
            explanation="Shape reranks candidates after imprint retrieval.",
        )

    def _score_color(self, pill: RecognitionInput, candidate: CandidateRecord) -> FieldScore | None:
        if not pill.color or not candidate.primary_color:
            return None
        distribution = {normalize_color(key): value for key, value in pill.color.distribution.items()}
        primary_match = distribution.get(candidate.primary_color, 0.0)
        if candidate.secondary_color:
            secondary_match = distribution.get(candidate.secondary_color, 0.0)
            match = 0.7 * primary_match + 0.3 * secondary_match
        else:
            match = primary_match
        if not distribution and pill.color.primary:
            match = 1.0 if pill.color.primary == candidate.primary_color else 0.0
        quality_multiplier = _color_quality_multiplier(pill)
        idf_weight = self.idf_statistics.get_weight("primary_color", candidate.primary_color)
        return self._field_score(
            field="color",
            cv_value=pill.color.primary,
            db_value=candidate.primary_color,
            match_score=match,
            idf_weight=idf_weight,
            confidence=pill.color.confidence,
            quality_multiplier=quality_multiplier,
            explanation="Color is soft evidence and is reduced when lighting is unreliable.",
        )

    def _score_dosage_form(self, pill: RecognitionInput, candidate: CandidateRecord) -> FieldScore | None:
        if not pill.dosage_form or not pill.dosage_form.label or not candidate.dosage_form:
            return None
        cv_form = normalize_dosage_form(pill.dosage_form.label)
        db_form = normalize_dosage_form(candidate.dosage_form)
        match = 1.0 if cv_form == db_form else 0.0
        idf_weight = self.idf_statistics.get_weight("dosage_form", db_form)
        return self._field_score(
            field="dosage_form",
            cv_value=cv_form,
            db_value=db_form,
            match_score=match,
            idf_weight=idf_weight,
            confidence=pill.dosage_form.confidence,
            quality_multiplier=1.0,
            explanation="Dosage form can reject clear tablet/capsule contradictions.",
        )

    def _score_scoreline(self, pill: RecognitionInput, candidate: CandidateRecord) -> FieldScore | None:
        if not pill.scoreline or pill.scoreline.visible is None:
            return None
        match = 1.0 if pill.scoreline.visible == candidate.score_line else 0.0
        if pill.scoreline.confidence < 0.5 and match == 0.0:
            match = 0.5
        idf_weight = self.idf_statistics.get_weight("score_line", candidate.score_line)
        return self._field_score(
            field="scoreline",
            cv_value=pill.scoreline.visible,
            db_value=candidate.score_line,
            match_score=match,
            idf_weight=idf_weight,
            confidence=pill.scoreline.confidence,
            quality_multiplier=1.0,
            explanation="Scoreline is auxiliary visual evidence.",
        )

    def _score_logo(self, pill: RecognitionInput, candidate: CandidateRecord) -> FieldScore | None:
        if not pill.logo_or_symbol or pill.logo_or_symbol.visible is None:
            return None
        match = 1.0 if pill.logo_or_symbol.visible == candidate.logo_or_symbol else 0.0
        if pill.logo_or_symbol.confidence < 0.5 and match == 0.0:
            match = 0.5
        idf_weight = self.idf_statistics.get_weight("logo_or_symbol", candidate.logo_or_symbol)
        return self._field_score(
            field="logo_or_symbol",
            cv_value=pill.logo_or_symbol.visible,
            db_value=candidate.logo_or_symbol,
            match_score=match,
            idf_weight=idf_weight,
            confidence=pill.logo_or_symbol.confidence,
            quality_multiplier=1.0,
            explanation="Logo boolean is weak auxiliary evidence.",
        )

    @staticmethod
    def _field_score(
        *,
        field: str,
        cv_value: object,
        db_value: object,
        match_score: float,
        idf_weight: float,
        confidence: float,
        quality_multiplier: float,
        explanation: str,
    ) -> FieldScore:
        bounded_match = max(0.0, min(1.0, match_score))
        bounded_confidence = max(0.0, min(1.0, confidence))
        bounded_quality = max(0.0, min(1.0, quality_multiplier))
        max_score = idf_weight * bounded_confidence * bounded_quality
        evidence_score = max_score * bounded_match
        return FieldScore(
            field=field,
            cv_value=cv_value,
            db_value=db_value,
            match_score=bounded_match,
            idf_weight=idf_weight,
            confidence=bounded_confidence,
            quality_multiplier=bounded_quality,
            evidence_score=evidence_score,
            max_score=max_score,
            explanation=explanation,
        )


def _shape_consistency(cv_shape: str | None, db_shape: str | None) -> float:
    cv_normalized = normalize_shape(cv_shape)
    db_normalized = normalize_shape(db_shape)
    if not cv_normalized or not db_normalized:
        return 0.0
    if cv_normalized == db_normalized:
        return 1.0
    similar_pairs = {frozenset(("OVAL", "OBLONG")), frozenset(("CAPSULE", "OVAL"))}
    if frozenset((cv_normalized, db_normalized)) in similar_pairs:
        return 0.7
    return 0.0


def _color_quality_multiplier(pill: RecognitionInput) -> float:
    multiplier = 1.0
    if pill.color and pill.color.lighting_warning:
        multiplier *= 0.5
    if pill.image_quality.lighting_warning:
        multiplier *= 0.5
    if pill.image_quality.glare_detected:
        multiplier *= 0.7
    if "minor_glare" in pill.quality_flags:
        multiplier *= 0.85
    return multiplier


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default
