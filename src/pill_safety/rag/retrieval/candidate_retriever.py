from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from pill_safety.database.models import DrugAppearance, DrugProduct
from pill_safety.rag.retrieval.normalization import (
    normalize_color,
    normalize_dosage_form,
    normalize_imprint,
    normalize_shape,
    normalize_token,
)
from pill_safety.rag.retrieval.similarity import multi_aspect_imprint_similarity, weighted_edit_similarity
from pill_safety.rag.retrieval.types import CandidateRecord, RecognitionInput, RetrievalDiagnostics


class CandidateRetriever:
    def __init__(self, db: Session) -> None:
        self.db = db

    def load_active_candidates(
        self,
        *,
        market: str | None = None,
        min_len: int | None = None,
        max_len: int | None = None,
    ) -> list[CandidateRecord]:
        from sqlalchemy import func
        statement = (
            select(DrugAppearance, DrugProduct)
            .join(DrugProduct, DrugProduct.drug_id == DrugAppearance.drug_id)
            .where(DrugProduct.active.is_(True))
        )
        if market:
            statement = statement.where(DrugProduct.market == normalize_token(market))
        if min_len is not None:
            statement = statement.where(func.length(DrugAppearance.imprint_normalized) >= min_len)
        if max_len is not None:
            statement = statement.where(func.length(DrugAppearance.imprint_normalized) <= max_len)

        rows = self.db.execute(statement).all()
        return [self._to_candidate(appearance, product) for appearance, product in rows]

    def retrieve(
        self,
        pill: RecognitionInput,
        *,
        idf_statistics: Any = None,
        limit: int = 20,
        fuzzy_threshold: float = 0.40,
    ) -> tuple[RetrievalDiagnostics, list[CandidateRecord]]:
        queried_imprints = [candidate.text for candidate in pill.imprint_candidates]

        if queried_imprints:
            all_candidates = self.load_active_candidates(market=pill.market)

            matched: list[tuple[float, CandidateRecord]] = []
            for record in all_candidates:
                best_similarity = max(
                    (
                        multi_aspect_imprint_similarity(
                            imprint,
                            record.imprint_normalized,
                            imprint_raw=record.imprint_raw,
                            imprint_side_a=record.imprint_side_a,
                            imprint_side_b=record.imprint_side_b,
                        )
                        for imprint in queried_imprints
                    ),
                    default=0.0,
                )
                if best_similarity >= fuzzy_threshold:
                    matched.append((best_similarity, record))

            if matched:
                matched.sort(key=lambda item: item[0], reverse=True)
                candidates = self._dedupe([record for _, record in matched])[:limit]
                diagnostics = RetrievalDiagnostics(
                    strategy="imprint_first",
                    queried_imprints=queried_imprints,
                    num_records_before_dedup=len(matched),
                    num_records_after_dedup=len(candidates),
                )
                return diagnostics, candidates

        all_candidates = self.load_active_candidates(market=pill.market)
        fallback = self._fallback_by_attributes(
            pill,
            all_candidates,
            idf_statistics=idf_statistics,
            limit=limit,
        )
        diagnostics = RetrievalDiagnostics(
            strategy="attribute_fallback",
            queried_imprints=queried_imprints,
            num_records_before_dedup=len(fallback),
            num_records_after_dedup=len(fallback),
        )
        return diagnostics, fallback

    def _fallback_by_attributes(
        self,
        pill: RecognitionInput,
        records: list[CandidateRecord],
        *,
        idf_statistics: Any = None,
        limit: int,
    ) -> list[CandidateRecord]:
        scored: list[tuple[float, CandidateRecord]] = []
        for record in records:
            score = 0.0
            if idf_statistics is not None:
                if pill.dosage_form and pill.dosage_form.label == record.dosage_form:
                    score += idf_statistics.get_weight("dosage_form", record.dosage_form)
                if pill.shape and pill.shape.label == record.shape:
                    score += idf_statistics.get_weight("shape", record.shape)
                if pill.color and pill.color.primary == record.primary_color:
                    score += idf_statistics.get_weight("primary_color", record.primary_color)
            else:
                if pill.dosage_form and pill.dosage_form.label == record.dosage_form:
                    score += 2.0
                if pill.shape and pill.shape.label == record.shape:
                    score += 1.0
                if pill.color and pill.color.primary == record.primary_color:
                    score += 1.0

            if score > 0.0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return self._dedupe([record for _, record in scored])[:limit]

    def _dedupe(self, records: list[CandidateRecord]) -> list[CandidateRecord]:
        seen: set[int] = set()
        deduped: list[CandidateRecord] = []
        for record in records:
            if record.appearance_id in seen:
                continue
            seen.add(record.appearance_id)
            deduped.append(record)
        return deduped

    @staticmethod
    def _to_candidate(appearance: DrugAppearance, product: DrugProduct) -> CandidateRecord:
        return CandidateRecord(
            appearance_id=appearance.appearance_id,
            drug_id=product.drug_id,
            product_code=product.product_code,
            product_name=product.name,
            imprint_normalized=normalize_imprint(appearance.imprint_normalized or appearance.imprint),
            shape=normalize_shape(appearance.shape),
            primary_color=normalize_color(appearance.primary_color or appearance.color),
            secondary_color=normalize_color(appearance.secondary_color),
            color_pattern=normalize_color(appearance.color_pattern),
            score_line=bool(appearance.score_line),
            logo_or_symbol=bool(appearance.logo_or_symbol),
            size_mm=appearance.size_mm,
            dosage_form=normalize_dosage_form(product.dosage_form),
            market=normalize_color(product.market),
            source_name=appearance.source_name or product.source_name,
            source_reference=appearance.source_reference or product.source_reference,
            imprint_raw=appearance.imprint_raw,
            imprint_side_a=appearance.imprint_side_a,
            imprint_side_b=appearance.imprint_side_b,
        )
