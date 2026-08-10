from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from pill_safety.rag.ranking.evidence_scorer import EvidenceScorer
from pill_safety.rag.ranking.safety_gate import SafetyGate
from pill_safety.rag.retrieval.candidate_retriever import CandidateRetriever
from pill_safety.rag.retrieval.cv_input_adapter import adapt_cv_pill_to_recognition_input
from pill_safety.rag.retrieval.idf_statistics import IdfStatisticsBuilder
from pill_safety.rag.retrieval.types import CandidateScore, FieldScore, RecognitionInput, RetrievalDiagnostics


class IdentificationService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.retriever = CandidateRetriever(db)
        self.safety_gate = SafetyGate()

    def identify(self, rag_request: dict[str, Any]) -> dict[str, Any]:
        cv_output = _dict(rag_request.get("cv_output"))
        idf_statistics = IdfStatisticsBuilder.get_cached_statistics(self.db)
        scorer = EvidenceScorer(idf_statistics)

        pill_results = []
        for pill in _list(cv_output.get("pills")):
            if not isinstance(pill, dict):
                continue
            normalized = adapt_cv_pill_to_recognition_input(
                rag_request=rag_request,
                cv_output=cv_output,
                pill=pill,
            )
            pill_results.append(
                self._identify_one(
                    normalized=normalized,
                    scorer=scorer,
                    idf_statistics_version=idf_statistics.version,
                )
            )

        return {
            "schema_version": "rag_identification_v0",
            "request_id": rag_request.get("request_id") or cv_output.get("request_id"),
            "session_id": rag_request.get("session_id") or cv_output.get("session_id"),
            "pill_results": pill_results,
        }

    def manual_bind_unresolved_pill(
        self,
        *,
        instance_id: str,
        manual_drug_name: str | None = None,
        product_id: str | None = None,
    ) -> dict[str, Any]:
        from pill_safety.database.services.drug_service import DrugService
        drug_service = DrugService(self.db)
        
        drug = None
        if product_id:
            from pill_safety.rag.ddi.ddi_lookup_service import DdiLookupService
            drug_id = DdiLookupService._parse_drug_id(product_id)
            if drug_id is not None:
                drug = drug_service.get_drug(drug_id)
            if drug is None:
                drug = drug_service.repository.get_by_product_code(product_id)

        if drug is None and manual_drug_name:
            results = drug_service.search(imprint=manual_drug_name)
            if not results:
                # Search by drug name in DB
                from sqlalchemy import select
                from pill_safety.database.models import DrugProduct
                stmt = select(DrugProduct).where(DrugProduct.name.ilike(f"%{manual_drug_name}%"))
                drug_obj = self.db.scalars(stmt).first()
                if drug_obj:
                    drug = drug_service._detail(drug_obj)
            elif results:
                drug = results[0]

        if drug is None:
            raise ValueError(f"Không tìm thấy sản phẩm thuốc phù hợp cho '{manual_drug_name or product_id}'.")

        drug_id_val = drug.get("drug_id") if isinstance(drug, dict) else drug.drug_id
        product_code_val = drug.get("product_code") if isinstance(drug, dict) else drug.product_code
        name_val = drug.get("name") if isinstance(drug, dict) else drug.name

        return {
            "instance_id": instance_id,
            "identification_status": "identified",
            "required_action": "none",
            "accepted_product": {
                "drug_id": drug_id_val,
                "product_code": product_code_val,
                "product_name": name_val,
            },
            "decision_reasons": ["manual_user_override"],
        }


    def _identify_one(
        self,
        *,
        normalized: RecognitionInput,
        scorer: EvidenceScorer,
        idf_statistics_version: str,
    ) -> dict[str, Any]:
        pre_decision = self.safety_gate.pre_retrieval_decision(normalized)
        if pre_decision is not None:
            return {
                "instance_id": normalized.instance_id,
                "instance_token": normalized.instance_token,
                "identification_status": pre_decision.identification_status,
                "required_action": pre_decision.required_action,
                "candidate_generation": {
                    "strategy": "not_queried",
                    "queried_imprints": [],
                    "num_records_before_dedup": 0,
                    "num_records_after_dedup": 0,
                },
                "ranking_method": "idf_weighted_evidence_v1",
                "idf_statistics_version": idf_statistics_version,
                "top_candidates": [],
                "accepted_product": None,
                "scope_warning": pre_decision.scope_warning,
                "decision_reasons": pre_decision.reasons,
            }

        diagnostics, candidates = self.retriever.retrieve(normalized, idf_statistics=scorer.idf_statistics)
        all_ranked = sorted(
            (scorer.score(normalized, candidate) for candidate in candidates),
            key=lambda item: item.final_score,
            reverse=True,
        )
        seen_drugs: set[int] = set()
        ranked: list[CandidateScore] = []
        for score_item in all_ranked:
            if score_item.candidate.drug_id not in seen_drugs:
                seen_drugs.add(score_item.candidate.drug_id)
                ranked.append(score_item)

        decision = self.safety_gate.decide(normalized, ranked)
        top_candidates = [
            self._serialize_candidate_score(score, rank=index + 1, top2_margin=_top2_margin(ranked, index))
            for index, score in enumerate(ranked[:5])
        ]

        accepted_product = None
        if decision.accepted is not None:
            accepted_product = {
                "drug_id": decision.accepted.candidate.drug_id,
                "product_code": decision.accepted.candidate.product_code,
                "product_name": decision.accepted.candidate.product_name,
            }

        return {
            "instance_id": normalized.instance_id,
            "instance_token": normalized.instance_token,
            "identification_status": decision.identification_status,
            "required_action": decision.required_action,
            "candidate_generation": self._serialize_diagnostics(diagnostics),
            "ranking_method": "idf_weighted_evidence_v1",
            "idf_statistics_version": idf_statistics_version,
            "top_candidates": top_candidates,
            "accepted_product": accepted_product,
            "scope_warning": decision.scope_warning,
            "decision_reasons": decision.reasons,
        }

    @staticmethod
    def _serialize_diagnostics(diagnostics: RetrievalDiagnostics) -> dict[str, Any]:
        return {
            "strategy": diagnostics.strategy,
            "queried_imprints": diagnostics.queried_imprints,
            "num_records_before_dedup": diagnostics.num_records_before_dedup,
            "num_records_after_dedup": diagnostics.num_records_after_dedup,
        }

    @staticmethod
    def _serialize_candidate_score(score: CandidateScore, *, rank: int, top2_margin: float | None) -> dict[str, Any]:
        candidate = score.candidate
        return {
            "rank": rank,
            "appearance_id": candidate.appearance_id,
            "drug_id": candidate.drug_id,
            "product_code": candidate.product_code,
            "product_name": candidate.product_name,
            "final_score": round(score.final_score, 4),
            "evidence": {
                "best_imprint_candidate": score.best_imprint_candidate,
                "imprint_match_score": round(score.imprint_match_score, 4),
                "shape_score": _rounded_field_score(score.field_scores.get("shape")),
                "color_score": _rounded_field_score(score.field_scores.get("color")),
                "dosage_form_score": _rounded_field_score(score.field_scores.get("dosage_form")),
                "scoreline_score": _rounded_field_score(score.field_scores.get("scoreline")),
                "logo_score": _rounded_field_score(score.field_scores.get("logo_or_symbol")),
                "top1_top2_margin": round(top2_margin, 4) if top2_margin is not None else None,
                "hard_reject": score.hard_reject,
                "hard_reject_reasons": score.hard_reject_reasons,
                "fields": {
                    field: _serialize_field_score(field_score)
                    for field, field_score in score.field_scores.items()
                },
            },
        }


def _serialize_field_score(score: FieldScore) -> dict[str, Any]:
    return {
        "cv_value": score.cv_value,
        "db_value": score.db_value,
        "match_score": round(score.match_score, 4),
        "idf_weight": round(score.idf_weight, 4),
        "confidence": round(score.confidence, 4),
        "quality_multiplier": round(score.quality_multiplier, 4),
        "evidence_score": round(score.evidence_score, 4),
        "max_score": round(score.max_score, 4),
        "explanation": score.explanation,
    }


def _rounded_field_score(score: FieldScore | None) -> float | None:
    if score is None:
        return None
    return round(score.evidence_score, 4)


def _top2_margin(ranked: list[CandidateScore], index: int) -> float | None:
    if index != 0:
        return None
    if len(ranked) < 2:
        return ranked[0].final_score if ranked else None
    return ranked[0].final_score - ranked[1].final_score


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
