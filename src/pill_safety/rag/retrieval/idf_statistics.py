from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from sqlalchemy.orm import Session

from pill_safety.rag.retrieval.candidate_retriever import CandidateRetriever
from pill_safety.rag.retrieval.types import CandidateRecord


@dataclass(frozen=True)
class IdfStatistics:
    weights: dict[str, dict[Any, float]]
    version: str

    def get_weight(self, field: str, value: Any, default: float = 0.2) -> float:
        if value is None:
            return default
        return self.weights.get(field, {}).get(value, default)

_cached_statistics: IdfStatistics | None = None


class IdfStatisticsBuilder:
    fields = (
        "imprint_normalized",
        "shape",
        "primary_color",
        "secondary_color",
        "color_pattern",
        "score_line",
        "logo_or_symbol",
        "dosage_form",
        "market",
    )

    @classmethod
    def from_database(cls, db: Session) -> IdfStatistics:
        records = CandidateRetriever(db).load_active_candidates()
        return cls.from_records(records)

    @classmethod
    def get_cached_statistics(cls, db: Session) -> IdfStatistics:
        global _cached_statistics
        if _cached_statistics is None:
            _cached_statistics = cls.from_database(db)
        return _cached_statistics

    @classmethod
    def invalidate_cache(cls) -> None:
        global _cached_statistics
        _cached_statistics = None

    @classmethod
    def from_records(cls, records: Iterable[CandidateRecord]) -> IdfStatistics:
        rows = list(records)
        n = len(rows)
        weights: dict[str, dict[Any, float]] = {}

        for field in cls.fields:
            values = [getattr(row, field) for row in rows if getattr(row, field) is not None]
            counter = Counter(values)
            if not counter or n == 0:
                weights[field] = {}
                continue

            raw_idf = {
                value: math.log((n + 1) / (count + 1)) + 1
                for value, count in counter.items()
            }
            min_idf = min(raw_idf.values())
            max_idf = max(raw_idf.values())
            field_weights: dict[Any, float] = {}
            for value, idf in raw_idf.items():
                if max_idf == min_idf:
                    field_weights[value] = 0.2
                else:
                    normalized = (idf - min_idf) / (max_idf - min_idf)
                    field_weights[value] = 0.2 + 0.8 * normalized
            weights[field] = field_weights

        return IdfStatistics(weights=weights, version=f"appearance_count_{n}")

