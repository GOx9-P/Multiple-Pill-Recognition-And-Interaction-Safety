#!/usr/bin/env python3
"""Script tối ưu hóa siêu tham số (Grid Search / Parameter Tuning) cho RAG Identification & Safety Gate."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sqlalchemy import create_engine
from sqlalchemy.orm import Session


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pill_safety.database.base import Base
from pill_safety.database.scripts.seed import seed_database
from pill_safety.rag.identification_service import IdentificationService
from pill_safety.rag.ranking.safety_gate import SafetyGate
from pill_safety.rag.retrieval.candidate_retriever import CandidateRetriever


def setup_in_memory_db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    seed_database(session)
    return session


def evaluate_parameter_set(
    benchmark_data: list[dict[str, Any]],
    db: Session,
    fuzzy_threshold: float,
    identified_threshold: float,
    margin_threshold: float,
    imprint_threshold: float,
    minimum_ocr_confidence: float,
) -> dict[str, Any]:
    """Đánh giá 1 bộ siêu tham số trên tập dữ liệu CV Output đã cache."""
    service = IdentificationService(db)
    # Tạm thời gán các ngưỡng vào SafetyGate
    SafetyGate.identified_threshold = identified_threshold
    SafetyGate.margin_threshold = margin_threshold
    SafetyGate.imprint_threshold = imprint_threshold
    SafetyGate.minimum_ocr_confidence = minimum_ocr_confidence

    total_samples = len(benchmark_data)
    top1_correct = 0
    top3_correct = 0
    identified_count = 0
    identified_correct = 0
    ambiguous_count = 0
    unknown_count = 0

    for item in benchmark_data:
        product_code = item.get("product_code")
        cv_dict = item.get("cv_output")
        if not cv_dict or not product_code:
            continue

        rag_req = {
            "schema_version": "rag_request_v1",
            "request_id": item.get("request_id", "bench"),
            "session_id": "sess_tune",
            "market": "US",
            "cv_output": cv_dict,
        }

        # Override fuzzy_threshold trong CandidateRetriever
        original_retrieve = service.retriever.retrieve
        def custom_retrieve(pill, **kwargs):
            kwargs["fuzzy_threshold"] = fuzzy_threshold
            return original_retrieve(pill, **kwargs)
        service.retriever.retrieve = custom_retrieve

        rag_resp = service.identify(rag_req)
        service.retriever.retrieve = original_retrieve

        pill_results = rag_resp.get("pill_results", [])
        if not pill_results:
            unknown_count += 1
            continue

        p_res = pill_results[0]
        status = p_res.get("identification_status")
        top_candidates = p_res.get("top_candidates", [])
        cand_codes = [c.get("product_code") for c in top_candidates]

        if cand_codes and cand_codes[0] == product_code:
            top1_correct += 1
        if product_code in cand_codes[:3]:
            top3_correct += 1

        if status == "identified":
            identified_count += 1
            if cand_codes and cand_codes[0] == product_code:
                identified_correct += 1
        elif status == "ambiguous":
            ambiguous_count += 1
        else:
            unknown_count += 1

    top1_acc = (top1_correct / total_samples * 100) if total_samples else 0.0
    top3_acc = (top3_correct / total_samples * 100) if total_samples else 0.0
    precision = (identified_correct / identified_count * 100) if identified_count else 0.0
    recall = (identified_correct / total_samples * 100) if total_samples else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "params": {
            "fuzzy_threshold": fuzzy_threshold,
            "identified_threshold": identified_threshold,
            "margin_threshold": margin_threshold,
            "imprint_threshold": imprint_threshold,
            "minimum_ocr_confidence": minimum_ocr_confidence,
        },
        "metrics": {
            "top1_accuracy": round(top1_acc, 2),
            "top3_accuracy": round(top3_acc, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1, 2),
            "identified_count": identified_count,
            "identified_correct": identified_correct,
            "ambiguous_count": ambiguous_count,
            "unknown_count": unknown_count,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Tune identification parameters using grid search.")
    parser.add_argument("--benchmark-file", type=Path, default=PROJECT_ROOT / "outputs" / "benchmark_verified_results.json")
    args = parser.parse_args()

    if not args.benchmark_file.exists():
        print(f"Không tìm thấy file benchmark: {args.benchmark_file}")
        print("Vui lòng chạy 'scripts/benchmark_verified_pills.py' trước để trích xuất dữ liệu.")
        sys.exit(1)

    data = json.loads(args.benchmark_file.read_text(encoding="utf-8"))
    detailed = data.get("detailed_results", [])
    print(f"Nạp {len(detailed)} mẫu đánh giá từ benchmark...")

    db = setup_in_memory_db()

    # Dải siêu tham số tìm kiếm
    fuzzy_grid = [0.40, 0.45, 0.50, 0.55]
    identified_grid = [0.75, 0.80, 0.85]
    margin_grid = [0.05, 0.08, 0.10]
    imprint_grid = [0.60, 0.65, 0.70]
    ocr_conf_grid = [0.30, 0.40]

    all_combos = list(itertools.product(fuzzy_grid, identified_grid, margin_grid, imprint_grid, ocr_conf_grid))
    print(f"Bắt đầu Grid Search trên {len(all_combos)} tổ hợp tham số...")

    best_score = -1.0
    best_config = None
    all_results = []

    for idx, (f_th, id_th, m_th, imp_th, ocr_th) in enumerate(all_combos, 1):
        res = evaluate_parameter_set(
            benchmark_data=detailed,
            db=db,
            fuzzy_threshold=f_th,
            identified_threshold=id_th,
            margin_threshold=m_th,
            imprint_threshold=imp_th,
            minimum_ocr_confidence=ocr_th,
        )
        f1 = res["metrics"]["f1_score"]
        prec = res["metrics"]["precision"]
        all_results.append(res)

        # Ưu tiên cấu hình có Precision cao (>= 90%) và F1 score lớn nhất
        score = f1 if prec >= 85.0 else f1 * 0.5
        if score > best_score:
            best_score = score
            best_config = res

    print("\n" + "=" * 80)
    print("KẾT QUẢ GRID SEARCH TỐI ƯU SIÊU THAM SỐ")
    print("=" * 80)
    print("CẤU HÌNH TỐI ƯU NHẤT (BEST CONFIGURATION):")
    print(json.dumps(best_config, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
