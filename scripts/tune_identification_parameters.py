#!/usr/bin/env python3
"""Script tối ưu hóa siêu tham số toàn diện (Grid Search & Confidence Calibration) cho RAG & Safety Gate."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
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
    ambiguous_threshold: float = 0.40,
) -> dict[str, Any]:
    """Đánh giá 1 bộ siêu tham số trên tập dữ liệu CV Output đã cache."""
    service = IdentificationService(db)
    
    # Gán các ngưỡng vào SafetyGate
    SafetyGate.identified_threshold = identified_threshold
    SafetyGate.margin_threshold = margin_threshold
    SafetyGate.imprint_threshold = imprint_threshold
    SafetyGate.minimum_ocr_confidence = minimum_ocr_confidence
    SafetyGate.ambiguous_threshold = ambiguous_threshold

    total_samples = len(benchmark_data)
    top1_correct = 0
    top3_correct = 0
    identified_count = 0
    identified_correct = 0
    identified_fp = 0
    ambiguous_count = 0
    ambiguous_with_top1_correct = 0
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

        is_top1 = (cand_codes and cand_codes[0] == product_code)
        if is_top1:
            top1_correct += 1
        if product_code in cand_codes[:3]:
            top3_correct += 1

        if status == "identified":
            identified_count += 1
            if is_top1:
                identified_correct += 1
            else:
                identified_fp += 1
        elif status == "ambiguous":
            ambiguous_count += 1
            if is_top1:
                ambiguous_with_top1_correct += 1
        else:
            unknown_count += 1

    top1_acc = (top1_correct / total_samples * 100) if total_samples else 0.0
    top3_acc = (top3_correct / total_samples * 100) if total_samples else 0.0
    precision = (identified_correct / identified_count * 100) if identified_count else 0.0
    recall = (identified_correct / total_samples * 100) if total_samples else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    user_friction = (ambiguous_with_top1_correct / total_samples * 100) if total_samples else 0.0

    return {
        "params": {
            "fuzzy_threshold": fuzzy_threshold,
            "identified_threshold": identified_threshold,
            "margin_threshold": margin_threshold,
            "imprint_threshold": imprint_threshold,
            "minimum_ocr_confidence": minimum_ocr_confidence,
            "ambiguous_threshold": ambiguous_threshold,
        },
        "metrics": {
            "top1_accuracy": round(top1_acc, 2),
            "top3_accuracy": round(top3_acc, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1, 2),
            "identified_count": identified_count,
            "identified_correct": identified_correct,
            "identified_false_positive": identified_fp,
            "ambiguous_count": ambiguous_count,
            "ambiguous_with_top1_correct": ambiguous_with_top1_correct,
            "unknown_count": unknown_count,
            "user_friction_pct": round(user_friction, 2),
        }
    }


def evaluate_scene_images(
    db: Session,
    params: dict[str, float],
    scene_dir: Path,
) -> list[dict[str, Any]]:
    """Đánh giá trực tiếp trên test_1.png và test_2.png."""
    from ui.model_loader import load_cv_pipeline
    from ui.adapters.pipeline_adapter import evaluate_safety_and_report, parse_cv_output

    # Áp dụng tham số
    SafetyGate.identified_threshold = params["identified_threshold"]
    SafetyGate.margin_threshold = params["margin_threshold"]
    SafetyGate.imprint_threshold = params["imprint_threshold"]
    SafetyGate.minimum_ocr_confidence = params["minimum_ocr_confidence"]
    SafetyGate.ambiguous_threshold = params.get("ambiguous_threshold", 0.40)

    cv_loader = load_cv_pipeline()
    if not cv_loader.available or cv_loader.pipeline is None:
        print("[WARNING] Không thể load CV Pipeline để chạy scene test.")
        return []

    cv_pipeline = cv_loader.pipeline
    id_service = IdentificationService(db)

    scene_files = [
        ("test_1", scene_dir / "test_1.png"),
        ("test_2", scene_dir / "test_2.png"),
    ]

    scene_results = []
    for scene_name, img_path in scene_files:
        if not img_path.exists():
            continue

        print(f"\n---> Chạy Scene Test trên: {scene_name} ({img_path.name})...")
        t0 = time.time()
        cv_out = cv_pipeline.predict({
            "request_id": f"scene_{scene_name}",
            "session_id": "sess_scene",
            "image_id": f"img_{scene_name}",
            "image_path": str(img_path),
        })
        cv_dict = cv_out.model_dump(mode="json")
        pills_cv = cv_dict.get("pills", [])

        # RAG identification
        rag_req = {
            "schema_version": "rag_request_v1",
            "request_id": f"scene_{scene_name}",
            "session_id": "sess_scene",
            "market": "US",
            "cv_output": cv_dict,
        }
        rag_resp = id_service.identify(rag_req)
        pill_results = rag_resp.get("pill_results", [])
        duration = time.time() - t0

        # UI Adapter & Safety Evaluation
        pills_vm, quality_vm = parse_cv_output(cv_out)
        safety_report = evaluate_safety_and_report(pills_vm)

        scene_results.append({
            "scene": scene_name,
            "image_path": str(img_path),
            "duration_sec": round(duration, 2),
            "num_pills_detected": len(pills_cv),
            "pill_results": pill_results,
            "safety_report": {
                "overall_severity": safety_report.overall_severity,
                "identified_drugs": safety_report.identified_drugs,
                "interactions_count": len(safety_report.interactions),
                "interactions": [
                    {
                        "drug_a": i.drug_a_name,
                        "drug_b": i.drug_b_name,
                        "severity": i.severity,
                        "message": i.message,
                    }
                    for i in safety_report.interactions
                ],
                "duplicate_warnings": [
                    {
                        "ingredient": d.ingredient_name,
                        "instances": d.source_instances,
                    }
                    for d in safety_report.duplicate_warnings
                ],
            }
        })

    return scene_results


def main():
    parser = argparse.ArgumentParser(description="Tune identification parameters and evaluate scenes.")
    parser.add_argument("--benchmark-file", type=Path, default=PROJECT_ROOT / "outputs" / "benchmark_verified_results.json")
    parser.add_argument("--scene-dir", type=Path, default=PROJECT_ROOT / "pill_images_verified")
    parser.add_argument("--run-scenes", action="store_true", help="Chạy kiểm thử trên test_1.png và test_2.png")
    args = parser.parse_args()

    if not args.benchmark_file.exists():
        print(f"Không tìm thấy file benchmark: {args.benchmark_file}")
        sys.exit(1)

    data = json.loads(args.benchmark_file.read_text(encoding="utf-8"))
    detailed = data.get("detailed_results", [])
    print(f"Nạp {len(detailed)} mẫu đánh giá từ benchmark verified pills...")

    db = setup_in_memory_db()

    # Dải siêu tham số tìm kiếm toàn diện
    fuzzy_grid = [0.35, 0.40, 0.45]
    identified_grid = [0.65, 0.70, 0.73, 0.75, 0.78, 0.80]
    margin_grid = [0.03, 0.05, 0.08]
    imprint_grid = [0.45, 0.50, 0.55, 0.60]
    ocr_conf_grid = [0.15, 0.20, 0.25]
    ambiguous_grid = [0.35, 0.40, 0.45]

    all_combos = list(itertools.product(
        fuzzy_grid, identified_grid, margin_grid, imprint_grid, ocr_conf_grid, ambiguous_grid
    ))
    print(f"Bắt đầu Grid Search trên {len(all_combos)} tổ hợp tham số toàn diện...")

    all_results = []
    best_balanced_score = -1.0
    best_balanced_config = None
    best_precision_config = None
    best_recall_config = None

    max_prec = -1.0
    max_rec = -1.0

    for idx, (f_th, id_th, m_th, imp_th, ocr_th, amb_th) in enumerate(all_combos, 1):
        res = evaluate_parameter_set(
            benchmark_data=detailed,
            db=db,
            fuzzy_threshold=f_th,
            identified_threshold=id_th,
            margin_threshold=m_th,
            imprint_threshold=imp_th,
            minimum_ocr_confidence=ocr_th,
            ambiguous_threshold=amb_th,
        )
        f1 = res["metrics"]["f1_score"]
        prec = res["metrics"]["precision"]
        rec = res["metrics"]["recall"]
        fp = res["metrics"]["identified_false_positive"]
        all_results.append(res)

        # 1. Best Precision (0 FP)
        if prec > max_prec or (prec == max_prec and rec > (best_precision_config["metrics"]["recall"] if best_precision_config else -1)):
            max_prec = prec
            best_precision_config = res

        # 2. Best Recall
        if rec > max_rec:
            max_rec = rec
            best_recall_config = res

        # 3. Best Balanced: Ưu tiên F1 cao và Precision cao (>= 90% hoặc 0 FP)
        # Điểm cân bằng = F1 - (FP * 5)
        balanced_score = f1 - (fp * 2.0)
        if balanced_score > best_balanced_score:
            best_balanced_score = balanced_score
            best_balanced_config = res

    # Sắp xếp top 5 cấu hình cân bằng tốt nhất
    all_results.sort(key=lambda r: (r["metrics"]["f1_score"] - r["metrics"]["identified_false_positive"] * 2.0), reverse=True)
    top_5 = all_results[:5]

    print("\n" + "=" * 80)
    print("KẾT QUẢ GRID SEARCH TỐI ƯU SIÊU THAM SỐ TOÀN DIỆN")
    print("=" * 80)
    print("\n[1] TOP 5 CẤU HÌNH CÂN BẰNG TỐT NHẤT (BEST BALANCED CONFIGURATIONS):")
    for i, cfg in enumerate(top_5, 1):
        p = cfg["params"]
        m = cfg["metrics"]
        print(f"Rank {i}: F1={m['f1_score']}% | Prec={m['precision']}% | Rec={m['recall']}% | Identified={m['identified_count']} (Đúng={m['identified_correct']}, Sai={m['identified_false_positive']}) | Ambiguous={m['ambiguous_count']}")
        print(f"       Params: id_th={p['identified_threshold']}, imp_th={p['imprint_threshold']}, margin={p['margin_threshold']}, ocr_conf={p['minimum_ocr_confidence']}, fuzzy={p['fuzzy_threshold']}, amb_th={p['ambiguous_threshold']}")

    print("\n" + "-" * 80)
    print("[2] CẤU HÌNH TỐI ƯU ĐƯỢC CHỌN (RECOMMENDED BEST CONFIG):")
    print(json.dumps(best_balanced_config, indent=2, ensure_ascii=False))
    print("=" * 80)

    # Lưu kết quả tối ưu vào file JSON
    opt_out_file = PROJECT_ROOT / "outputs" / "optimized_rag_parameters.json"
    opt_out_file.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_balanced_config": best_balanced_config,
        "best_precision_config": best_precision_config,
        "best_recall_config": best_recall_config,
        "top_5_configs": top_5,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Đã lưu cấu hình tối ưu vào: {opt_out_file}")

    # Chạy Scene Tests nếu được yêu cầu
    if args.run_scenes and best_balanced_config:
        print("\n" + "=" * 80)
        print("ĐÁNH GIÁ TRỰC TIẾP TRÊN ẢNH HIỆN TRƯỜNG TEST_1 & TEST_2")
        print("=" * 80)
        scene_res = evaluate_scene_images(db, best_balanced_config["params"], args.scene_dir)
        scene_out_file = PROJECT_ROOT / "outputs" / "scene_evaluation_results.json"
        scene_out_file.write_text(json.dumps(scene_res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n✓ Đã lưu kết quả Scene Test vào: {scene_out_file}")
        for s in scene_res:
            print(f"\n--- Scene: {s['scene']} ---")
            print(f"  Số viên phát hiện: {s['num_pills_detected']}")
            for p in s["pill_results"]:
                inst = p.get("instance_id")
                st = p.get("identification_status")
                prod = p.get("accepted_product", {})
                prod_name = prod.get("product_name") if prod else "N/A"
                top1_cand = p.get("top_candidates", [{}])[0].get("product_name", "None") if p.get("top_candidates") else "None"
                print(f"  - {inst}: Status={st} | Product={prod_name} | Top1_Candidate={top1_cand}")
            print(f"  Báo cáo an toàn tổng thể: {s['safety_report']['overall_severity']}")
            print(f"  Số cặp tương tác phát hiện: {s['safety_report']['interactions_count']}")


if __name__ == "__main__":
    main()
