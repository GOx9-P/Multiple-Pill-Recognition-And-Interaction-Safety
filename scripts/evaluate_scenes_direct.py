#!/usr/bin/env python3
"""Script đo lường trực tiếp kết quả nhận diện trên test_1.png và test_2.png bằng mô hình CV thật."""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from pill_safety.database.base import Base
from pill_safety.database.scripts.seed import seed_database
from pill_safety.rag.identification_service import IdentificationService
from pill_safety.rag.ranking.safety_gate import SafetyGate
from ui.adapters.pipeline_adapter import evaluate_safety_and_report, parse_cv_output
from ui.model_loader import load_cv_pipeline


def setup_in_memory_db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    seed_database(session)
    return session


def _find_scene_file(name: str, scene_dir: Path) -> Path | None:
    for cand in [
        scene_dir / f"{name}.png",
        scene_dir / f"{name}.jpg",
        scene_dir / "pill_images_verified" / f"{name}.png",
        scene_dir / "pill_images_verified" / f"{name}.jpg",
    ] + list(scene_dir.rglob(f"{name}.*")):
        if cand.exists() and cand.is_file():
            return cand
    return None


def main():
    print("=" * 80)
    print("KHỞI CHẠY ĐO LƯỜNG THỰC TẾ TRÊN ẢNH HIỆN TRƯỜNG TEST_1.PNG & TEST_2.PNG")
    print("=" * 80)

    # 1. Khởi tạo Database & RAG với cấu hình tối ưu
    db = setup_in_memory_db()
    id_service = IdentificationService(db)

    # Nạp cấu hình tối ưu đã lưu
    opt_file = PROJECT_ROOT / "outputs" / "optimized_rag_parameters.json"
    if opt_file.exists():
        try:
            opt_data = json.loads(opt_file.read_text(encoding="utf-8"))
            best_params = opt_data.get("best_balanced_config", {}).get("params", {})
            print(f"Áp dụng cấu hình siêu tham số tối ưu: {json.dumps(best_params, indent=2)}")
            SafetyGate.identified_threshold = best_params.get("identified_threshold", 0.65)
            SafetyGate.margin_threshold = best_params.get("margin_threshold", 0.03)
            SafetyGate.imprint_threshold = best_params.get("imprint_threshold", 0.45)
            SafetyGate.minimum_ocr_confidence = best_params.get("minimum_ocr_confidence", 0.15)
            SafetyGate.ambiguous_threshold = best_params.get("ambiguous_threshold", 0.35)
        except Exception as e:
            print(f"⚠️ Không thể đọc file cấu hình: {e}")

    # 2. Khởi tạo Pipeline CV thật
    print("\n[1/3] Đang nạp các mô hình Deep Learning CV thật...")
    t0 = time.time()
    cv_loader = load_cv_pipeline()
    if not cv_loader.available or cv_loader.pipeline is None:
        print(f"[ERROR] Không thể nạp CV Pipeline: {cv_loader.error}")
        return
    cv_pipeline = cv_loader.pipeline
    print(f"✓ Nạp mô hình CV thành công trong {time.time() - t0:.2f} giây!\n")

    # 3. Quét qua test_1.png và test_2.png
    scene_dir = PROJECT_ROOT / "pill_images_verified"
    scene_names = ["test_1", "test_2"]

    all_scene_results = []

    for idx, scene_name in enumerate(scene_names, 1):
        img_path = _find_scene_file(scene_name, scene_dir)
        if not img_path or not img_path.exists():
            print(f"[WARNING] Không tìm thấy file {scene_name}.png trong {scene_dir}")
            continue

        print("=" * 80)
        print(f"[{idx}/2] QUÉT HÌNH ẢNH: {scene_name.upper()} ({img_path.name})")
        print("=" * 80)
        t_scene = time.time()

        # A. Chạy CV Pipeline thật
        print(f" -> Đang chạy YOLOv11 segmentation, ResNet-18 attributes, PaddleOCR...")
        req = {
            "request_id": f"scene_{scene_name}",
            "session_id": "sess_real",
            "image_id": f"img_{scene_name}",
            "image_path": str(img_path),
        }
        cv_out = cv_pipeline.predict(req)
        cv_dict = cv_out.model_dump(mode="json")
        pills_cv = cv_dict.get("pills", [])
        print(f" -> CV phát hiện tổng cộng: {len(pills_cv)} viên thuốc trong ảnh.")

        # B. Chạy RAG Retrieval & Safety Gate
        rag_req = {
            "schema_version": "rag_request_v1",
            "request_id": f"scene_{scene_name}",
            "session_id": "sess_real",
            "market": "US",
            "cv_output": cv_dict,
        }
        rag_resp = id_service.identify(rag_req)
        pill_results = rag_resp.get("pill_results", [])

        # C. Chuyển đổi sang ViewModel & Đánh giá DDI
        pills_vm, quality_vm = parse_cv_output(cv_out)
        safety_report = evaluate_safety_and_report(pills_vm)

        scene_duration = time.time() - t_scene
        print(f"✓ Xử lý hoàn tất trong {scene_duration:.2f} giây.\n")

        # Thống kê chi tiết từng viên thuốc
        identified_pills = []
        ambiguous_pills = []
        unknown_pills = []

        print("--- CHI TIẾT TỪNG VIÊN THUỐC ĐƯỢC PHÁT HIỆN ---")
        for p_idx, p_res in enumerate(pill_results):
            inst_id = p_res.get("instance_id")
            st = p_res.get("identification_status")
            top_candidates = p_res.get("top_candidates", [])
            
            # Thuộc tính CV
            p_cv = pills_cv[p_idx] if p_idx < len(pills_cv) else {}
            shape_cv = (p_cv.get("shape") or {}).get("label", "N/A")
            color_cv = (p_cv.get("color") or {}).get("primary", "N/A")
            ocr_raw = (p_cv.get("imprint") or {}).get("raw", "None")
            ocr_conf = (p_cv.get("imprint") or {}).get("confidence", 0.0)

            top1_name = top_candidates[0].get("product_name") if top_candidates else "Không tìm thấy"
            top1_score = top_candidates[0].get("final_score", 0.0) if top_candidates else 0.0

            print(f"• [{inst_id}] CV: Shape={shape_cv}, Color={color_cv}, OCR='{ocr_raw}' (conf={ocr_conf:.2f})")
            print(f"    Trạng thái: {st.upper()} | Top 1: {top1_name} (Điểm={top1_score:.4f})")

            if st == "identified":
                identified_pills.append({
                    "instance_id": inst_id,
                    "drug_name": top1_name,
                    "score": top1_score,
                    "shape": shape_cv,
                    "color": color_cv,
                    "imprint": ocr_raw,
                })
            elif st == "ambiguous":
                cands_summary = [f"{c.get('product_name')} ({c.get('final_score'):.2f})" for c in top_candidates[:3]]
                ambiguous_pills.append({
                    "instance_id": inst_id,
                    "top_candidates": cands_summary,
                    "shape": shape_cv,
                    "color": color_cv,
                    "imprint": ocr_raw,
                })
                print(f"    -> Gợi ý Top ứng viên: {', '.join(cands_summary)}")
            else:
                unknown_pills.append({
                    "instance_id": inst_id,
                    "shape": shape_cv,
                    "color": color_cv,
                    "imprint": ocr_raw,
                })

        print("\n--- BÁO CÁO TỔNG HỢP SCENE ---")
        print(f"Tổng số viên phát hiện: {len(pills_cv)}")
        print(f"  + Số viên IDENTIFIED (Chắc chắn 100%): {len(identified_pills)}")
        for ip in identified_pills:
            print(f"     -> {ip['instance_id']}: {ip['drug_name']} (Điểm: {ip['score']:.4f})")
        print(f"  + Số viên AMBIGUOUS (Gợi ý Top 3): {len(ambiguous_pills)}")
        print(f"  + Số viên UNKNOWN (Chưa rõ): {len(unknown_pills)}")
        print(f"Mức độ rủi ro tương tác thuốc (DDI): {safety_report.overall_severity.upper()}")
        print(f"Số cặp tương tác thuốc phát hiện: {len(safety_report.interactions)}")
        for inter in safety_report.interactions:
            print(f"  ⚡ [{inter.severity.upper()}] {inter.drug_a_name} + {inter.drug_b_name}: {inter.message}")
        if safety_report.duplicate_warnings:
            for dup in safety_report.duplicate_warnings:
                print(f"  🔄 [TRÙNG HOẠT CHẤT] {dup.ingredient_name} ({', '.join(dup.source_instances)})")

        all_scene_results.append({
            "scene": scene_name,
            "duration_sec": round(scene_duration, 2),
            "num_pills": len(pills_cv),
            "num_identified": len(identified_pills),
            "identified_pills": identified_pills,
            "num_ambiguous": len(ambiguous_pills),
            "ambiguous_pills": ambiguous_pills,
            "num_unknown": len(unknown_pills),
            "overall_severity": safety_report.overall_severity,
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
        })

    # Lưu kết quả
    out_file = PROJECT_ROOT / "outputs" / "scene_evaluation_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_scene_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Đã lưu toàn bộ kết quả đo lường thực tế vào: {out_file}")


if __name__ == "__main__":
    main()
