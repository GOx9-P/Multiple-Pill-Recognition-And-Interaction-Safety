#!/usr/bin/env python3
"""Script benchmark toàn diện hệ thống nhận diện thuốc trên tập ảnh pill_images_verified."""

from __future__ import annotations

import argparse
import json
import logging
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
from pill_safety.database.models import DrugAppearance, DrugProduct, Ingredient, ProductIngredient, DrugInteraction
from pill_safety.database.scripts.seed import seed_database
from pill_safety.rag.identification_service import IdentificationService
from pill_safety.rag.retrieval.normalization import normalize_color, normalize_imprint, normalize_shape
from ui.model_loader import load_cv_pipeline


def setup_in_memory_db() -> Session:
    """Khởi tạo SQLite in-memory và seed toàn bộ 35 loại thuốc chuẩn."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    seed_database(session)
    return session


def run_benchmark(
    verified_dir: Path,
    output_json: Path | None = None,
    max_drugs: int | None = None,
) -> dict[str, Any]:
    print("=" * 80)
    print("KHỞI CHẠY BENCHMARK HỆ THỐNG NHẬN DIỆN THUỐC TRÊN PILL_IMAGES_VERIFIED")
    print("=" * 80)

    # 1. Khởi tạo Database & RAG
    print("[1/4] Khởi tạo Cơ sở dữ liệu và RAG Identification Service...")
    db_session = setup_in_memory_db()
    id_service = IdentificationService(db_session)

    # 2. Khởi tạo CV Pipeline (YOLOv11-seg, ResNet-18, PaddleOCR)
    print("[2/4] Đang nạp các mô hình AI (YOLOv11, ResNet-18, PaddleOCR)...")
    cv_loader = load_cv_pipeline()
    if not cv_loader.available or cv_loader.pipeline is None:
        raise RuntimeError(f"Không thể khởi tạo CV Pipeline: {cv_loader.error}")
    cv_pipeline = cv_loader.pipeline
    print("✓ Khởi tạo toàn bộ mô hình thành công!\n")

    # 3. Đọc manifest
    manifest_path = verified_dir / "pill_images_verified" / "manifest.json"
    if not manifest_path.exists():
        manifest_path = verified_dir / "manifest.json"
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Không tìm thấy manifest.json tại: {manifest_path}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if max_drugs:
        manifest_data = manifest_data[:max_drugs]

    print(f"[3/4] Bắt đầu quét qua {len(manifest_data)} danh mục thuốc...")

    results: list[dict[str, Any]] = []
    category_counts = {
        "both_sides": {"total": 0, "top1_match": 0, "top3_match": 0, "identified": 0, "identified_correct": 0, "ambiguous": 0, "unknown": 0},
        "face_1": {"total": 0, "top1_match": 0, "top3_match": 0, "identified": 0, "identified_correct": 0, "ambiguous": 0, "unknown": 0},
        "face_2": {"total": 0, "top1_match": 0, "top3_match": 0, "identified": 0, "identified_correct": 0, "ambiguous": 0, "unknown": 0},
    }

    feature_stats = {
        "shape_correct": 0,
        "color_correct": 0,
        "scoreline_correct": 0,
        "imprint_detected": 0,
        "imprint_matched": 0,
        "total_instances": 0,
    }

    start_time = time.time()

    for idx, drug_entry in enumerate(manifest_data, 1):
        product_code = drug_entry.get("product_code")
        drug_name = drug_entry.get("drug")
        folder_name = drug_entry.get("folder")
        expected_imprint = drug_entry.get("expected_imprint", "")
        expected_shape = normalize_shape(drug_entry.get("expected_shape"))
        expected_color = normalize_color(drug_entry.get("expected_color"))
        expected_score_line = bool(drug_entry.get("score_line"))

        folder_path = verified_dir / "pill_images_verified" / folder_name
        if not folder_path.exists():
            folder_path = verified_dir / folder_name

        image_tasks = [
            ("both_sides", folder_path / "01_both_sides_original.jpg"),
            ("face_1", folder_path / "02_face_1_crop.png"),
            ("face_2", folder_path / "03_face_2_crop.png"),
        ]

        print(f"[{idx}/{len(manifest_data)}] Đang xử lý: {drug_name} ({product_code})...")

        for img_type, img_path in image_tasks:
            if not img_path.exists():
                # Thử tìm đuôi mở rộng khác (png/jpg)
                alts = list(folder_path.glob(f"{img_path.stem}.*"))
                if alts:
                    img_path = alts[0]
                else:
                    continue

            category_counts[img_type]["total"] += 1
            img_start = time.time()
            print(f"   -> Đang xử lý {img_type}: {img_path.name}...", flush=True)

            try:
                # 1. Chạy CV Pipeline
                req_id = f"bench_{product_code}_{img_type}"
                cv_output = cv_pipeline.predict({
                    "request_id": req_id,
                    "session_id": "sess_bench",
                    "image_id": f"img_{product_code}_{img_type}",
                    "image_path": str(img_path),
                })
                
                cv_dict = cv_output.model_dump(mode="json")
                pills = cv_dict.get("pills", [])

                # 2. Chạy RAG Identification
                rag_req = {
                    "schema_version": "rag_request_v1",
                    "request_id": req_id,
                    "session_id": "sess_bench",
                    "market": "US",
                    "cv_output": cv_dict,
                }
                rag_resp = id_service.identify(rag_req)
                pill_results = rag_resp.get("pill_results", [])

                img_duration = time.time() - img_start

                # Đánh giá kết quả cho từng viên thuốc phát hiện được
                detected_top1_match = False
                detected_top3_match = False
                is_identified = False
                is_identified_correct = False
                is_ambiguous = False
                is_unknown = False

                for p_idx, p_res in enumerate(pill_results):
                    feature_stats["total_instances"] += 1
                    status = p_res.get("identification_status")
                    top_candidates = p_res.get("top_candidates", [])
                    cand_product_codes = [c.get("product_code") for c in top_candidates]

                    # Trích xuất đặc trưng dự đoán từ CV
                    if p_idx < len(pills):
                        pill_cv = pills[p_idx]
                        pred_shape = normalize_shape((pill_cv.get("shape") or {}).get("label"))
                        pred_color = normalize_color((pill_cv.get("color") or {}).get("primary"))
                        pred_scoreline = (pill_cv.get("scoreline") or {}).get("visible")
                        imprint_info = pill_cv.get("imprint") or {}
                        imprint_raw = imprint_info.get("raw")

                        if pred_shape == expected_shape:
                            feature_stats["shape_correct"] += 1
                        if pred_color == expected_color:
                            feature_stats["color_correct"] += 1
                        if pred_scoreline is not None and bool(pred_scoreline) == expected_score_line:
                            feature_stats["scoreline_correct"] += 1
                        if imprint_raw:
                            feature_stats["imprint_detected"] += 1
                            if normalize_imprint(imprint_raw) in normalize_imprint(expected_imprint) or normalize_imprint(expected_imprint) in normalize_imprint(imprint_raw):
                                feature_stats["imprint_matched"] += 1

                    top1_code = cand_product_codes[0] if cand_product_codes else None
                    if top1_code == product_code:
                        detected_top1_match = True

                    if product_code in cand_product_codes[:3]:
                        detected_top3_match = True

                    if status == "identified":
                        is_identified = True
                        if top1_code == product_code:
                            is_identified_correct = True
                    elif status == "ambiguous":
                        is_ambiguous = True
                    else:
                        is_unknown = True

                if detected_top1_match:
                    category_counts[img_type]["top1_match"] += 1
                if detected_top3_match:
                    category_counts[img_type]["top3_match"] += 1
                if is_identified:
                    category_counts[img_type]["identified"] += 1
                    if is_identified_correct:
                        category_counts[img_type]["identified_correct"] += 1
                elif is_ambiguous:
                    category_counts[img_type]["ambiguous"] += 1
                elif is_unknown or not pills:
                    category_counts[img_type]["unknown"] += 1

                results.append({
                    "product_code": product_code,
                    "drug_name": drug_name,
                    "img_type": img_type,
                    "img_path": str(img_path.relative_to(PROJECT_ROOT) if img_path.is_relative_to(PROJECT_ROOT) else img_path),
                    "num_pills_detected": len(pills),
                    "top1_match": detected_top1_match,
                    "top3_match": detected_top3_match,
                    "status": pill_results[0].get("identification_status") if pill_results else "no_pill_detected",
                    "cv_output": cv_dict,
                    "top_candidates": [
                        {
                            "rank": c.get("rank"),
                            "product_code": c.get("product_code"),
                            "product_name": c.get("product_name"),
                            "final_score": c.get("final_score"),
                            "imprint_match_score": c.get("evidence", {}).get("imprint_match_score"),
                        }
                        for c in (pill_results[0].get("top_candidates", []) if pill_results else [])[:3]
                    ],
                    "duration_sec": round(img_duration, 3),
                })

            except Exception as exc:
                print(f"   [LỖI] Xử lý ảnh {img_path.name}: {exc}")
                category_counts[img_type]["unknown"] += 1
                results.append({
                    "product_code": product_code,
                    "drug_name": drug_name,
                    "img_type": img_type,
                    "img_path": str(img_path),
                    "error": str(exc),
                })

    # Đánh giá 2 ảnh test ngữ cảnh đa viên
    scene_tests = [
        ("test_1", verified_dir / "test_1.png"),
        ("test_2", verified_dir / "test_2.png"),
    ]
    scene_results = []
    print("\n[4/4] Đang quét qua các ảnh test đa viên tổng hợp (test_1.png, test_2.png)...", flush=True)
    for scene_name, scene_path in scene_tests:
        if scene_path.exists():
            try:
                print(f"   -> Đang xử lý {scene_name}: {scene_path.name}...", flush=True)
                sc_start = time.time()
                cv_out = cv_pipeline.predict({
                    "request_id": f"bench_{scene_name}",
                    "session_id": "sess_bench_scene",
                    "image_id": f"img_{scene_name}",
                    "image_path": str(scene_path),
                })
                cv_dict = cv_out.model_dump(mode="json")
                rag_resp = id_service.identify({
                    "schema_version": "rag_request_v1",
                    "request_id": f"bench_{scene_name}",
                    "session_id": "sess_bench_scene",
                    "cv_output": cv_dict,
                })
                sc_duration = time.time() - sc_start
                pills = cv_dict.get("pills", [])
                p_results = rag_resp.get("pill_results", [])
                scene_results.append({
                    "scene": scene_name,
                    "pills_detected": len(pills),
                    "identified_pills": [
                        {
                            "instance_id": pr.get("instance_id"),
                            "status": pr.get("identification_status"),
                            "accepted_product": pr.get("accepted_product"),
                            "top_candidates": [
                                {"product_name": c.get("product_name"), "score": c.get("final_score")}
                                for c in pr.get("top_candidates", [])[:2]
                            ]
                        }
                        for pr in p_results
                    ],
                    "duration_sec": round(sc_duration, 3),
                })
                print(f"✓ Hoàn thành {scene_name}: phát hiện {len(pills)} viên thuốc trong {round(sc_duration, 2)}s.", flush=True)
            except Exception as e:
                print(f"Lỗi khi xử lý {scene_name}: {e}", flush=True)
                import traceback
                traceback.print_exc()

    total_time = time.time() - start_time

    # Tổng hợp các chỉ số
    total_imgs = sum(c["total"] for c in category_counts.values())
    total_top1 = sum(c["top1_match"] for c in category_counts.values())
    total_top3 = sum(c["top3_match"] for c in category_counts.values())
    total_identified = sum(c["identified"] for c in category_counts.values())
    total_identified_correct = sum(c["identified_correct"] for c in category_counts.values())
    total_ambiguous = sum(c["ambiguous"] for c in category_counts.values())
    total_unknown = sum(c["unknown"] for c in category_counts.values())

    top1_acc = (total_top1 / total_imgs * 100) if total_imgs else 0.0
    top3_acc = (total_top3 / total_imgs * 100) if total_imgs else 0.0
    precision = (total_identified_correct / total_identified * 100) if total_identified else 0.0
    recall = (total_identified_correct / total_imgs * 100) if total_imgs else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    inst_total = feature_stats["total_instances"] or 1
    shape_acc = (feature_stats["shape_correct"] / inst_total) * 100
    color_acc = (feature_stats["color_correct"] / inst_total) * 100
    scoreline_acc = (feature_stats["scoreline_correct"] / inst_total) * 100
    imprint_det_rate = (feature_stats["imprint_detected"] / inst_total) * 100
    imprint_match_rate = (feature_stats["imprint_matched"] / (feature_stats["imprint_detected"] or 1)) * 100

    summary = {
        "total_images_evaluated": total_imgs,
        "total_drugs": len(manifest_data),
        "total_duration_sec": round(total_time, 2),
        "overall_metrics": {
            "top1_accuracy": round(top1_acc, 2),
            "top3_accuracy": round(top3_acc, 2),
            "precision_identified": round(precision, 2),
            "recall_identified": round(recall, 2),
            "f1_score": round(f1, 2),
            "identification_rate": round((total_identified / total_imgs) * 100, 2) if total_imgs else 0.0,
            "ambiguity_rate": round((total_ambiguous / total_imgs) * 100, 2) if total_imgs else 0.0,
            "unknown_rate": round((total_unknown / total_imgs) * 100, 2) if total_imgs else 0.0,
        },
        "feature_metrics": {
            "shape_accuracy": round(shape_acc, 2),
            "color_accuracy": round(color_acc, 2),
            "scoreline_accuracy": round(scoreline_acc, 2),
            "imprint_detection_rate": round(imprint_det_rate, 2),
            "imprint_match_rate": round(imprint_match_rate, 2),
        },
        "per_category_breakdown": {
            cat: {
                "total": data["total"],
                "top1_accuracy": round((data["top1_match"] / data["total"] * 100) if data["total"] else 0.0, 2),
                "top3_accuracy": round((data["top3_match"] / data["total"] * 100) if data["total"] else 0.0, 2),
                "identified_count": data["identified"],
                "identified_correct": data["identified_correct"],
                "ambiguous_count": data["ambiguous"],
                "unknown_count": data["unknown"],
            }
            for cat, data in category_counts.items()
        },
        "scene_tests": scene_results,
    }

    print("\n" + "=" * 80, flush=True)
    print("BÁO CÁO KẾT QUẢ BENCHMARK HỆ THỐNG NHẬN DIỆN THUỐC", flush=True)
    print("=" * 80, flush=True)
    print(f"Tổng số ảnh đánh giá: {total_imgs} ảnh ({len(manifest_data)} loại thuốc)", flush=True)
    print(f"Thời gian hoàn thành: {round(total_time, 2)} giây (trung bình {round(total_time / (total_imgs or 1), 2)}s/ảnh)\n", flush=True)
    print("--- CHỈ SỐ NHẬN DIỆN TỔNG THỂ (OVERALL METRICS) ---", flush=True)
    print(f"• Top-1 Accuracy:            {top1_acc:.2f}% ({total_top1}/{total_imgs})", flush=True)
    print(f"• Top-3 Accuracy:            {top3_acc:.2f}% ({total_top3}/{total_imgs})", flush=True)
    print(f"• Precision (Identified):    {precision:.2f}% (Tỷ lệ đúng tuyệt đối khi hệ thống báo Identified)", flush=True)
    print(f"• Recall (Identified):       {recall:.2f}%", flush=True)
    print(f"• F1-Score:                  {f1:.2f}%", flush=True)
    print(f"• Tỷ lệ Xác Định (Identified): {summary['overall_metrics']['identification_rate']:.2f}%", flush=True)
    print(f"• Tỷ lệ Cần Check (Ambiguous): {summary['overall_metrics']['ambiguity_rate']:.2f}%", flush=True)
    print(f"• Tỷ lệ Không Rõ (Unknown):   {summary['overall_metrics']['unknown_rate']:.2f}%\n", flush=True)

    print("--- ĐỘ CHÍNH XÁC TỪNG ĐẶC TRƯNG THỊ GIÁC (CV FEATURES) ---", flush=True)
    print(f"• Shape Accuracy:            {shape_acc:.2f}%", flush=True)
    print(f"• Color Accuracy:            {color_acc:.2f}%", flush=True)
    print(f"• Scoreline Accuracy:        {scoreline_acc:.2f}%", flush=True)
    print(f"• Imprint Detection Rate:    {imprint_det_rate:.2f}%", flush=True)
    print(f"• Imprint Match Rate:        {imprint_match_rate:.2f}%\n", flush=True)

    print("--- PHÂN BỐ THEO LOẠI ẢNH (CATEGORY BREAKDOWN) ---", flush=True)
    for cat, data in summary["per_category_breakdown"].items():
        print(f"• {cat.upper():12s}: Total={data['total']}, Top-1 Acc={data['top1_accuracy']}%, Top-3 Acc={data['top3_accuracy']}%, Identified={data['identified_count']} ({data['identified_correct']} đúng), Ambiguous={data['ambiguous_count']}", flush=True)

    print("=" * 80, flush=True)

    report_payload = {
        "summary": summary,
        "detailed_results": results,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✓ Đã lưu kết quả chi tiết tại: {output_json}", flush=True)

    return report_payload


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark pill recognition on verified images.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "pill_images_verified",
        help="Thư mục chứa pill_images_verified",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "benchmark_verified_results.json",
        help="Đường dẫn file JSON lưu kết quả.",
    )
    parser.add_argument(
        "--max-drugs",
        type=int,
        default=None,
        help="Giới hạn số thuốc để test nhanh (mặc định toàn bộ 35).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        run_benchmark(
            verified_dir=args.dataset_dir,
            output_json=args.output,
            max_drugs=args.max_drugs,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.exit(1)
