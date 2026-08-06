"""
check_data.py (tuỳ chọn)
---------------------------
Không có trong script gốc của bạn — thêm để kiểm tra nhanh kết quả sau khi
chạy prepare_data.py: đếm ảnh/label mỗi split, cảnh báo ảnh thiếu label,
label rỗng. KHÔNG bắt buộc phải chạy, chỉ để đối chiếu số liệu.

Usage:
    python data_preparation/check_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.segmentation.config import OUTPUT_DIR


def main():
    print(f"Kiểm tra dataset tại: {OUTPUT_DIR}\n")
    for split in ["train", "val", "test"]:
        img_dir = OUTPUT_DIR / "images" / split
        lbl_dir = OUTPUT_DIR / "labels" / split

        if not img_dir.exists():
            print(f"[{split}] !! không tồn tại {img_dir} — chạy prepare_data.py trước.")
            continue

        images = sorted(img_dir.glob("*.*"))
        n_missing_label = 0
        n_empty_label = 0
        n_instances = 0

        for img_path in images:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                n_missing_label += 1
                continue
            lines = [l for l in lbl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                n_empty_label += 1
            n_instances += len(lines)

        print(f"[{split}] {len(images)} ảnh | {n_instances} instance | "
              f"thiếu label: {n_missing_label} | label rỗng: {n_empty_label}")

    data_yaml = OUTPUT_DIR / "data.yaml"
    print(f"\ndata.yaml: {'OK - ' + str(data_yaml) if data_yaml.exists() else 'CHƯA CÓ - chạy prepare_data.py'}")


if __name__ == "__main__":
    main()
