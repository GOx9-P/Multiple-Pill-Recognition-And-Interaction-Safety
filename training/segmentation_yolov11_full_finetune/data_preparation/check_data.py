from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.pill_safety.cv.segmentation.utils.config import OUTPUT_DIR


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
