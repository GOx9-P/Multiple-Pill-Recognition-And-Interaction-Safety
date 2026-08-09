from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.pill_safety.cv.segmentation.utils.config import OUTPUT_DIR, CLASS_AGNOSTIC, RAW_IMAGES_DIR


# ----------------------------------------------------------------------------
# 1. LOAD COCO ANNOTATIONS
# ----------------------------------------------------------------------------
def load_coco(ann_path: Path):
    import json

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {c["id"]: c["name"] for c in coco["categories"]}

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    print(f"Loaded {len(images)} images, {len(coco['annotations'])} annotations, "
          f"{len(categories)} classes.")
    return images, anns_by_image, categories


def drop_empty_annotation_images(images: dict, anns_by_image: dict):
    """Loại bỏ hoàn toàn các ảnh không có annotation nào (vd 2 ảnh rỗng phát hiện ở bước EDA)."""
    empty_ids = [img_id for img_id in images if not anns_by_image.get(img_id)]
    for img_id in empty_ids:
        print(f"  [drop] {images[img_id]['file_name']} (không có annotation)")
        del images[img_id]
        anns_by_image.pop(img_id, None)
    print(f"Đã loại {len(empty_ids)} ảnh rỗng annotation. Còn lại {len(images)} ảnh.")
    return images, anns_by_image


# ----------------------------------------------------------------------------
# 2. SPLIT TRAIN / VAL / TEST
# ----------------------------------------------------------------------------
def dominant_class(anns):
    """Class xuất hiện nhiều nhất trong ảnh — dùng để stratify cho cân bằng."""
    counts = Counter(a["category_id"] for a in anns)
    return counts.most_common(1)[0][0]


def split_dataset(images: dict, anns_by_image: dict, ratios, seed):
    image_ids = list(images.keys())
    strat_labels = [dominant_class(anns_by_image[i]) if anns_by_image[i] else -1
                    for i in image_ids]

    label_counts = Counter(strat_labels)
    strat_labels = [lbl if label_counts[lbl] >= 2 else "rare" for lbl in strat_labels]
    n_rare = sum(1 for l in strat_labels if l == "rare")
    if n_rare:
        print(f"  [info] gộp {n_rare} ảnh thuộc class hiếm (<2 mẫu, gồm cả ảnh rỗng) "
              f"vào nhóm 'rare' để stratify được.")

    train_ratio, val_ratio, test_ratio = ratios
    train_ids, temp_ids, train_lbl, temp_lbl = train_test_split(
        image_ids, strat_labels,
        train_size=train_ratio, random_state=seed, stratify=strat_labels,
    )
    # chia phần còn lại (val+test) theo tỉ lệ tương ứng
    rel_val = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids, train_size=rel_val, random_state=seed, stratify=temp_lbl,
    )

    print(f"Split -> train: {len(train_ids)} | val: {len(val_ids)} | test: {len(test_ids)}")
    return {"train": train_ids, "val": val_ids, "test": test_ids}


# ----------------------------------------------------------------------------
# 3. COCO polygon -> YOLO-seg normalized polygon lines
# ----------------------------------------------------------------------------
def coco_ann_to_yolo_lines(anns, img_w, img_h, class_agnostic=CLASS_AGNOSTIC, class_map=None):
    lines = []
    for ann in anns:
        seg = ann.get("segmentation")
        if not seg or isinstance(seg, dict):  # bỏ qua RLE / rỗng
            continue
        class_id = 0 if class_agnostic else class_map[ann["category_id"]]
        for poly in seg:  # 1 instance có thể có nhiều polygon (đa vùng)
            if len(poly) < 6:
                continue
            norm = []
            for i in range(0, len(poly), 2):
                x = min(max(poly[i] / img_w, 0), 1)
                y = min(max(poly[i + 1] / img_h, 0), 1)
                norm.extend([f"{x:.6f}", f"{y:.6f}"])
            lines.append(f"{class_id} " + " ".join(norm))
    return lines


# ----------------------------------------------------------------------------
# 4. BUILD FOLDER STRUCTURE + WRITE ORIGINAL (unaugmented) SPLITS
# ----------------------------------------------------------------------------
def build_splits(images, anns_by_image, categories, splits):
    class_map = {cid: idx for idx, cid in enumerate(sorted(categories.keys()))}

    for split_name, ids in splits.items():
        img_out = OUTPUT_DIR / "images" / split_name
        lbl_out = OUTPUT_DIR / "labels" / split_name

        # Xóa dữ liệu cũ trước khi tạo lại split
        for out_dir in (img_out, lbl_out):
            if out_dir.exists():
                for item in out_dir.iterdir():
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)

            out_dir.mkdir(parents=True, exist_ok=True)

        for img_id in ids:
            img_info = images[img_id]
            src = RAW_IMAGES_DIR / img_info["file_name"]
            if not src.exists():
                print(f"  [warn] missing file: {src}")
                continue
            dst = img_out / img_info["file_name"]
            shutil.copy2(src, dst)

            lines = coco_ann_to_yolo_lines(
                anns_by_image[img_id], img_info["width"], img_info["height"],
                class_map=class_map,
            )
            (lbl_out / (Path(img_info["file_name"]).stem + ".txt")).write_text(
                "\n".join(lines), encoding="utf-8"
            )

    print("Original (unaugmented) train/val/test folders written.")
    return class_map


# ----------------------------------------------------------------------------
# data.yaml cho Ultralytics
# ----------------------------------------------------------------------------
def write_data_yaml(categories, class_map):
    if CLASS_AGNOSTIC:
        names = ["pill"]
    else:
        names = [categories[cid] for cid, idx in sorted(class_map.items(), key=lambda x: x[1])]

    yaml_content = (
        f"path: {OUTPUT_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    (OUTPUT_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"data.yaml written -> {OUTPUT_DIR / 'data.yaml'}")
