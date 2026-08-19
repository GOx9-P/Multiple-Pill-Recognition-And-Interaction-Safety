from __future__ import annotations

import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.pill_safety.cv.segmentation.utils.config import OUTPUT_DIR, CLASS_AGNOSTIC, RAW_IMAGES_DIR


class SplitResult(dict):
    """Split IDs plus the strategy evidence used to produce them."""

    def __init__(self, splits: dict[str, list], split_strategy: dict):
        super().__init__(splits)
        self.split_strategy = split_strategy


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


def _split_sizes(n_samples: int, train_ratio: float) -> tuple[int, int]:
    """Match sklearn's float train_size rounding for the second split."""
    n_train = math.floor(n_samples * train_ratio)
    return n_train, n_samples - n_train


def _deterministic_temp_fallback(
    temp_ids: list,
    temp_lbl: list,
    rel_val: float,
    seed: int,
    label_counts: Counter,
) -> tuple[list, list]:
    """Allocate per-label quotas without assuming every label can span both splits."""
    n_val, n_test = _split_sizes(len(temp_ids), rel_val)
    grouped_ids = defaultdict(list)
    for image_id, label in zip(temp_ids, temp_lbl):
        grouped_ids[label].append(image_id)

    rng = random.Random(seed)
    labels = sorted(grouped_ids, key=str)
    for label in labels:
        rng.shuffle(grouped_ids[label])

    rare_labels = {label for label, count in label_counts.items() if count < 2}
    val_ids, test_ids = [], []
    for label in labels:
        if label not in rare_labels:
            continue
        image_id = grouped_ids[label][0]
        if len(val_ids) < n_val and (not n_test or len(val_ids) / n_val <= len(test_ids) / n_test):
            val_ids.append(image_id)
        else:
            test_ids.append(image_id)

    quotas = {}
    for label in labels:
        quotas[label] = 0 if label in rare_labels else math.floor(len(grouped_ids[label]) * rel_val)

    remaining_val_slots = n_val - len(val_ids) - sum(quotas.values())
    if remaining_val_slots > 0:
        candidates = sorted(
            (label for label in labels if label not in rare_labels and quotas[label] < len(grouped_ids[label])),
            key=lambda label: (-(len(grouped_ids[label]) * rel_val - quotas[label]), str(label)),
        )
        for label in candidates[:remaining_val_slots]:
            quotas[label] += 1

    for label in labels:
        if label in rare_labels:
            continue
        quota = quotas[label]
        val_ids.extend(grouped_ids[label][:quota])
        test_ids.extend(grouped_ids[label][quota:])

    if len(val_ids) != n_val or len(test_ids) != n_test:
        raise RuntimeError("Deterministic fallback could not satisfy the requested val/test split sizes.")
    return val_ids, test_ids


def _validate_split_coverage(image_ids: list, splits: dict[str, list]) -> None:
    train_ids, val_ids, test_ids = (set(splits[name]) for name in ("train", "val", "test"))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Split overlap detected; an image was assigned to more than one split.")
    assigned_ids = train_ids | val_ids | test_ids
    if len(assigned_ids) != len(image_ids) or assigned_ids != set(image_ids):
        raise RuntimeError("Split coverage mismatch; images were lost or duplicated.")


def split_dataset(images: dict, anns_by_image: dict, ratios, seed):
    image_ids = list(images.keys())
    strat_labels = [dominant_class(anns_by_image[i]) if anns_by_image[i] else -1
                    for i in image_ids]

    label_counts = Counter(strat_labels)
    strat_labels = [lbl if label_counts[lbl] >= 2 else "rare" for lbl in strat_labels]
    n_rare = sum(1 for label in strat_labels if label == "rare")
    if n_rare:
        print(f"  [info] gộp {n_rare} ảnh thuộc class hiếm (<2 mẫu, gồm cả ảnh rỗng) "
              f"vào nhóm 'rare' để stratify được.")

    train_ratio, val_ratio, test_ratio = ratios
    train_ids, temp_ids, _, temp_lbl = train_test_split(
        image_ids, strat_labels,
        train_size=train_ratio, random_state=seed, stratify=strat_labels,
    )
    rel_val = val_ratio / (val_ratio + test_ratio)
    temp_counts = Counter(temp_lbl)
    n_val, n_test = _split_sizes(len(temp_ids), rel_val)
    rare_temp_labels = {label: count for label, count in temp_counts.items() if count < 2}
    can_stratify_temp = not rare_temp_labels and n_val >= len(temp_counts) and n_test >= len(temp_counts)

    if can_stratify_temp:
        val_ids, test_ids = train_test_split(
            temp_ids, train_size=rel_val, random_state=seed, stratify=temp_lbl,
        )
        split_strategy = {
            "method": "stratified",
            "fallback_used": False,
            "fallback_reason": None,
            "rare_classes": {},
            "seed": seed,
        }
    else:
        val_ids, test_ids = _deterministic_temp_fallback(
            temp_ids, temp_lbl, rel_val, seed, temp_counts,
        )
        split_strategy = {
            "method": "deterministic_fallback",
            "fallback_used": True,
            "fallback_reason": "rare_class_in_temp" if rare_temp_labels else "insufficient_temp_split_size",
            "rare_classes": {str(label): count for label, count in rare_temp_labels.items()},
            "temp_label_counts": {str(label): count for label, count in temp_counts.items()},
            "seed": seed,
        }

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    _validate_split_coverage(image_ids, splits)
    print(f"Split -> train: {len(train_ids)} | val: {len(val_ids)} | test: {len(test_ids)}")
    return SplitResult(splits, split_strategy)


# ----------------------------------------------------------------------------
# 3. COCO polygon -> YOLO-seg normalized polygon lines
# ----------------------------------------------------------------------------
def _new_conversion_stats() -> Counter:
    return Counter(
        total_annotations=0,
        polygon_annotations=0,
        rle_annotations=0,
        multi_polygon_annotations=0,
        converted_annotations=0,
        skipped_invalid_annotations=0,
        coco_input_instances=0,
        yolo_output_instances=0,
    )


def _record_invalid(stats: Counter, ann: dict, reason: str) -> None:
    stats["skipped_invalid_annotations"] += 1
    stats.setdefault("skip_reasons", Counter())[reason] += 1
    print(f"  [invalid segmentation] annotation_id={ann.get('id')}: {reason}")


def _segmentation_kind(segmentation) -> str:
    if isinstance(segmentation, dict):
        return "rle"
    if isinstance(segmentation, list):
        return "polygon"
    return "invalid"


def _normalise_polygon(points, img_w: int, img_h: int) -> list[str] | None:
    if not isinstance(points, (list, tuple)) or len(points) < 6 or len(points) % 2:
        return None

    normalised = []
    for x, y in zip(points[::2], points[1::2]):
        if isinstance(x, bool) or isinstance(y, bool):
            return None
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        normalised.extend((f"{min(max(x / img_w, 0), 1):.6f}", f"{min(max(y / img_h, 0), 1):.6f}"))
    return normalised


def _decode_coco_rle(segmentation: dict, img_w: int, img_h: int):
    """Decode COCO's uncompressed or compressed RLE into a binary mask."""
    import numpy as np

    size = segmentation.get("size")
    counts = segmentation.get("counts")
    if size != [img_h, img_w] or counts is None:
        raise ValueError("RLE size/counts do not match the annotated image")

    if isinstance(counts, str):
        decoded_counts = []
        position = 0
        while position < len(counts):
            value = 0
            shift = 0
            more = True
            while more:
                if position >= len(counts):
                    raise ValueError("truncated compressed RLE counts")
                code = ord(counts[position]) - 48
                if code < 0 or code > 63:
                    raise ValueError("invalid compressed RLE character")
                position += 1
                value |= (code & 0x1F) << shift
                more = bool(code & 0x20)
                shift += 5
                if not more and code & 0x10:
                    value |= -1 << shift
            if len(decoded_counts) > 2:
                value += decoded_counts[-2]
            decoded_counts.append(value)
        counts = decoded_counts

    if not isinstance(counts, list) or any(not isinstance(count, int) or count < 0 for count in counts):
        raise ValueError("RLE counts must be non-negative integers")

    mask = np.zeros(img_h * img_w, dtype=np.uint8)
    index = 0
    value = 0
    for count in counts:
        next_index = index + count
        if next_index > mask.size:
            raise ValueError("RLE counts exceed mask size")
        if value:
            mask[index:next_index] = 1
        index = next_index
        value = 1 - value
    if index != mask.size:
        raise ValueError("RLE counts do not cover the full mask")
    return mask.reshape((img_h, img_w), order="F")


def _mask_to_single_polygon(mask, img_w: int, img_h: int) -> tuple[list[str] | None, str | None]:
    import cv2

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None and any(parent >= 0 for parent in hierarchy[0, :, 3]):
        return None, "mask contains holes, which YOLO polygon labels cannot represent safely"
    if len(contours) != 1:
        return None, "mask contains multiple disconnected components; one COCO annotation must remain one YOLO instance"
    polygon = contours[0].reshape(-1, 2).flatten().tolist()
    normalised = _normalise_polygon(polygon, img_w, img_h)
    if normalised is None:
        return None, "mask contour is not a valid polygon"
    return normalised, None


def coco_ann_to_yolo_lines(
    anns,
    img_w,
    img_h,
    class_agnostic=CLASS_AGNOSTIC,
    class_map=None,
    conversion_stats: Counter | None = None,
):
    """Convert each safely representable COCO annotation into exactly one YOLO instance line."""
    stats = conversion_stats if conversion_stats is not None else _new_conversion_stats()
    lines = []
    for ann in anns:
        stats["total_annotations"] += 1
        stats["coco_input_instances"] += 1
        segmentation = ann.get("segmentation")
        kind = _segmentation_kind(segmentation)
        if kind == "invalid":
            _record_invalid(stats, ann, "missing or unsupported segmentation type")
            continue
        if kind == "rle":
            stats["rle_annotations"] += 1
            try:
                mask = _decode_coco_rle(segmentation, img_w, img_h)
                polygon, reason = _mask_to_single_polygon(mask, img_w, img_h)
            except (ImportError, ValueError) as exc:
                polygon, reason = None, f"RLE conversion failed: {exc}"
        else:
            stats["polygon_annotations"] += 1
            if not segmentation:
                polygon, reason = None, "empty polygon segmentation"
            elif len(segmentation) == 1:
                polygon, reason = _normalise_polygon(segmentation[0], img_w, img_h), None
                if polygon is None:
                    reason = "invalid polygon coordinates"
            else:
                stats["multi_polygon_annotations"] += 1
                try:
                    import cv2
                    import numpy as np

                    mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    polygons = []
                    for component in segmentation:
                        if _normalise_polygon(component, img_w, img_h) is None:
                            raise ValueError("multi-polygon contains invalid coordinates")
                        polygons.append(np.asarray(component, dtype=np.float32).reshape(-1, 2).round().astype(np.int32))
                    cv2.fillPoly(mask, polygons, 1)
                    polygon, reason = _mask_to_single_polygon(mask, img_w, img_h)
                except (ImportError, ValueError) as exc:
                    polygon, reason = None, f"multi-polygon conversion failed: {exc}"

        if polygon is None:
            _record_invalid(stats, ann, reason or "unknown conversion failure")
            continue
        class_id = 0 if class_agnostic else class_map[ann["category_id"]]
        lines.append(f"{class_id} " + " ".join(polygon))
        stats["converted_annotations"] += 1
        stats["yolo_output_instances"] += 1
    return lines


# ----------------------------------------------------------------------------
# 4. BUILD FOLDER STRUCTURE + WRITE ORIGINAL (unaugmented) SPLITS
# ----------------------------------------------------------------------------
def build_splits(images, anns_by_image, categories, splits):
    class_map = {cid: idx for idx, cid in enumerate(sorted(categories.keys()))}
    conversion_stats = _new_conversion_stats()

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
                conversion_stats=conversion_stats,
            )
            (lbl_out / (Path(img_info["file_name"]).stem + ".txt")).write_text(
                "\n".join(lines), encoding="utf-8"
            )

    conversion_report = dict(conversion_stats)
    conversion_report["skip_reasons"] = dict(conversion_stats.get("skip_reasons", {}))
    conversion_report["instance_count_matches"] = (
        conversion_stats["coco_input_instances"] == conversion_stats["yolo_output_instances"]
    )
    conversion_report["instance_count_note"] = (
        "Each converted COCO annotation produced exactly one YOLO instance."
        if conversion_report["instance_count_matches"]
        else "Counts differ because unsafe annotations were rejected; see skip_reasons."
    )
    conversion_report["split_strategy"] = getattr(splits, "split_strategy", None)
    report_path = OUTPUT_DIR / "conversion_stats.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(conversion_report, handle, indent=2, ensure_ascii=False)

    print(f"Conversion statistics -> {report_path}")
    for key, value in conversion_report.items():
        if key != "skip_reasons":
            print(f"  {key}: {value}")
    if conversion_stats["skipped_invalid_annotations"]:
        raise ValueError(
            "COCO-to-YOLO conversion rejected unsafe annotations; see "
            f"{report_path} for skip reasons."
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
