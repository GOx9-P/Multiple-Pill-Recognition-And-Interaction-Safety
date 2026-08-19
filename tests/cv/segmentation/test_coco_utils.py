from collections import defaultdict

from src.pill_safety.cv.segmentation.datasets.coco_utils import split_dataset


def test_rare_label_in_temp_uses_deterministic_fallback_without_overlap():
    images = {image_id: {"id": image_id} for image_id in range(1, 11)}
    anns_by_image = defaultdict(list)
    # Every class can participate in the first stratified split, but the
    # three-image temp set has one image per class.
    categories = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3]
    for image_id, category_id in zip(images, categories):
        anns_by_image[image_id].append({"category_id": category_id})

    first = split_dataset(images, anns_by_image, ratios=(0.7, 0.15, 0.15), seed=42)
    second = split_dataset(images, anns_by_image, ratios=(0.7, 0.15, 0.15), seed=42)

    assert first == second
    assert first.split_strategy["fallback_used"] is True
    assert first.split_strategy["fallback_reason"] == "rare_class_in_temp"
    assert first.split_strategy["seed"] == 42
    assert first.split_strategy["rare_classes"]

    train_ids, val_ids, test_ids = (set(first[name]) for name in ("train", "val", "test"))
    assert not train_ids & val_ids
    assert not train_ids & test_ids
    assert not val_ids & test_ids
    assert train_ids | val_ids | test_ids == set(images)
