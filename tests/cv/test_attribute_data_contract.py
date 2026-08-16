"""Regression tests cho schema va anti-leakage cua split attribute."""

import pandas as pd
import pytest

from pill_safety.cv.attribute.training.data_contract import COLOR_COLUMNS, source_image_group, validate_attribute_data


def test_source_image_group_removes_side_and_offline_variant():
    """Variant AUG/ADV cua hai mat cung anh goc phai co mot group duy nhat."""
    assert source_image_group("00002-3228-30_RXNAVIMAGE10_391E1C80_1_AUG_2.jpg") == "00002-3228-30_RXNAVIMAGE10_391E1C80"
    assert source_image_group("00002-3228-30_RXNAVIMAGE10_391E1C80_2.jpg") == "00002-3228-30_RXNAVIMAGE10_391E1C80"


def test_validator_rejects_synthetic_validation_sample(tmp_path):
    """Val/test khong duoc chua offline augmentation tu train."""
    paths = {"shape_image_dir": tmp_path, "color_image_dir": tmp_path}
    for split in ("train", "val", "test"):
        shape_path = tmp_path / f"shape_{split}.csv"
        color_path = tmp_path / f"color_{split}.csv"
        pd.DataFrame({"rximageFileName": [f"shape_{split}.jpg"], "shape": ["ROUND"], "label_shape": [4]}).to_csv(shape_path, index=False)
        pd.DataFrame({"rximageFileName": [f"color_{split}.jpg"], "is_synthetic": [0], **{column: [0] for column in COLOR_COLUMNS}}).to_csv(color_path, index=False)
        paths[f"shape_{split}_csv"] = shape_path
        paths[f"color_{split}_csv"] = color_path
    pd.DataFrame({"rximageFileName": ["color_2_AUG_1.jpg"], "is_synthetic": [1], **{column: [0] for column in COLOR_COLUMNS}}).to_csv(paths["color_val_csv"], index=False)
    with pytest.raises(ValueError, match="val contains synthetic samples"):
        validate_attribute_data(paths, verify_images=False)
