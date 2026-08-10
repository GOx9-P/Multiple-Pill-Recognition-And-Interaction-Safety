import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from pill_safety.cv.attribute.utils.leakage import check_split_leakage
from pill_safety.cv.attribute.labels.label_mapping import build_label_mapping
from pill_safety.cv.attribute.utils.checkpoint import save_checkpoint, load_checkpoint
from pill_safety.cv.attribute.utils.config import AttributeConfig

def test_leakage_passed(tmp_path):
    """Test that split leakage returns True when no overlapping groups exist."""
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    
    pd.DataFrame({"rxnav_rxcui": ["A", "B"]}).to_csv(train_csv, index=False)
    pd.DataFrame({"rxnav_rxcui": ["C", "D"]}).to_csv(val_csv, index=False)
    pd.DataFrame({"rxnav_rxcui": ["E", "F"]}).to_csv(test_csv, index=False)
    
    passed, details = check_split_leakage(str(train_csv), str(val_csv), str(test_csv), group_key="rxnav_rxcui")
    assert passed is True, f"Expected passed=True, got passed=False. Details: {details}"
    assert details.get("overlap_train_val", 0) == 0

def test_leakage_failed(tmp_path):
    """Test that split leakage returns False when overlapping groups exist."""
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    
    # "C" is overlapping between train and val
    pd.DataFrame({"rxnav_rxcui": ["A", "B", "C"]}).to_csv(train_csv, index=False)
    pd.DataFrame({"rxnav_rxcui": ["C", "D"]}).to_csv(val_csv, index=False)
    pd.DataFrame({"rxnav_rxcui": ["E", "F"]}).to_csv(test_csv, index=False)
    
    passed, details = check_split_leakage(str(train_csv), str(val_csv), str(test_csv), group_key="rxnav_rxcui")
    assert passed is False
    assert details.get("overlap_train_val", 0) > 0

def test_leakage_keyerror(tmp_path):
    """Test that a KeyError is raised or handled when group_key is missing."""
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    
    pd.DataFrame({"WRONG_COLUMN": ["A", "B"]}).to_csv(train_csv, index=False)
    pd.DataFrame({"WRONG_COLUMN": ["C", "D"]}).to_csv(val_csv, index=False)
    pd.DataFrame({"WRONG_COLUMN": ["E", "F"]}).to_csv(test_csv, index=False)
    
    with pytest.raises(KeyError):
        check_split_leakage(str(train_csv), str(val_csv), str(test_csv), group_key="rxnav_rxcui")


def test_build_label_mapping():
    """Test that build_label_mapping sorts classes consistently."""
    class DummyDataset:
        def __init__(self):
            # df has out-of-order shapes
            self.df = pd.DataFrame({"shape": ["ROUND", "CAPSULE", "OVAL"]})
            # mock color cols
            self.color_cols = ["color_RED", "color_BLUE"]

    train_dataset = DummyDataset()
    mapping, shape_names, color_names = build_label_mapping(train_dataset, 3, 2)
    
    assert shape_names == ["CAPSULE", "OVAL", "ROUND"], "Shapes should be sorted alphabetically."
    assert color_names == ["color_RED", "color_BLUE"], "Colors should match mlb classes."
    assert mapping["shape"] == ["CAPSULE", "OVAL", "ROUND"]


def test_checkpoint_save_load(tmp_path):
    """Test saving and loading checkpoint preserves mapping hash and model weights."""
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    ckpt_path = tmp_path / "test_ckpt.pt"
    dummy_mapping = {"shape": {"A": 0}}
    dummy_hash = "abcdef123456"
    
    save_checkpoint(
        path=ckpt_path,
        epoch=5,
        model=model,
        optimizer=optimizer,
        best_metric=0.85,
        label_mapping=dummy_mapping,
        mapping_hash=dummy_hash
    )
    
    assert ckpt_path.exists()
    
    # Load
    model_loaded = nn.Linear(10, 2)
    optimizer_loaded = optim.SGD(model_loaded.parameters(), lr=0.1)
    
    ckpt_data = load_checkpoint(
        path=ckpt_path,
        device=torch.device("cpu"),
        expected_mapping_hash=dummy_hash
    )
    model_loaded.load_state_dict(ckpt_data["model_state_dict"])
    optimizer_loaded.load_state_dict(ckpt_data["optimizer_state_dict"])
    
    assert ckpt_data["epoch"] == 5
    assert ckpt_data["best_metric"] == 0.85
    assert ckpt_data["label_mapping"] == dummy_mapping
    
    # Check weights match
    for p1, p2 in zip(model.parameters(), model_loaded.parameters()):
        assert torch.allclose(p1, p2)


def test_artifact_paths_no_duplicate_run_id():
    """Test that AttributeConfig generates paths with run_id as leaf folder."""
    paths = AttributeConfig.get_experiment_paths("attribute_resnet18_head_tune", "test_run_123")
    
    pred_path = str(paths["predictions"]).replace("\\", "/")
    
    assert "test_run_123" == pred_path.split("/")[-1], "run_id SHOULD be the leaf folder for predictions."
