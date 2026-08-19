import json
import pandas as pd
from pathlib import Path

def create_and_save_label_mapping(train_csv_path: Path, color_cols: list, output_json_path: Path):
    shape_df = pd.read_csv(train_csv_path)
    shape_map = shape_df.groupby("shape_label")["shape"].first().to_dict()
    
    label_mapping = {
        "shape": {str(k): str(v) for k, v in shape_map.items()},
        "color": [c.replace("color_", "") for c in color_cols]
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
    
    return label_mapping