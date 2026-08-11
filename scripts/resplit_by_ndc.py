import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import sys

def main():
    parser = argparse.ArgumentParser(description="Resplit attribute dataset strictly by NDC11 to prevent leakage.")
    parser.add_argument("--data_dir", type=str, default="data/splits/nih_attribute", help="Directory containing original CSVs.")
    parser.add_argument("--out_dir", type=str, default="data/splits/nih_attribute", help="Output directory.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all data
    csv_files = ["train_combined_crop.csv", "val_combined_crop.csv", "test_combined_crop.csv"]
    dfs = []
    for f in csv_files:
        path = data_dir / f
        if path.exists():
            dfs.append(pd.read_csv(path))
        else:
            print(f"Warning: {path} not found.")

    if not dfs:
        print("Error: No input CSV files found!")
        sys.exit(1)

    df_all = pd.concat(dfs, ignore_index=True)
    
    # Drop duplicates if any image exists in multiple splits (which causes leakage)
    df_all = df_all.drop_duplicates(subset=["rxnavImageFileName"])

    # 2. Extract NDC11 group key
    # Filename format: "33342-0031-10_RXNAVIMAGE10_FD3E7E93_1.jpg" -> "33342-0031-10"
    df_all["NDC11"] = df_all["rxnavImageFileName"].apply(lambda x: x.split("_")[0])

    print(f"Total images: {len(df_all)}")
    print(f"Total unique NDCs: {df_all['NDC11'].nunique()}")

    # 3. Split 1: Train+Val (85%) vs Test (15%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.random_state)
    train_val_idx, test_idx = next(gss1.split(df_all, groups=df_all["NDC11"]))
    
    df_train_val = df_all.iloc[train_val_idx].reset_index(drop=True)
    df_test = df_all.iloc[test_idx].reset_index(drop=True)

    # 4. Split 2: Train (70/85 ≈ 82.35%) vs Val (15/85 ≈ 17.65%)
    # test_size here is relative to the remaining 85%
    test_size_val = 15 / 85
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_size_val, random_state=args.random_state)
    train_idx, val_idx = next(gss2.split(df_train_val, groups=df_train_val["NDC11"]))
    
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    # Print summary
    print("\nSplit Results (Images):")
    print(f"  Train: {len(df_train)} ({len(df_train)/len(df_all)*100:.1f}%)")
    print(f"  Val:   {len(df_val)} ({len(df_val)/len(df_all)*100:.1f}%)")
    print(f"  Test:  {len(df_test)} ({len(df_test)/len(df_all)*100:.1f}%)")
    
    print("\nSplit Results (Unique NDCs):")
    print(f"  Train: {df_train['NDC11'].nunique()}")
    print(f"  Val:   {df_val['NDC11'].nunique()}")
    print(f"  Test:  {df_test['NDC11'].nunique()}")

    # 5. Save clean CSVs
    df_train.to_csv(out_dir / "train_clean.csv", index=False)
    df_val.to_csv(out_dir / "val_clean.csv", index=False)
    df_test.to_csv(out_dir / "test_clean.csv", index=False)
    
    print("\nSaved train_clean.csv, val_clean.csv, test_clean.csv successfully!")

if __name__ == "__main__":
    main()
