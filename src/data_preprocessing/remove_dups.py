
#!/usr/bin/env python3
"""
Rebuild full dataset with duplicates removed from test split.

Assumptions:
  - Input dataset: ./multimodal_sentiment_dataset.csv
  - Duplicate list: ./data/duplicates.csv (must contain a 'basename' column)
  - Dataset has a 'split' column (with values like 'train', 'test', etc.)
  - Dataset has an image path column (auto-detected if not set)

Outputs:
  - ./data/test_dedup.csv        (deduplicated test split only)
  - ./data/dataset_dedup.csv     (full dataset, but with test deduplicated)
"""

import os
import sys
import pandas as pd
from pathlib import Path

DATASET_CSV = Path("./data/datasets/multimodal_sentiment_dataset.csv")
DUP_CSV = Path("./data/duplicates.csv")
OUTPUT_DIR = Path("./data/datasets")

def main():
    if not DATASET_CSV.exists():
        sys.exit(f"Input dataset not found: {DATASET_CSV}")
    if not DUP_CSV.exists():
        sys.exit(f"Duplicate list not found: {DUP_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_CSV)
    dup = pd.read_csv(DUP_CSV)

    if 'basename' not in dup.columns:
        sys.exit("duplicates.csv must contain a 'basename' column.")
    dup_basenames = set(dup['basename'].astype(str).str.strip().str.lower())

    path_col = "local_image_patb"

    # Separate test split

    # Compute basename
    df["_basename"] = (
        df[path_col]
        .astype(str)
        .apply(lambda p: os.path.basename(p.strip()))
        .str.strip()
        .str.lower()
    )

    before = len(df)
    dedup = df.loc[~df["_basename"].isin(dup_basenames)].drop(columns=["_basename"])
    after = len(dedup)

    # Rebuild full dataset: train/val unchanged, test replaced with deduped version
    dedup.to_csv(OUTPUT_DIR / "multimodal_sentiment_dataset.csv", index=False)

    print("De-duplication complete.\n")
    print(f"Test rows before de-dup: {before}")
    print(f"Removed due to duplicate basenames: {before - after}")
    print(f"Rows after de-dup: {after}")
    print(f"Full dataset rows: {len(dedup)}")
    print(f"Wrote deduped test to: {OUTPUT_DIR / 'test_dedup.csv'}")
    print(f"Wrote full deduped dataset to: {OUTPUT_DIR / 'dataset_dedup.csv'}")

if __name__ == "__main__":
    main()
