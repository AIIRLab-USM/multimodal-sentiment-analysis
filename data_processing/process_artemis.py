import os
import unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split

"""
ArtEmis Dataset Processor

This script processes the ArtEmis dataset as described in:
Achlioptas, P., Ovsjankiov, M., Haydarov, K., Elhoseiny, M., & Guibas, L. (2021).
ArtEmis: Affective Language for Visual Art.
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Original dataset available at: https://artemisdataset.org/#dataset

This Python script processes the ArtEmis dataset by:
    1. Extracting relevant columns
    2. Generate local file paths for associated WikiArt images
    3. Save generated CapIma dataset to a new CSV file
    
Author: Clayton Durepos
Contact: clayton.durepos@maine.edu
"""

# Default ArtEmis dataset filename
INPUT_FILE = f"..{os.path.sep}original_data{os.path.sep}artemis_dataset_release_v0.csv"

OUTPUT_FILE = f"..{os.path.sep}processed_data{os.path.sep}custom_artemis.csv"

TRAIN_RATIO = 0.7
EVAL_RATIO = 0.2
TEST_RATIO = 0.1

def main():
    """
    Process ArtEmis dataset:
    1. Read original CSV
    2. Extract painting and emotion columns
    3. Create the local image paths
    4. Save to a new CSV
    """

    # Check if ArtEmis dataset is present
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError("Missing ArtEmis data"
                                " - Refer to https://www.artemisdataset.org/#dataset")

    try:
        artemis_df = pd.read_csv(INPUT_FILE)

    except Exception as e:
        print(f"Error reading {INPUT_FILE} : {e}")
        return

    # Extract necessary data to new DataFrame
    new = artemis_df[["painting", "emotion"]].copy()

    # Generate local image paths of WikiArt pieces
    new["local_image_path"] = artemis_df.apply(
        lambda row: os.path.join("wikiart", row["art_style"], row["painting"] + ".jpg"), axis=1
    )

    # Label train, eval, test
    train_df, temp_df = train_test_split(new, test_size=(1-TRAIN_RATIO), random_state=42)
    eval_df, test_df = train_test_split(temp_df, test_size=(TEST_RATIO / (EVAL_RATIO + TEST_RATIO)), random_state=42)

    train_df["split"] = "train"
    eval_df["split"] = "eval"
    test_df["split"] = "test"

    new = pd.concat([train_df, eval_df, test_df])

    # Reorder columns (Author's personal preference), remove "painting"
    new = new[["local_image_path", "split", "emotion"]]

    # Save generated data to a new CSV file
    try:
        new.to_csv(OUTPUT_FILE, index=False)

    except Exception as e:
        print(f"Error saving {OUTPUT_FILE} : {e}")

if __name__ == '__main__':
    main()