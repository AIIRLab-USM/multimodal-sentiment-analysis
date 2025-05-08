import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
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
    3. Save generated dataset to a new CSV file
    
Author: Clayton Durepos
Version: 04.11.2025
Contact: clayton.durepos@maine.edu
"""


# Default ArtEmis dataset filename
ARTEMIS_PATH = os.path.join( 'data', 'datasets', 'artemis_dataset_release_v0.csv')
CONTRASTIVE_PATH = os.path.join( 'data', 'datasets', 'Contrastive.csv')

OUTPUT_FILE = os.path.join('data', 'datasets', 'custom_artemis.csv')

LABEL_MAP = {
    'amusement': 0,
    'anger': 1,
    'awe': 2,
    'contentment': 3,
    'disgust': 4,
    'excitement': 5,
    'fear': 6,
    'sadness': 7,
    'something else': 8
}

TRAIN_RATIO = 0.8
EVAL_RATIO = 0.1
TEST_RATIO = 0.1

tqdm.pandas()

def main():
    """
    Process ArtEmis dataset:
    1. Read original CSV
    2. Extract painting and emotion columns
    3. Create the local image paths
    4. Save to a new CSV
    """

    # Check if ArtEmis dataset is present
    if not os.path.exists(ARTEMIS_PATH):
        raise FileNotFoundError("Missing ArtEmis data"
                                " - Refer to https://www.artemisdataset.org/#dataset")
    try:
        artemis_df = pd.read_csv(ARTEMIS_PATH)
    except Exception as e:
        print(f"Error reading {ARTEMIS_PATH} : {e}")
        return

    # Check if ArtEmis V2.0 data is present
    if not os.path.exists(CONTRASTIVE_PATH):
        raise FileNotFoundError("Missing ArtEmis V2.0 data"
                                " - Refer to https://www.artemisdataset-v2.org/")
    try:
        contrastive_df = pd.read_csv(CONTRASTIVE_PATH)
    except Exception as e:
        print(f"Error reading {CONTRASTIVE_PATH} : {e}")
        return

    # Concat data for ArtEmis V2.0 DataFrame
    artemis_df = pd.concat([artemis_df, contrastive_df])

    # Scrap 'utterance'
    artemis_df = artemis_df[["art_style", "painting", "emotion"]].copy()

    # Generate image paths for temporary use
    artemis_df['local_image_path']  = artemis_df.progress_apply(
        lambda row: os.path.join("wikiart", row["art_style"], row["painting"] + ".jpg"), axis=1
    )

    # Count number of labels per emotion for each painting
    label_count = artemis_df.groupby(['painting', 'emotion']).size().reset_index(name='count')

    # Get total amount of labels per image
    total_counts = label_count.groupby('painting')['count'].sum().reset_index(name='total_count')

    # Merge
    label_count = label_count.merge(total_counts, on='painting')

    # Filter to only retain samples where dominant emotion accounts for >=50% of labels
    label_count.sort_values(by=['painting', 'count'], ascending=[True, False], inplace=True)
    dominant_emotions = label_count.groupby('painting').first().reset_index()
    dominant_emotions = dominant_emotions[
        dominant_emotions['count'] / dominant_emotions['total_count'] >= 0.5
    ]

    # Merge with original dataframe
    artemis_df = artemis_df.merge(dominant_emotions[['painting', 'emotion']], on=['painting', 'emotion'])
    artemis_df = artemis_df.drop_duplicates(subset=['painting'])

    # Customized DF
    new = artemis_df[["emotion","local_image_path"]].copy()
    new = new.rename(columns={'emotion':'labels', 'local_image_path':'local_image_path'})

    # Label train, eval, test
    train_df, temp_df = train_test_split(new, test_size=(1-TRAIN_RATIO), random_state=42)
    eval_df, test_df = train_test_split(temp_df, test_size=(TEST_RATIO / (EVAL_RATIO + TEST_RATIO)), random_state=42)

    train_df["split"] = "train"
    eval_df["split"] = "eval"
    test_df["split"] = "test"

    new = pd.concat([train_df, eval_df, test_df])

    # Reorder columns (Author's personal preference), remove "painting"
    new = new[["local_image_path", "split", "labels"]]

    # Save generated data to a new CSV file
    try:
        new.to_csv(OUTPUT_FILE, index=False)

    except Exception as e:
        print(f"Error saving {OUTPUT_FILE} : {e}")

if __name__ == '__main__':
    main()