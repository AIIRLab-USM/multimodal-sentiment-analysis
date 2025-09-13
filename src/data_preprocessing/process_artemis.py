import os
import unicodedata
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

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
Version: 09.12.2025
Contact: clayton.durepos@maine.edu
"""


# Default ArtEmis dataset filename
ARTEMIS_PATH = os.path.join( 'data', 'datasets', 'original_data', 'artemis_dataset_release_v0.csv')
CONTRASTIVE_PATH = os.path.join( 'data', 'datasets', 'original_data', 'Contrastive.csv')

OUTPUT_FILE = os.path.join('data', 'datasets', 'artemis-temp.csv')

DUP_CSV = os.path.join('data', 'duplicates.csv')

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
        artemis_df = pd.read_csv(ARTEMIS_PATH, encoding='utf-8')
    except Exception as e:
        print(f"Error reading {ARTEMIS_PATH} : {e}")
        return

    # Check if ArtEmis V2.0 data is present
    if not os.path.exists(CONTRASTIVE_PATH):
        raise FileNotFoundError("Missing ArtEmis V2.0 data"
                                " - Refer to https://www.artemisdataset-v2.org/")
    try:
        contrastive_df = pd.read_csv(CONTRASTIVE_PATH, encoding='utf-8')
    except Exception as e:
        print(f"Error reading {CONTRASTIVE_PATH} : {e}")
        return

    # Concat data for ArtEmis V2.0 DataFrame
    artemis_df = pd.concat([artemis_df, contrastive_df])

    # There exist unicode mismatches between ArtEmis V1.0 and V2.0 - get rid of these to ensure consistent IDs
    artemis_df['painting'] = artemis_df['painting'].apply(
        lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8')
    )

    # Scrap 'utterance'
    artemis_df = artemis_df[["art_style", "painting", "emotion"]].copy()

    # Map emotion labels to index
    artemis_df['label'] = artemis_df['emotion'].map(LABEL_MAP)

    grouped = artemis_df.groupby(['painting'])
    def compute_soft_label_vec(group):
        counts = np.zeros(len(LABEL_MAP), dtype=np.float32)
        for idx in group['label']:
            counts[idx] += 1
        return counts / counts.sum()

    soft_label_df = grouped.apply(lambda g: pd.Series({
        'probs': compute_soft_label_vec(g).tolist()
    })).reset_index()

    artemis_df = artemis_df.merge(soft_label_df, on=['painting'], how='left')

    # Label train, eval, test
    train_df, temp_df = train_test_split(
        artemis_df,
        test_size=(1-TRAIN_RATIO),
        random_state=42,
        stratify=artemis_df["label"]
    )

    eval_df, test_df = train_test_split(
        temp_df,
        test_size=(TEST_RATIO / (EVAL_RATIO + TEST_RATIO)),
        random_state=42,
        stratify=temp_df["label"]
    )

    train_df["split"] = "train"
    eval_df["split"] = "eval"
    test_df["split"] = "test"

    artemis_df = pd.concat([train_df, eval_df, test_df])

    # Reserve only confident labels in test and evaluation
    artemis_df["dominant_label"] = artemis_df["probs"].apply(lambda p: int(np.argmax(p)))
    artemis_df["confidence"] = artemis_df["probs"].apply(lambda p: float(np.max(p)))

    # Keep all train samples + confident eval/test
    artemis_df = artemis_df[
        (artemis_df["split"] == "train") |
        (
                (artemis_df["split"] != "train")
                & (artemis_df["label"] == artemis_df["dominant_label"])
                & (artemis_df["confidence"] > 0.5)
        )
    ]

    artemis_df = artemis_df[["art_style", "painting", "probs", "split"]]

    try:
        artemis_df.to_csv(OUTPUT_FILE, encoding='utf-8', index=False)
    except Exception as e:
        print(f"Error saving {OUTPUT_FILE} : {e}")

if __name__ == '__main__':
    main()