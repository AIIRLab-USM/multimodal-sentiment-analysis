import os
import unicodedata
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from collections import Counter
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
Version: 07.17.2025
Contact: clayton.durepos@maine.edu
"""


# Default ArtEmis dataset filename
ARTEMIS_PATH = os.path.join( 'data', 'datasets', 'original_data', 'artemis_dataset_release_v0.csv')
CONTRASTIVE_PATH = os.path.join( 'data', 'datasets', 'original_data', 'Contrastive.csv')

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

# ... previous imports and constants remain the same ...

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

    # Scrap 'utterance'
    artemis_df = artemis_df[["art_style", "painting", "emotion"]].copy()

    # Generate image paths for temporary use
    artemis_df['local_image_path']  = artemis_df.progress_apply(
        lambda row: unicodedata.normalize('NFC',
                        os.path.join("wikiart", row["art_style"], row["painting"] + ".jpg")
                    ), axis=1
    )

    # Map emotion labels to index
    artemis_df['ground_truth'] = artemis_df['emotion'].map(LABEL_MAP)

    grouped = artemis_df.groupby(['local_image_path'])
    def compute_soft_label_vec(group):
        counts = np.zeros(len(LABEL_MAP), dtype=np.float32)
        for idx in group['ground_truth']:
            counts[idx] += 1
        return counts / counts.sum()


    soft_label_df = grouped.apply(lambda g: pd.Series({
        'labels': compute_soft_label_vec(g).tolist()
    })).reset_index()

    artemis_df = artemis_df.merge(soft_label_df, on=['local_image_path'], how='left')

    # Label train, eval, test
    train_df, temp_df = train_test_split(artemis_df, test_size=(1-TRAIN_RATIO), random_state=42)
    eval_df, test_df = train_test_split(temp_df, test_size=(TEST_RATIO / (EVAL_RATIO + TEST_RATIO)), random_state=42)

    train_df["split"] = "train"
    eval_df["split"] = "eval"
    test_df["split"] = "test"

    full_df = pd.concat([train_df, eval_df, test_df])

    # Compute dominant labels and filter eval/test samples
    group_stats = (
        full_df.groupby(['local_image_path'])['ground_truth']
        .agg(lambda x: (x.value_counts().idxmax(), x.value_counts().max() / len(x)))
        .apply(pd.Series)
    )
    group_stats.columns = ['dominant_label', 'confidence']
    group_stats = group_stats[group_stats['confidence'] > 0.5]
    group_stats.reset_index(inplace=True)

    # Merge to get confident eval/test samples
    full_df = full_df.merge(group_stats, on='local_image_path', how='left')

    # Keep all train samples + confident eval/test
    filtered_df = full_df[
        (full_df['split'] == 'train') |
        ((full_df['split'] != 'train') & (full_df['ground_truth'] == full_df['dominant_label']))
    ]

    final_df = filtered_df[["local_image_path", "split", "labels", "ground_truth"]]

    try:
        final_df.to_csv(OUTPUT_FILE, encoding='utf-8', index=False)
    except Exception as e:
        print(f"Error saving {OUTPUT_FILE} : {e}")

if __name__ == '__main__':
    main()