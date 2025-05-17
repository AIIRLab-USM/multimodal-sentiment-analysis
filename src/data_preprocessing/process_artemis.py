import os
import unicodedata
import pandas as pd
from tqdm import tqdm
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
ARTEMIS_PATH = os.path.join( 'data', 'original_data', 'datasets', 'artemis_dataset_release_v0.csv')
CONTRASTIVE_PATH = os.path.join( 'data', 'original_data', 'datasets', 'Contrastive.csv')

OUTPUT_FILE = os.path.join('data', 'datasets', 'custom_artemis.csv')
BALANCED_OUTPUT_FILE = os.path.join('data', 'datasets', 'balanced_artemis.csv')

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

    # Scrap 'utterance'
    artemis_df = artemis_df[["art_style", "painting", "emotion"]].copy()

    # Generate image paths for temporary use
    artemis_df['local_image_path']  = artemis_df.progress_apply(
        lambda row: unicodedata.normalize('NFC',
                        os.path.join("wikiart", row["art_style"], row["painting"] + ".jpg")
                    ), axis=1
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
    artemis_df = artemis_df.rename(columns={'emotion':'labels', 'local_image_path':'local_image_path'})

    # Label train, eval, test
    train_df, temp_df = train_test_split(artemis_df, test_size=(1-TRAIN_RATIO), random_state=42)
    eval_df, test_df = train_test_split(temp_df, test_size=(TEST_RATIO / (EVAL_RATIO + TEST_RATIO)), random_state=42)

    train_df["split"] = "train"
    eval_df["split"] = "eval"
    test_df["split"] = "test"

    artemis_df = pd.concat([train_df, eval_df, test_df])

    # Reorder columns (Author's personal preference), remove "painting"
    artemis_df = artemis_df[["local_image_path", "labels", "split"]]

    # Save generated data to a new CSV file
    try:
        artemis_df.to_csv(OUTPUT_FILE, encoding='utf-8', index=False)
        print(f"Standard dataset saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving {OUTPUT_FILE} : {e}")

    # Create balanced dataset from the filtered dataset
    # (Samples with label that accounts for >=50% label volume)

    # Find count of least frequent class
    min_samples = artemis_df['labels'].value_counts().min()

    # Add numeric check for least frequent label
    print(f"Least frequent emotion has {min_samples} samples")

    # Create balanced dataset with equal samples per class
    balanced_df = pd.DataFrame()
    for emotion in LABEL_MAP.keys():

        # Get samples for this emotion
        emotion_samples = artemis_df[artemis_df['labels'] == emotion].copy()

        # Sample randomly instead of taking the most confident ones
        # If we have fewer samples than min_samples, take all of them
        if len(emotion_samples) <= min_samples:
            sampled_emotion = emotion_samples

        else:
            sampled_emotion = emotion_samples.sample(n=min_samples, random_state=42)

        # Add to balanced dataframe
        balanced_df = pd.concat([balanced_df, sampled_emotion])

    # Label train, eval, test
    train_df, temp_df = train_test_split(balanced_df, test_size=(1 - TRAIN_RATIO), random_state=42)
    eval_df, test_df = train_test_split(temp_df, test_size=(TEST_RATIO / (EVAL_RATIO + TEST_RATIO)), random_state=42)

    train_df["split"] = "train"
    eval_df["split"] = "eval"
    test_df["split"] = "test"
    balanced_df = pd.concat([train_df, eval_df, test_df])

    # Reorder columns (Author's preference)
    balanced_df = balanced_df[["local_image_path", "labels", "split"]]

    # Print class distribution stats
    print("\nOriginal dataset class distribution:")
    print(artemis_df['labels'].value_counts())

    print("\nBalanced dataset class distribution:")
    print(balanced_df['labels'].value_counts())

    # Save balanced dataset
    try:
        balanced_df.to_csv(BALANCED_OUTPUT_FILE, encoding='utf-8', index=False)
        print(f"Balanced dataset saved to {BALANCED_OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving {BALANCED_OUTPUT_FILE} : {e}")


if __name__ == '__main__':
    main()