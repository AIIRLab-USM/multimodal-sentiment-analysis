import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

"""
A short script for creating visual data using model evaluation data

Author: Clayton Durepos
Version: 07.17.2025
Contact: clayton.durepos@maine.edu
"""

result_types = ["text",
                "image",
                "multimodal"
                ]
label_map = {
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

def artemis_distribution_chart(label_counts, file_name):
    plt.figure(figsize=(10, 6))
    plt.bar(label_map.keys(), label_counts)
    plt.xticks(rotation=45)
    plt.title("Class Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(os.path.join('data', 'plot', 'class_distribution.png'))


def main():
    os.makedirs(f'data{os.path.sep}plot', exist_ok=True)

    # Total class distribution chart
    df = pd.read_csv( os.path.join('data', 'datasets', 'multimodal_sentiment_dataset.csv') )
    label_counts = df['ground_truth'].value_counts().sort_index()
    artemis_distribution_chart(label_counts, os.path.join('data', 'plot', 'global_class_distribution.png'))

    for split in ['train', 'eval', 'test']:
        curr_df = df.loc[df['split'] == split]
        label_counts = curr_df['ground_truth'].value_counts().sort_index()
        artemis_distribution_chart(label_counts, os.path.join('data', 'plot', f'{split}_class_distribution.png'))

    # Confusion matrices for each modality
    for result_type in result_types:
        curr_data_path = os.path.join('data', 'evaluation', result_type)
        result_df = pd.read_csv(f'{curr_data_path}_results.csv')

        y_true = result_df['true_label']
        y_pred = result_df['pred_label']

        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Reds",
                    xticklabels=list( label_map.keys() ),
                    yticklabels=list( label_map.keys() ))

        plt.xlabel("Prediction")
        plt.ylabel("Ground Truth")
        plt.title(f"{ result_type.capitalize() } Confusion Matrix")
        plt.tight_layout()
        plt.savefig( os.path.join('data', 'plot', f'{result_type}_matrix_2.png') )

if __name__ == "__main__":
    main()

