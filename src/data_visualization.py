import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

"""
A short script for creating visual data using model evaluation data

Author: Clayton Durepos
Version: 08.01.2025
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

def artemis_distribution_chart(title, label_counts, file_name, file_type):
    plt.figure(figsize=(10, 6))
    plt.bar(label_map.keys(), label_counts)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.xlabel("Emotion")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(f'{file_name}.{file_type}', format=file_type)


def main():
    os.makedirs(f'data{os.path.sep}plot', exist_ok=True)
    os.makedirs( os.path.join('data', 'plot', 'distribution'), exist_ok=True)

    # Total class distribution chart
    df = pd.read_csv( os.path.join('data', 'datasets', 'multimodal_sentiment_dataset.csv') )
    label_counts = df['ground_truth'].value_counts().sort_index()
    artemis_distribution_chart(
        "Global Class Distribution",
        label_counts,
        os.path.join('data', 'plot', 'distribution', 'global_class_distribution'),
        file_type='pdf'
    )

    for split in ['train', 'eval', 'test']:
        curr_df = df.loc[df['split'] == split]
        label_counts = curr_df['ground_truth'].value_counts().sort_index()
        artemis_distribution_chart(
            f"{split.capitalize()} Class Distribution",
            label_counts,
            os.path.join('data', 'plot', 'distribution', f'{split}_class_distribution'),
            file_type='pdf'
        )

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
        plt.savefig( os.path.join('data', 'plot', f'{result_type}_matrix.pdf'), format="pdf" )

        # Additional: Attention weight visualization for multimodal model
        if result_type == "multimodal":
            if "image_weight" in result_df.columns and "text_weight" in result_df.columns:
                # Per-emotion average attention weights
                label_names = list(label_map.keys())
                result_df["emotion"] = result_df["true_label"].map(lambda x: label_names[x])

                # Compute per-emotion averages
                avg_weights = result_df.groupby("emotion")[["image_weight", "text_weight"]].mean().reset_index()
                avg_weights = avg_weights.melt(id_vars="emotion", var_name="Modality", value_name="Attention Weight")

                # Plot
                plt.figure(figsize=(10, 6))
                sns.barplot(data=avg_weights, x="emotion", y="Attention Weight", hue="Modality",
                            palette=["skyblue", "salmon"])
                # plt.title("Average Attention Weights per Emotion")
                plt.xlabel("Emotion")
                plt.ylabel("Average Attention Weight")
                plt.xticks(rotation=45)
                plt.ylim(0, 1)
                plt.tight_layout()
                plt.savefig(os.path.join('data', 'plot', 'class_attentions.pdf'), format="pdf")

if __name__ == "__main__":
    main()

