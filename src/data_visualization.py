import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

result_types = ["text", "image", "multimodal"]
training_types = ["", "bal-training_"]

dataset_paths = [
    os.path.join('data', 'datasets', 'multimodal_sentiment_dataset.csv'),
    os.path.join('data', 'datasets', 'bal_multimodal_sentiment_dataset.csv')
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

def main():
    os.makedirs(f'data{os.path.sep}plot', exist_ok=True)

    for i, path in enumerate(dataset_paths):
        # Class distribution chart
        df = pd.read_csv( path )
        label_counts = df['labels'].value_counts().sort_index()

        plt.figure(figsize=(10, 6))
        plt.bar(label_map.keys(), label_counts)
        plt.xticks(rotation=45)
        plt.title("Class Distribution")
        plt.xlabel("Emotion")
        plt.ylabel("Number of Samples")
        plt.tight_layout()
        plt.savefig(os.path.join('data', 'plot', 'class_distribution.png' \
                                 if i == 0 else 'balanced_class_distribution.png'))

    # Confusion matrices for each modality
    for result_type in result_types:
        for training_type in training_types:
            curr_data_path = os.path.join('data', 'evaluation', f'{training_type}{result_type}')
            result_df = pd.read_csv(f'{curr_data_path}-results.csv')

            y_true = result_df['labels']
            y_pred = result_df['prediction']

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
            plt.savefig( os.path.join('data', 'plot', f'{training_type}{result_type}-matrix.png') )

if __name__ == "__main__":
    main()

