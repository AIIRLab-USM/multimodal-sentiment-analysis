import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

result_types = ["text", "image", "multimodal"]
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
    for result_type in result_types:
        curr_dir = f'data{os.path.sep}plot{os.path.sep}{result_type}'
        os.makedirs(curr_dir, exist_ok=True)

        curr_data_path = os.path.join('data', 'evaluation', result_type)
        result_df = pd.read_csv(f'{curr_data_path}_results.csv')

        # Heatmap for confusion matrix
        inverse_label_map = {v: k for k, v in label_map.items()}

        y_true = result_df['labels']
        y_pred = result_df['prediction']

        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Reds",
                    xticklabels=[inverse_label_map[i] for i in range(len(label_map))],
                    yticklabels=[inverse_label_map[i] for i in range(len(label_map))])

        plt.xlabel("Prediction")
        plt.ylabel("Ground Truth")
        plt.title("Normalized Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{curr_dir}{os.path.sep}confusion_matrix.png")

if __name__ == "__main__":
    main()

