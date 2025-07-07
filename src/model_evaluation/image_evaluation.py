import os
import ast
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from src.classification_models import ImageClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dict_path = f'models{os.path.sep}vit-dict.pt'
data_path = os.path.join('data', 'datasets', 'multimodal_sentiment_dataset.csv')

# Load processor
processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

def round_list(lst):
    return [round(float(x), 3) for x in lst]

# Custom dataset
class ImageProcessingDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        with Image.open(row['local_image_path']) as img:
            inputs = processor(images=img, return_tensors='pt')

        return (
            inputs['pixel_values'].squeeze(0),  # pixel_values
            row['ground_truth'],                # true_labels
            row['labels']                       # true_dists
        )

def main():
    os.makedirs(f'data{os.path.sep}evaluation', exist_ok=True)

    # Load dataset
    df = pd.read_csv(data_path)

    # Parse labels into lists, convert to tensors
    df['labels'] = df['labels'].apply(
        lambda x : torch.tensor( ast.literal_eval (x) )
    )

    # Compute dominant label + confidence
    group_stats = (
        df.groupby(['local_image_path'])['ground_truth']
        .agg(lambda x: (x.value_counts().idxmax(), x.value_counts().max() / len(x)))
        .apply(pd.Series)
    )
    group_stats.columns = ['dominant_label', 'confidence']
    group_stats = group_stats[group_stats['confidence'] >= 0.5].reset_index()

    # Filter to confident test samples
    df = df.merge(group_stats, on=['local_image_path'], how='inner')
    test_df = df[
        (df['split'] == 'test') &
        (df['ground_truth'] == df['dominant_label'])
    ][['local_image_path', 'ground_truth', 'labels']].reset_index(drop=True)

    # Build dataset + loader
    test_data = ImageProcessingDataset(test_df)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = ImageClassifier(base_model='google/vit-base-patch16-224', num_classes=9)
    model.load_state_dict(torch.load(state_dict_path))
    model.to(device)
    model.eval()

    # Evaluation loop
    all_true_labels = []
    all_true_dists = []
    all_pred_labels = []
    all_pred_dists = []

    with torch.no_grad():
        for pixel_values, true_labels, true_dists in tqdm(test_loader, desc="Evaluating"):
            logits = model(pixel_values= pixel_values.to(device) )['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=1)

            all_true_labels.extend(true_labels.cpu().numpy().tolist())
            all_true_dists.extend(true_dists.cpu().numpy().tolist())
            all_pred_labels.extend(preds.cpu().numpy().tolist())
            all_pred_dists.extend(probs.cpu().numpy().tolist())

    # Compute metrics
    f1 = f1_score(all_true_labels, all_pred_labels, average='macro')
    precision = precision_score(all_true_labels, all_pred_labels, average='macro')
    recall = recall_score(all_true_labels, all_pred_labels, average='macro')
    acc = accuracy_score(all_true_labels, all_pred_labels)

    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"Accuracy: {acc:.4f}")

    # Save metrics
    metrics_df = pd.DataFrame({
        'f1': [f1],
        'precision': [precision],
        'recall': [recall],
        'accuracy': [acc]
    })
    metrics_df.to_csv(os.path.join('data', 'evaluation', 'image_metrics.csv'), index=False)

    all_true_dists = [json.dumps( round_list(vec) ) for vec in all_true_dists]
    all_pred_dists = [json.dumps( round_list(vec) ) for vec in all_pred_dists]

    print(len(list(test_df['local_image_path'])))
    print(len(all_true_labels))
    print(len(all_pred_labels))
    print(len(all_true_dists))
    print(len(all_pred_dists))

    result_df = pd.DataFrame({
        'image_path': list(test_df['local_image_path']),
        'true_label': all_true_labels,
        'pred_label': all_pred_labels,
        'true_dist': all_true_dists,
        'pred_dist': all_pred_dists
    })

    result_df.to_csv(os.path.join('data', 'evaluation', 'image_results.csv'), index=False)

if __name__ == "__main__":
    main()
