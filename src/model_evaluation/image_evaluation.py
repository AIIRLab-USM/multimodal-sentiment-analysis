import os
import ast
import json
import torch
import unicodedata
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from src.classification_models import ImageClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

"""
A short script for evaluating a fine-tuned ViT model

Author: Clayton Durepos
Version: 08.22.2025
Contact: clayton.durepos@maine.edu
"""

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

        with Image.open(
                os.path.join('wikiart', row['art_style'], f'{unicodedata.normalize("NFC", row["painting"])}.jpg')
        ) as img:
            inputs = processor(images=img, return_tensors='pt')

        return (
            inputs['pixel_values'].squeeze(0),  # pixel_values
            row['label'],                       # true_labels
            torch.tensor(row['probs'])          # true_dists
        )

def main():
    os.makedirs(f'data{os.path.sep}evaluation', exist_ok=True)

    # Load dataset
    df = pd.read_csv(data_path)
    test_df = df.loc[df['split'] == 'test'][['art_style', 'painting', 'label', 'probs']]
    test_df['probs'] = test_df['probs'].apply( ast.literal_eval )

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

    # Class names for labeling
    class_names = [
        'amusement', 'anger', 'awe', 'contentment', 'disgust',
        'excitement', 'fear', 'sadness', 'something else'
    ]

    # Calculate global and per-class metrics
    f1_macro = f1_score(all_true_labels, all_pred_labels, average='macro')
    precision_macro = precision_score(all_true_labels, all_pred_labels, average='macro')
    recall_macro = recall_score(all_true_labels, all_pred_labels, average='macro')
    accuracy = accuracy_score(all_true_labels, all_pred_labels)

    f1_per_class = f1_score(all_true_labels, all_pred_labels, average=None, labels=range(len(class_names)))
    precision_per_class = precision_score(all_true_labels, all_pred_labels, average=None,
                                          labels=range(len(class_names)))
    recall_per_class = recall_score(all_true_labels, all_pred_labels, average=None, labels=range(len(class_names)))

    # Structure data into a single dictionary
    metrics_dict = {
        'global': {
            'f1_score': f1_macro,
            'precision': precision_macro,
            'recall': recall_macro,
            'accuracy': accuracy
        }
    }

    for i, name in enumerate(class_names):
        metrics_dict[name] = {
            'f1_score': f1_per_class[i],
            'precision': precision_per_class[i],
            'recall': recall_per_class[i]
        }

    with open( os.path.join('data', 'evaluation', 'image_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    result_df = pd.DataFrame({
        'art_style': list(test_df['art_style']),
        'painting': list(test_df['painting']),
        'true_label': all_true_labels,
        'pred_label': all_pred_labels,
        'true_dist': all_true_dists,
        'pred_dist': all_pred_dists
    })

    result_df.to_csv(os.path.join('data', 'evaluation', 'image_results.csv'), index=False)

if __name__ == "__main__":
    main()
