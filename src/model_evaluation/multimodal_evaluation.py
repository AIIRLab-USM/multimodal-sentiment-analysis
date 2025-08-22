import os
import ast
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import unicodedata
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer
from src.classification_models import MultimodalClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

"""
A short script for evaluating a fine-tuned ViT & BERT multimodal model

Author: Clayton Durepos
Version: 08.21.2025
Contact: clayton.durepos@maine.edu
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained('google/vit-base-patch16-224')
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')

data_path = os.path.join('data', 'datasets', 'multimodal_sentiment_dataset.csv')
dict_path = f'models{os.path.sep}multimodal-dict.pt'

def round_list(lst):
    return [round(float(x), 3) for x in lst]

class MMProcessingDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(
                os.path.join('wikiart', row['art_style'], unicodedata.normalize("NFC", row['painting']), '.jpg')
        ) as img:
            img_inputs = processor(images=img, return_tensors='pt')
            txt_inputs = tokenizer(text=row['caption'],
                                   padding='max_length',
                                   truncation=True,
                                   max_length=256,
                                   add_special_tokens=True,
                                   return_tensors="pt")

        return (
            img_inputs['pixel_values'].squeeze(0),                      # pixeL_values
            txt_inputs['input_ids'].squeeze(0),                         # input_ids
            txt_inputs['attention_mask'].squeeze(0),                    # attention_mask
            row['label'],                                               # true_labels
            torch.tensor(row['probs']),                                 # true_dists
        )

def main():
    os.makedirs(f'data{os.path.sep}evaluation', exist_ok=True)

    # Load testing data
    df = pd.read_csv(data_path)
    test_df = df.loc[df['split'] == 'test'][['local_image_path', 'caption', 'ground_truth', 'labels']]
    test_df['probs'] = test_df['probs'].apply( ast.literal_eval )

    # Build dataset + loader
    test_data = MMProcessingDataset(test_df)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Load model
    model = MultimodalClassifier()
    model.load_state_dict(torch.load(dict_path))
    model.to(device)
    model.eval()

    all_true_labels, all_pred_labels = [], []
    all_true_dists, all_pred_dists = [], []
    all_attn_weights_text, all_attn_weights_image = [], []

    with torch.no_grad():
        for pixel_values, input_ids, attention_mask, true_labels, true_dists in tqdm(test_loader, desc="Evaluating"):
            outputs, attn_weights = model(
                pixel_values=pixel_values.to(device),
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                return_weights=True
            )

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=1)

            all_true_labels.extend(true_labels.cpu().numpy().tolist())
            all_true_dists.extend(true_dists.cpu().numpy().tolist())
            all_pred_labels.extend(preds.cpu().numpy().tolist())
            all_pred_dists.extend(probs.cpu().numpy().tolist())

            # Collect attention weights
            attn_weights_np = attn_weights.cpu().numpy()
            all_attn_weights_text.extend(attn_weights_np[:, 0].tolist())
            all_attn_weights_image.extend(attn_weights_np[:, 1].tolist())

    # 1. Define class names for labeling
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

    with open( os.path.join('data', 'evaluation', 'multimodal_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    result_df = pd.DataFrame({
        'local_image_path': list(test_df['local_image_path']),
        'caption': list(test_df['caption']),
        'true_label': all_true_labels,
        'pred_label': all_pred_labels,
        'true_dist': [json.dumps(round_list(vec)) for vec in all_true_dists],
        'pred_dist': [json.dumps(round_list(vec)) for vec in all_pred_dists],
        'text_weight': [round(float(w), 4) for w in all_attn_weights_text],
        'image_weight': [round(float(w), 4) for w in all_attn_weights_image]
    })
    result_df.to_csv(os.path.join('data', 'evaluation', 'multimodal_results.csv'), index=False)


if __name__ == "__main__":
    main()