import os
import ast
import json
from cProfile import label

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from src.classification_models import TextClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

"""
A short script for evaluating a fine-tuned BERT model

Author: Clayton Durepos
Version: 08.22.2025
Contact: clayton.durepos@maine.edu
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = os.path.join('data', 'datasets', 'multimodal_sentiment_dataset.csv')
dict_path = f'models{os.path.sep}bert-dict.pt'

def round_list(lst):
    """Round list of floats to 3 decimal places."""
    return [round(float(x), 3) for x in lst]

def main():
    os.makedirs(f'data{os.path.sep}evaluation', exist_ok=True)

    # Load dataset
    df = pd.read_csv(data_path)
    test_df = df.loc[df['split'] == 'test'][['art_style', 'painting', 'caption', 'label', 'probs']]
    test_df['probs'] = test_df['probs'].apply( ast.literal_eval )

    # Prepare tensors
    captions = list(test_df['caption'])
    label_tensor = torch.tensor(test_df['label'].values, dtype=torch.long)
    prob_tensor = torch.stack([torch.tensor(x, dtype=torch.float32) for x in test_df['probs'].tolist()])

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
    tokens = tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    test_dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], label_tensor, prob_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = TextClassifier(base_model='google-bert/bert-base-cased', num_classes=9)
    model.load_state_dict(torch.load(dict_path))
    model.to(device)
    model.eval()

    # Evaluation loop
    all_true_labels = []
    all_true_dists = []
    all_pred_labels = []
    all_pred_dists = []

    with torch.no_grad():
        for input_ids, attention_mask, true_labels, true_dists in tqdm(test_loader, desc="Evaluating"):
            logits = model(input_ids= input_ids.to(device), attention_mask=attention_mask.to(device) )['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=1)

            all_true_labels.extend(true_labels.cpu().numpy().tolist())
            all_true_dists.extend(true_dists.cpu().numpy().tolist())
            all_pred_labels.extend(preds.cpu().numpy().tolist())
            all_pred_dists.extend(probs.cpu().numpy().tolist())

    # Class names for labeling the dictionary keys
    class_names = [
        'amusement', 'anger', 'awe', 'contentment', 'disgust',
        'excitement', 'fear', 'sadness', 'something else'
    ]

    # Global (macro-averaged) metrics
    f1_macro = f1_score(all_true_labels, all_pred_labels, average='macro')
    precision_macro = precision_score(all_true_labels, all_pred_labels, average='macro')
    recall_macro = recall_score(all_true_labels, all_pred_labels, average='macro')
    accuracy = accuracy_score(all_true_labels, all_pred_labels)

    # Per-class metrics
    f1_per_class = f1_score(all_true_labels, all_pred_labels, average=None, labels=range(len(class_names)))
    precision_per_class = precision_score(all_true_labels, all_pred_labels, average=None,
                                          labels=range(len(class_names)))
    recall_per_class = recall_score(all_true_labels, all_pred_labels, average=None, labels=range(len(class_names)))

    # Structure the data into a single dictionary
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

    with open( os.path.join('data', 'evaluation', 'text_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    # Prepare rounded + JSON strings for vector columns
    all_true_dists = [json.dumps(round_list(vec)) for vec in all_true_dists]
    all_pred_dists = [json.dumps(round_list(vec)) for vec in all_pred_dists]


    result_df = pd.DataFrame({
        'art_style': list(test_df['art_style']),
        'painting': list(test_df['painting']),
        'caption': captions,
        'true_label': all_true_labels,
        'pred_label': all_pred_labels,
        'true_dist': all_true_dists,
        'pred_dist': all_pred_dists
    })

    results_path = os.path.join('data', 'evaluation', 'text_results.csv')
    result_df.to_csv(results_path, index=False)

if __name__ == '__main__':
    main()