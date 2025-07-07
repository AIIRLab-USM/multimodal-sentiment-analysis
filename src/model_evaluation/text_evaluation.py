import os
import ast
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from src.classification_models import TextClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

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

    # Compute dominant label + confidence
    group_stats = (
        df.groupby(['local_image_path'])['ground_truth']
        .agg(lambda x: (x.value_counts().idxmax(), x.value_counts().max() / len(x)))
        .apply(pd.Series)
    )
    group_stats.columns = ['dominant_label', 'confidence']
    group_stats = group_stats[group_stats['confidence'] >= 0.5].reset_index()

    # Merge directly into df to get confident samples
    df = df.merge(group_stats, on=['local_image_path'], how='inner')
    test_df = df[
        (df['split'] == 'test') &
        (df['ground_truth'] == df['dominant_label'])
    ][['caption', 'ground_truth', 'labels']]

    # Parse soft label vectors from string to list
    test_df['labels'] = test_df['labels'].apply(ast.literal_eval)

    # Prepare tensors
    captions = list(test_df['caption'])
    ground_truth_tensor = torch.tensor(test_df['ground_truth'].values, dtype=torch.long)
    prob_tensor = torch.stack([torch.tensor(x, dtype=torch.float32) for x in test_df['labels'].tolist()])

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
    tokens = tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    test_dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], ground_truth_tensor, prob_tensor)
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

    # Compute metrics
    f1 = f1_score(all_true_labels, all_pred_labels, average='macro')
    precision = precision_score(all_true_labels, all_pred_labels, average='macro')
    recall = recall_score(all_true_labels, all_pred_labels, average='macro')
    acc = accuracy_score(all_true_labels, all_pred_labels)

    print(f"\nF1 Score (Macro): {f1:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"Accuracy: {acc:.4f}")

    # Save metrics CSV
    pd.DataFrame({
        'f1': [f1],
        'precision': [precision],
        'recall': [recall],
        'accuracy': [acc]
    }).to_csv(os.path.join('data', 'evaluation', 'text_metrics.csv'), index=False)

    # Prepare rounded + JSON strings for vector columns
    all_true_dists = [json.dumps( round_list(vec) ) for vec in all_true_dists]
    all_pred_dists = [json.dumps( round_list(vec) ) for vec in all_pred_dists]


    result_df = pd.DataFrame({
        'caption': captions,
        'true_label': all_true_labels,
        'pred_label': all_pred_labels,
        'true_dist': all_true_dists,
        'pred_dist': all_pred_dists
    })

    result_df.to_csv(os.path.join('data', 'evaluation', 'text_results.csv'), index=False)

if __name__ == '__main__':
    main()
