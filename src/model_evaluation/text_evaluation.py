import ast
import os
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

def main():
    os.makedirs(f'data{os.path.sep}evaluation', exist_ok=True)

    # Load and pre-process testing data
    df = pd.read_csv(data_path)

    # Compute dominant label and confidence for each (image, caption)
    group_stats = (
        df.groupby(['local_image_path'])['labels']
        .agg(lambda x: (x.value_counts().idxmax(), x.value_counts().max() / len(x)))
        .apply(pd.Series)
    )

    group_stats.columns = ['dominant_label', 'confidence']
    group_stats = group_stats[group_stats['confidence'] >= 0.5]
    group_stats.reset_index(inplace=True)

    # Merge directly into df to get confident samples
    df = df.merge(group_stats, on=['local_image_path'], how='inner')
    test_df = df[
        (df['split'] == 'test') &
        (df['labels'] == df['dominant_label'])
    ][['caption', 'labels']]

    # Load model & tokenizer
    model = TextClassifier(base_model='google-bert/bert-base-cased', num_classes=9)
    model.load_state_dict(torch.load(dict_path))
    model.to(device)
    model.eval()

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
    tokens = tokenizer(
        list(test_df['caption']),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    labels_tensor = torch.stack( test_df['labels'].apply( lambda x: torch.tensor(x, dtype=torch.long) ).tolist() )
    test_dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(test_loader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)['logits']
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())


    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Metrics
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)

    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"Accuracy: {acc:.4f}")

    # Save metrics
    metric_dict = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': acc
    }

    pd.DataFrame(metric_dict, index=['1']).to_csv( os.path.join('data', 'evaluation', 'text_metrics.csv'), index=False)

    # Convert to integer for ease-of-use in reading
    test_df['prediction'] = all_preds.astype(int).tolist()
    test_df['labels'] = all_labels.astype(int).tolist()

    # Save direct results
    test_df.to_csv( os.path.join('data', 'evaluation', 'text_results.csv'), index=False)

if __name__ == '__main__':
    main()
