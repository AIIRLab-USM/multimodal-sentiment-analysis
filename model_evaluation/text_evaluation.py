import os
import gc
import ast
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from classification_models import TextClassifier
from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

data_path = os.path.join('..', 'data', 'multimodal_sentiment_dataset.csv')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dict_path = os.path.join('..', 'models', 'bert-dict.pt')
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
    # Load and pre-process testing data
    df = pd.read_csv(data_path)
    test_df = df[df['split'] == 'test'].copy()[['caption', 'labels']]
    test_df['labels'] = test_df['labels'].apply(lambda x: label_map[x])


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

    labels_tensor = torch.stack(test_df['labels'].tolist())
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

    pd.DataFrame(metric_dict, index=['1']).to_csv('text_metrics.csv')

    # Convert to integer for ease-of-use in reading
    test_df['prediction'] = all_preds.astype(int).tolist()
    test_df['labels'] = all_labels.astype(int).tolist()

    # Save direct results
    test_df.to_csv('text_results.csv', index=False)

if __name__ == '__main__':
    main()
