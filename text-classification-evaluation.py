import os
import ast
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join('models', 'roberta-classifier.pt')
data_path = os.path.join('data', 'multimodal_sentiment_dataset.csv')

# Load model & tokenizer
model = torch.load(model_path)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')

# Load and pre-process testing data
df = pd.read_csv(data_path)
test_df = df[df['split'] == 'test'].copy()[['caption', 'labels']]
test_df['labels'] = test_df['labels'].apply(lambda x: torch.tensor(ast.literal_eval(x), dtype=torch.float))

# Tokenize
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
        preds = (torch.sigmoid(logits) >= 0.5).float()

        all_preds.extend(preds.cpu())
        all_labels.extend(labels.cpu())


all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

f1 = f1_score(all_labels, all_preds, average='weighted')
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
acc = accuracy_score(all_labels, all_preds)

print(f"F1 Score (Weighted): {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {acc:.4f}")
