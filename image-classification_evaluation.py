import os
import ast
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoImageProcessor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

tqdm.pandas()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join('models', 'vit-classifier.pt')
data_path = os.path.join('data', 'multimodal_sentiment_dataset.csv')

# Load model & tokenizer
model = torch.load(model_path)
model.to(device)
model.eval()

processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Load and pre-process testing data
df = pd.read_csv(data_path)
test_df = df[df['split'] == 'test'].copy()[['local_image_path', 'labels']]
test_df['labels'] = test_df['labels'].apply(lambda x: ast.literal_eval(x))

test_data = Dataset.from_pandas(test_df)
def process_fn(examples):
    return processor(Image.open(examples['local_image_path']), return_tensors='pt')

test_data = test_data.map(process_fn, batched=False)
test_data = test_data.with_format(type='torch', columns=['pixel_values', 'labels'])
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        logits = model(pixel_values=pixel_values)['logits']
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
