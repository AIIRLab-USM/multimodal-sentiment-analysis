import os
import ast
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from classification_models import ImageClassifier
from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score

tqdm.pandas()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict_path = os.path.join('..','models', 'vit-dict.pt')
data_path = os.path.join('..', 'data', 'multimodal_sentiment_dataset.csv')

# Load model & tokenizer
model = ImageClassifier(base_model='google/vit-base-patch16-224', num_classes=9)
model.load_state_dict(torch.load(state_dict_path))
model.to(device)
model.eval()

processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Load and pre-process testing data
df = pd.read_csv(data_path)
test_df = df[df['split'] == 'test'].copy()[['local_image_path', 'labels']]
test_df['labels'] = test_df['labels'].apply(lambda x: ast.literal_eval(x))
test_df['local_image_path'] = test_df.apply(
    lambda row: '../' + row['local_image_path'], axis=1
)

test_data = Dataset.from_pandas(test_df).remove_columns(['__index_level_0__'])
def process_fn(examples):
    with Image.open(examples['local_image_path']) as img:
        return {
            'pixel_values': processor(images=img, return_tensors='pt')['pixel_values'].squeeze(0)
        }

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

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())


all_preds = torch.cat(all_preds, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()

# Metrics
f1 = f1_score(all_labels, all_preds, average='weighted')
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')

# Accuracy
ham_acc = 1 - hamming_loss(all_labels, all_preds)   # Label-level
acc = (all_labels == all_preds).all(axis=1).mean()  # Sample-level

print(f"F1 Score (Weighted): {f1:.4f}")
print(f"Precision (Weighted): {precision:.4f}")
print(f"Recall (Weighted): {recall:.4f}")
print(f"Hamming Accuracy: {ham_acc:.4f}")
print(f"Accuracy: {acc:.4f}")
