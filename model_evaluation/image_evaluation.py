import os
import ast
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from classification_models import ImageClassifier
from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score, accuracy_score

tqdm.pandas()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict_path = os.path.join('..','models', 'vit-dict.pt')
data_path = os.path.join('..', 'data', 'multimodal_sentiment_dataset.csv')
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

# Load model & tokenizer
model = ImageClassifier(base_model='google/vit-base-patch16-224', num_classes=9)
model.load_state_dict(torch.load(state_dict_path))
model.to(device)
model.eval()

processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Custom dataset for memory efficiency
class ImageProcessingDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(row['local_image_path']) as img:
            inputs = processor(images=img, return_tensors='pt')

        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': label_map[ row['labels'] ]
        }

def main():
    # Load testing data
    df = pd.read_csv(data_path)
    test_df = df[df['split'] == 'test'].copy()[['local_image_path', 'labels']]

    # Adjust image_paths for directory structure
    test_df['local_image_path'] = test_df.apply(
        lambda row: '../' + row['local_image_path'], axis=1
    )

    test_data = ImageProcessingDataset(test_df)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Remove original DataFrame to free memory
    del df

    # Evaluation
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            logits = model(pixel_values=pixel_values)['logits']
            preds = logits.argmax(dim=1)

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

    pd.DataFrame(metric_dict, index=['1']).to_csv('image_metrics.csv', index=False)

    # Convert to integer for ease-of-use in reading
    test_df['prediction'] = all_preds.astype(int).tolist()
    test_df['labels'] = all_labels.astype(int).tolist()

    # Save direct results
    test_df.to_csv('image_results.csv', index=False)

if __name__ == "__main__":
    main()
