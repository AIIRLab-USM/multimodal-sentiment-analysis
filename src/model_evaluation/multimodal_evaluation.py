import os
import ast
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer
from src.classification_models import MultimodalClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

"""
A short script for evaluating a fine-tuned ViT & BERT multimodal model

Author: Clayton Durepos
Version: 07.17.2025
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
        with Image.open(row['local_image_path']) as img:
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
            row['ground_truth'],                                        # true_labels
            row['labels'],                                              # true_dists
        )

def main():
    os.makedirs(f'data{os.path.sep}evaluation', exist_ok=True)

    # Load testing data
    df = pd.read_csv(data_path)
    test_df = df.loc[df['split'] == 'test']['local_image_path', 'caption', 'ground_truth', 'labels']
    test_df['labels'] = test_df['labels'].apply( ast.literal_eval )

    # Build dataset + loader
    test_data = MMProcessingDataset(test_df)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Load model
    model = MultimodalClassifier()
    model.load_state_dict(torch.load(dict_path))
    model.to(device)
    model.eval()

    # Run evaluation
    all_true_labels = []
    all_true_dists = []
    all_pred_labels = []
    all_pred_dists = []

    with torch.no_grad():
        for pixel_values, input_ids, attention_mask, true_labels, true_dists in tqdm(test_loader, desc="Evaluating"):
            logits = model(pixel_values= pixel_values.to(device) , input_ids= input_ids.to(device) , attention_mask= attention_mask.to(device) )['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=1)

            all_true_labels.extend(true_labels.cpu().numpy().tolist())
            all_true_dists.extend(true_dists.cpu().numpy().tolist())
            all_pred_labels.extend(preds.cpu().numpy().tolist())
            all_pred_dists.extend(probs.cpu().numpy().tolist())

    f1 = f1_score(all_true_labels, all_pred_labels, average='macro')
    precision = precision_score(all_true_labels, all_pred_labels, average='macro')
    recall = recall_score(all_true_labels, all_pred_labels, average='macro')
    acc = accuracy_score(all_true_labels, all_pred_labels)

    print(f"\nF1 Score (Macro): {f1:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"Accuracy: {acc:.4f}")

    # Save metrics
    pd.DataFrame({
        'f1': [f1],
        'precision': [precision],
        'recall': [recall],
        'accuracy': [acc]
    }).to_csv(os.path.join('data', 'evaluation', 'multimodal_metrics.csv'), index=False)

    all_true_dists = [json.dumps(round_list(vec)) for vec in all_true_dists]
    all_pred_dists = [json.dumps(round_list(vec)) for vec in all_pred_dists]

    result_df = pd.DataFrame({
        'local_image_path': list(test_df['local_image_path']),
        'caption': list(test_df['caption']),
        'true_label': all_true_labels,
        'pred_label': all_pred_labels,
        'true_dist': all_true_dists,
        'pred_dist': all_pred_dists
    })
    result_df.to_csv(os.path.join('data', 'evaluation', 'multimodal_results.csv'), index=False)

if __name__ == "__main__":
    main()
