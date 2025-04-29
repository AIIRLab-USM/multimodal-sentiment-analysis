import os
import gc
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer
from classification_models import MultimodalClassifier
from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score

tqdm.pandas()
processor = AutoProcessor.from_pretrained('google/vit-base-patch16-224')
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = os.path.join('..', 'data', 'multimodal_sentiment_dataset.csv')

class MMProcessingDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df

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

        return {
            'pixel_values': img_inputs['pixel_values'].squeeze(0),
            'input_ids': txt_inputs['input_ids'].squeeze(0),
            'attention_mask': txt_inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(eval(row['labels']), dtype=torch.float)
        }


def main():

    # Load testing data
    df = pd.read_csv(data_path)
    test_df = df[df['split'] == 'test'].copy()[['local_image_path', 'caption', 'labels']]

    # Adjust image_paths to directory structure
    test_df['local_image_path'] = test_df.apply(
        lambda row: '../' + row['local_image_path'], axis=1
    )

    test_data = MMProcessingDataset(test_df)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Remove original DataFrame to free memory
    del df

    model = MultimodalClassifier()
    model.load_state_dict( torch.load( os.path.join('..', 'models', f'multimodal-dict.pt')   ) )
    model = model.to(device).eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)['logits']
            preds = (torch.sigmoid(logits) >= 0.5).float()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Metrics
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # Accuracy
    ham_acc = 1 - hamming_loss(all_labels, all_preds)  # Label-level
    acc = (all_labels == all_preds).all(axis=1).mean()  # Sample-level

    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"Hamming Accuracy: {ham_acc:.4f}")
    print(f"Accuracy: {acc:.4f}")

    # Memory management
    del model, all_preds, all_labels
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
