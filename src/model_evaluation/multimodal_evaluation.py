import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer
from src.classification_models import MultimodalClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained('google/vit-base-patch16-224')
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')

state_dict_paths = [
    f'models{os.path.sep}multimodal-dict.pt',
    f'models{os.path.sep}bal-training_multimodal-dict.pt'
]

data_paths = [
    os.path.join('data', 'datasets', 'multimodal_sentiment_dataset.csv'),
    os.path.join('data', 'datasets', 'bal_multimodal_sentiment_dataset.csv')
]


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

tqdm.pandas()

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
            'labels': label_map[ row['labels'] ],
        }


def main():
    os.makedirs(f'data{os.path.sep}evaluation', exist_ok=True)

    for state_dict_path, data_path in zip(state_dict_paths, data_paths):
        save_name = 'multimodal' if state_dict_path==state_dict_paths[0] \
            else 'bal-training_multimodal'

        # Load testing data
        df = pd.read_csv(data_path)

        test_df = df[df['split'] == 'test'].copy()[['local_image_path', 'caption', 'labels']]
        test_data = MMProcessingDataset(test_df)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # Remove original DataFrame to free memory
        del df

        # Load model
        model = MultimodalClassifier()
        model.load_state_dict( torch.load( state_dict_path )   )
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

        pd.DataFrame(metric_dict, index=['1']).to_csv( os.path.join('data', 'evaluation', f'{save_name}-metrics.csv'), index=False)

        # Convert to integer for ease-of-use in reading
        test_df['prediction'] = all_preds.astype(int).tolist()
        test_df['labels'] = all_labels.astype(int).tolist()

        # Save direct results
        test_df.to_csv( os.path.join('data', 'evaluation', f'{save_name}-results.csv'), index=False)

if __name__ == "__main__":
    main()
