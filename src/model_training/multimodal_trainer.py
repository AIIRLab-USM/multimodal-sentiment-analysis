import os
import ast
import torch
import unicodedata
import pandas as pd
from PIL import Image
from src.classification_models import MultimodalClassifier
from transformers import AutoImageProcessor, AutoTokenizer
from src.model_training.training import get_args, compute_metrics, early_stopping_callback, KLTrainer

"""
A short script for fine-tuning a multimodal classification model on a sentiment classification task

Author: Clayton Durepos
Version: 08.21.2025
Contact: clayton.durepos@maine.edu
"""

DATA_PATH = os.path.join('data', 'datasets', 'multimodal_sentiment_dataset.csv')

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Custom dataset for memory efficiency
class MMProcessingDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(
                os.path.join( 'wikiart', row['art_style'], unicodedata.normalize("NFC", row['painting']), '.jpg')
        ) as img:
            img_inputs = processor(images=img, return_tensors='pt')
            txt_inputs = tokenizer(text=row['caption'],
                                   padding='max_length',
                                   truncation=True,
                                   max_length=256,
                                   add_special_tokens=True,
                                   return_tensors="pt")

        inputs = {
            'pixel_values': img_inputs['pixel_values'].squeeze(0),
            'input_ids': txt_inputs['input_ids'].squeeze(0),
            'attention_mask': txt_inputs['attention_mask'].squeeze(0),
            'labels': row['probs']
        }

        if 'ground_truth' in self.df.columns:
            inputs['ground_truth'] = row['ground_truth']

        return inputs


def main():
    os.makedirs('models', exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Train Data pre-processing
    train_data = df.loc[df['split'] == 'train'][['art_style', 'painting', 'caption', 'label', 'probs']].copy()
    train_data['probs'] = train_data['probs'].apply( lambda x: ast.literal_eval(x) )
    train_data = MMProcessingDataset( train_data[['local_image_path', 'caption', 'probs']] )

    # Evaluation Data pre-processing
    eval_data = df.loc[df['split'] == 'eval'][['art_style', 'painting', 'caption', 'label', 'probs']].copy()
    eval_data['probs'] = eval_data['probs'].apply( lambda x: ast.literal_eval(x) )
    eval_data = MMProcessingDataset( eval_data )

    # Delete original DataFrame to free memory
    del df

    model = MultimodalClassifier()
    training_args = get_args(learning_rate=1e-5)         # Used for multimodal models in
                                                         # LXMERT, Tan and Bansal, EMNLP-IJCNLP 2019
                                                         # UNITER, Chan et al. ECCV 2020
    trainer = KLTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=eval_data,
        callbacks=[early_stopping_callback]
    )

    trainer.train()
    torch.save(model.state_dict(), f'models{os.path.sep}multimodal-dict.pt')

if __name__ == "__main__":
    main()
