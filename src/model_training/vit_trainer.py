import os
import ast
import pandas as pd
from PIL import Image
import unicodedata
from src.classification_models import *
from transformers import AutoImageProcessor
from src.model_training.training import get_args, compute_metrics, early_stopping_callback, KLTrainer

"""
A short script for fine-tuning the ViT model for multi-label sentiment classification

Author: Clayton Durepos
Version: 08.22.2025
Contact: clayton.durepos@maine.edu
"""

DATA_PATH = os.path.join('data', 'datasets', 'multimodal-sentiment-dataset.csv')
MODEL_NAME = 'google/vit-base-patch16-224'

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# Custom dataset for memory efficiency
class ImageProcessingDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(
                os.path.join('wikiart', row['art_style'], f'{unicodedata.normalize("NFC", row["painting"])}.jpg')
        ) as img:
            inputs = processor(images=img, return_tensors='pt')

        inputs = {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'probs': row['probs']
        }

        if 'label' in self.df.columns:
            inputs['label'] = row['label']

        return inputs

def main():
    os.makedirs('models', exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Train Data pre-processing
    train_data = df.loc[df['split'] == 'train'][['art_style','painting', 'probs']]
    train_data['probs'] = train_data['probs'].apply(lambda x: ast.literal_eval(x))  # String to array
    train_data = ImageProcessingDataset( train_data )

    # Evaluation Data pre-processing
    eval_data = df.loc[df['split'] == 'eval'][['art_style','painting', 'label', 'probs']].copy()
    eval_data['probs'] = eval_data['probs'].apply( lambda x: ast.literal_eval(x) )
    eval_data = ImageProcessingDataset( eval_data )

    # Delete original DataFrame to free memory
    del df

    model = ImageClassifier(base_model=MODEL_NAME, num_classes=9)
    training_args = get_args(learning_rate=1e-4)    # As used in ViT, Dosovitskiy et al., ICLR 2019

    # Train
    trainer = KLTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=eval_data,
        callbacks=[early_stopping_callback]
    )

    trainer.train()

    # Save
    torch.save(model.state_dict(), f'models{os.path.sep}vit-dict.pt')

if __name__ == "__main__":
    main()
