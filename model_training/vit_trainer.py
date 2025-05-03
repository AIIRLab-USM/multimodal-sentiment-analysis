import ast
import pandas as pd
from PIL import Image
from classification_models import *
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, Trainer
from training import get_args, compute_metrics, early_stopping_callback

"""
A short script for fine-tuning the ViT model for multi-label sentiment classification

Author: Clayton Durepos
Version: 04.30.2025
Contact: clayton.durepos@maine.edu
"""

DATA_PATH = '../data/multimodal_sentiment_dataset.csv'
MODEL_NAME = 'google/vit-base-patch16-224'
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
        with Image.open(row['local_image_path']) as img:
            inputs = processor(images=img, return_tensors='pt')

        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(
                        label_map[ row['labels'] ],
                        dtype=torch.long
                     )
        }

def main():
    df = pd.read_csv(DATA_PATH)[['split', 'local_image_path', 'labels']]

    # Modify image_paths to comply with directory structure
    df['local_image_path'] = df.apply(
        lambda row: '../' + row['local_image_path'], axis=1
    )

    # Train dataset initialization
    train_data = df.loc[df['split'] == 'train'][['local_image_path', 'labels']].copy()
    # eval_df = eval_df.iloc[:int(len(eval_df) * 0.01)]       # For testing

    train_data = ImageProcessingDataset(train_data)

    # Evaluation dataset initialization
    eval_data = df.loc[df['split'] == 'eval'][['local_image_path', 'labels']].copy()
    # eval_df = eval_df.iloc[:int(len(eval_df) * 0.01)]    # For testing

    eval_data = ImageProcessingDataset(eval_data)

    # Delete original DataFrame to free memory
    del df

    model = ImageClassifier(base_model=MODEL_NAME, num_classes=9)
    training_args = get_args(output_dir=f"./vit_test_trainer",
                             learning_rate=1e-4)    # As used in ViT, Dosovitskiy et al., ICLR 2019

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=eval_data,
        callbacks=[early_stopping_callback]
    )

    trainer.train()

    # Save
    torch.save(model.state_dict(), f"../models/vit-dict.pt")

if __name__ == "__main__":
    main()