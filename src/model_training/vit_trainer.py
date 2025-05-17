import os
import numpy as np
import pandas as pd
from PIL import Image
from src.classification_models import *
from transformers import AutoImageProcessor
from sklearn.utils.class_weight import compute_class_weight
from src.model_training.training import get_args, compute_metrics, early_stopping_callback, WeightedTrainer

"""
A short script for fine-tuning the ViT model for multi-label sentiment classification

Author: Clayton Durepos
Version: 05.04.2025
Contact: clayton.durepos@maine.edu
"""

DATA_PATH = os.path.join('data', 'datasets', 'multimodal_sentiment_dataset.csv')
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
            'labels': label_map[ row['labels'] ]
        }

def main():
    os.makedirs('models', exist_ok=True)

    df = pd.read_csv(DATA_PATH)[['split', 'local_image_path', 'labels']]

    # Train dataset initialization
    train_df = df.loc[df['split'] == 'train'][['local_image_path', 'labels']].copy()
    # train_df = train_df.iloc[:int(len(train_df) * 0.01)]       # For testing

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.arange(len(label_map)),
                                         y=train_df['labels'].map(label_map).tolist()
                                         )

    train_data = ImageProcessingDataset(train_df)

    # Evaluation dataset initialization
    eval_df = df.loc[df['split'] == 'eval'][['local_image_path', 'labels']].copy()
    # eval_df = eval_dfiloc[:int(len(eval_df) * 0.01)]    # For testing

    eval_data = ImageProcessingDataset(eval_df)

    # Delete original DataFrame to free memory
    del df

    model = ImageClassifier(base_model=MODEL_NAME, num_classes=9)
    training_args = get_args(learning_rate=1e-4)    # As used in ViT, Dosovitskiy et al., ICLR 2019

    # Train
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=eval_data,
        class_weights=class_weights,
        callbacks=[early_stopping_callback]
    )

    trainer.train()

    # Save
    torch.save(model.state_dict(), f'models{os.path.sep}vit-dict.pt')

if __name__ == "__main__":
    main()