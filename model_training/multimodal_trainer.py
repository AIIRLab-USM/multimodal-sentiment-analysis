import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import ast
import torch
import pandas as pd
from PIL import Image
from classification_models import MultimodalClassifier
from transformers import AutoImageProcessor, Trainer, AutoTokenizer
from training import training_args, compute_metrics, early_stopping_callback, alpha_monitoring_callback, writer

"""
A short script for fine-tuning a multimodal classification model on a sentiment classification task

Author: Clayton Durepos
Version: 04.29.2025
Contact: clayton.durepos@maine.edu
"""

DATA_PATH = f'..{os.path.sep}data{os.path.sep}multimodal_sentiment_dataset.csv'

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
            'labels': torch.tensor(
                ast.literal_eval(row['labels']),
                dtype=torch.float
            )
        }


def main():
    df = pd.read_csv(DATA_PATH)

    # Modify image_paths to comply with directory structure
    df['local_image_path'] = df.apply(
        lambda row: '../' + row['local_image_path'], axis=1
    )

    # Train Data pre-processing
    train_df = df.loc[df['split'] == 'train'][['local_image_path', 'caption', 'labels']].copy()
    # eval_df = eval_df.iloc[:int(len(eval_df) * 0.01)]       # For testing

    train_data = MMProcessingDataset(train_df)

    # Evaluation Data pre-processing
    eval_df = df.loc[df['split'] == 'eval'][['local_image_path', 'caption', 'labels']].copy()
    # eval_df = eval_df.iloc[:int(len(eval_df) * 0.01)]    # For testing

    eval_data = MMProcessingDataset(eval_df)

    # Delete original DataFrame to free memory
    del df

    model = MultimodalClassifier()

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=eval_data,
        callbacks=[
            early_stopping_callback,
            alpha_monitoring_callback
        ]
    )

    trainer.train()
    torch.save(model.state_dict(), f"../models/multimodal-dict.pt")

if __name__ == "__main__":
    main()
    writer.close()
