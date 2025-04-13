import torch
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from custom_models import *
from transformers import AutoImageProcessor

"""
A short script for fine-tuning the ViT model for multi-label sentiment classification

Author: Clayton Durepos
Version: 04.11.2025
Contact: clayton.durepos@maine.edu
"""

DATA_PATH = '../data/multimodal_sentiment_dataset.csv'
MODEL_NAME = 'google/vit-base-patch16-224'

def main():
    model = ImageClassifier(MODEL_NAME, 9)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    df = pd.read_csv(DATA_PATH)[['split', 'local_image_path', 'label_vector']]

    # Modify image_paths to comply with directory structure
    df['local_image_path'] = df.apply(
        lambda row: '../' + row['local_image_path'], axis=1
    )

    # Get training and evaluation designated data from pre-generated dataset
    train_data = Dataset.from_pandas(df.loc[df['split'] == 'train'][['local_image_path', 'label_vector']]).remove_columns(['__index_level_0__'])
    eval_data = Dataset.from_pandas(df.loc[df['split'] == 'eval'][['local_image_path', 'label_vector']]).remove_columns(['__index_level_0__'])

    # Process images
    def process_fn(batch):
        return processor(Image.open(batch['local_image_path']), return_tensors="pt")

    train_data, eval_data = train_data.map(process_fn, batched=False), eval_data.map(process_fn, batched=False)

    training_args = TrainingArguments(
        output_dir="./test_trainer",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    trainer.train()

    # Save
    torch.save(model, f"../models/vit-classifier.pt")
    processor.save_pretrained("../models/vit-image-processor")

if __name__ == "__main__":
    main()