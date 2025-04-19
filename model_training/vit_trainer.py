import ast
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from classification_models import *
from transformers import AutoImageProcessor

"""
A short script for fine-tuning the ViT model for multi-label sentiment classification

Author: Clayton Durepos
Version: 04.18.2025
Contact: clayton.durepos@maine.edu
"""

DATA_PATH = '../data/multimodal_sentiment_dataset.csv'
MODEL_NAME = 'google/vit-base-patch16-224'

def main():
    model = ImageClassifier(MODEL_NAME, 9)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    df = pd.read_csv(DATA_PATH)[['split', 'local_image_path', 'labels']]

    # Modify image_paths to comply with directory structure
    df['local_image_path'] = df.apply(
        lambda row: '../' + row['local_image_path'], axis=1
    )

    # Train Data pre-processing
    train_data = df.loc[df['split'] == 'train'][['local_image_path', 'labels']]
    train_data['labels'] = train_data.apply(
        lambda row: [float(x) for x in ast.literal_eval(row['labels'])],
        axis=1
    )

    train_data = Dataset.from_pandas(train_data).remove_columns(['__index_level_0__'])

    # Evaluation Data pre-processing
    eval_data = df.loc[df['split'] == 'eval'][['local_image_path', 'labels']]
    eval_data['labels'] = eval_data.apply(
        lambda row: [float(x) for x in ast.literal_eval(row['labels'])],
        axis=1
    )

    eval_data = Dataset.from_pandas(eval_data).remove_columns(['__index_level_0__'])

    # Process images
    def process_fn(batch):
        with Image.open(batch['local_image_path']) as img:
            return {
                # Remove batch dimension (Will be re-added in Trainer)
                'pixel_values': processor(images=[img], return_tensors='pt')['pixel_values'][0]
            }

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
        eval_dataset=eval_data
    )

    trainer.train()

    # Save
    torch.save(model, f"../models/vit-classifier.pt")

if __name__ == "__main__":
    main()