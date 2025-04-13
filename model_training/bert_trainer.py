import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from custom_models import *
from transformers import Trainer, TrainingArguments, AutoTokenizer

"""
A short script for fine-tuning BERT and RoBERTa models for multi-label sentiment classification

Author: Clayton Durepos
Version: 04.11.2025
Contact: clayton.durepos@maine.edu
"""

MODEL_NAMES = ['google-bert/bert-base-cased', 'FacebookAI/roberta-base']
DATA_PATH = f'..{os.path.sep}data{os.path.sep}multimodal_sentiment_dataset.csv'

def main():
    df = pd.read_csv(DATA_PATH)

    # Get training and evaluation designated data from pre-generated dataset
    train_data = Dataset.from_pandas(df.loc[df['split'] == 'train'][['caption', 'label_vector']]).remove_columns(['__index_level_0__'])
    eval_data = Dataset.from_pandas(df.loc[df['split'] == 'eval'][['caption', 'label_vector']]).remove_columns(['__index_level_0__'])

    for name in tqdm(MODEL_NAMES, desc="Model Number", total=len(MODEL_NAMES)):
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = TextClassifier(name, 9)


        # Tokenize captions
        def tokenize_fn(batch):
            return tokenizer(batch['caption'],
                             padding='max_length',
                             truncation=True,
                             max_length=256,
                             add_special_tokens=True,
                             return_tensors="pt")

        train_data = train_data.map(tokenize_fn, batched=True)
        eval_data = eval_data.map(tokenize_fn, batched=True)

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
        save_name = 'roberta' if name == 'FacebookAI/roberta-base' else 'bert'
        torch.save(model, f"../models/{save_name}-classifier.pt")
        tokenizer.save_pretrained(f"../models/{save_name}-tokenizer")

if __name__ == "__main__":
    main()