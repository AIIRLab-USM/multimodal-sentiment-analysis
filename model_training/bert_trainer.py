import ast
import os
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from classification_models import *
from transformers import Trainer, TrainingArguments, AutoTokenizer

"""
A short script for fine-tuning BERT and RoBERTa models for multi-label sentiment classification

Author: Clayton Durepos
Version: 04.18.2025
Contact: clayton.durepos@maine.edu
"""

MODEL_NAMES = ['google-bert/bert-base-cased', 'FacebookAI/roberta-base']
DATA_PATH = f'..{os.path.sep}data{os.path.sep}multimodal_sentiment_dataset.csv'

def main():
    df = pd.read_csv(DATA_PATH)

    # Train Data pre-processing
    train_data = df.loc[df['split'] == 'train'][['caption', 'labels']]
    train_data['labels'] = train_data.apply(
        lambda row: [float(x) for x in ast.literal_eval(row['labels'])],
        axis=1
    )

    train_data = Dataset.from_pandas(train_data).remove_columns(['__index_level_0__'])

    # Evaluation Data pre-processing
    eval_data = df.loc[df['split'] == 'eval'][['caption', 'labels']]
    eval_data['labels'] = eval_data.apply(
        lambda row: [float(x) for x in ast.literal_eval(row['labels'])],
        axis=1
    )

    eval_data = Dataset.from_pandas(eval_data).remove_columns(['__index_level_0__'])

    # Training loop
    for name in tqdm(MODEL_NAMES, desc="Model Number", total=len(MODEL_NAMES)):
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = TextClassifier(num_classes=9, base_model=name)


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

if __name__ == "__main__":
    main()