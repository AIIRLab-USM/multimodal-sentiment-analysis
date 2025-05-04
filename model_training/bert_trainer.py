import os
import ast
import pandas as pd
from datasets import Dataset
from classification_models import *
from transformers import Trainer, AutoTokenizer
from model_training.training import compute_metrics, get_args, early_stopping_callback

"""
A short script for fine-tuning BERT and RoBERTa models for multi-label sentiment classification

Author: Clayton Durepos
Version: 05.04.2025
Contact: clayton.durepos@maine.edu
"""

MODEL_NM = 'google-bert/bert-base-cased'
DATA_PATH = f'data{os.path.sep}multimodal_sentiment_dataset.csv'
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

def main():
    df = pd.read_csv(DATA_PATH)

    # Train Data pre-processing
    train_data = df.loc[df['split'] == 'train'][['caption', 'labels']]
    train_data['labels'] = train_data.apply(
        lambda row: label_map[ row['labels'] ],
        axis=1
    )

    train_data = Dataset.from_pandas(train_data).remove_columns(['__index_level_0__'])

    # Evaluation Data pre-processing
    eval_data = df.loc[df['split'] == 'eval'][['caption', 'labels']]
    eval_data['labels'] = eval_data.apply(
        lambda row: label_map[ row['labels'] ],
        axis=1
    )

    eval_data = Dataset.from_pandas(eval_data).remove_columns(['__index_level_0__'])

    # Tokenize captions
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NM)
    def tokenize_fn(batch):
        return tokenizer(batch['caption'],
                         padding='max_length',
                         truncation=True,
                         max_length=256,
                         add_special_tokens=True,
                         return_tensors="pt")

    train_data = train_data.map(tokenize_fn, batched=True)
    eval_data = eval_data.map(tokenize_fn, batched=True)

    model = TextClassifier(num_classes=9, base_model=MODEL_NM)
    training_args = get_args(output_dir=f"model_training{os.path.sep}bert_test_trainer",
                             learning_rate=2e-5)    # As used in BERT - Devlin et al., NAACL 2019

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
    torch.save(model.state_dict(), f"models{os.path.sep}bert-dict.pt")

if __name__ == "__main__":
    main()