import os
import ast
import pandas as pd
from datasets import Dataset
from src.classification_models import *
from transformers import AutoTokenizer
from src.model_training.training import compute_metrics, get_args, early_stopping_callback, KLTrainer

"""
A short script for fine-tuning BERT and RoBERTa models for multi-label sentiment classification

Author: Clayton Durepos
Version: 08.22.2025
Contact: clayton.durepos@maine.edu
"""

MODEL_NM = 'google-bert/bert-base-cased'
DATA_PATH = os.path.join('data', 'datasets', 'multimodal_sentiment_dataset.csv')

def main():
    os.makedirs('models', exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Train Data pre-processing
    train_data = df.loc[df['split'] == 'train'][['caption', 'probs']]
    train_data['probs'] = train_data['probs'].progress_apply( lambda x: ast.literal_eval(x) ) # String to array
    train_data = Dataset.from_pandas( train_data, preserve_index=False)

    # Evaluation Data pre-processing
    eval_data = df.loc[df['split'] == 'eval'][['caption', 'label', 'probs']].copy()
    eval_data = Dataset.from_pandas(eval_data, preserve_index=False)

    # Tokenize captions
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NM)
    def tokenize_fn(batch):
        return tokenizer(batch['caption'],
                         padding='max_length',
                         truncation=True,
                         max_length=256,
                         add_special_tokens=True)

    train_data = train_data.map(tokenize_fn, batched=True).with_format("torch")
    eval_data = eval_data.map(tokenize_fn, batched=True).with_format("torch")

    # Tokenizer no longer needed, free memory
    del tokenizer

    model = TextClassifier(num_classes=9, base_model=MODEL_NM)
    training_args = get_args(learning_rate=2e-5)    # As used in BERT - Devlin et al., NAACL 2019

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
    torch.save(model.state_dict(), f'models{os.path.sep}bert-dict.pt')

if __name__ == "__main__":
    main()