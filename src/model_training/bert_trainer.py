import os
import numpy as np
import pandas as pd
from datasets import Dataset
from src.classification_models import *
from transformers import AutoTokenizer, EarlyStoppingCallback
from sklearn.utils.class_weight import compute_class_weight
from src.model_training.training import compute_metrics, get_args, WeightedTrainer, CELTrainer

"""
A short script for fine-tuning BERT and RoBERTa models for multi-label sentiment classification

Author: Clayton Durepos
Version: 05.04.2025
Contact: clayton.durepos@maine.edu
"""

MODEL_NM = 'google-bert/bert-base-cased'
DATA_PATHS = [
    os.path.join('data', 'datasets', 'multimodal_sentiment_dataset.csv'),
    os.path.join('data', 'datasets', 'bal_multimodal_sentiment_dataset.csv')
]
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
    os.makedirs('models', exist_ok=True)

    for model_num, path in enumerate(DATA_PATHS, start=1):
        df = pd.read_csv(path)

        # Train Data pre-processing
        train_df = df.loc[df['split'] == 'train'][['caption', 'labels']]
        train_df['labels'] = train_df['labels'].apply(lambda x: label_map[x])

        train_data = Dataset.from_pandas(train_df).remove_columns(['__index_level_0__'])

        # Evaluation Data pre-processing
        eval_df = df.loc[df['split'] == 'eval'][['caption', 'labels']]
        eval_df['labels'] = eval_df['labels'].apply(lambda x: label_map[x])

        eval_data = Dataset.from_pandas(eval_df).remove_columns(['__index_level_0__'])

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

        # Train
        if model_num == 1:
            training_args = get_args(learning_rate=2e-5,    # As used in BERT - Devlin et al., NAACL 2019
                                     num_train_epochs=10)

            class_weights = compute_class_weight(class_weight='balanced',
                                                 classes=np.arange(len(label_map)),
                                                 y=train_df['labels'].tolist())

            trainer = WeightedTrainer(
                model=model,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=train_data,
                eval_dataset=eval_data,
                class_weights=class_weights,
                callbacks=[EarlyStoppingCallback(early_stopping_threshold=0.001, early_stopping_patience=2)]
            )
        elif model_num == 2:
            training_args = get_args(learning_rate=2e-5,  # As used in BERT - Devlin et al., NAACL 2019
                                     num_train_epochs=50)

            trainer = CELTrainer(
                model=model,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=train_data,
                eval_dataset=eval_data,
                callbacks=[EarlyStoppingCallback(early_stopping_threshold=0.001, early_stopping_patience=10)]
            )

        else:
            raise NotImplementedError

        trainer.train()

        # Save
        torch.save(model.state_dict(), f'models{os.path.sep}bert-dict.pt' if model_num==1 \
                   else f'models{os.path.sep}bal-training_bert-dict.pt')

if __name__ == "__main__":
    main()