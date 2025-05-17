import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from src.classification_models import MultimodalClassifier
from transformers import AutoImageProcessor, AutoTokenizer, EarlyStoppingCallback
from src.model_training.training import (
    get_args, compute_metrics, WeightedTrainer, CELTrainer
)

"""
A short script for fine-tuning a multimodal classification model on a sentiment classification task

Author: Clayton Durepos
Version: 05.04.2025
Contact: clayton.durepos@maine.edu
"""

DATA_PATHS = [
    # os.path.join('data', 'datasets', 'multimodal_sentiment_dataset.csv'),
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
            'labels': label_map[ row['labels'] ]
        }


def main():
    os.makedirs('models', exist_ok=True)

    for model_num, path in enumerate(DATA_PATHS, start=1):
        df = pd.read_csv(path)[['split', 'local_image_path', 'caption', 'labels']]

        # Train dataset initialization
        train_df = df.loc[df['split'] == 'train'][['local_image_path', 'caption', 'labels']].copy()
        # train_df = train_df.iloc[:int(len(train_df) * 0.01)]       # For testing

        train_data = MMProcessingDataset(train_df)

        # Evaluation dataset initialization
        eval_df = df.loc[df['split'] == 'eval'][['local_image_path', 'caption', 'labels']].copy()
        # eval_df = eval_dfiloc[:int(len(eval_df) * 0.01)]    # For testing

        eval_data = MMProcessingDataset(eval_df)

        # Delete original DataFrame to free memory
        del df

        model = MultimodalClassifier()

        # Train
        # Imbalanced data
        if model_num == 1:
            training_args = get_args(learning_rate=1e-5,    # Used for multimodal models in
                                                            # LXMERT, Tan and Bansal, EMNLP-IJCNLP 2019
                                                            # UNITER, Chan et al. ECCV 2020
                                     num_train_epochs=10)

            class_weights = compute_class_weight(class_weight='balanced',
                                                 classes=np.arange(len(label_map)),
                                                 y=train_df['labels'].map(label_map).tolist()
                                                 )

            trainer = WeightedTrainer(
                model=model,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=train_data,
                eval_dataset=eval_data,
                class_weights=class_weights,
                callbacks=[EarlyStoppingCallback(early_stopping_threshold=0.001, early_stopping_patience=2)],
            )

        # Balanced data
        elif model_num == 2:
            training_args = get_args(learning_rate=1e-5,    # Used for multimodal models in
                                                            # LXMERT, Tan and Bansal, EMNLP-IJCNLP 2019
                                                            # UNITER, Chan et al. ECCV 2020
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
        torch.save(model.state_dict(),
                   f'models{os.path.sep}multimodal-dict.pt' if model_num == 1 \
                       else f'models{os.path.sep}bal-training_multimodal-dict.pt')

if __name__ == "__main__":
    main()
