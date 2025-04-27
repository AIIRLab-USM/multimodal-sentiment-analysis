import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import ast
import pandas as pd
from PIL import Image
from datasets import Dataset
from classification_models import *
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, Trainer, AutoTokenizer
from training import training_args, compute_metrics, early_stop_callback

"""
A short script for fine-tuning CLIP on a multimodal sentiment classification task

Author: Clayton Durepos
Version: 04.24.2025
Contact: clayton.durepos@maine.edu
"""

DATA_PATH = f'..{os.path.sep}data{os.path.sep}multimodal_sentiment_dataset.csv'

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
model_dict = {
    'fused': FusedMMClassifier,
    'unified': UnifiedMMClassifier
}

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

    for model_type, model_class in model_dict.items():
        model = model_class()

        # Adjust LR, weight decay, and warm-up for lightweight, fused-logits model
        if model_type == 'fused':
            training_args.learning_rate = 1e-2
            training_args.weight_decay = 0.0
            training_args.warmup_ratio = 0

        else:
            training_args.learning_rate = 1e-5      # Used for multimodal models in
                                                    # LXMERT, Tan and Bansal, EMNLP-IJCNLP 2019
                                                    # UNITER, Chan et al. ECCV 2020

            training_args.weight_decay = 0.01
            training_args.warmup_ratio = 0.1

        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_data,
            eval_dataset=eval_data,
            callbacks=early_stop_callback
        )

        trainer.train()
        torch.save(model.state_dict(), f"../models/{model_type}-dict.pt")

        # Memory management
        del model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
