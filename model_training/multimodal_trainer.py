import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import ast
import pandas as pd
from PIL import Image
from datasets import Dataset
from classification_models import *
from training import training_args, compute_metrics, early_stop_callback
from transformers import AutoImageProcessor, Trainer, AutoTokenizer

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


def main():
    df = pd.read_csv(DATA_PATH)

    # Modify image_paths to comply with directory structure
    df['local_image_path'] = df.apply(
        lambda row: '../' + row['local_image_path'], axis=1
    )

    # Train Data pre-processing
    train_data = df.loc[df['split'] == 'train'][['local_image_path', 'caption', 'labels']]
    train_data['labels'] = train_data.apply(
        lambda row: [float(x) for x in ast.literal_eval(row['labels'])],
        axis=1
    )

    train_data = Dataset.from_pandas(train_data).remove_columns(['__index_level_0__'])
    # train_data = train_data.select(range(int(len(train_data)*0.01)))      # For testing

    # Evaluation Data pre-processing
    eval_data = df.loc[df['split'] == 'eval'][['local_image_path', 'caption', 'labels']]
    eval_data['labels'] = eval_data.apply(
        lambda row: [float(x) for x in ast.literal_eval(row['labels'])],
        axis=1
    )

    eval_data = Dataset.from_pandas(eval_data).remove_columns(['__index_level_0__'])
    # eval_data = eval_data.select(range(int(len(eval_data) * 0.01)))       # For testing

    # Process images
    def process_fn(batch):
        with Image.open(batch['local_image_path']) as img:
            img_inputs = processor(images=[img], return_tensors='pt')
            txt_inputs = tokenizer(text=[batch['caption']],
                                   padding='max_length',
                                   truncation=True,
                                   max_length=256,
                                   add_special_tokens=True,
                                   return_tensors="pt")

            return {
                'pixel_values': img_inputs['pixel_values'][0],
                'input_ids': txt_inputs['input_ids'][0],
                'attention_mask': txt_inputs['attention_mask'][0],
                'token_type_ids': txt_inputs['token_type_ids'][0]
            }


    train_data, eval_data = train_data.map(process_fn, batched=False), eval_data.map(process_fn, batched=False)

    for model_type, model_class in model_dict.items():
        model = model_class()

        # Adjust LR, weight decay, and warm-up for lightweight, fused-logits model
        if model_type == 'fused':
            training_args.learning_rate = 5e-5
            training_args.weight_decay = 0.0
            training_args.warmup_steps = 0
        else:
            training_args.learning_rate = 1e-5
            training_args.weight_decay = 0.01
            training_args.warmup_steps = 512

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
