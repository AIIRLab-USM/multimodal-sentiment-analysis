import os
import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers import TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

"""
A small file for shared arguments across model training scripts

Author: Clayton Durepos
Version: 05.18.2025
Contact: clayton.durepos@maine.edu
"""

# Custom trainer for KL Divergence Loss
class KLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = F.kl_div( input= F.log_softmax(logits, dim=1), target=labels.float(), reduction='mean' )
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        labels = inputs.get("ground_truth")
        return None, logits, labels


# Additional metrics for monitoring model performance
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
    }

# Function for maintaining common arguments where necessary
def get_args(learning_rate:float):
    return TrainingArguments(
            output_dir= os.path.join( 'src', 'model_training', 'trainer_output' ),
            remove_unused_columns=False,        # Model doesn't take labels - They should remain in inputs for loss_fn

            # Evaluation & Saving
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="f1",
            load_best_model_at_end=True,
            greater_is_better=True,
            save_total_limit=3,

            # Hyperparameters                   Reasoning, Citation

            num_train_epochs=10,                # Used in UNITER, Chen et al., ECCV 2020
                                                # ViLT, Kim et al., ICML 2021
                                                # BERT, Devlin et al., NAACL 2019

            warmup_ratio=0.1,                   # Used in UNITER, Chen et al., ECCV 2020
                                                # ViLT, Kim et al., ICML 2021
                                                # BERT, Devlin et al., NAACL 2019

            learning_rate=learning_rate,

            weight_decay=0.01,                  # Used in UNITER, Chen et al., ECCV 2020
                                                # ViLT, Kim et al., ICML 2021
                                                # BERT, Devlin et al., NAACL 2019

            fp16=True,                          # Speed-up training

            per_device_train_batch_size=16,     # Maximum sized allowed by local compute resources
            per_device_eval_batch_size=32,      # (2x NVIDIA GeForce RTX 2080 Ti)

            # Logging
            logging_dir=os.path.join( 'src', 'model_training', 'logs'),
            logging_steps=64,
            report_to="tensorboard"             # For visualizing metrics & performance
        )


# Callbacks
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001)