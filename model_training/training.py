import os
import torch
from transformers import Trainer
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, EarlyStoppingCallback, TrainerCallback
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

"""
A small file for shared arguments across model training scripts

Author: Clayton Durepos
Version: 04.29.2025
Contact: clayton.durepos@maine.edu
"""

# Tensorboard monitoring
writer = SummaryWriter(log_dir=f"model_training{os.path.sep}logs")

# Custom trainer for weighted classes
class WeightedTrainer(Trainer):

    # Overload to store class_weights - Constructor uses raw weights from sklearn.utils.class_weight.compute_class_weight for ease of use
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Initialize loss_function, dynamically move class_weights
        loss_fn = torch.nn.CrossEntropyLoss( weight=self.class_weights.to( logits.device ) )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

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
            # Evaluation & Saving
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="accuracy",
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
            logging_dir=f"model_training{os.path.sep}logs",
            logging_steps=64,
            report_to="tensorboard"             # For visualizing metrics & performance
        )

# Custom callback class for monitoring modal weight in multi-modal models
class AlphaCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        model = kwargs['model']
        if hasattr(model, 'alpha'):
            alpha_val = torch.sigmoid(model.alpha).item()

            if alpha_val > 0.5:
                dom = 'language-dominant'
            elif alpha_val < 0.5:
                dom = 'vision-dominant'
            else:
                dom = 'balanced'

            print(f'\nAlpha (modal weight): {alpha_val:.4f} ({dom})')
            if writer is not None:
                writer.add_scalar('alpha',
                                  alpha_val,
                                  global_step=state.global_step)

        return control


# Callbacks
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001)
alpha_monitoring_callback = AlphaCallback()