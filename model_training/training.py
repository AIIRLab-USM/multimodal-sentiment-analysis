import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, EarlyStoppingCallback, TrainerCallback
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss

"""
A small file for shared arguments across model training scripts

Author: Clayton Durepos
Version: 04.29.2025
Contact: clayton.durepos@maine.edu
"""

# Tensorboard monitoring
writer = SummaryWriter(log_dir="./logs")

# Additional metrics for monitoring model performance
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) >= 0.5).float().numpy()

    return {
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
        "hamming_loss": hamming_loss(labels, preds)
    }

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
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)
alpha_monitoring_callback = AlphaCallback()

# Training arguments
training_args = TrainingArguments(
            output_dir="./test_trainer",

            # Evaluation & Saving
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="f1",
            load_best_model_at_end=True,
            greater_is_better=True,
            save_total_limit=2,

            # Hyperparameters                   Reasoning, Citation

            num_train_epochs=10,                # Used in UNITER, Chen et al., ECCV 2020
                                                # ViLT, Kim et al., ICML 2021
                                                # BERT, Devlin et al., NAACL 2019

            warmup_ratio=0.1,                   # Used in UNITER, Chen et al., ECCV 2020
                                                # ViLT, Kim et al., ICML 2021
                                                # BERT, Devlin et al., NAACL 2019

            learning_rate=1e-5,                 # Used for multimodal models in
                                                # LXMERT, Tan and Bansal, EMNLP-IJCNLP 2019
                                                # UNITER, Chan et al. ECCV 2020

            weight_decay=0.01,                  # Used in UNITER, Chen et al., ECCV 2020
                                                # ViLT, Kim et al., ICML 2021
                                                # BERT, Devlin et al., NAACL 2019

            fp16=True,                          # Speed-up training

            per_device_train_batch_size=16,     # Maximum sized allowed by local compute resources
            per_device_eval_batch_size=32,      # (2x NVIDIA GeForce RTX 2080 Ti)

            # Logging
            logging_dir="./logs",
            logging_steps=64,
            report_to="tensorboard"             # For visualizing metrics & performance
        )