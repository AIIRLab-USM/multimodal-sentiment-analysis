import os
import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers import TrainingArguments
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

"""
A small file for shared arguments across model training scripts

Author: Clayton Durepos
Version: 04.29.2025
Contact: clayton.durepos@maine.edu
"""

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Logarithm of probabilities \[ log(p_t) \]
        logpt = F.log_softmax(inputs, dim=1)

        # Retrieve \[ log(p_t) \] for ground-truth label
        logpt = logpt.gather(1, targets.unsqueeze(1))
        logpt = logpt.view(-1)

        # Undo logarithm to retrieve \[ p_t \]
        pt = torch.exp(logpt)

        # Compute with focal loss formula
        # \[ -w_t (1-p_t)^\gamma log(p_t) \]
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            focal_loss = self.weight.gather(0, targets) * focal_loss

        return focal_loss.mean()


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
        loss_fn = FocalLoss( gamma=2.0, weight=self.class_weights.to( logits.device ) )
        loss = loss_fn(inputs=logits, targets=labels)
        return (loss, outputs) if return_outputs else loss

# Custom Trainer for normal, external Cross Entropy Loss
class CELTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.criterion(input=logits, target=labels)
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
def get_args(learning_rate:float, num_train_epochs:int=10):
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

            num_train_epochs=num_train_epochs,                # 10 epochs Used in UNITER, Chen et al., ECCV 2020
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