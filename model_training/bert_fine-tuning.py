import pandas as pd
import evaluate
import numpy as np

from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = 'google-bert/bert-base-uncased'
DATA_PATH = '../data/caption_data/mistica_dataset.csv'
LABEL_MAP = {
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=9).to("cuda")

df = pd.read_csv(DATA_PATH).drop(columns=['index'])
df['label'] = df['label'].map(LABEL_MAP)

train_data = Dataset.from_pandas(df.loc[df['split'] == 'train'][['caption', 'label']])
eval_data = Dataset.from_pandas(df.loc[df['split'] == 'eval'][['caption', 'label']])

def tokenize_fn(batch):
    return tokenizer(batch['caption'], padding='max_length', truncation=True, max_length=256, return_tensors="pt")

train_data, eval_data = train_data.map(tokenize_fn, batched=True), eval_data.map(tokenize_fn, batched=True)

training_args = TrainingArguments(
    output_dir="/test_trainer",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# TRAINING
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=eval_data,
)

trainer.train()

# SAVING
model.save_pretrained("../ft_bert")
tokenizer.save_pretrained("../ft_bert")