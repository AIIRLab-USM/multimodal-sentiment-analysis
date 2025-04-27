import os
import ast
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from classification_models import TextClassifier
from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

data_path = os.path.join('..', 'data', 'multimodal_sentiment_dataset.csv')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict_paths = {
    'google-bert/bert-base-cased': os.path.join('..', 'models', 'bert-dict.pt'),
    'FacebookAI/roberta-base': os.path.join('..', 'models', 'roberta-dict.pt')
}

def main():
    # Load and pre-process testing data
    df = pd.read_csv(data_path)
    test_df = df[df['split'] == 'test'].copy()[['caption', 'labels']]
    test_df['labels'] = test_df['labels'].apply(lambda x: torch.tensor(ast.literal_eval(x), dtype=torch.float))

    for base_model, dict_path in state_dict_paths.items():
        # Load model & tokenizer
        model = TextClassifier(base_model=base_model, num_classes=9)
        model.load_state_dict(torch.load(dict_path))
        model.to(device)
        model.eval()

        # Tokenize
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokens = tokenizer(
            list(test_df['caption']),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        labels_tensor = torch.stack(test_df['labels'].tolist())
        test_dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Evaluation
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(test_loader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask)['logits']
                preds = (torch.sigmoid(logits) >= 0.5).float()

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())


        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # Metrics
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')

        # Accuracy
        ham_acc = 1 - hamming_loss(all_labels, all_preds)   # Label-level
        acc = (all_labels == all_preds).all(axis=1).mean()  # Sample-level

        print(f"F1 Score (Weighted): {f1:.4f}")
        print(f"Precision (Weighted): {precision:.4f}")
        print(f"Recall (Weighted): {recall:.4f}")
        print(f"Hamming Accuracy: {ham_acc:.4f}")
        print(f"Accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()
