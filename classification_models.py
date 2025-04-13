from torch import nn
from transformers import AutoModel


class MLPHeader(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class TextClassifier(nn.Module):
    def __init__(self, base_model:str, num_classes):
        super().__init__()
        self.base = AutoModel.from_pretrained(base_model)
        self.classifier = MLPHeader(self.base.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        pooled = self.base(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids
                           ).pooler_output
        return self.classifier(pooled)

class ImageClassifier(nn.Module):
    def __init__(self, base_model:str, num_classes):
        super().__init__()
        self.base = AutoModel.from_pretrained(base_model)
        self.classifier = MLPHeader(self.base.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        pooled = self.base(pixel_values=pixel_values.squeeze(1)).pooler_output
        return self.classifier(pooled)