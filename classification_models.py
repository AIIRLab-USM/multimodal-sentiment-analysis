import torch
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class MLPHeader(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Dropout(0.3),                    # Used in VGG16-MLP, Mogan et al., Appl. Sci. 2022
            nn.Linear(dim // 4, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class Classifier(nn.Module):
    def __init__(self, base_model:str):
        super().__init__()
        self.base = AutoModel.from_pretrained(base_model)
        self.classifier = None


class TextClassifier(Classifier):
    def __init__(self,  num_classes:int, base_model:str):
        super().__init__(base_model)
        self.classifier = MLPHeader(self.base.config.hidden_size, num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.base(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)

        logits = self.classifier(outputs.pooler_output)
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

class ImageClassifier(Classifier):
    def __init__(self, num_classes:int, base_model:str):
        super().__init__(base_model)
        self.classifier = MLPHeader(self.base.config.hidden_size, num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pixel_values, labels=None):
        outputs = self.base(pixel_values=pixel_values)

        logits = self.classifier(outputs.pooler_output)
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


class MultimodalClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_model = AutoModel.from_pretrained('FacebookAI/roberta-base')
        self.text_norm = nn.LayerNorm(self.text_model.config.hidden_size)

        self.image_model = AutoModel.from_pretrained('google/vit-base-patch16-224')
        self.image_norm = nn.LayerNorm(self.image_model.config.hidden_size)

        # Ensure image_model and text_model embeddings are of identical vector space
        assert self.image_model.config.hidden_size == self.text_model.config.hidden_size

        # Alpha weight, classification head
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.classifier = MLPHeader(self.text_model.config.hidden_size,9)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        image_outputs = self.image_model(pixel_values)
        text_outputs = self.text_model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)

        # Clamp learnable weight to [0, 1]
        a = torch.sigmoid(self.alpha)
        logits = self.classifier(
                    # Calculate the weighted sum of normalized image and text embeddings
                    a  * self.text_norm(text_outputs.pooler_output) +  (1 - a) * self.image_norm(image_outputs.pooler_output)
        )

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )