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

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.base(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)

        logits = self.classifier(outputs.pooler_output)
        return SequenceClassifierOutput(logits=logits)

class ImageClassifier(Classifier):
    def __init__(self, num_classes:int, base_model:str):
        super().__init__(base_model)
        self.classifier = MLPHeader(self.base.config.hidden_size, num_classes)

    def forward(self, pixel_values, **kwargs):
        outputs = self.base(pixel_values=pixel_values)

        logits = self.classifier(outputs.pooler_output)
        return SequenceClassifierOutput(logits=logits)


class MultimodalClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_model = AutoModel.from_pretrained('google-bert/bert-base-cased')
        self.text_norm = nn.LayerNorm(self.text_model.config.hidden_size)

        self.image_model = AutoModel.from_pretrained('google/vit-base-patch16-224')
        self.image_norm = nn.LayerNorm(self.image_model.config.hidden_size)

        # Ensure image_model and text_model embeddings are of identical vector space
        assert self.image_model.config.hidden_size == self.text_model.config.hidden_size

        # Classification head
        self.classifier = MLPHeader(self.text_model.config.hidden_size+self.image_model.config.hidden_size,9)

    def forward(self, pixel_values, input_ids, attention_mask, token_type_ids=None, **kwargs):
        image_outputs = self.image_model(pixel_values)
        text_outputs = self.text_model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)

        logits = self.classifier(                                   # Fused embeddings through concatenation
            torch.cat([                                      # As seen in ALBEF, Li et al., 2021
                self.text_norm(text_outputs.pooler_output),
                self.image_norm(image_outputs.pooler_output)
            ], dim=1)
        )

        return SequenceClassifierOutput(logits=logits)