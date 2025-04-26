import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import AutoModel


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

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.base(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)

        logits = self.classifier(outputs.pooler_output)
        if labels is not None:
            loss_fn = BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return {"logits": logits, "loss": loss}

        return {"logits": logits}

class ImageClassifier(Classifier):
    def __init__(self, num_classes:int, base_model:str):
        super().__init__(base_model)
        self.classifier = MLPHeader(self.base.config.hidden_size, num_classes)

    def forward(self, pixel_values, labels=None):
        outputs = self.base(pixel_values=pixel_values)

        logits = self.classifier(outputs.pooler_output)
        if labels is not None:
            loss_fn = BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return {"logits": logits, "loss": loss}

        return {"logits": logits}

class FusedMMClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_model = TextClassifier(base_model='google-bert/bert-base-cased', num_classes=9)
        self.text_model.load_state_dict(torch.load('../models/bert-dict.pt'))
        for p in self.text_model.parameters():
            p.requires_grad = False

        self.image_model = ImageClassifier(base_model='google/vit-base-patch16-224', num_classes=9)
        self.image_model.load_state_dict(torch.load('../models/vit-dict.pt'))
        for p in self.image_model.parameters():
            p.requires_grad = False

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        with torch.no_grad():
            image_outputs = self.image_model(pixel_values)
            text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        a = torch.sigmoid(self.alpha)
        logits = a * text_outputs['logits'] + (1 - a) * image_outputs['logits']
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fn = BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return {"logits": logits, "loss": loss}

        return {"logits": logits}


class UnifiedMMClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_model = AutoModel.from_pretrained('google-bert/bert-base-cased')
        self.image_model = AutoModel.from_pretrained('google/vit-base-patch16-224')

        assert self.image_model.config.hidden_size == self.text_model.config.hidden_size

        self.classifier = MLPHeader(self.text_model.config.hidden_size,9)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        image_outputs = self.image_model(pixel_values)
        text_outputs = self.text_model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)

        a = torch.sigmoid(self.alpha)
        logits = self.classifier(
            a * image_outputs.pooler_output + (1 - a) * text_outputs.pooler_output
        )

        if labels is not None:
            labels = labels.to(logits.device)
            loss_fn = BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return {"logits": logits, "loss": loss}

        return {"logits": logits}