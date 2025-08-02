# Multimodal Sentiment Classification

This repository contains the codebase for our research, exploring whether generating textual descriptions of artworks
and classifying those captions — alone or accompanied by visual input — more effectively captures sentiment than relying
solely on vision for classification.

---

## Abstract

Understanding sentiment in fine-art images is a complex task that may benefit from human-like reasoning through 
language. In this study, we investigate whether the intermediate step of **caption generation** — transforming image 
content into affective language — enhances sentiment classification performance. We evaluate:

- **Language** Classification
- **Vision** Classification
- **Multimodal** Classification

This pipeline allows us to assess whether converting visual signals into descriptive language leads to better emotional 
understanding by machine learning models.

---

## Method Overview

Our study consists of three parallel pipelines:

1. **Image-only**: A Vision Transformer (ViT) classifies emotion directly from artwork images.
2. **Caption-only**: We generate affective captions using LLaVA and classify them using a fine-tuned BERT model.
3. **Multimodal Fusion**: Language and vision embeddings are fused with attention and passed through an MLP for classification.

---

## Dataset

We use a modified version of the **ArtEmis** dataset:

- Images are from WikiArt.
- Affective captions are generated using a pre-trained **LLaVA** model.
-  Only the most dominant emotion is used, for each sample. Those with $>1$ dominant emotions are removed to avoid 
ambiguity.

---

## Installation

Any user wishing to replicate the experiments done using the code in this repository will need to download the ArtEmis V2.0 dataset. You can request access to this dataset at https://www.artemisdataset-v2.org/

```
git clone https://github.io/cdurepos/multimodal-sentiment-analysis.git
cd multimodal-sentiment-analysis
pip install -r requirements.txt
```

---

## Use
```
python main.py
```

The main script for this repository will run scripts to preprocess data, train the models mentioned prior, and
evaluate them. Evaluation metrics and sample-level results will be saved locally, as well as confusion matrices
for each modality and a class-distribution graph.

It is important to note that caption generation and model training can take especially long.

---

## Directory Structure

The components of the ArtEmis V2.0 dataset should be placed within the original_data directory

```
multimodal-sentiment-analysis
|--- data/
|     |--- datasets/
            |--- original_data/
|     |--- evaluation/
|     |--- plot/    
|--- models/
|--- src/
|--- main.py
|--- README.md
|--- requirements.txt
```

---

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [HuggingFace Transformers](https://huggingface.co/)
- [ArtEmis Dataset](https://www.artemisdataset.org)

