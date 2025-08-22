import pandas as pd
import torch
import os
import unicodedata
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from transformers import BitsAndBytesConfig
from transformers import pipeline

# Local variable set to allow larger batch sizes on the local device
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

class CaptionGenerator:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", max_caption_length=256):
        self.max_new_tokens = max_caption_length
        self.model_id = model_id
        self.pipe = None


    def setup(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.pipe = pipeline("image-text-to-text",
                             model=self.model_id,
                             trust_remote_code=True,
                             model_kwargs={"quantization_config": quantization_config,
                                           "torch_dtype": torch.float16},
                             device_map="auto")


    def __call__(self,
                 data: str | list[str],
                 prompt="USER: <image>\nCaption this image\nASSISTANT:"):
        """
        Generates captions for an image or a list of images

        Args:
            data (str | list):  A single image path or a list of image paths

        Returns:
            A list of captions generated for image(s) provided
        """

        if isinstance(data, str):
            data = [data]

        images = [Image.open(path) for path in data]
        outputs = self.pipe(images=images, text=[prompt] * len(images),
                            generate_kwargs={"max_new_tokens": self.max_new_tokens})

        captions = [output["generated_text"].replace(prompt, "") for output in outputs]
        [image.close() for image in images]
        return captions


    def from_csv(self,
                          in_path:str | Path,
                          out_path:str | Path,
                          prompt="USER: <image>\nCaption this image\nASSISTANT:",
                          batch_size=16):
        """
        Generates captions for a CSV file with image paths

        Args:
            in_path (str):  path of the existing csv file
            out_path (str): path of the csv file for new dataset to be saved
            prompt (str):   a prompt for the LLM
            batch_size (int):  The number of images to be processed and captioned at a time
        """
        df = pd.read_csv(in_path)

        if os.path.exists(out_path):
            new_df = pd.read_csv(out_path)
        else:
            new_df = df.copy()
            new_df["caption"] = ""

        images = new_df.loc[
            new_df["caption"].isna() | (new_df["caption"] == ""),
            ["art_style", "painting"]
        ].drop_duplicates(subset=["painting"])[['art_style', 'painting']]

        images["local_image_path"] = images.apply(
            lambda row: os.path.join('wikiart',
                            row["art_style"],
                            f'{unicodedata.normalize("NFC", row["painting"])}.jpg'
                        ),
            axis=1
        )

        for i in tqdm(range(0, len( images['local_image_path'] ), batch_size), desc="Generating captions"):
            batch = images.iloc[i:i + batch_size]
            captions = self(data=batch["local_image_path"].tolist(), prompt=prompt)

            for (_, row), caption in zip(batch.iterrows(), captions):
                new_df.loc[
                    new_df["painting"] == row["painting"],
                    "caption"
                ] = caption

            new_df.to_csv(out_path, index=False)


def main():
    # Generate captions for 80K WikiArt paintings in the ArtEmis dataset
    in_path = os.path.join('data', 'datasets', 'artemis-temp.csv')
    out_path = os.path.join('data', 'datasets', 'multimodal-sentiment-dataset.csv')

    cap_gen = CaptionGenerator("llava-hf/llava-1.5-7b-hf")
    cap_gen.setup()
    cap_gen.from_csv(in_path,
                     out_path,
                     prompt="USER: <image>\nCaption this image\nASSISTANT:",
                     batch_size=16) # Adjust to GPU (VRAM) capacity

if __name__ == "__main__":
    # Use this block for testing
    main()



