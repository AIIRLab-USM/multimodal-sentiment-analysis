import pandas as pd
import torch
import os
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from transformers import pipeline

from data_customization import process_artemis
from utils.image_dataset import ImageDataset
from torch.utils.data import DataLoader

import data_customization.process_artemis

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
                 prompt="USER: <image>\nCaption this image\nASSISTANT:",
                 batch_size=16):
        """
        Generates captions for an image or a list of images

        Args:
            data (str | list):  A single image path or a list of image paths
            batch_size (int):   The number of images to be processed and captioned at a time

        Returns:
            A list of captions generated for image(s) provided
        """
        captions = []

        if isinstance(data, str):
            data = [data]

        img_dataset = ImageDataset(data)
        dataloader = DataLoader(img_dataset, batch_size=batch_size, num_workers=4, persistent_workers=False, shuffle=True, collate_fn=lambda x: x)

        progress_bar = tqdm(
            total=len(img_dataset),
            desc="Generating captions"
        )

        for batch_images in dataloader:
            outputs = self.pipe(images=batch_images, text=[prompt] * len(batch_images),
                                generate_kwargs={"max_new_tokens": self.max_new_tokens})

            batch_captions = [output["generated_text"].replace(prompt, "") for output in outputs]
            captions.extend(batch_captions)

            progress_bar.update(len(batch_images))

        return captions


    def from_csv(self,
                          in_path,
                          out_path=None,
                          prompt="USER: <image>\nCaption this painting\nASSISTANT:",
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
        captions = []

        if "local_image_path" not in df.columns:
            raise ValueError("CSV file must have 'local_image_path' column")

        if os.path.exists(out_path):
            new_df = pd.read_csv(out_path)
        else:
            new_df = df.copy()
            new_df["caption"] = ""

        img_paths = new_df.loc[
            new_df["caption"].isna() | (new_df["caption"] == ""), "local_image_path"].unique().tolist()

        img_dataset = ImageDataset(img_paths)
        dataloader = DataLoader(img_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

        progress_bar = tqdm(
            total=len(img_dataset),
            desc="Generating captions"
        )

        for batch in dataloader:
            batch_images, batch_paths = zip(*batch)
            outputs = self.pipe(images=batch_images,
                                text=[prompt] * len(batch_images),
                                generate_kwargs={"max_new_tokens": self.max_new_tokens}
                                )

            batch_captions = [output["generated_text"].replace(prompt, "") for output in outputs]
            captions.extend(batch_captions)

            for img_path, caption in zip(batch_paths, batch_captions):
                new_df.loc[new_df["local_image_path"] == img_path, "caption"] = caption

            if out_path:
                new_df.to_csv(out_path, index=False)
            progress_bar.update(len(batch_images))

        if not out_path:
            print(captions)


def main():
    # Generate proper data file
    process_artemis.main()

    # Generate captions for 80K WikiArt paintings in the ArtEmis dataset
    in_path = f"data{os.path.sep}custom_data{os.path.sep}custom_artemis.csv"
    out_path = f"data{os.path.sep}caption_data{os.path.sep}mistica_dataset.csv"

    cap_gen = CaptionGenerator("llava-hf/llava-1.5-7b-hf")
    cap_gen.setup()
    cap_gen.from_csv(in_path, out_path, batch_size=16) # Adjust to GPU (VRAM) capacity

if __name__ == "__main__":
    # Use this block for testing
    main()



