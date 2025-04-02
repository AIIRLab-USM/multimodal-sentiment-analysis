from transformers import AutoImageProcessor
from PIL import Image
import pandas as pd
import os

class ImageProcessor:
    def __init__(self, processor:AutoImageProcessor=None, model_nm:str=None):
        self.processor = processor if processor is not None else AutoImageProcessor.from_pretrained(model_nm)

    def __call__(self, img_path:str):
        img = Image.open(img_path)
        return self.processor(img)

    def from_csv(self, in_path:str, out_path:str):
        if os.path.exists(out_path):
            return

        img_df = pd.read_csv(in_path)
        pxl_df = img_df.copy()

        try:
            pxl_df = pxl_df.drop_duplicates(subset='local_image_path')
        except ValueError as e:
            print(f'Input data is missing \'local_image_path\' column - {e}')
            return

        # Process images
        pxl_df["pxl_val"] = pxl_df.progress_apply(
            lambda row: [self.processor(Image.open(row["local_image_path"]), return_tensors="pt")], axis=1
        )

        # Drop unnecessary values
        pxl_df = pxl_df[["local_image_path", "pxl_val"]]

        # Merge dataframes
        img_df = img_df.merge(pxl_df, on="local_image_path", how="left")

        # Drop unnecessary columns (again)
        img_df = img_df[["split", "pxl_val", "emotion"]]

        # Save data
        img_df.to_csv(out_path, index=False)
