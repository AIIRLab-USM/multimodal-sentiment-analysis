import requests
import pandas as pd
from tqdm import tqdm
import os

INPUT_PATH = f'processed_data{os.path.sep}custom_artpedia.csv'

def main():
    try:
        ap_df = pd.read_csv(INPUT_PATH)
    except Exception as e:
        print("Missing custom Artpedia file - Use 'process_artpedia.py' first.")
        return

    os.makedirs('artpedia', exist_ok=True)

    unavailable_indices = []
    with (tqdm(total=len(ap_df)) as pbar):
        for index, row in ap_df.iterrows():
            response = requests.get(row["img_url"], headers={"User-Agent": "Mozilla/5.0"}, stream=True)
            if not response.ok:
                unavailable_indices.append(index)
                continue

            with open(row["local_image_path"], "wb") as handler:
                handler.write(response.content)

            pbar.update(1)

    ap_df.drop(index=unavailable_indices, inplace=True)
    ap_df.to_csv(f'processed_data{os.path.sep}custom_artpedia.csv')

if __name__ == "__main__":
    main()