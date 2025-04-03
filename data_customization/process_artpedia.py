import pandas as pd
import hashlib
import os


INPUT_FILE = f"..{os.path.sep}data{os.path.sep}original_data{os.path.sep}artpedia.json"

OUTPUT_FILE = f"..{os.path.sep}data{os.path.sep}custom_data{os.path.sep}custom_artpedia.csv"

def main():
    # Open input file
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError("Missing Artpedia data"
                                " - Refer to https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=35")

    try:
        artpedia_df = pd.read_json(INPUT_FILE, orient='index')

    except Exception as e:
        print(f"Error reading {INPUT_FILE} : {e}")
        return

    # Copy data to new dataframe
    new_df = artpedia_df[["img_url", "visual_sentences"]].copy()

    # Generate image IDs for all images in Artpedia dataset
    new_df["img_id"] = [hashlib.sha256(title.encode()).hexdigest()[:8] for title in artpedia_df["title"]]

    # Create local image paths
    new_df["local_image_path"] = new_df.apply(
        lambda row: f'artpedia{os.path.sep}{row["img_id"]}.jpg', axis=1
    )

    # Rearrange columns (Author's preference)
    new_df = new_df[["local_image_path", "img_id", "img_url", "visual_sentences"]]

    # Save custom data file
    try:
        new_df.to_csv(OUTPUT_FILE, index=False)

    except Exception as e:
        print(f"Error saving {OUTPUT_FILE} : {e}")

if __name__  == "__main__":
    main()
