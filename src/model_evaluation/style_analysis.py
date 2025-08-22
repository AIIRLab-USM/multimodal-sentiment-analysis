import pandas as pd

def main():
    # Load results
    df = pd.read_csv("../../data/evaluation/multimodal_results.csv")

    # Extract genre from path: "wikiart/{genre}/{image_name}.jpg"
    df["genre"] = df["local_image_path"].apply(lambda x: x.split("/")[1] if isinstance(x, str) else None)

    # Check total samples
    total = len(df)

    results = []
    for genre, gdf in df.groupby("genre"):
        count = len(gdf)
        # % of total dataset
        percentage = (count / total) * 100

        # Misclassification rate
        misclassified = (gdf["true_label"] != gdf["pred_label"]).sum()
        mis_rate = (misclassified / count) * 100

        results.append((genre, percentage, mis_rate))

    print(f"{'Genre':<20}{'% of Dataset':>15}{'Misclass. Rate':>20}")
    print("-" * 55)
    for genre, perc, mis in sorted(results, key=lambda x: x[0]):
        print(f"{genre:<20}{perc:15.2f}{mis:20.2f}")

if __name__ == "__main__":
    main()
