import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def main():
    # This is the first of the two plot scripts to be ran, make sure plot dir exists
    os.makedirs(f"data{os.path.sep}plot", exist_ok=True)

    # Make bootstrap_plotting dir
    os.makedirs(os.path.join( "data", "plot", "bootstrap_testing"), exist_ok=True)

    # Load predictions
    image_df = pd.read_csv("data/evaluation/image_results.csv")
    text_df = pd.read_csv("data/evaluation/text_results.csv")
    multi_df = pd.read_csv("data/evaluation/multimodal_results.csv")

    # Ensure alignment
    assert all(image_df["true_label"] == text_df["true_label"])
    assert all(image_df["true_label"] == multi_df["true_label"])

    true = np.array(image_df["true_label"])
    image_pred = np.array(image_df["pred_label"])
    text_pred = np.array(text_df["pred_label"])
    multi_pred = np.array(multi_df["pred_label"])

    # Bootstrap function
    def bootstrap_diff(true_labels, model_a_preds, model_b_preds, num_samples=10000):
        n = len(true_labels)
        rng = np.random.default_rng(seed=42)
        acc_diffs = []

        for _ in range(num_samples):
            idx = rng.integers(0, n, n)  # sample with replacement
            acc_a = accuracy_score(true_labels[idx], model_a_preds[idx])
            acc_b = accuracy_score(true_labels[idx], model_b_preds[idx])
            acc_diffs.append(acc_a - acc_b)

        acc_diffs = np.array(acc_diffs)
        ci_lower = np.percentile(acc_diffs, 2.5)
        ci_upper = np.percentile(acc_diffs, 97.5)
        p_val = np.mean(acc_diffs <= 0) if np.mean(acc_diffs) > 0 else np.mean(acc_diffs >= 0)

        return acc_diffs, ci_lower, ci_upper, p_val

    # Compare models
    comparisons = [
        ("Image vs Text", image_pred, text_pred),
        ("Multimodal vs Image", multi_pred, image_pred),
        ("Multimodal vs Text", multi_pred, text_pred)
    ]

    for name, a, b in comparisons:
        assert len(a) == len(b)

        diffs, lower, upper, p = bootstrap_diff(true, a, b, num_samples= len(a) )

        print(f"\n{name}")
        print(f"  95% CI of accuracy difference: [{lower:.4f}, {upper:.4f}]")
        print(f"  p-value: {p:.12f} {'(Significant)' if p < 0.05 else '(Not Significant)'}")

        # Histogram
        plt.figure()
        plt.hist(diffs, bins=50, alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', label='no difference')
        plt.axvline(x=np.mean(diffs), color='blue', linestyle='-', label=f"mean difference")

        plt.title(name)
        plt.xlabel("Accuracy Difference (A - B)")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.legend()

        plt.savefig(
            os.path.join("data", "plot", "bootstrap_testing",
                         f"{name.replace(' ', '_').lower()}.png")
        )
