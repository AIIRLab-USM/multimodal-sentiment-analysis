import os
import math
import json
from datetime import datetime

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join("data", "evaluation")
os.makedirs( f'{RESULTS_DIR}{os.path.sep}mcnemar_tests', exist_ok=True )

OUT_GLOBAL_CSV = os.path.join("data", "evaluation", "mcnemar_tests", "global.csv")
OUT_BYCLASS_JSON = os.path.join("data", "evaluation", "mcnemar_tests", "by-class.json")

# ArtEmis classes
CLASS_MAP = {
    0: "amusement",
    1: "anger",
    2: "awe",
    3: "contentment",
    4: "disgust",
    5: "excitement",
    6: "fear",
    7: "sadness",
    8: "something else",
}

PAIRS = [
    ("Image vs Text",       "image_results.csv",      "text_results.csv"),
    ("Multimodal vs Image", "multimodal_results.csv", "image_results.csv"),
    ("Multimodal vs Text",  "multimodal_results.csv", "text_results.csv"),
]

# helpers
def _binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided p-value for Binomial(n, 0.5) via doubled smaller tail."""
    if n == 0:
        return 1.0
    k = int(k); n = int(n)
    k_tail = min(k, n - k)
    log_half_n = -n * math.log(2.0)

    def logC(nn, rr):
        return math.log(math.comb(nn, rr))

    logsum = None
    for i in range(k_tail + 1):
        val = logC(n, i) + log_half_n
        logsum = val if logsum is None else math.log(math.exp(logsum) + math.exp(val))
    p = 2.0 * math.exp(logsum)
    return min(1.0, p)

def mcnemar_pvalue(b: int, c: int):
    """Return (test_type, statistic, p_value). Saved files will only keep p-values."""
    n = b + c
    if n == 0:
        return "degenerate", 0.0, 1.0
    if n < 25:
        p = _binom_two_sided_p(min(b, c), n)
        return "exact", float("nan"), p
    # chi-square with continuity correction
    stat = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(stat / 2.0))  # df=1
    return "chi2_cc", stat, p

def holm_bonferroni(pvals):
    """Holm–Bonferroni adjusted p-values (step-down)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order, start=1):
        adj = min(1.0, (m - rank + 1) * pvals[i])
        running = max(running, adj)
        out[i] = running
    return out

# core testing
def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("true_label", "pred_label"):
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {csv_path}")
    df["true_label"] = pd.to_numeric(df["true_label"], errors="raise").astype(int)
    df["pred_label"] = pd.to_numeric(df["pred_label"], errors="raise").astype(int)
    return df[["true_label", "pred_label"]].copy()

def ensure_alignment(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str):
    if len(a) != len(b):
        raise ValueError(f"Length mismatch between {name_a} and {name_b}")
    if not np.array_equal(a["true_label"].values, b["true_label"].values):
        raise ValueError(f"true_label sequences differ between {name_a} and {name_b}")

def counts_global(true_, pred_a, pred_b):
    correct_a = (pred_a == true_)
    correct_b = (pred_b == true_)
    b = int(((correct_a) & (~correct_b)).sum())  # A correct, B wrong
    c = int(((~correct_a) & (correct_b)).sum())  # B correct, A wrong
    return b, c

def counts_by_class(true_, pred_a, pred_b, cls_id: int):
    mask = (true_ == cls_id)
    if mask.sum() == 0:
        return 0, 0
    a_hit = (pred_a == cls_id)
    b_hit = (pred_b == cls_id)
    b = int(((a_hit) & (~b_hit) & mask).sum())  # A correct-for-class, B not
    c = int(((~a_hit) & (b_hit) & mask).sum())  # B correct-for-class, A not
    return b, c

def main():
    os.makedirs(os.path.dirname(OUT_GLOBAL_CSV), exist_ok=True)

    # Load once
    image_df = load_results(os.path.join(RESULTS_DIR, "image_results.csv"))
    text_df  = load_results(os.path.join(RESULTS_DIR, "text_results.csv"))
    multi_df = load_results(os.path.join(RESULTS_DIR, "multimodal_results.csv"))

    # Align sets
    ensure_alignment(image_df, text_df, "image", "text")
    ensure_alignment(image_df, multi_df, "image", "multimodal")

    true = image_df["true_label"].values
    n_samples = int(len(true))
    preds = {
        "Image": image_df["pred_label"].values,
        "Text": text_df["pred_label"].values,
        "Multimodal": multi_df["pred_label"].values,
    }

    # global csv
    global_rows_for_save = []
    debug_rows = []

    tmp = []
    for name, a_key, b_key in [
        ("Image vs Text", "Image", "Text"),
        ("Multimodal vs Image", "Multimodal", "Image"),
        ("Multimodal vs Text", "Multimodal", "Text"),
    ]:
        b, c = counts_global(true, preds[a_key], preds[b_key])
        ttype, stat, p = mcnemar_pvalue(b, c)
        tmp.append(p)
        debug_rows.append((name, ttype, b, c, b + c, stat, p))

    adj = holm_bonferroni(tmp)
    for (name, _ttype, b, c, n_disc, stat, p), p_holm in zip(debug_rows, adj):
        # Save only pair, p_value, p_holm
        global_rows_for_save.append({"pair": name, "p_value": p, "p_holm": p_holm})

    pd.DataFrame(global_rows_for_save).to_csv(OUT_GLOBAL_CSV, index=False)

    # Save class-level JSON
    present_classes = sorted(np.unique(true).tolist())
    per_class_payload = {
        "meta": {
            "method": "McNemar",
            "adjustment": "Holm-Bonferroni (per class, 3 pairs)",
            "created": datetime.utcnow().isoformat() + "Z",
            "n_samples": n_samples,
            "pairs": ["Image vs Text", "Multimodal vs Image", "Multimodal vs Text"],
        },
        "classes": []
    }

    for cls_id in present_classes:
        cls_name = CLASS_MAP[cls_id]
        pvals = []
        tests_tmp = []

        for name, a_key, b_key in [
            ("Image vs Text", "Image", "Text"),
            ("Multimodal vs Image", "Multimodal", "Image"),
            ("Multimodal vs Text", "Multimodal", "Text"),
        ]:
            b, c = counts_by_class(true, preds[a_key], preds[b_key], cls_id)
            ttype, stat, p = mcnemar_pvalue(b, c)
            pvals.append(p)
            tests_tmp.append((name, p))

        adj = holm_bonferroni(pvals)
        per_class_payload["classes"].append({
            "class_id": int(cls_id),
            "class_name": cls_name,
            "tests": [
                {"pair": name, "p_value": float(p), "p_holm": float(p_adj)}
                for (name, p), p_adj in zip(tests_tmp, adj)
            ]
        })

    with open(OUT_BYCLASS_JSON, "w", encoding="utf-8") as f:
        json.dump(per_class_payload, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
