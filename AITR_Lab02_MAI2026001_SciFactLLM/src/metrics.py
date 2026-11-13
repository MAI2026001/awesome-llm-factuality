"""Metrics and claim-level bootstrap. All functions take a DataFrame with gold/pred columns."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

LABELS = ["SUPPORT", "CONTRADICT", "NEI"]


def check_alignment(pred_df, eval_pairs):
    """Raise if predictions do not cover exactly the evaluation pairs, or gold labels disagree."""
    if pred_df["pair_id"].duplicated().any():
        raise ValueError("duplicate pair_ids in predictions")
    if set(pred_df["pair_id"]) != set(eval_pairs["pair_id"]):
        raise ValueError("prediction pair_ids differ from evaluation pairs")
    merged = pred_df.merge(eval_pairs[["pair_id", "gold"]], on="pair_id", suffixes=("", "_ref"))
    if (merged["gold"] != merged["gold_ref"]).any():
        raise ValueError("gold labels in predictions differ from the reference pairs")
    if pred_df[["gold", "pred"]].isna().any().any():
        raise ValueError("missing gold or pred values")


def macro_f1(df):
    """Macro-F1 over the labels that occur in gold or predictions.

    On the full evaluation set all three labels occur in gold, so this equals the plain 3-class macro-F1.
    The restriction only matters inside bootstrap resamples, where a class can be absent by chance and
    should not be scored as 0. INVALID predictions are never counted as a label (they count as errors).
    """
    present = [l for l in LABELS if (df["gold"] == l).any() or (df["pred"] == l).any()]
    return f1_score(df["gold"], df["pred"], labels=present, average="macro", zero_division=0)


def accuracy(df):
    return accuracy_score(df["gold"], df["pred"])


def per_class(df):
    p, r, f, s = precision_recall_fscore_support(df["gold"], df["pred"], labels=LABELS, zero_division=0)
    return {l: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])} for i, l in enumerate(LABELS)}


def nei_metrics(df):
    """NEI recall, NEI precision, and the confident-wrong rate on gold-NEI pairs."""
    gold_nei = df[df["gold"] == "NEI"]
    pred_nei = df[df["pred"] == "NEI"]
    recall = (gold_nei["pred"] == "NEI").mean() if len(gold_nei) else float("nan")
    precision = (pred_nei["gold"] == "NEI").mean() if len(pred_nei) else float("nan")
    confident_wrong = gold_nei["pred"].isin(["SUPPORT", "CONTRADICT"]).mean() if len(gold_nei) else float("nan")
    return {"nei_recall": recall, "nei_precision": precision, "confident_wrong_rate": confident_wrong}


def invalid_rate(df):
    return (df["pred"] == "INVALID").mean()


def error_rate(df):
    return (df["pred"] != df["gold"]).mean() if len(df) else float("nan")


def numeric_gap(df):
    """error rate on numeric claims minus error rate on non-numeric claims."""
    num = df[df["is_numeric"] == True]  # noqa: E712
    non = df[df["is_numeric"] == False]  # noqa: E712
    if num.empty or non.empty:
        return float("nan")
    return error_rate(num) - error_rate(non)


def ece(df, bins=10):
    """Expected calibration error over the three label probabilities (argmax of the probabilities, not the parsed
    prediction). NaN if probabilities are missing."""
    cols = ["p_support", "p_contradict", "p_nei"]
    if not all(c in df.columns for c in cols) or df[cols].isna().any().any():
        return float("nan")
    probs = df[cols].to_numpy(dtype=float)
    conf = probs.max(axis=1)
    pred_lab = np.array(LABELS)[probs.argmax(axis=1)]
    correct = (pred_lab == df["gold"].to_numpy()).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return total


def bootstrap_ci(df, stat_fn, n=1000, seed=2026):
    """95% percentile CI, resampling claims (all pairs of a claim stay together)."""
    df = df.reset_index(drop=True)
    groups = df.groupby("claim_id").indices
    keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        picks = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[i]] for i in picks])
        vals.append(stat_fn(df.iloc[idx]))
    vals = np.array(vals, dtype=float)
    if np.isnan(vals).all():
        return float("nan"), float("nan")
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def paired_bootstrap_diff(df_a, df_b, stat_fn, n=1000, seed=2026):
    """stat(A) - stat(B) on the same pairs; returns (point, lo, hi)."""
    a = df_a.sort_values("pair_id").reset_index(drop=True)
    b = df_b.sort_values("pair_id").reset_index(drop=True)
    if not (a["pair_id"].to_numpy() == b["pair_id"].to_numpy()).all():
        raise ValueError("methods are not evaluated on the same pairs")
    groups = a.groupby("claim_id").indices
    keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        picks = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[i]] for i in picks])
        diffs.append(stat_fn(a.iloc[idx]) - stat_fn(b.iloc[idx]))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return stat_fn(a) - stat_fn(b), lo, hi
