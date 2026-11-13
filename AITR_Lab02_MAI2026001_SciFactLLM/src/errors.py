"""Sample wrong predictions for manual error analysis (the reading and grouping is done by a person)."""
import pandas as pd


def sample_errors(pred_df, pairs_df, n=20, seed=13, abstract_chars=700):
    """Return up to n wrong predictions (pred != gold) joined with claim and abstract text.

    pred_df: prediction file (pair_id, gold, pred, raw, p_*). pairs_df: output of build_pairs for the same split.
    A blank `category` column is added for the reader to fill in by hand.
    """
    pred_df = pred_df.drop_duplicates("pair_id", keep="first")
    wrong = pred_df[pred_df["pred"] != pred_df["gold"]]
    wrong = wrong.sample(n=min(n, len(wrong)), random_state=seed)
    keep = pairs_df[["pair_id", "claim", "title", "abstract"]].copy()
    keep["abstract"] = keep["abstract"].str.slice(0, abstract_chars)
    cols = [c for c in ["pair_id", "gold", "pred", "raw", "p_support", "p_contradict", "p_nei", "is_numeric"] if c in wrong.columns]
    out = wrong[cols].merge(keep, on="pair_id", how="left")
    out["category"] = ""
    return out
