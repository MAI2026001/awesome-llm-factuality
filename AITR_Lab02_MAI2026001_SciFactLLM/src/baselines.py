import pandas as pd


def majority_label(train_pairs):
    return train_pairs["gold"].value_counts().idxmax()


def predict_majority(train_pairs, eval_pairs):
    lab = majority_label(train_pairs)
    out = eval_pairs[["pair_id", "claim_id", "doc_id", "gold", "is_numeric"]].copy()
    out["pred"] = lab
    out["raw"] = lab
    return out
