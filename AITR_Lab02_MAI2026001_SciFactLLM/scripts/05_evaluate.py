"""Compute metrics for every prediction file and write results/results.csv. Real predictions only."""
import re

import pandas as pd

from _common import get_pairs, load_config, resolve
from src import metrics as M

cfg = load_config()
dev, _ = get_pairs(cfg, cfg["data"]["eval_split"])
pred_dir = resolve(cfg["paths"]["predictions"])
n_boot, b_seed = cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"]

rows = []
files = sorted(p for p in pred_dir.glob("*.csv") if "fewshot_ids" not in p.name and "smoke" not in p.name)
if not files:
    raise SystemExit("No prediction files found in " + str(pred_dir))
for path in files:
    df = pd.read_csv(path)
    dup = df[df["pair_id"].duplicated(keep=False)]
    if len(dup):   # files written before the duplicate-cited-doc fix contain one repeated row
        n_disagree = int((dup.groupby("pair_id")["pred"].nunique() > 1).sum())
        print(f"{path.name}: {dup['pair_id'].nunique()} pair_id(s) repeated; keeping the first row. "
              f"Repeated rows with different predictions: {n_disagree}")
        df = df.drop_duplicates("pair_id", keep="first")
    df = df.merge(dev[["pair_id", "is_numeric"]].rename(columns={"is_numeric": "is_numeric_ref"}), on="pair_id", how="left")
    df["is_numeric"] = df["is_numeric_ref"]
    M.check_alignment(df, dev)
    m = re.match(r"(?P<method>.+?)(?:_(?P<template>T\d)_(?P<shots>\d)shot_seed(?P<seed>\d+))?$", path.stem)
    row = {"run": path.stem, "method": m["method"], "template": m["template"], "shots": m["shots"], "seed": m["seed"],
           "n_pairs": len(df), "n_claims": df["claim_id"].nunique()}
    row["macro_f1"] = M.macro_f1(df)
    row["macro_f1_lo"], row["macro_f1_hi"] = M.bootstrap_ci(df, M.macro_f1, n_boot, b_seed)
    row["accuracy"] = M.accuracy(df)
    for lab, d in M.per_class(df).items():
        row[f"f1_{lab}"] = d["f1"]
    row.update(M.nei_metrics(df))
    row["invalid_rate"] = M.invalid_rate(df)
    nm = df[df["is_numeric"] == True]   # noqa: E712
    nn = df[df["is_numeric"] == False]  # noqa: E712
    row["n_numeric_pairs"], row["err_numeric"], row["err_nonnumeric"] = len(nm), M.error_rate(nm), M.error_rate(nn)
    row["err_gap"] = M.numeric_gap(df)
    row["err_gap_lo"], row["err_gap_hi"] = M.bootstrap_ci(df, M.numeric_gap, n_boot, b_seed)
    row["ece"] = M.ece(df)
    row["truncated_rate"] = df["truncated"].mean() if "truncated" in df else float("nan")
    row["mean_seconds"] = df["seconds"].mean() if "seconds" in df else float("nan")
    rows.append(row)

out = resolve(cfg["paths"]["results_csv"])
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out, "with", len(rows), "runs")
