"""Agreement between the numeric rule and your hand labels (results/numeric_rule_check.csv).

Fill the column `human_says_quantitative` with True/False for all 30 rows first. A label means: the claim itself
states or compares a quantity (a number, percentage, rate, dose, fold change, 'most', 'twice' ...), NOT a digit that
is only part of a gene, protein or chemical name (for example APOE4, p53, Ca2+).
Writes results/numeric_rule_agreement.json.
"""
import json
import sys

import pandas as pd

from _common import ROOT

path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "numeric_rule_check.csv"
d = pd.read_csv(path)
h = d["human_says_quantitative"].map(lambda x: str(x).strip().lower() in ("true", "1", "yes") if pd.notna(x) and str(x).strip() else None)
if h.isna().any():
    raise SystemExit(f"{int(h.isna().sum())} of {len(d)} rows have no hand label yet")
r = d["rule_says_numeric"].astype(bool)
h = h.astype(bool)
tp, fp, fn, tn = int((r & h).sum()), int((r & ~h).sum()), int((~r & h).sum()), int((~r & ~h).sum())
n = len(d)
po = (tp + tn) / n
pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
res = {"n": n, "rule_true": int(r.sum()), "human_true": int(h.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
       "agreement": po, "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
       "cohen_kappa": (po - pe) / (1 - pe) if pe != 1 else None}
(ROOT / "results" / "numeric_rule_agreement.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
