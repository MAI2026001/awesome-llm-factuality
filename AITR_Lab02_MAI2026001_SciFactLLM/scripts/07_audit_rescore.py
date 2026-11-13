"""Independent audit of results/ (no raw data or GPU needed).

1. Re-scores every prediction file with scikit-learn directly and with src.metrics, compares with results/results.csv.
2. Checks alignment (same pairs, same gold labels in every run), INVALID outputs, majority label vs train counts.
3. Checks the few-shot example sets against the dev set (claims and abstracts).
4. Paired claim-level bootstrap of macro-F1: NLI minus each large-LLM run (same pairs).
Writes results/audit_report.json and results/paired_comparisons.csv.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import metrics as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRED = os.path.join(ROOT, "results", "predictions")
LABELS = M.LABELS
stored = pd.read_csv(os.path.join(ROOT, "results", "results.csv")).set_index("run")
dv = json.load(open(os.path.join(ROOT, "results", "data_validation.json")))
report = {}


def load(run):
    return pd.read_csv(os.path.join(PRED, run + ".csv")).drop_duplicates("pair_id", keep="first").reset_index(drop=True)


# 1 + 2: re-score and alignment
frames = {run: load(run) for run in stored.index}
ref = frames["nli"].set_index("pair_id")["gold"]
worst = 0.0
for run, d in frames.items():
    s = d.set_index("pair_id")["gold"]
    assert set(s.index) == set(ref.index) and (s.reindex(ref.index) == ref).all(), f"misaligned: {run}"
    assert d["pred"].isin(LABELS).all(), f"INVALID or unknown prediction in {run}"
    mf = f1_score(d["gold"], d["pred"], labels=LABELS, average="macro", zero_division=0)
    worst = max(worst, abs(mf - stored.loc[run, "macro_f1"]), abs(accuracy_score(d["gold"], d["pred"]) - stored.loc[run, "accuracy"]),
                abs(M.macro_f1(d) - stored.loc[run, "macro_f1"]), abs(M.numeric_gap(d) - stored.loc[run, "err_gap"]))
    if "p_nei" in d and d["p_nei"].notna().all():
        worst = max(worst, abs(M.ece(d) - stored.loc[run, "ece"]))
assert worst < 1e-9, worst
report["runs_checked"] = len(frames)
report["max_abs_diff_vs_results_csv"] = worst
report["n_pairs"] = int(len(ref))
report["n_claims"] = int(frames["nli"]["claim_id"].nunique())
report["raw_rows_per_file"] = int(len(pd.read_csv(os.path.join(PRED, "nli.csv"))))
report["gold_counts_after_dedup"] = frames["nli"]["gold"].value_counts().to_dict()

train_major = max(dv["train_pairs"]["gold_counts"], key=dv["train_pairs"]["gold_counts"].get)
report["majority_label_in_file"] = frames["majority"]["pred"].iloc[0]
report["majority_label_from_train_counts"] = train_major
assert report["majority_label_in_file"] == train_major

# 3: few-shot sets versus dev
devclaims = set(frames["nli"]["claim_id"].astype(int))
devdocs = set(frames["nli"]["doc_id"].astype(int))
fs = {}
for f in sorted(glob.glob(os.path.join(PRED, "*_fewshot_ids.csv"))):
    if "smoke" in f:
        continue
    seed = int(os.path.basename(f).split("seed")[1].split("_")[0])
    ids = tuple(pd.read_csv(f)["pair_id"])
    fs.setdefault(seed, set()).add(ids)
report["fewshot_distinct_sets_per_seed"] = {s: len(v) for s, v in fs.items()}
report["fewshot_vs_dev"] = {}
for seed, sets in fs.items():
    ids = list(next(iter(sets)))
    report["fewshot_vs_dev"][seed] = {
        "pair_ids": ids,
        "claims_also_in_dev": [i for i in ids if int(i.split("_")[0]) in devclaims],
        "abstracts_also_in_dev": [i for i in ids if int(i.split("_")[1]) in devdocs]}

# numeric rule check status
chk = pd.read_csv(os.path.join(ROOT, "results", "numeric_rule_check.csv"))
report["numeric_rule_check_rows"] = int(len(chk))
report["numeric_rule_check_human_labels_filled"] = int(chk["human_says_quantitative"].notna().sum())


# 4: paired bootstrap on per-claim confusion counts (same resampling scheme as src.metrics.bootstrap_ci)
def claim_counts(d, pred):
    idx = {l: i for i, l in enumerate(LABELS)}
    claims = sorted(d["claim_id"].unique())
    pos = {c: i for i, c in enumerate(claims)}
    out = np.zeros((len(claims), 3, 3))
    for c, g, p in zip(d["claim_id"], d["gold"], pred):
        out[pos[c], idx[g], idx[p]] += 1
    return out


def f1_from_conf(c):
    tp = np.einsum("...ii->...i", c)
    den = 2 * tp + (c.sum(-2) - tp) + (c.sum(-1) - tp)
    return np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0).mean(-1)


a = frames["nli"].sort_values("pair_id").reset_index(drop=True)
rows = []
for run in [r for r in frames if r.startswith("llm_large")]:
    b = frames[run].sort_values("pair_id").reset_index(drop=True)
    ca, cb = claim_counts(a, a["pred"]), claim_counts(a, b["pred"])
    rng = np.random.default_rng(2026)
    diffs = []
    for _ in range(1000):
        pick = rng.integers(0, len(ca), len(ca))
        diffs.append(f1_from_conf(ca[pick].sum(0)) - f1_from_conf(cb[pick].sum(0)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    rows.append({"comparison": f"nli - {run}", "macro_f1_diff": f1_from_conf(ca.sum(0)) - f1_from_conf(cb.sum(0)), "ci_lo": lo, "ci_hi": hi,
                 "ci_includes_zero": bool(lo <= 0 <= hi)})
pc = pd.DataFrame(rows)
pc.to_csv(os.path.join(ROOT, "results", "paired_comparisons.csv"), index=False)
report["paired_all_ci_include_zero"] = bool(pc["ci_includes_zero"].all())
json.dump(report, open(os.path.join(ROOT, "results", "audit_report.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in report.items() if k != "fewshot_vs_dev"}, indent=1))
print("fewshot_vs_dev:", json.dumps(report["fewshot_vs_dev"]))
print(pc.round(4).to_string(index=False))
print("AUDIT OK")
