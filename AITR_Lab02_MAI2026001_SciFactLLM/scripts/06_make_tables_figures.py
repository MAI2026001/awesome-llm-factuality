"""Build LaTeX tables and figures from results/results.csv and the prediction files."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import load_config, resolve
from src.metrics import LABELS

cfg = load_config()
res = pd.read_csv(resolve(cfg["paths"]["results_csv"]))
tab_dir, fig_dir = resolve(cfg["paths"]["tables"]), resolve(cfg["paths"]["figures"])
tab_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)


def fmt(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def write_table(name, header, body, caption, label):
    lines = ["\\begin{table}[t]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\resizebox{\\columnwidth}{!}{%",
             "\\begin{tabular}{" + "l" * 2 + "r" * (len(header) - 2) + "}", "\\hline",
             " & ".join(header) + " \\\\", "\\hline"]
    lines += [" & ".join(r) + " \\\\" for r in body]
    lines += ["\\hline", "\\end{tabular}}", "\\end{table}"]
    (tab_dir / name).write_text("\n".join(lines))


res["config"] = res["template"].fillna("-") + "/" + res["shots"].astype("Int64").astype(str).replace("<NA>", "-")
grp = res.groupby(["method", "template", "shots"], dropna=False)
body1, body2, body3 = [], [], []
for (method, template, shots), g in grp:
    first = g.sort_values("seed").iloc[0]
    sd = g["macro_f1"].std() if len(g) > 1 else float("nan")
    cfgname = "-" if pd.isna(template) else f"{template}, {int(shots)}-shot"
    body1.append([method.replace("_", "\\_"), cfgname, str(len(g)),
                  fmt(g["macro_f1"].mean()), fmt(sd),
                  f"{fmt(first['macro_f1'])} [{fmt(first['macro_f1_lo'])}, {fmt(first['macro_f1_hi'])}]"])
    body2.append([method.replace("_", "\\_"), cfgname, str(len(g)), fmt(g["nei_recall"].mean()),
                  fmt(g["nei_precision"].mean()), fmt(g["confident_wrong_rate"].mean()), fmt(g["ece"].mean())])
    body3.append([method.replace("_", "\\_"), cfgname, str(len(g)), fmt(g["err_numeric"].mean()),
                  fmt(g["err_nonnumeric"].mean()), f"{first['err_gap']:+.3f}",
                  f"[{first['err_gap_lo']:+.3f}, {first['err_gap_hi']:+.3f}]"])
write_table("table1_main.tex", ["Method", "Setting", "Runs", "Mean", "SD", "Seed 13 [95\\% CI]"], body1,
            "Macro-F1 on SciFact dev pairs. Mean and SD are over the runs listed (3-shot: seeds 13, 42, 2026; 0-shot: seed 13 only). The interval is the claim-level bootstrap CI of the seed-13 run, shown next to that run's own score.", "tab:main")
write_table("table2_nei.tex", ["Method", "Setting", "Runs", "NEI recall", "NEI prec.", "Conf.-wrong", "ECE"], body2,
            "Handling of NOT ENOUGH INFO pairs (means over runs). Conf.-wrong is the share of gold-NEI pairs labelled SUPPORT or CONTRADICT. ECE: 10 bins, confidence = highest label probability. No run produced an INVALID output.", "tab:nei")
write_table("table3_numeric.tex", ["Method", "Setting", "Runs", "Err. num.", "Err. other", "Gap (seed 13)", "95\\% CI"], body3,
            "Error rate on claims tagged numeric versus other claims (means over runs). Gap = numeric minus other, for the seed-13 run, with its claim-level bootstrap CI. The numeric tag failed its hand check (precision 1/11 on 30 claims; see text), so the groups are not numeric versus non-numeric content.", "tab:numeric")
print("wrote tables to", tab_dir)

# Confusion matrices for zero-shot T2/T3, NLI and majority (first seed)
pred_dir = resolve(cfg["paths"]["predictions"])
want = [p for p in sorted(pred_dir.glob("*.csv"))
        if p.stem in ("majority", "nli") or ("_0shot_" in p.stem and ("_T2_" in p.stem or "_T3_" in p.stem))]
want = [p for p in want if "smoke" not in p.name and "fewshot_ids" not in p.name]
for p in want:
    df = pd.read_csv(p).drop_duplicates("pair_id", keep="first")
    labs = LABELS + ["INVALID"]
    cm = np.array([[((df["gold"] == g) & (df["pred"] == q)).sum() for q in labs] for g in LABELS])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labs)), labs, rotation=30)
    ax.set_yticks(range(len(LABELS)), LABELS)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(p.stem, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"confusion_{p.stem}.png", dpi=200)
    fig.savefig(fig_dir / f"confusion_{p.stem}.pdf")
    plt.close(fig)

# Error rate: numeric vs other, first seed per configuration
first_runs = res.sort_values("seed").groupby(["method", "template", "shots"], dropna=False).first().reset_index()
labels = [f"{r.method}\n{'' if pd.isna(r.template) else r.template + '/' + str(int(r.shots))}" for r in first_runs.itertuples()]
x = np.arange(len(first_runs))
fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(x)), 3.6))
ax.bar(x - 0.2, first_runs["err_numeric"], 0.4, label="numeric claims")
ax.bar(x + 0.2, first_runs["err_nonnumeric"], 0.4, label="other claims")
ax.set_xticks(x, labels, rotation=75, fontsize=6)
ax.set_ylabel("error rate")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(fig_dir / "error_numeric_vs_other.png", dpi=200)
fig.savefig(fig_dir / "error_numeric_vs_other.pdf")
plt.close(fig)
print("wrote figures to", fig_dir)

# Macro-F1 per configuration: individual runs (dots), mean (bar), NLI and majority as reference lines
llm = res[res["method"].str.startswith("llm")].copy()
llm["label"] = llm["method"].str.replace("llm_", "") + " " + llm["template"] + " " + llm["shots"].astype(int).astype(str) + "-shot"
order = list(dict.fromkeys(llm.sort_values(["method", "template", "shots"], ascending=[False, True, True])["label"]))
fig, ax = plt.subplots(figsize=(7.2, 3.4))
for i, lab in enumerate(order):
    g = llm[llm["label"] == lab]
    ax.scatter([i] * len(g), g["macro_f1"], s=24, color="#3b5b92", zorder=3)
    ax.hlines(g["macro_f1"].mean(), i - 0.3, i + 0.3, color="#3b5b92")
for name, col, ls in (("nli", "#b5472f", "--"), ("majority", "#555555", ":")):
    v = res[res["method"] == name]["macro_f1"].iloc[0]
    ax.axhline(v, color=col, ls=ls, lw=1.2, label=f"{name} ({v:.3f})")
ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=7)
ax.set_ylabel("Macro-F1 (dev)")
ax.set_ylim(0, 0.85)
ax.legend(fontsize=7, loc="lower left")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(fig_dir / "macro_f1_by_config.png", dpi=200)
fig.savefig(fig_dir / "macro_f1_by_config.pdf")
plt.close(fig)

# Copy into paper/ so the paper folder can be uploaded to Overleaf on its own
import shutil
for src_dir, name in ((tab_dir, "tables"), (fig_dir, "figures")):
    dst = resolve("paper") / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in src_dir.glob("*"):
        if f.suffix in (".tex", ".pdf", ".png"):
            shutil.copy(f, dst / f.name)
print("copied tables and figures into paper/")
