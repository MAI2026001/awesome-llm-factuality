"""Write a sample of wrong predictions per language-model run, for you to read and group by hand.

Needs the raw data (run scripts/00_download_data.py first). Output: results/error_samples/<run>.csv
Usage: python scripts/08_error_sample.py [--runs RUN1 RUN2 ...] [--n 20]
"""
import argparse

import pandas as pd

from _common import ROOT, get_pairs, load_config, resolve
from src.errors import sample_errors

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["llm_small_T2_0shot_seed13", "llm_large_T2_0shot_seed13"])
ap.add_argument("--n", type=int, default=20)
args = ap.parse_args()

cfg = load_config()
dev, _ = get_pairs(cfg, cfg["data"]["eval_split"])
pred_dir = resolve(cfg["paths"]["predictions"])
out_dir = ROOT / "results" / "error_samples"
out_dir.mkdir(parents=True, exist_ok=True)
for run in args.runs:
    df = sample_errors(pd.read_csv(pred_dir / f"{run}.csv"), dev, n=args.n, seed=cfg["seeds"][0])
    df.to_csv(out_dir / f"{run}.csv", index=False)
    print("wrote", out_dir / f"{run}.csv", len(df), "rows")
print("Read each row, write a short category in the `category` column, then count the categories for the paper.")
