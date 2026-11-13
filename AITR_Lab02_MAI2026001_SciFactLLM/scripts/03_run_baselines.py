"""Majority-class floor and NLI cross-encoder on the dev pairs."""
import argparse

from _common import ROOT, get_pairs, load_config, resolve
from src.baselines import predict_majority
from src.seed import set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--skip-nli", action="store_true")
args = ap.parse_args()

cfg = load_config()
pred_dir = resolve(cfg["paths"]["predictions"])
pred_dir.mkdir(parents=True, exist_ok=True)
set_seed(cfg["seeds"][0])
train, _ = get_pairs(cfg, "train")
dev, _ = get_pairs(cfg, cfg["data"]["eval_split"])

maj = predict_majority(train, dev)
maj.to_csv(pred_dir / "majority.csv", index=False)
print("majority label:", maj["pred"].iloc[0], "| wrote majority.csv")

if not args.skip_nli:
    from src.nli import run_nli
    out = run_nli(cfg["models"]["nli"], dev)
    out.to_csv(pred_dir / "nli.csv", index=False)
    print("wrote nli.csv | truncated pairs:", int(out["truncated"].sum()), "of", len(out))
