"""Write 30 random dev claims for hand-checking the numeric rule (protocol B1)."""
from _common import ROOT, get_pairs, load_config

cfg = load_config()
df, _ = get_pairs(cfg, cfg["data"]["eval_split"])
claims = df.drop_duplicates("claim_id")[["claim_id", "claim", "is_numeric"]]
sample = claims.sample(n=min(30, len(claims)), random_state=cfg["seeds"][0]).rename(columns={"is_numeric": "rule_says_numeric"})
sample["human_says_quantitative"] = ""   # fill in by hand: True / False
out = ROOT / "results" / "numeric_rule_check.csv"
sample.to_csv(out, index=False)
print("wrote", out, "- fill the last column by hand, then report agreement in baseline_validation.md")
