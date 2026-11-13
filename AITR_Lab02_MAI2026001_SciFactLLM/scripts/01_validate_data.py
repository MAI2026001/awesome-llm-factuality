"""Checklist items C1-C4: counts, label structure, NEI pairs, dev-train abstract overlap."""
import json

from _common import ROOT, get_pairs, load_config, resolve
from src.data import find_data_dir, load_claims

cfg = load_config()
data_dir = find_data_dir(resolve(cfg["data"]["dir"]))
report = {}
for split in ("train", "dev", "test"):
    claims = load_claims(data_dir, split)
    with_ev = sum(1 for c in claims if c.get("evidence"))
    with_cited = sum(1 for c in claims if c.get("cited_doc_ids"))
    report[f"{split}_claims"] = {"n": len(claims), "with_evidence": with_ev, "with_cited_doc_ids": with_cited}

pairs = {}
for split in ("train", "dev"):
    df, stats = get_pairs(cfg, split)
    pairs[split] = df
    report[f"{split}_pairs"] = {
        "n_pairs": len(df),
        "n_claims": int(df["claim_id"].nunique()),
        "gold_counts": df["gold"].value_counts().to_dict(),
        "numeric_share": float(df["is_numeric"].mean()),
        "exclusions": dict(stats),
    }

overlap = set(pairs["train"]["doc_id"]) & set(pairs["dev"]["doc_id"])
report["dev_train_abstract_overlap"] = {
    "docs_in_both": len(overlap),
    "dev_docs": int(pairs["dev"]["doc_id"].nunique()),
    "dev_pairs_with_overlapping_doc": int(pairs["dev"]["doc_id"].isin(overlap).sum()),
}
out = ROOT / "results" / "data_validation.json"
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print("wrote", out)
if report["dev_pairs"]["gold_counts"].get("NEI", 0) == 0:
    print("WARNING: no NEI pairs in dev. RQ2 cannot be run as designed. Log this in research_decisions.md and see docs/STUDENT_TODO.md.")
