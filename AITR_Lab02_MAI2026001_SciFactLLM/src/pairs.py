"""Claim-abstract pair construction, label mapping and numeric tagging (protocol B1)."""
import random
import re
from collections import Counter

import pandas as pd
import yaml

LABELS = ["SUPPORT", "CONTRADICT", "NEI"]
RAW_TO_LABEL = {"SUPPORT": "SUPPORT", "CONTRADICT": "CONTRADICT"}


def load_numeric_rule(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return re.compile("|".join(f"(?:{p})" for p in cfg["patterns"]), re.IGNORECASE)


def is_numeric(text, rule):
    return bool(rule.search(text))


def build_pairs(claims, corpus, split, numeric_rule=None, allow_test=False):
    """One row per (claim, cited abstract). Returns (DataFrame, Counter of exclusions).

    Gold label: evidence label for that document, or NEI if the cited document has no evidence.
    Pairs with mixed evidence labels are excluded and counted.
    """
    if split == "test" and not allow_test:
        raise ValueError("test split is not used in this study (labels are held out by the dataset authors)")
    rows, stats = [], Counter()
    for c in claims:
        cited_raw = c.get("cited_doc_ids") or []
        cited = list(dict.fromkeys(int(d) for d in cited_raw))   # a claim can list the same abstract twice
        if len(cited) < len(cited_raw):
            stats["duplicate_cited_docs"] += len(cited_raw) - len(cited)
        if not cited:
            stats["claims_without_cited_docs"] += 1
            continue
        evidence = c.get("evidence") or {}
        for d in cited:
            d = int(d)
            if d not in corpus:
                stats["missing_corpus_doc"] += 1
                continue
            raw = {e["label"] for e in evidence.get(str(d), [])}
            if len(raw) > 1:
                stats["mixed_label_pairs"] += 1
                continue
            if not raw:
                gold = "NEI"
            else:
                r = next(iter(raw))
                if r not in RAW_TO_LABEL:
                    stats["unknown_label_pairs"] += 1
                    continue
                gold = RAW_TO_LABEL[r]
            doc = corpus[d]
            rows.append({
                "pair_id": f"{c['id']}_{d}",
                "claim_id": c["id"],
                "doc_id": d,
                "claim": c["claim"],
                "title": doc["title"],
                "abstract": " ".join(doc["abstract"]),
                "gold": gold,
                "is_numeric": is_numeric(c["claim"], numeric_rule) if numeric_rule else None,
                "split": split,
            })
    return pd.DataFrame(rows), stats


def select_fewshot(train_pairs, seed, max_chars=1500):
    """One training pair per label, drawn with a seeded RNG from short abstracts. Train only."""
    if not (train_pairs["split"] == "train").all():
        raise ValueError("few-shot examples must come from the train split only")
    rng = random.Random(seed)
    chosen = []
    for lab in LABELS:
        pool = train_pairs[(train_pairs["gold"] == lab) & (train_pairs["abstract"].str.len() <= max_chars)]
        if pool.empty:
            raise ValueError(f"no short train example for label {lab}")
        chosen.append(pool.iloc[rng.randrange(len(pool))])
    return pd.DataFrame(chosen).reset_index(drop=True)
