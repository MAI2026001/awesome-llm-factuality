"""Zero-shot NLI cross-encoder: premise = abstract, hypothesis = claim."""
import time

import pandas as pd

NLI_TO_LABEL = {"entailment": "SUPPORT", "contradiction": "CONTRADICT", "neutral": "NEI"}


def run_nli(model_name, pairs, max_length=None, batch_size=8):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    id2label = {int(i): l.lower() for i, l in model.config.id2label.items()}
    unknown = [l for l in id2label.values() if l not in NLI_TO_LABEL]
    if unknown:
        raise ValueError(f"unexpected NLI labels {unknown}; fix NLI_TO_LABEL before running")
    max_length = max_length or min(tok.model_max_length, 1024)

    rows = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs.iloc[start:start + batch_size]
        premises = (chunk["title"] + ". " + chunk["abstract"]).tolist()
        hyps = chunk["claim"].tolist()
        full_len = [len(tok(p, h).input_ids) for p, h in zip(premises, hyps)]
        enc = tok(premises, hyps, truncation="only_first", max_length=max_length,
                  padding=True, return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        per_item = (time.time() - t0) / len(chunk)
        for i, (_, r) in enumerate(chunk.iterrows()):
            by_label = {NLI_TO_LABEL[id2label[j]]: float(probs[i, j]) for j in range(probs.shape[1])}
            pred = max(by_label, key=by_label.get)
            rows.append({
                "pair_id": r["pair_id"], "claim_id": r["claim_id"], "doc_id": r["doc_id"],
                "gold": r["gold"], "is_numeric": r["is_numeric"], "pred": pred, "raw": pred,
                "p_support": by_label["SUPPORT"], "p_contradict": by_label["CONTRADICT"],
                "p_nei": by_label["NEI"], "truncated": full_len[i] > max_length, "seconds": per_item,
            })
    return pd.DataFrame(rows)
