"""LLM inference: greedy decoding, strict output parsing, label probabilities."""
import re
import time

import pandas as pd

from .pairs import LABELS
from .prompts import build_messages

_PARSE = re.compile(r"^\W*(SUPPORT|CONTRADICT|NEI)\b", re.IGNORECASE)


def parse_label(text):
    """Return SUPPORT / CONTRADICT / NEI, or INVALID. Only a label at the start of the output counts."""
    m = _PARSE.match(text or "")
    return m.group(1).upper() if m else "INVALID"


def load_model(name, load_4bit=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    kw = {"device_map": "auto"}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    else:
        # newer transformers versions call this argument `dtype`, older ones `torch_dtype`
        try:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, **kw)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, **kw)
    model.eval()
    return tok, model


def label_token_ids(tok):
    """First token of each label word; None if two labels share a first token."""
    ids = {lab: tok.encode(lab, add_special_tokens=False)[0] for lab in LABELS}
    return ids if len(set(ids.values())) == len(LABELS) else None


def _fit_prompt(tok, prompts, template, row, fewshot, max_tokens):
    abstract = row["abstract"]
    truncated = False
    text = ""
    for _ in range(6):
        msgs = build_messages(prompts, template, row, fewshot, abstract_override=abstract)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        n = len(tok(text).input_ids)
        if n <= max_tokens:
            return text, truncated
        truncated = True
        abstract = abstract[: int(len(abstract) * max_tokens / n * 0.95)]
    return text, True


def run_llm(tok, model, pairs, prompts, template, fewshot, max_new_tokens=8, max_input_tokens=3072):
    import torch

    ids = label_token_ids(tok)
    rows = []
    for _, r in pairs.iterrows():
        text, truncated = _fit_prompt(tok, prompts, template, r, fewshot, max_input_tokens)
        inputs = tok(text, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True,
                                 pad_token_id=tok.eos_token_id)
        secs = time.time() - t0
        gen = out.sequences[0, inputs["input_ids"].shape[1]:]
        raw = tok.decode(gen, skip_special_tokens=True)
        probs = [None, None, None]
        if ids is not None:
            logits = torch.stack([out.scores[0][0][ids[l]] for l in LABELS]).float()
            probs = torch.softmax(logits, dim=0).tolist()
        rows.append({
            "pair_id": r["pair_id"], "claim_id": r["claim_id"], "doc_id": r["doc_id"],
            "gold": r["gold"], "is_numeric": r["is_numeric"], "pred": parse_label(raw), "raw": raw,
            "p_support": probs[0], "p_contradict": probs[1], "p_nei": probs[2],
            "truncated": truncated, "seconds": secs,
        })
    return pd.DataFrame(rows)
