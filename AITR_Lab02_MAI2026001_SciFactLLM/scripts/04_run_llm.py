"""Run one LLM over all templates and shot settings. Existing prediction files are skipped."""
import argparse

from _common import get_pairs, load_config, resolve
from src.llm import load_model, run_llm
from src.pairs import select_fewshot
from src.prompts import load_prompts
from src.seed import set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--model-key", required=True, choices=["llm_small", "llm_large"])
ap.add_argument("--limit", type=int, default=None, help="smoke test on the first N pairs; output goes to a _smoke file")
args = ap.parse_args()

cfg = load_config()
pred_dir = resolve(cfg["paths"]["predictions"])
pred_dir.mkdir(parents=True, exist_ok=True)
prompts = load_prompts(resolve(cfg["prompts"]))
train, _ = get_pairs(cfg, "train")
dev, _ = get_pairs(cfg, cfg["data"]["eval_split"])
if args.limit:
    dev = dev.iloc[: args.limit]

name = cfg["models"][args.model_key]
tok, model = load_model(name, load_4bit=args.model_key in cfg["llm"]["load_4bit"])

for template in cfg["llm"]["templates"]:
    for shots in cfg["llm"]["shots"]:
        seeds = cfg["seeds"] if shots > 0 else cfg["seeds"][:1]   # 0-shot has no sampled examples
        for seed in seeds:
            tag = f"{args.model_key}_{template}_{shots}shot_seed{seed}" + ("_smoke" if args.limit else "")
            path = pred_dir / f"{tag}.csv"
            if path.exists():
                print("skip (exists):", tag)
                continue
            set_seed(seed)
            fewshot = select_fewshot(train, seed) if shots > 0 else None
            if fewshot is not None:
                fewshot[["pair_id"]].to_csv(pred_dir / f"{tag}_fewshot_ids.csv", index=False)
            out = run_llm(tok, model, dev, prompts, template, fewshot,
                          cfg["llm"]["max_new_tokens"], cfg["llm"]["max_input_tokens"])
            out.to_csv(path, index=False)
            print("wrote", tag, "| INVALID:", int((out["pred"] == "INVALID").sum()), "of", len(out))
