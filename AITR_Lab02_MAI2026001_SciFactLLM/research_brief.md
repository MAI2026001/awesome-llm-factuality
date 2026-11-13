# Research brief (Gate G1)

| Field | Entry |
|---|---|
| Review topic and key papers | Factual accuracy benchmarks for LLMs in scientific domains. Papers to verify from the originals: SciFact (Wadden et al. 2020), SciFact-Open (Wadden et al. 2022), SimpleQA (Wei et al. 2024), PubMedQA (Jin et al. 2019), FELM (Chen et al. 2023). So far these are checked only against my own review text, plus the SciFact release page for data facts. |
| Research gap | The review reports that headline accuracy can hide fine-grained errors (Sec. 4.4), that abstention matters in high-stakes science (Sec. 3.3 and 6), and that quantitative claims are under-tested (Sec. 5). The third point is partly my own reading. Not yet checked against original papers or a fresh literature search. |
| Research question | Primary: on SciFact dev with the cited abstract supplied, how do a majority floor, an NLI cross-encoder and two open-weight LLMs differ in macro-F1? Secondary: how do they handle NOT ENOUGH INFO, and is the error rate higher on numeric claims? |
| Study pathway | B, a controlled comparative study, with a prompt-level abstention ablation. |
| Dataset | SciFact. Source and licence in `data/README.md`. Train for few-shot examples and the floor, dev for evaluation, test not used. |
| Baseline and comparators | Majority class; NLI cross-encoder (`facebook/bart-large-mnli`, zero-shot); `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-7B-Instruct` (4-bit). Model cards and licences still to be read. |
| Primary metric and control | Macro-F1 on SUPPORT / CONTRADICT / NEI. Same pairs, label set, parser and metric code for every method; same prompts, greedy decoding and quantisation across LLM runs. |
| Computing and risks | Free Colab T4. Risks: Colab session limits, prompt sensitivity, few numeric claims leading to wide intervals, NEI pairs possibly absent from the data (checked by `01_validate_data.py`). Fallbacks in `research_decisions.md`. |
| First executable milestone | Run `python -m src.env_check` and `scripts/01_validate_data.py` and record the output. |

**What I intend to investigate versus what is already known.** The review reports that SciFact supports claim verification against abstracts, that SimpleQA frames answering versus abstaining, and that LLM judges are unreliable. I do not claim a new method or a new finding. If a comparison like this already exists, mine is a small controlled replication of that kind of study. A quick search on 2026-11-11 found close work (abstention-aware verification on SciFact; small LLMs fine-tuned on SciFact and HealthVer; zero-shot small LLMs on SciFact and HealthVer). Judged from abstracts only, none is the same comparison, but I have not read them in full and the search was not systematic. See D18 in `research_decisions.md`.
