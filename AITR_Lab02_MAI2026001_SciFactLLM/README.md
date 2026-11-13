# Small Open Language Models on Scientific Claim Verification

**Course:** AI Tools for Research — Lab 02  
**Student:** Shingala Smit Mukeshbhai  
**Roll No.:** MAI2026001

A controlled comparison of a majority-class floor, an NLI cross-encoder, and two small open instruction-tuned language models on SciFact claim–abstract verification. The study focuses on abstention behavior and errors involving quantitative claims.

**GitHub:** https://github.com/MAI2026001/Small-Open-Language-Models-on-Scientific-Claim-Verification

## Research question and models

The experiment uses 339 unique SciFact dev claim–abstract pairs from 300 claims. Each model receives the cited abstract as evidence and predicts SUPPORT, CONTRADICT, or NOT ENOUGH INFO (NEI).

Compared methods:

1. Majority-class floor (always predicts the most frequent training label, SUPPORT).
2. `facebook/bart-large-mnli` (zero-shot NLI cross-encoder).
3. `Qwen/Qwen2.5-1.5B-Instruct`.
4. `Qwen/Qwen2.5-7B-Instruct` loaded in 4-bit.

The study compares macro-F1, NEI/abstention behavior, and error rates for pairs tagged by the numeric-pattern rule. It also includes a manual 30-claim audit of that rule and a 40-error manual categorization.

## Main results

| Method | Dev macro-F1 |
|---|---:|
| Majority floor | 0.193 |
| `facebook/bart-large-mnli`, zero-shot | 0.729 (95% CI: 0.675–0.774) |
| Qwen2.5-7B-Instruct, 4-bit (six configurations) | 0.682–0.715 |
| Qwen2.5-1.5B-Instruct (six configurations) | 0.285–0.473 |

The NLI baseline's advantage over individual 7B runs is +0.002 to +0.048; all 12 paired 95% intervals include zero. The T3 abstention sentence did not consistently improve NEI recall.

### Manual validation results

The 30-claim numeric-rule check produced 11 rule positives and 4 human-positive labels: TP=2, FP=9, FN=2, TN=17; agreement=63.3%, precision=18.2%, recall=50.0%, and Cohen's kappa=0.0884. The rule frequently mistakes digits inside biomedical names (for example APOE4, p53, Ca2+, and UBC13) for quantities.

For 20 wrong zero-shot T2, seed-13 predictions per LLM, the 7B sample was dominated by over-abstention (12/20, 60%); the 1.5B sample was dominated by false-positive SUPPORT predictions (18/20, 90%). See `baseline_validation.md` for the full category breakdown and limitations.

## Repository structure

```text
AITR_Lab02_MAI2026001_SciFactLLM/
├── .gitignore
├── README.md
├── ai_assistance_log.md
├── baseline_validation.md
├── research_brief.md
├── research_decisions.md
├── requirements.txt
├── requirements.lock.txt
├── configs/
│   ├── config.yaml
│   ├── numeric_rule.yaml
│   └── prompts.yaml
├── data/
│   └── README.md                 # Raw SciFact JSONL files are excluded by .gitignore
├── docs/
│   ├── STUDENT_TODO.md
│   └── architecture.mmd
├── notebooks/
│   └── 01_colab_run_all.ipynb
├── paper/
│   ├── main.tex
│   ├── main.pdf                  # Include only after final IEEEtran PDF is downloaded from Overleaf
│   ├── references.bib
│   ├── figures/
│   └── tables/
├── results/
│   ├── audit_report.json
│   ├── data_validation.json
│   ├── env_check.json
│   ├── numeric_rule_agreement.json
│   ├── numeric_rule_check.csv
│   ├── paired_comparisons.csv
│   ├── python_version.txt
│   ├── results.csv
│   ├── error_samples/
│   │   ├── llm_large_T2_0shot_seed13.csv
│   │   └── llm_small_T2_0shot_seed13.csv
│   ├── figures/
│   ├── predictions/
│   └── tables/
├── scripts/
│   ├── _common.py
│   ├── 00_download_data.py
│   ├── 01_validate_data.py
│   ├── 02_numeric_rule_sample.py
│   ├── 03_run_baselines.py
│   ├── 04_run_llm.py
│   ├── 05_evaluate.py
│   ├── 06_make_tables_figures.py
│   ├── 07_audit_rescore.py
│   ├── 08_error_sample.py
│   ├── 09_numeric_agreement.py
│   └── freeze_env.sh
├── src/
│   ├── __init__.py
│   ├── baselines.py
│   ├── common.py
│   ├── data.py
│   ├── env_check.py
│   ├── errors.py
│   ├── llm.py
│   ├── metrics.py
│   ├── nli.py
│   ├── pairs.py
│   ├── prompts.py
│   └── seed.py
└── tests/
    ├── __init__.py
    └── test_all.py
```

`paper/figures/`, `paper/tables/`, `results/figures/`, `results/predictions/`, and `results/tables/` contain generated artifacts. The exact generated files depend on the completed run. The raw SciFact JSONL files are not committed.

## Environment and setup

The experiments were run in Google Colab with a Tesla T4 GPU. The 7B model uses 4-bit loading. Do not reinstall or downgrade Colab's preinstalled PyTorch unless the environment check identifies a problem.

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m src.env_check
python -m unittest discover -s tests -v
```

For Colab, install only missing packages if the base runtime already contains compatible versions:

```bash
pip install -q accelerate bitsandbytes pyyaml
```

The exact environment used for the reported runs is recorded in `requirements.lock.txt` and `results/env_check.json`.

## Dataset

The project uses the authors' SciFact release. Follow `data/README.md` for the source, download details, and licence notes. Raw data is stored under the ignored raw-data directory and must not be committed.

```bash
python scripts/00_download_data.py
python scripts/01_validate_data.py
```

The final evaluation split contains 339 unique claim–abstract pairs from 300 dev claims: 138 SUPPORT, 71 CONTRADICT, and 130 NEI. The test split is not used because its labels are held out by the dataset authors.

## Reproduce the experiment

Run commands from the repository root.

### 1. Numeric-rule sample and baselines

```bash
python scripts/02_numeric_rule_sample.py
python scripts/03_run_baselines.py
```

### 2. LLM inference

A short smoke test can be run before the full jobs:

```bash
python scripts/04_run_llm.py --model-key llm_small --limit 10
```

Run the full model configurations:

```bash
python scripts/04_run_llm.py --model-key llm_small
python scripts/04_run_llm.py --model-key llm_large
```

### 3. Evaluation, tables, figures, and audit

```bash
python scripts/05_evaluate.py
python scripts/06_make_tables_figures.py
python scripts/07_audit_rescore.py
```

### 4. Manual audits

Generate the two 20-error samples, inspect each case, and complete the `category` column in both CSV files:

```bash
python scripts/08_error_sample.py --n 20
```

Complete the 30 human labels in `results/numeric_rule_check.csv`, then compute agreement:

```bash
python scripts/09_numeric_agreement.py
```

The reported numeric-rule metrics are agreement 63.3%, precision 18.2%, recall 50.0%, and Cohen's kappa 0.0884. Manual error-category counts are documented in `baseline_validation.md`.

### 5. Tests and environment freeze

```bash
python -m unittest discover -s tests -v
bash scripts/freeze_env.sh
```

`freeze_env.sh` records the installed package versions in `requirements.lock.txt`.

## Experimental protocol

- Evaluation split: SciFact dev only.
- Unit of evaluation: unique claim–cited-abstract pair.
- Labels: SUPPORT, CONTRADICT, NEI.
- Primary metric: three-class macro-F1.
- Confidence intervals: 1,000 bootstrap resamples at claim level.
- Few-shot seeds: 13, 42, and 2026.
- Zero-shot runs: seed 13.
- Prompt templates: T1, T2, and T3; T3 adds an abstention instruction.
- The test split is not evaluated.

## Limitations

- Evidence is oracle-selected: the cited abstract is supplied to the model.
- The study uses one biomedical claim-verification dataset.
- The NLI neutral-to-NEI mapping is an assumption.
- Three of nine few-shot example abstracts overlap with dev abstracts, although no example claim overlaps.
- The numeric-pattern rule is noisy: its manual precision is 18.2% and Cohen's kappa is 0.0884.
- The manual error analysis is based on 20 wrong predictions per model for one zero-shot configuration and is descriptive.
- Pretraining contamination cannot be ruled out.
- GPU runtime was measured; peak memory was not.

## Research and provenance files

- `research_brief.md`: motivation, research questions, and scope.
- `research_decisions.md`: protocol decisions, changes, numeric audit closure, manual error analysis, and repository record.
- `baseline_validation.md`: validation evidence and final metrics.
- `ai_assistance_log.md`: disclosure of AI assistance and student verification.

## Citation

SciFact: Wadden et al., “Fact or Fiction: Verifying Scientific Claims,” EMNLP 2020. Full bibliographic information is provided in `paper/references.bib`.
