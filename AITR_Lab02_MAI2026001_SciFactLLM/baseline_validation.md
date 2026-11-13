# Baseline and pipeline validation report

**Project:** Small Open Language Models on Scientific Claim Verification  
**Student:** Shingala Smit Mukeshbhai  
**Roll No.:** MAI2026001  
**Evaluation split:** SciFact dev — 339 claim–abstract pairs from 300 claims  
**Validation status:** Final experimental and manual-audit checks recorded on 12 November 2026.

A status of **VERIFIED** means that the check was completed using the real Colab experiment outputs or the documented manual review. Assistant-sandbox tests on toy data are identified separately and are not presented as real-data validation.

## A. Dataset and evaluation facts

| # | Item | Status | Evidence / result |
|---|---|---|---|
| 1 | SciFact release and source files | VERIFIED | Authors' release and README; `data/README.md` records source and download procedure. |
| 2 | Test labels are held out | VERIFIED | The authors direct test evaluation through the AI2 leaderboard. This project evaluates on dev only. |
| 3 | Evaluation unit | VERIFIED | One row per unique `(claim, cited abstract)` pair. Repeated cited abstract IDs within a claim are collapsed. |
| 4 | Final dev size | VERIFIED | 300 claims and 339 claim–abstract pairs after duplicate-pair correction. |
| 5 | Final dev gold-label counts | VERIFIED | 138 SUPPORT, 71 CONTRADICT, and 130 NEI pairs. |
| 6 | Train data for majority floor and few-shot examples | VERIFIED | Train contains 809 claims and 919 pairs before evaluation. The majority floor predicts SUPPORT, the most frequent train label (370 of 919). |
| 7 | Dev–train abstract overlap | VERIFIED | The audit found 181 abstracts in both splits. This is disclosed as a dataset limitation; it is not claim-label leakage. |
| 8 | Raw dataset handling | VERIFIED | Raw SciFact JSONL files are kept outside version control under the ignored raw-data directory. |

## B. Pipeline and result validation

| # | Item | Status | Evidence / result |
|---|---|---|---|
| 9 | Real-data Colab execution | VERIFIED | The experiment pipeline was run in Google Colab on a Tesla T4. The environment record is stored in `results/env_check.json`. |
| 10 | Unit and regression tests | VERIFIED | The student ran the project tests in Colab. The duplicate-pair regression case was added after claim 1245 was found to cite abstract 7662395 twice. |
| 11 | Majority-class floor | VERIFIED | Always predicts SUPPORT; macro-F1 = 0.193, matching the hand calculation. |
| 12 | NLI baseline | VERIFIED | `facebook/bart-large-mnli` was run zero-shot; the stored dev macro-F1 is 0.729 (95% CI: 0.675–0.774). |
| 13 | LLM experiment runs | VERIFIED | `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-7B-Instruct` (4-bit) were evaluated under the recorded configurations. |
| 14 | Independent metric audit | VERIFIED | All 26 runs in `results/results.csv` were independently re-scored from saved predictions; maximum absolute difference was \(1\times10^{-16}\). |
| 15 | Prediction/gold alignment | VERIFIED | The audit confirmed that the evaluated runs use the same 339 pairs and identical gold labels, with no INVALID or unknown predictions in the full runs. |
| 16 | Paired NLI-versus-7B comparisons | VERIFIED | The 12 paired claim-level bootstrap comparisons show differences from +0.002 to +0.048; every 95% interval includes zero. |
| 17 | Few-shot overlap check | VERIFIED WITH LIMITATION | No dev claim appears among the few-shot example claims. However, 3 of 9 few-shot abstracts overlap with dev abstracts: seed 13 has 1, seed 42 has 2, and seed 2026 has 0. This is reported as a limitation. |
| 18 | Runtime and environment | VERIFIED | Colab environment: Tesla T4 (15.64 GB), Python 3.13.16, torch 2.11.0+cu130, transformers 5.20.2; `ENV CHECK: PASS`. GPU memory usage was not measured. |

## C. Numeric-rule manual validation

**VERIFIED — 30 claims manually labeled**

The numeric-pattern rule was checked against 30 claims manually labeled by the student. A claim is considered quantitative when it states or compares a quantity; digits that occur only inside a gene, protein, or chemical name do not make the claim quantitative.

| Metric | Final result |
|---|---:|
| Sample size | 30 |
| Rule-positive claims | 11 |
| Human-positive claims | 4 |
| True positives (TP) | 2 |
| False positives (FP) | 9 |
| False negatives (FN) | 2 |
| True negatives (TN) | 17 |
| Agreement | 19/30 = 63.3% |
| Precision | 2/11 = 18.2% |
| Recall | 2/4 = 50.0% |
| Cohen's kappa | 0.0884 |

**Interpretation:** The rule has weak agreement with the manual labels. Nine of its eleven positive flags are false positives because digits occur inside gene, protein, or chemical symbols, including APOE4, p53, Ca2+, PGE 2, Rad53, ALDH1, and UBC13. The numeric-rule flag is therefore a noisy heuristic, not a reliable semantic detector of quantitative claims.

**Evidence files:** `results/numeric_rule_check.csv` and `results/numeric_rule_agreement.json`; computation script: `scripts/09_numeric_agreement.py`.

## D. Manual error analysis

**VERIFIED — 40 errors manually categorized**

The student manually categorized 20 wrong predictions per LLM for the zero-shot T2 configuration, seed 13 (40 error samples total).

### Qwen2.5-7B-Instruct — 20 wrong predictions

| Error category | Count | Share | Description |
|---|---:|---:|---|
| Over-abstention on clear paraphrased evidence | 12 | 60% | 7 CONTRADICT→NEI and 5 SUPPORT→NEI |
| Polarity / direction reversal | 5 | 25% | The predicted relation reverses the claim's evidence polarity or direction. |
| Causal-link or trend nuance misread | 2 | 10% | The model misses a causal relationship or the direction/trend nuance. |
| Topical overlap mistaken for support | 1 | 5% | Related subject matter is treated as supporting evidence. |
| **Total** | **20** | **100%** | |

### Qwen2.5-1.5B-Instruct — 20 wrong predictions

| Error category | Count | Share | Description |
|---|---:|---:|---|
| Unsupported entity / topical overlap (`NEI→SUPPORT`) | 12 | 60% | The model predicts SUPPORT from shared entities or topic despite insufficient evidence. It predicts SUPPORT for opposite claims `268_970012` and `269_970012` when paired with the same abstract. |
| Polarity / negation ignored (`CONTRADICT→SUPPORT`) | 6 | 30% | The model overlooks negation or the opposite direction of the evidence. |
| Other label reversals / errors | 2 | 10% | One `CONTRADICT→NEI` and one `NEI→CONTRADICT`. |
| **Total** | **20** | **100%** | |

**Interpretation:** In this manually reviewed sample, the 7B model's largest error group is over-abstention on paraphrased evidence. The 1.5B model's errors are dominated by false-positive SUPPORT predictions, especially topical/entity overlap and ignored polarity. These are descriptive proportions from 20 sampled errors per model; they are not population-level estimates.

**Evidence files:** `results/error_samples/llm_large_T2_0shot_seed13.csv` and `results/error_samples/llm_small_T2_0shot_seed13.csv`, with the student's `category` labels completed.

## E. Main measured results

The primary metric is three-class macro-F1 on the 339-pair SciFact dev set.

| Method | Macro-F1 |
|---|---:|
| Majority floor (always SUPPORT) | 0.193 |
| `facebook/bart-large-mnli`, zero-shot | 0.729 (95% CI: 0.675–0.774) |
| Qwen2.5-7B-Instruct, 4-bit (six configurations) | 0.682–0.715 |
| Qwen2.5-1.5B-Instruct (six configurations) | 0.285–0.473 |

The NLI baseline exceeds each 7B run by +0.002 to +0.048, but all 12 paired 95% intervals include zero. The T3 abstention sentence did not produce a consistent increase in NEI recall. Numeric-versus-other error-rate differences remain descriptive because the rule's manual validation shows low precision and low kappa; the numeric-rule label should not be treated as a high-quality semantic annotation.

## F. Discrepancies and limitations retained in the report

1. **Repeated cited abstract:** claim 1245 lists abstract 7662395 twice. The pair builder now collapses duplicate cited IDs; the final dev set therefore has 339 rather than 340 pairs. This does not change the evidence label or prompt.
2. **Older validation counts:** any earlier report showing 340 dev pairs and 131 NEI pairs predates the duplicate correction. The final counts are 339 pairs: 138 SUPPORT, 71 CONTRADICT, and 130 NEI.
3. **Few-shot abstract overlap:** three of nine example abstracts overlap with dev abstracts, although no example claim overlaps. This is disclosed as a limitation and is not described as claim-label leakage.
4. **NLI mapping:** mapping the NLI model's neutral output to NEI is a modeling assumption.
5. **Numeric rule:** the manual check shows only 18.2% precision and Cohen's kappa 0.0884. Numeric subgroup conclusions are interpreted cautiously.
6. **GPU memory:** runtime was recorded, but peak memory was not measured.
7. **Paper compilation:** the supplied project archive contains a stand-in-class preview PDF, not an IEEEtran PDF downloaded from Overleaf. The Overleaf-produced IEEEtran PDF should be included only after it has actually been compiled and downloaded.

## G. Reproduction and audit commands

Run from the repository root, after installing dependencies and obtaining the raw SciFact files:

```bash
python scripts/00_download_data.py
python scripts/01_validate_data.py
python scripts/02_numeric_rule_sample.py
python scripts/03_run_baselines.py
python scripts/04_run_llm.py --model-key llm_small
python scripts/04_run_llm.py --model-key llm_large
python scripts/05_evaluate.py
python scripts/06_make_tables_figures.py
python scripts/07_audit_rescore.py
python scripts/08_error_sample.py --n 20
python scripts/09_numeric_agreement.py
python -m unittest discover -s tests -v
```

The model inference commands require a compatible GPU/runtime. The audit, manual-label summaries, and tests should be run separately from the expensive inference stage when appropriate.
