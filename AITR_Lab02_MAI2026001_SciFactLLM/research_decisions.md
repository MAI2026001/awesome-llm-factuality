# Research decisions log

**Project:** Small Open Language Models on Scientific Claim Verification  
**Student:** Shingala Smit Mukeshbhai (MAI2026001)  
**Last updated:** 12 November 2026

Evidence labels: **VERIFIED** = supported by a primary source, real experiment run, completed manual review, or reproducible audit; **INFERRED** = plausible but not fully checked; **UNKNOWN** = not checked.

## Main decisions

| ID | Date | Decision | Reason | Evidence / status |
|---|---|---|---|---|
| D1 | 2026-10-29 | Use pathway B: a controlled comparison | Inference-only study fits the available compute and addresses abstention and numeric-claim errors. | Protocol decision; literature novelty remains limited by the searches described in D18. |
| D2 | 2026-10-29 | Use SciFact dev for evaluation | Public scientific claim-verification benchmark; test labels are held out by the dataset authors. | VERIFIED from the authors' project README and real data validation. |
| D3 | 2026-10-29 | Use one unique pair per `(claim, cited abstract)` | This makes the evidence unit explicit and supports pair-level prediction. Gold label is NEI when the cited abstract has no evidence annotation for that claim. | VERIFIED in the implementation and real-data pair construction. |
| D4 | 2026-10-29 | Exclude pairs with mixed evidence labels and count exclusions | A single pair-level gold label cannot represent conflicting evidence annotations. | Implemented in `src/pairs.py`; unit-tested. |
| D5 | 2026-10-29 | Primary metric: three-class macro-F1; bootstrap CIs resample claims | Class balance is uneven and pairs from one claim are not independent. | Frozen protocol; 1,000 claim-level bootstrap resamples. |
| D6 | 2026-10-29 | Average macro-F1 over labels present in gold or predictions | A class absent from a bootstrap resample should not automatically receive an artificial zero score. | Unit-test issue identified and corrected; full dev set contains all three labels. |
| D7 | 2026-10-29 | Compare majority floor, `facebook/bart-large-mnli`, Qwen2.5-1.5B-Instruct, and Qwen2.5-7B-Instruct | Provides a simple floor, an NLI cross-encoder, and two open-weight instruction models at different sizes. The 7B model is loaded in 4-bit for T4 feasibility. | Models and runs recorded in `results/results.csv`; official model-card licence review remains a separate limitation. |
| D8 | 2026-10-29 | Freeze prompts T1, T2, T3 and 0-/3-shot settings | T3 adds an abstention sentence so its effect can be compared with the other templates. | Prompt definitions in `configs/prompts.yaml`. |
| D9 | 2026-10-29 | Use a simple digit/word-pattern numeric flag and manually audit 30 claims | A transparent heuristic is easy to reproduce but may mistake digits in biomedical names for quantities. | Manual audit completed; see D17. |
| D10 | 2026-10-29 | Disable test-split evaluation | Test labels are held out by the authors. | Enforced by `build_pairs`; evaluation is on dev. |
| D11 | 2026-10-29 | Use seeds 13, 42, and 2026; zero-shot uses seed 13 | Greedy decoding does not sample; seeds affect few-shot example selection. | Recorded in `configs/config.yaml` and prediction files. |
| D12 | 2026-10-29 | Collapse repeated cited abstract IDs within each claim | Claim 1245 cites abstract 7662395 twice, producing a duplicate pair. | Implemented in `src/pairs.py`; final dev count is 339 pairs. |

## Protocol change log

| Date | Change | Reason | Results seen before change? | Required reruns / status |
|---|---|---|---|---|
| 2026-10-29 | P1: collapse repeated cited abstract IDs; retain one pair per unique claim–abstract combination | Claim 1245 listed abstract 7662395 twice, which would otherwise count the same pair twice. | No evaluation metric had been computed or viewed when the change was made; only the first 10 smoke-test rows had been seen. | Final evaluation uses 339 unique pairs. |
| 2026-11-11 | P2: update `scripts/06_make_tables_figures.py` | Corrected table pairing so a seed-13 score is shown beside its own interval; updated ECE presentation and overview figure. | Yes. This was a presentation correction after results existed. | Re-ran table/figure generation only; no values in `results.csv` changed. |

## Findings from the results audit

| ID | Decision / finding | Evidence | Status |
|---|---|---|---|
| D13 | All 26 reported runs reproduce from saved predictions; maximum absolute metric difference is \(1\times10^{-16}\). | `scripts/07_audit_rescore.py`, `results/audit_report.json` | VERIFIED |
| D14 | NLI versus Qwen2.5-7B is described as not statistically distinguishable under the paired comparisons, not as a tie or a model win. | 12 paired claim-level bootstrap intervals in `results/paired_comparisons.csv`; all include zero. | VERIFIED |
| D15 | The T3 abstention sentence did not consistently raise NEI recall. | `results/results.csv`; e.g. 7B zero-shot 0.877 to 0.885, 7B three-shot 0.774 to 0.800, while the 1.5B results do not show a consistent increase. | VERIFIED |
| D16 | Few-shot sets contain one example per label in the fixed order SUPPORT, CONTRADICT, NEI; the same sets are used for both LLMs and all templates. Three of nine example abstracts overlap with dev abstracts, but no example claim overlaps. | `src/pairs.py::select_fewshot`, saved `*_fewshot_ids.csv`, and audit. | VERIFIED; overlap is disclosed as a limitation. |
| **D17** | **Close the numeric-rule validation using the completed 30-claim manual audit.** Rule-positive = 11; human-positive = 4; TP = 2; FP = 9; FN = 2; TN = 17. Agreement = 19/30 = 63.3%; precision = 2/11 = 18.2%; recall = 2/4 = 50.0%; Cohen's kappa = 0.0884. | `results/numeric_rule_check.csv`, `results/numeric_rule_agreement.json`, and `scripts/09_numeric_agreement.py`. Nine of eleven rule positives contain digits only inside gene/protein/chemical symbols, including APOE4, p53, Ca2+, PGE 2, Rad53, ALDH1, and UBC13. | **VERIFIED — CLOSED** |
| D18 | A quick novelty search identified close related work: Abdaljalil et al. (2026; abstention-aware verification on SciFact and PubMedQA) and Kumar (2026; QLoRA small LLMs on SciFact and HealthVer), plus a 2025 BioNLP study of small LLMs on scientific verification datasets. | Search-result abstracts and metadata only; the two closest papers have not been read in full. | Search recorded; no novelty claim is made. |

## Manual error-analysis decision

| ID | Date | Decision | Evidence | Status |
|---|---|---|---|---|
| D19 | 2026-11-12 | Complete a manual error analysis of 20 wrong zero-shot T2, seed-13 predictions per LLM (40 errors total). | `results/error_samples/llm_large_T2_0shot_seed13.csv` and `results/error_samples/llm_small_T2_0shot_seed13.csv`, with category labels completed. | VERIFIED — CLOSED |

The 7B sample contains 12/20 (60%) over-abstentions on clear paraphrased evidence (7 CONTRADICT→NEI, 5 SUPPORT→NEI), 5/20 (25%) polarity/direction reversals, 2/20 (10%) causal-link or trend nuance errors, and 1/20 (5%) topical-overlap error.

The 1.5B sample contains 18/20 (90%) false-positive SUPPORT predictions: 12/20 (60%) unsupported entity/topical overlap (`NEI→SUPPORT`), 6/20 (30%) ignored polarity/negation (`CONTRADICT→SUPPORT`), and 2/20 (10%) other label errors (`CONTRADICT→NEI` and `NEI→CONTRADICT`). The model predicted SUPPORT for opposite claims `268_970012` and `269_970012` paired with the same abstract.

These percentages describe the manually selected sample of wrong predictions; they are not estimates of population error prevalence.

## Repository record

- **GitHub repository:** https://github.com/MAI2026001/Small-Open-Language-Models-on-Scientific-Claim-Verification
- **Default branch:** `main`
- **Visibility:** Public
- **Repository state checked:** 12 November 2026
- **Commit hash:** No commit exists yet. GitHub reports that this repository has an empty Git history (HTTP 409 when listing commits). A commit hash cannot be truthfully recorded until the project files are pushed.
- **Student access:** GitHub reports push/admin permissions for the connected account.

This repository record intentionally states the verified empty-repository status instead of inventing a commit identifier.
