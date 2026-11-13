# What you still have to do (step by step)

## Status after the audit of your results (2026-11-11). Boxes below are still yours to tick.

Evidence found in `results/` that these were done: Colab run of scripts 00 to 05 (env check passed, data validation, 26 scored runs, 24 full LLM runs with no INVALID outputs), duplicate-pair fix (P1).
**Still open, nothing in the archive shows them done:** A2 licence line and model cards (A3), literature search (A4), original-paper checks (A5), numeric hand labels (B8), config commit hash (B9), repeat-run check (B13), manual error analysis (C14), all of D, E and F.
Update 2026-11-12: numeric hand labels were given (rows 18 and 19 numeric) and processed: the rule fails (precision 1/11), so RQ3 is reported as not answered; the error samples were received and a draft grouping is in the paper and in `results/error_samples/` (column `claude_suggested_category`). You still have to read the full abstracts for those 40 rows, fill the `category` column yourself and edit the paper subsection 'Error analysis'.
Added after the second pass: `results/numeric_rule_check_for_review.csv` holds the 30 claims with **suggested** labels and reasons from the assistant; read each claim yourself, copy your final True/False into `human_says_quantitative` in `results/numeric_rule_check.csv`, then run `python scripts/09_numeric_agreement.py`. For the error analysis run `python scripts/08_error_sample.py` in Colab (needs the data), read the 20 rows per model, and write a short category in the `category` column. Licence/model-card/novelty lookups are summarised in `research_decisions.md` (D7, D18) but you must still open the official pages.
New items from the audit: (a) decide what to do about the 3 few-shot abstracts that also occur in dev (baseline_validation check 21; if you rerun, log it as a post-hoc protocol change); (b) rerun `01_validate_data.py` so `data_validation.json` shows 339 pairs; (c) `.gitignore` excludes `results/env_check.json`, but the assignment asks for environment evidence: remove that line before committing; (d) `scripts/06` was edited (P2): add your commit hash.


Do these in order. Tick each box only when it has really been done. Use the Prompt 11 text with Claude only when you have a real error message.

## A. Before running anything

1. [ ] Read `research_brief.md`, `research_decisions.md` and `configs/`. Be able to explain every choice in your own words.
2. [ ] Open the original SciFact paper (EMNLP 2020) and the SciFact repository. Check the split sizes, label names and the licence line. Paste what you find into `data/README.md` and `baseline_validation.md` (items 3 to 5).
3. [ ] Open the model cards for `facebook/bart-large-mnli`, `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-7B-Instruct`. Note licence, gating, training data and maximum length. Record it in `research_decisions.md` (D7). If you swap a model, edit `configs/config.yaml` and log the change.
4. [ ] Run the literature searches from the Phase I document (Part B). Read the closest 5 to 10 papers. Fill the novelty matrix honestly, and soften the claims in `research_brief.md` and the paper if something close already exists.
5. [ ] Verify each number you cite from your review (GPQA, TruthfulQA, PubMedQA, FActScore, SimpleQA) in the original paper before you reuse it.

## B. Colab run

6. [ ] Open `notebooks/01_colab_run_all.ipynb` in Colab, choose a GPU runtime, and run the cells in order. Paste the real output of the environment check and `01_validate_data.py` into your chat with Claude if anything fails.
7. [ ] If `01_validate_data.py` shows zero NEI pairs in dev, stop. Write the problem in the protocol change log, then decide with your supervisor or by your own reasoning how to continue. Do not edit the design after looking at model results.
8. [ ] Run `scripts/02_numeric_rule_sample.py`, label the 30 claims by hand, and record the agreement in `baseline_validation.md` (item 18).
9. [ ] Commit `configs/` to git **before** the first dev run. Write the commit hash in `research_decisions.md`.
10. [ ] Run the baselines, then the LLM smoke test (`--limit 10`). Open a few raw outputs and read them. If many are INVALID, fix the parser or prompt only on train data, not on dev, and log it.
11. [ ] Run the full LLM jobs. If Colab disconnects, rerun the same command; finished files are skipped.
12. [ ] Run `05_evaluate.py` and `06_make_tables_figures.py`, then `scripts/freeze_env.sh`. Download the results zip before closing Colab.
13. [ ] Repeat one configuration with the same seed to see how much the output varies. Note it in `baseline_validation.md` (item 19).

## C. Reading the results (yours, not the AI's)

14. [ ] Read at least 20 wrong predictions per LLM and group them yourself. Write the groups and counts in the paper's error analysis.
15. [ ] Answer each research question from the tables, using the decision rules in the brief. If the result is negative or unclear, say so.
16. [ ] Copy `results/` outputs into `paper/` (the script already does this for tables and figures).

## D. Paper

17. [ ] Upload `paper/` to Overleaf. Change the document class if your instructor gives a template. Fill every red `[TO FILL]`. Write the abstract last.
18. [ ] Check every citation against the original source. Add the model and dataset references you used, with verified metadata.
19. [ ] Rewrite the text in your own voice. The drafts are a starting point; you must be able to explain every sentence at the viva.
20. [ ] Ask Claude for a peer review (Prompt 16) on your finished PDF and log what you changed.

## E. Repository and submission

21. [ ] Create a GitHub repository. Upload everything in this folder. Check that `.gitignore` is present, no tokens or raw data are committed, and `results/` contains only real outputs.
22. [ ] Replace every `[TEST REQUIRED]` in the README with the real result, or leave it if you did not run the command.
23. [ ] Choose a licence for your code.
24. [ ] Add your own entries to `ai_assistance_log.md` using this template: date, module, exact error text, what the AI suggested, what you checked, the fix you applied, what needed rerunning.
25. [ ] Rename the folder if your instructor wants `AITR_Lab02_EnrollmentNo_TopicShortName`. Submit through Google Classroom.
26. [ ] Practice the 5-minute demo: show the pipeline, one result table, one error example, and be ready to rerun one component.

## F. What to upload to GitHub (matches Section 9 of the assignment)

README.md, requirements.txt, requirements.lock.txt (after your run), configs/, data/README.md, notebooks/, src/, scripts/, results/ (real outputs only), paper/ (LaTeX, bib, figures, final PDF), research_brief.md, research_decisions.md, baseline_validation.md, ai_assistance_log.md. Extra folders `tests/` and `docs/` are fine to keep.
