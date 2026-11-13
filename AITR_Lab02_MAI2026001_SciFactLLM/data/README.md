# Data access

**Dataset:** SciFact (Wadden et al., EMNLP 2020).

**Source (from the authors' GitHub README, found 29 Oct 2026):** the project page is github.com/allenai/scifact, and the release tarball is
`https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`.

**Files after extraction (as listed in the README):** `corpus.jsonl`, `claims_train.jsonl`, `claims_dev.jsonl`, `claims_test.jsonl`, plus smaller `_1` and `_5` variants that this project does not use.

**Test labels:** the README says test-set evaluation is done by submitting to the AI2 leaderboard, so test labels are not public. This project uses train (few-shot examples, majority floor) and dev (evaluation).

**Licence:** the Hugging Face dataset card for `allenai/scifact` states CC BY-NC 2.0 and points to LICENSE.md in the authors' repository (seen in a web search on 2026-11-11). The repository's LICENSE.md itself has not been read: open it and paste the line here: [STUDENT TO FILL]. Non-commercial research use only.

**Download:**

```bash
python scripts/00_download_data.py     # [TEST REQUIRED]
```

Counts from your Colab run (`results/data_validation.json`, written 29 Oct 2026): train 809 claims / 919 pairs (SUPPORT 370, NEI 355, CONTRADICT 194); dev 300 claims / 340 pairs before the duplicate fix (339 after; NEI 131 before, 130 after); test 300 claims with no evidence and no cited abstracts (labels held out). 181 abstracts occur in both train and dev (217 of 340 pre-fix dev pairs).

Date accessed and the exact file listing printed by the script: [STUDENT TO FILL].

Raw data is git-ignored. Do not commit it.
