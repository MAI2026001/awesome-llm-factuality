"""Unit tests on tiny hand-made fixtures. They check code behaviour, not experimental results."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import metrics as M  # noqa: E402
from src.baselines import majority_label, predict_majority  # noqa: E402
from src.data import find_data_dir, load_claims, load_corpus  # noqa: E402
from src.llm import parse_label  # noqa: E402
from src.pairs import LABELS, build_pairs, is_numeric, load_numeric_rule, select_fewshot  # noqa: E402
from src.prompts import build_messages, load_prompts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

CORPUS = [
    {"doc_id": 1, "title": "T1", "abstract": ["A one.", "A two."]},
    {"doc_id": 2, "title": "T2", "abstract": ["B one."]},
    {"doc_id": 3, "title": "T3", "abstract": ["C one."]},
]
CLAIMS = [
    {"id": 10, "claim": "Drug X lowers risk by 20%.", "cited_doc_ids": [1], "evidence": {"1": [{"sentences": [0], "label": "SUPPORT"}]}},
    {"id": 11, "claim": "Gene Y causes disease.", "cited_doc_ids": [2], "evidence": {"2": [{"sentences": [0], "label": "CONTRADICT"}]}},
    {"id": 12, "claim": "Protein Z binds DNA.", "cited_doc_ids": [3], "evidence": {}},
    {"id": 13, "claim": "No cited docs here.", "cited_doc_ids": [], "evidence": {}},
    {"id": 14, "claim": "Mixed labels.", "cited_doc_ids": [1], "evidence": {"1": [{"sentences": [0], "label": "SUPPORT"}, {"sentences": [1], "label": "CONTRADICT"}]}},
]


def write_jsonl(path, rows):
    Path(path).write_text("\n".join(json.dumps(r) for r in rows))


class TestData(unittest.TestCase):
    def test_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "data"
            sub.mkdir()
            write_jsonl(sub / "corpus.jsonl", CORPUS)
            write_jsonl(sub / "claims_dev.jsonl", CLAIMS)
            data_dir = find_data_dir(d)
            self.assertEqual(len(load_claims(data_dir, "dev")), 5)
            self.assertEqual(set(load_corpus(data_dir)), {1, 2, 3})
            with self.assertRaises(ValueError):
                load_claims(data_dir, "validation")


class TestPairs(unittest.TestCase):
    def setUp(self):
        self.corpus = {int(d["doc_id"]): {"title": d["title"], "abstract": d["abstract"]} for d in CORPUS}
        self.rule = load_numeric_rule(ROOT / "configs" / "numeric_rule.yaml")

    def test_labels_and_exclusions(self):
        df, stats = build_pairs(CLAIMS, self.corpus, "dev", self.rule)
        gold = dict(zip(df["claim_id"], df["gold"]))
        self.assertEqual(gold, {10: "SUPPORT", 11: "CONTRADICT", 12: "NEI"})
        self.assertEqual(stats["claims_without_cited_docs"], 1)
        self.assertEqual(stats["mixed_label_pairs"], 1)
        self.assertEqual(df.loc[df.claim_id == 10, "abstract"].iloc[0], "A one. A two.")

    def test_repeated_cited_doc_gives_one_pair(self):
        claims = [{"id": 20, "claim": "X.", "cited_doc_ids": [1, 1], "evidence": {}}]
        df, stats = build_pairs(claims, self.corpus, "dev", self.rule)
        self.assertEqual(len(df), 1)
        self.assertEqual(df["pair_id"].is_unique, True)
        self.assertEqual(stats["duplicate_cited_docs"], 1)

    def test_test_split_refused(self):
        with self.assertRaises(ValueError):
            build_pairs(CLAIMS, self.corpus, "test", self.rule)

    def test_numeric_rule(self):
        self.assertTrue(is_numeric("Risk falls by 20%", self.rule))
        self.assertTrue(is_numeric("A twofold increase", self.rule))
        self.assertTrue(is_numeric("Dose doubles survival", self.rule))
        self.assertFalse(is_numeric("Gene Y causes disease", self.rule))

    def test_fewshot(self):
        rows = []
        for i in range(30):
            rows.append({"pair_id": f"{i}", "claim_id": i, "doc_id": i, "claim": f"c{i}", "title": "t",
                         "abstract": "a" * 50, "gold": LABELS[i % 3], "is_numeric": False, "split": "train"})
        train = pd.DataFrame(rows)
        a = select_fewshot(train, seed=13)
        b = select_fewshot(train, seed=13)
        c = select_fewshot(train, seed=42)
        self.assertEqual(list(a["gold"]), LABELS)
        self.assertEqual(list(a["pair_id"]), list(b["pair_id"]))
        self.assertTrue(set(a["pair_id"]) | set(c["pair_id"]))
        dev = train.copy()
        dev["split"] = "dev"
        with self.assertRaises(ValueError):
            select_fewshot(dev, seed=13)

    def test_majority(self):
        train = pd.DataFrame({"gold": ["SUPPORT", "SUPPORT", "NEI"]})
        self.assertEqual(majority_label(train), "SUPPORT")
        ev = pd.DataFrame({"pair_id": ["a", "b"], "claim_id": [1, 2], "doc_id": [1, 2],
                           "gold": ["NEI", "SUPPORT"], "is_numeric": [False, False]})
        self.assertEqual(set(predict_majority(train, ev)["pred"]), {"SUPPORT"})


class TestParser(unittest.TestCase):
    def test_cases(self):
        self.assertEqual(parse_label("SUPPORT"), "SUPPORT")
        self.assertEqual(parse_label("  contradict."), "CONTRADICT")
        self.assertEqual(parse_label("NEI\n"), "NEI")
        self.assertEqual(parse_label("Answer: SUPPORT"), "INVALID")
        self.assertEqual(parse_label("I think it is NEI"), "INVALID")
        self.assertEqual(parse_label(""), "INVALID")
        self.assertEqual(parse_label(None), "INVALID")
        self.assertEqual(parse_label("SUPPORTS the claim"), "INVALID")


class TestPrompts(unittest.TestCase):
    def setUp(self):
        self.prompts = load_prompts(ROOT / "configs" / "prompts.yaml")
        self.row = pd.Series({"claim_id": 99, "pair_id": "99_5", "claim": "CLAIMTEXT", "title": "TITLE",
                              "abstract": "ABSTRACTTEXT", "gold": "CONTRADICT"})

    def test_no_leak_and_structure(self):
        msgs = build_messages(self.prompts, "T2", self.row)
        joined = " ".join(m["content"] for m in msgs)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertIn("CLAIMTEXT", joined)
        self.assertNotIn("99_5", joined)
        self.assertNotIn("claim_id", joined)

    def test_t3_has_abstention_t2_does_not(self):
        t2 = build_messages(self.prompts, "T2", self.row)[0]["content"]
        t3 = build_messages(self.prompts, "T3", self.row)[0]["content"]
        self.assertNotIn("instead of guessing", t2)
        self.assertIn("instead of guessing", t3)

    def test_fewshot_turns(self):
        fs = pd.DataFrame([{"claim": f"c{i}", "title": "t", "abstract": "a", "gold": g} for i, g in enumerate(LABELS)])
        msgs = build_messages(self.prompts, "T1", self.row, fewshot=fs)
        self.assertEqual(len(msgs), 1 + 6 + 1)
        self.assertEqual([m["content"] for m in msgs[2:7:2]], LABELS)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "pair_id": list("abcd"), "claim_id": [1, 2, 3, 4],
            "gold": ["SUPPORT", "SUPPORT", "CONTRADICT", "NEI"],
            "pred": ["SUPPORT", "CONTRADICT", "CONTRADICT", "SUPPORT"],
            "is_numeric": [True, True, False, False]})

    def test_macro_f1_hand_computed(self):
        # SUPPORT F1=0.5, CONTRADICT F1=2/3, NEI F1=0 -> mean 0.388888...
        self.assertAlmostEqual(M.macro_f1(self.df), (0.5 + 2 / 3 + 0) / 3, places=6)

    def test_nei_and_error_metrics(self):
        m = M.nei_metrics(self.df)
        self.assertEqual(m["nei_recall"], 0.0)
        self.assertEqual(m["confident_wrong_rate"], 1.0)
        self.assertTrue(np.isnan(m["nei_precision"]))
        self.assertEqual(M.error_rate(self.df), 0.5)
        self.assertEqual(M.numeric_gap(self.df), 0.5 - 0.5)

    def test_invalid_counts_as_wrong(self):
        df = self.df.copy()
        df.loc[0, "pred"] = "INVALID"
        self.assertEqual(M.invalid_rate(df), 0.25)
        self.assertEqual(M.error_rate(df), 0.75)

    def test_perfect_predictions_ci(self):
        df = self.df.copy()
        df["pred"] = df["gold"]
        self.assertEqual(M.macro_f1(df), 1.0)
        self.assertEqual(M.bootstrap_ci(df, M.macro_f1, n=50, seed=1), (1.0, 1.0))

    def test_alignment_checks(self):
        ref = self.df[["pair_id", "gold"]]
        M.check_alignment(self.df, ref)
        with self.assertRaises(ValueError):
            M.check_alignment(self.df.iloc[:3], ref)
        dup = pd.concat([self.df, self.df.iloc[[0]]])
        with self.assertRaises(ValueError):
            M.check_alignment(dup, ref)
        bad = self.df.copy()
        bad.loc[0, "gold"] = "NEI"
        with self.assertRaises(ValueError):
            M.check_alignment(bad, ref)

    def test_ece_calibrated_and_overconfident(self):
        df = self.df.copy()
        df[["p_support", "p_contradict", "p_nei"]] = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]]
        # confidence 1.0 everywhere; argmax is correct for 3 of 4 rows -> ECE = |0.75 - 1.0| = 0.25
        self.assertAlmostEqual(M.ece(df), 0.25, places=6)
        df[["p_support", "p_contradict", "p_nei"]] = np.nan
        self.assertTrue(np.isnan(M.ece(df)))

    def test_ece_without_probability_columns(self):
        self.assertTrue(np.isnan(M.ece(self.df)))   # e.g. the majority-class baseline has none

    def test_paired_diff_same_data_is_zero(self):
        point, lo, hi = M.paired_bootstrap_diff(self.df, self.df, M.macro_f1, n=50, seed=1)
        self.assertEqual((point, lo, hi), (0.0, 0.0, 0.0))


class TestErrorSample(unittest.TestCase):
    def test_only_wrong_rows_with_text_and_empty_category(self):
        from src.errors import sample_errors
        pairs = pd.DataFrame({"pair_id": ["1_1", "2_2", "3_3"], "claim": ["a", "b", "c"], "title": ["t"] * 3, "abstract": ["x" * 900] * 3})
        pred = pd.DataFrame({"pair_id": ["1_1", "2_2", "3_3", "3_3"], "gold": ["NEI", "SUPPORT", "NEI", "NEI"],
                             "pred": ["NEI", "NEI", "SUPPORT", "SUPPORT"], "raw": ["NEI", "NEI", "SUPPORT", "SUPPORT"]})
        out = sample_errors(pred, pairs, n=5)
        self.assertEqual(sorted(out["pair_id"]), ["2_2", "3_3"])
        self.assertTrue((out["abstract"].str.len() == 700).all())
        self.assertTrue((out["category"] == "").all())


if __name__ == "__main__":
    unittest.main()
