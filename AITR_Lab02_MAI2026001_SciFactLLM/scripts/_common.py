import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ROOT, load_config, resolve  # noqa: E402,F401
from src.data import find_data_dir, load_claims, load_corpus  # noqa: E402,F401
from src.pairs import build_pairs, load_numeric_rule  # noqa: E402,F401


def get_pairs(cfg, split):
    data_dir = find_data_dir(resolve(cfg["data"]["dir"]))
    rule = load_numeric_rule(resolve(cfg["numeric_rule"]))
    return build_pairs(load_claims(data_dir, split), load_corpus(data_dir), split, rule)
