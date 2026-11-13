"""Loading and downloading SciFact (official release tarball)."""
import json
import tarfile
import urllib.request
from pathlib import Path


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_data_dir(root):
    hits = list(Path(root).rglob("claims_dev.jsonl"))
    if not hits:
        raise FileNotFoundError(f"claims_dev.jsonl not found under {root}")
    return hits[0].parent


def download_scifact(url, dest):
    """Download and extract the tarball. Returns the directory with the jsonl files."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / "data.tar.gz"
    if not tar_path.exists():
        urllib.request.urlretrieve(url, tar_path)
    base = dest.resolve()
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():  # refuse path traversal
            if not str((dest / m.name).resolve()).startswith(str(base)):
                raise RuntimeError(f"unsafe path in archive: {m.name}")
        try:
            tf.extractall(dest, filter="data")   # safer extraction on Python versions that support it
        except TypeError:
            tf.extractall(dest)
    return find_data_dir(dest)


def load_claims(data_dir, split):
    if split not in ("train", "dev", "test"):
        raise ValueError(split)
    return _read_jsonl(Path(data_dir) / f"claims_{split}.jsonl")


def load_corpus(data_dir):
    """doc_id (int) -> {"title": str, "abstract": [sentences]}"""
    corpus = {}
    for d in _read_jsonl(Path(data_dir) / "corpus.jsonl"):
        corpus[int(d["doc_id"])] = {"title": d.get("title", ""), "abstract": d.get("abstract", [])}
    return corpus
