import pandas as pd

from src.data.analysis import profile_splits
from src.data.cleaning import build_clean_splits, clean_split, normalize_text


def _df(rows):
    return pd.DataFrame(rows, columns=["text", "label"])


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  hello \n  world\t") == "hello world"


def test_clean_split_drops_empty_rows():
    out = clean_split(_df([("   ", 0), (" ok  text ", 1)]))
    assert out["text"].tolist() == ["ok text"]


def test_build_clean_splits_removes_conflicts_duplicates_and_leakage():
    raw = {
        "train": _df(
            [
                ("a b", 0),
                ("a b", 0),
                ("conflict", 1),
                ("conflict", 2),
                ("leak", 3),
                ("ok", 4),
            ]
        ),
        "validation": _df([("leak", 3)]),
        "test": _df([("t", 0)]),
    }
    clean, stats = build_clean_splits(raw)
    assert sorted(clean["train"]["text"]) == ["a b", "ok"]
    assert stats["train_removed_conflicting_labels"] == 2
    assert stats["train_removed_duplicates"] == 1
    assert stats["train_removed_leakage"] == 1
    assert len(clean["validation"]) == 1
    assert len(clean["test"]) == 1


def test_profile_splits_counts_and_lengths():
    splits = {"train": _df([("a b c", 0), ("d", 1), ("e f", 1)])}
    report = profile_splits(splits, ["neg", "pos"])["train"]
    assert report["rows"] == 3
    assert report["class_counts"] == {"neg": 1, "pos": 2}
    assert report["imbalance_ratio"] == 2.0
    assert report["words"]["min"] == 1
    assert report["words"]["max"] == 3