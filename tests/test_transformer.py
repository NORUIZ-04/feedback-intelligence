import numpy as np
import torch

from src.transformer.analyze import top_confusions
from src.transformer.data import TextDataset, collate_fn
from src.transformer.token_lengths import length_stats, truncated_share


class FakeTokenizer:
    """Maps each word to a fixed id and truncates like a real tokenizer."""

    def __call__(self, texts, truncation=True, max_length=None, padding=False):
        ids = []
        for text in texts:
            row = [101] + [abs(hash(w)) % 1000 + 1000 for w in text.split()] + [102]
            if truncation and max_length is not None:
                row = row[:max_length]
            ids.append(row)
        return {"input_ids": ids, "attention_mask": [[1] * len(r) for r in ids]}


def test_text_dataset_truncates_and_keeps_labels():
    ds = TextDataset(["a b c d e f", "a"], [3, 1], FakeTokenizer(), max_length=4)
    assert len(ds) == 2
    assert len(ds[0]["input_ids"]) == 4
    assert ds[0]["labels"] == 3
    assert ds[1]["labels"] == 1


def test_collate_pads_to_longest_with_mask():
    batch = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": 0},
        {"input_ids": [4], "attention_mask": [1], "labels": 2},
    ]
    out = collate_fn(pad_token_id=0)(batch)
    assert out["input_ids"].shape == (2, 3)
    assert out["input_ids"][1].tolist() == [4, 0, 0]
    assert out["attention_mask"][1].tolist() == [1, 0, 0]
    assert out["labels"].tolist() == [0, 2]
    assert out["labels"].dtype == torch.long


def test_length_stats_and_truncated_share():
    lengths = np.array([5, 10, 10, 20, 40])
    stats = length_stats(lengths)
    assert stats["min"] == 5
    assert stats["max"] == 40
    assert stats["median"] == 10.0
    assert truncated_share(lengths, 20) == 20.0
    assert truncated_share(lengths, 40) == 0.0


def test_top_confusions_ignores_diagonal_and_sorts():
    cm = np.array([[9, 3, 0], [1, 8, 5], [0, 2, 7]])
    pairs = top_confusions(cm, ["a", "b", "c"], k=2)
    assert pairs[0] == {"true": "b", "predicted": "c", "count": 5}
    assert pairs[1] == {"true": "a", "predicted": "b", "count": 3}