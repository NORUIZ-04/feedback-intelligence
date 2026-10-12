"""Measure DistilBERT token lengths to choose max_length.

Run from the repository root:  python -m src.transformer.token_lengths
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.results import save_metrics

plt.switch_backend("Agg")
logger = get_logger(__name__)

CANDIDATES = [32, 48, 64, 96, 128]


def token_lengths(tokenizer, texts: list[str]) -> np.ndarray:
    enc = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
    return np.asarray([len(ids) for ids in enc["input_ids"]])


def length_stats(lengths: np.ndarray) -> dict:
    return {
        "min": int(lengths.min()),
        "mean": round(float(lengths.mean()), 2),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
    }


def truncated_share(lengths: np.ndarray, max_length: int) -> float:
    return round(float((lengths > max_length).mean() * 100), 3)


def main() -> None:
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_lengths = {}
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(f"data/processed/{split}.parquet")
        all_lengths[split] = token_lengths(tokenizer, df["text"].tolist())

    stats = {split: length_stats(v) for split, v in all_lengths.items()}
    truncation = {
        str(c): {split: truncated_share(v, c) for split, v in all_lengths.items()}
        for c in CANDIDATES
    }
    metrics = {"tokenizer": model_name, "stats": stats, "truncated_percent": truncation}
    out = save_metrics("token_lengths", metrics, results_dir=str(results_dir))
    logger.info("Saved metrics to %s", out)

    plot_path = results_dir / "plots" / "token_length_distribution.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    train = all_lengths["train"]
    ax.hist(train, bins=range(int(train.max()) + 2), color="#C44E52")
    ax.axvline(np.percentile(train, 95), color="black", linestyle="--", label="95th percentile")
    ax.set_title(f"Token length, train split ({model_name})")
    ax.set_xlabel("Tokens per example (with special tokens)")
    ax.set_ylabel("Number of examples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print()
    print(f"Tokenizer: {model_name}")
    print("| Split | Min | Mean | Median | P95 | P99 | Max |")
    print("|---|---|---|---|---|---|---|")
    for split, s in stats.items():
        print(
            f"| {split} | {s['min']} | {s['mean']} | {s['median']} | "
            f"{s['p95']} | {s['p99']} | {s['max']} |"
        )
    print()
    print("Share of examples truncated at each max_length (%)")
    print("| max_length | train | validation | test |")
    print("|---|---|---|---|")
    for c, row in truncation.items():
        print(f"| {c} | {row['train']} | {row['validation']} | {row['test']} |")


if __name__ == "__main__":
    main()