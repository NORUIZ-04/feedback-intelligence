from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")


def plot_class_distribution(split_profile: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(split_profile["class_counts"])
    counts = list(split_profile["class_counts"].values())
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts, color="#4C72B0")
    ax.bar_label(bars)
    ax.set_title("Class distribution (clean train split)")
    ax.set_ylabel("Number of examples")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_length_distribution(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    words = df["text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(words, bins=range(int(words.max()) + 2), color="#55A868")
    ax.axvline(
        words.quantile(0.95),
        color="red",
        linestyle="--",
        label="95th percentile",
    )
    ax.set_title("Text length (clean train split)")
    ax.set_xlabel("Words per example")
    ax.set_ylabel("Number of examples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)