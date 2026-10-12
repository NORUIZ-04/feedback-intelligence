import pandas as pd


def profile_splits(splits: dict[str, pd.DataFrame], labels: list[str]) -> dict:
    """Row counts, class balance and text-length stats for every split."""
    report = {}
    for name, df in splits.items():
        rows = len(df)
        counts = df["label"].value_counts()
        class_counts = {lab: int(counts.get(i, 0)) for i, lab in enumerate(labels)}
        class_pct = {k: round(100 * v / rows, 2) for k, v in class_counts.items()}
        non_zero = [c for c in class_counts.values() if c > 0]
        ratio = round(max(non_zero) / min(non_zero), 2) if non_zero else None
        words = df["text"].str.split().str.len()
        report[name] = {
            "rows": rows,
            "class_counts": class_counts,
            "class_pct": class_pct,
            "imbalance_ratio": ratio,
            "words": {
                "min": int(words.min()),
                "mean": round(float(words.mean()), 2),
                "median": float(words.median()),
                "p95": float(words.quantile(0.95)),
                "max": int(words.max()),
            },
        }
    return report


def render_markdown(
    raw: dict, clean: dict, stats: dict, labels: list[str], dataset_name: str
) -> str:
    """Render the measured profile as markdown tables for README and docs."""
    lines = [f"### Dataset: `{dataset_name}`", ""]

    lines += ["| Split | Raw rows | Clean rows |", "|---|---|---|"]
    for split in clean:
        lines.append(f"| {split} | {raw[split]['rows']} | {clean[split]['rows']} |")

    train = clean["train"]
    lines += ["", "**Class distribution (clean train split)**", ""]
    lines += ["| Label | Count | Share |", "|---|---|---|"]
    for lab in labels:
        count = train["class_counts"][lab]
        lines.append(f"| {lab} | {count} | {train['class_pct'][lab]}% |")
    lines += [
        "",
        f"Imbalance ratio (largest / smallest class): {train['imbalance_ratio']}",
    ]

    lines += ["", "**Text length in words (clean splits)**", ""]
    lines += ["| Split | Min | Mean | Median | P95 | Max |", "|---|---|---|---|---|---|"]
    for split, prof in clean.items():
        w = prof["words"]
        row = f"| {split} | {w['min']} | {w['mean']} | {w['median']} | "
        lines.append(row + f"{w['p95']} | {w['max']} |")

    empty_total = sum(stats["empty_rows_removed"].values())
    lines += ["", "**Cleaning report**", "", "| Check | Rows |", "|---|---|"]
    conflicts = stats["train_removed_conflicting_labels"]
    dups = stats["train_removed_duplicates"]
    leaks = stats["train_removed_leakage"]
    overlap = stats["validation_test_overlap_texts"]
    lines += [
        f"| Empty texts removed (all splits) | {empty_total} |",
        f"| Train rows with conflicting labels removed | {conflicts} |",
        f"| Train duplicate rows removed | {dups} |",
        f"| Train rows also in validation/test removed | {leaks} |",
        f"| Texts shared by validation and test (kept) | {overlap} |",
    ]

    lines += [
        "",
        "![Class distribution](results/plots/class_distribution.png)",
        "",
        "![Text length distribution](results/plots/text_length_distribution.png)",
        "",
    ]
    return "\n".join(lines)