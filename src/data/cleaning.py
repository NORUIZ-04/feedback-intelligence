import pandas as pd


def normalize_text(text: str) -> str:
    """Collapse runs of whitespace and strip both ends."""
    return " ".join(str(text).split())


def clean_split(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize whitespace and drop rows whose text is empty."""
    out = df.copy()
    out["text"] = out["text"].map(normalize_text)
    return out[out["text"] != ""].reset_index(drop=True)


def build_clean_splits(
    raw: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Clean all splits and apply extra train-only filtering.

    Validation and test are only whitespace-normalized, never filtered, so
    scores stay comparable with published results. The train split also loses:
      1. rows whose text appears with more than one label (label noise),
      2. exact duplicate texts (keeping the first),
      3. texts that also appear in validation or test (leakage).
    """
    cleaned = {name: clean_split(df) for name, df in raw.items()}
    stats: dict = {
        "empty_rows_removed": {n: len(raw[n]) - len(cleaned[n]) for n in raw},
    }

    train = cleaned["train"]

    label_sets = train.groupby("text")["label"].nunique()
    conflicting = label_sets[label_sets > 1].index
    before = len(train)
    train = train[~train["text"].isin(conflicting)]
    stats["train_removed_conflicting_labels"] = before - len(train)

    before = len(train)
    train = train.drop_duplicates(subset="text", keep="first")
    stats["train_removed_duplicates"] = before - len(train)

    eval_texts: set[str] = set()
    for name, df in cleaned.items():
        if name != "train":
            eval_texts |= set(df["text"])
    before = len(train)
    train = train[~train["text"].isin(eval_texts)]
    stats["train_removed_leakage"] = before - len(train)

    stats["validation_test_overlap_texts"] = len(
        set(cleaned["validation"]["text"]) & set(cleaned["test"]["text"])
    )

    cleaned["train"] = train.reset_index(drop=True)
    stats["train_rows_final"] = len(cleaned["train"])
    return cleaned, stats