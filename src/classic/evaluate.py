import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def evaluate_predictions(y_true, y_pred, labels: list[str]) -> dict:
    """Accuracy, macro/weighted F1 and per-class precision, recall, F1."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), zero_division=0
    )
    per_class = {
        lab: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, lab in enumerate(labels)
    }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "per_class": per_class,
    }


def time_predict(predict_fn, X, repeats: int = 3) -> dict:
    """Best-of-N prediction time for a whole set and per example, in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        predict_fn(X)
        best = min(best, time.perf_counter() - start)
    n = X.shape[0] if hasattr(X, "shape") else len(X)
    return {
        "predict_total_ms": round(best * 1000, 2),
        "predict_ms_per_example": round(best * 1000 / n, 4),
    }


def load_split(name: str):
    import pandas as pd

    df = pd.read_parquet(f"data/processed/{name}.parquet")
    return df["text"].tolist(), np.asarray(df["label"])