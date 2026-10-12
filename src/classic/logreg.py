"""TF-IDF + Logistic Regression baseline.

Run from the repository root:  python -m src.classic.logreg
"""

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.classic.evaluate import evaluate_predictions, load_split, time_predict
from src.data.loader import get_label_names, load_raw
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.results import save_metrics
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    labels = get_label_names(load_raw(cfg["dataset"]["name"]))

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    X_test, y_test = load_split("test")

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
    start = time.perf_counter()
    Xtr = vec.fit_transform(X_train)
    vectorize_seconds = round(time.perf_counter() - start, 4)
    Xva, Xte = vec.transform(X_val), vec.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=cfg["seed"])
    start = time.perf_counter()
    model.fit(Xtr, y_train)
    train_seconds = round(time.perf_counter() - start, 4)

    metrics = {
        "model": "tfidf_logreg",
        "features": int(Xtr.shape[1]),
        "vectorize_seconds": vectorize_seconds,
        "train_seconds": train_seconds,
        "validation": evaluate_predictions(y_val, model.predict(Xva), labels),
        "test": evaluate_predictions(y_test, model.predict(Xte), labels),
        "inference": time_predict(model.predict, Xte),
    }
    out = save_metrics("baseline_logreg", metrics, results_dir=cfg["paths"]["results_dir"])
    logger.info("Saved metrics to %s", out)

    print()
    print(f"TF-IDF + Logistic Regression ({metrics['features']} features)")
    print("| Split | Accuracy | Macro-F1 | Weighted-F1 |")
    print("|---|---|---|---|")
    for split in ("validation", "test"):
        m = metrics[split]
        print(f"| {split} | {m['accuracy']} | {m['macro_f1']} | {m['weighted_f1']} |")
    print()
    print("Per-class F1 (test)")
    print("| Class | Precision | Recall | F1 | Support |")
    print("|---|---|---|---|---|")
    for lab, c in metrics["test"]["per_class"].items():
        print(f"| {lab} | {c['precision']} | {c['recall']} | {c['f1']} | {c['support']} |")
    print()
    print(f"Vectorize: {vectorize_seconds}s | Train: {train_seconds}s | "
          f"Inference: {metrics['inference']['predict_ms_per_example']} ms/example")


if __name__ == "__main__":
    main()