"""Logistic Regression C sweep, selected on validation macro-F1 only.

Run from the repository root:  python -m src.classic.logreg_sweep
"""

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.classic.evaluate import evaluate_predictions, load_split
from src.data.loader import get_label_names, load_raw
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.results import save_metrics
from src.utils.seed import set_seed

logger = get_logger(__name__)

C_VALUES = [0.1, 1, 10, 100]


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    labels = get_label_names(load_raw(cfg["dataset"]["name"]))

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    X_test, y_test = load_split("test")

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform(X_train)
    Xva, Xte = vec.transform(X_val), vec.transform(X_test)

    runs = []
    for weight in (None, "balanced"):
        for c in C_VALUES:
            model = LogisticRegression(
                C=c, max_iter=2000, class_weight=weight, random_state=seed
            )
            start = time.perf_counter()
            model.fit(Xtr, y_train)
            seconds = round(time.perf_counter() - start, 4)
            val = evaluate_predictions(y_val, model.predict(Xva), labels)
            runs.append(
                {
                    "class_weight": weight or "none",
                    "C": c,
                    "train_seconds": seconds,
                    "validation_accuracy": val["accuracy"],
                    "validation_macro_f1": val["macro_f1"],
                    "model": model,
                }
            )

    best = max(runs, key=lambda r: r["validation_macro_f1"])
    best_test = evaluate_predictions(y_test, best["model"].predict(Xte), labels)

    saved_runs = [{k: v for k, v in r.items() if k != "model"} for r in runs]
    metrics = {
        "model": "logreg_sweep",
        "selection_metric": "validation_macro_f1",
        "runs": saved_runs,
        "best": {k: v for k, v in best.items() if k != "model"},
        "best_test": best_test,
    }
    save_metrics("baseline_logreg_sweep", metrics, results_dir=cfg["paths"]["results_dir"])

    print()
    print("Logistic Regression sweep (selected on validation macro-F1)")
    print("| Class weight | C | Val accuracy | Val macro-F1 | Train s |")
    print("|---|---|---|---|---|")
    for r in saved_runs:
        print(
            f"| {r['class_weight']} | {r['C']} | {r['validation_accuracy']} | "
            f"{r['validation_macro_f1']} | {r['train_seconds']} |"
        )
    print()
    print(f"Best on validation: class_weight={best['class_weight']}, C={best['C']}")
    print(
        f"Test with best setting: accuracy {best_test['accuracy']}, "
        f"macro-F1 {best_test['macro_f1']}"
    )


if __name__ == "__main__":
    main()