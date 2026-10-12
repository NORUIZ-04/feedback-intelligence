"""Majority-class baseline: always predict the most common training label.

Run from the repository root:  python -m src.classic.majority
"""

import time

from sklearn.dummy import DummyClassifier

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

    model = DummyClassifier(strategy="most_frequent")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = round(time.perf_counter() - start, 4)

    metrics = {
        "model": "majority_class",
        "predicted_label": labels[int(model.predict(["x"])[0])],
        "train_seconds": train_seconds,
        "validation": evaluate_predictions(y_val, model.predict(X_val), labels),
        "test": evaluate_predictions(y_test, model.predict(X_test), labels),
        "inference": time_predict(model.predict, X_test),
    }
    out = save_metrics("baseline_majority", metrics, results_dir=cfg["paths"]["results_dir"])
    logger.info("Saved metrics to %s", out)

    print()
    print("Majority-class baseline (always predicts:", metrics["predicted_label"] + ")")
    print("| Split | Accuracy | Macro-F1 | Weighted-F1 |")
    print("|---|---|---|---|")
    for split in ("validation", "test"):
        m = metrics[split]
        print(f"| {split} | {m['accuracy']} | {m['macro_f1']} | {m['weighted_f1']} |")


if __name__ == "__main__":
    main()