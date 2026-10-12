"""TF-IDF + Linear SVM and TF-IDF + Random Forest baselines.

Run from the repository root:  python -m src.classic.svm_rf
"""

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.classic.evaluate import evaluate_predictions, load_split, time_predict
from src.data.loader import get_label_names, load_raw
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.results import save_metrics
from src.utils.seed import set_seed

logger = get_logger(__name__)


def run_model(name, model, Xtr, y_train, Xva, y_val, Xte, y_test, labels, results_dir):
    start = time.perf_counter()
    model.fit(Xtr, y_train)
    train_seconds = round(time.perf_counter() - start, 4)
    metrics = {
        "model": name,
        "features": int(Xtr.shape[1]),
        "train_seconds": train_seconds,
        "validation": evaluate_predictions(y_val, model.predict(Xva), labels),
        "test": evaluate_predictions(y_test, model.predict(Xte), labels),
        "inference": time_predict(model.predict, Xte),
    }
    save_metrics(f"baseline_{name}", metrics, results_dir=results_dir)
    return metrics


def print_result(metrics):
    print()
    print(f"{metrics['model']} ({metrics['features']} features)")
    print("| Split | Accuracy | Macro-F1 | Weighted-F1 |")
    print("|---|---|---|---|")
    for split in ("validation", "test"):
        m = metrics[split]
        print(f"| {split} | {m['accuracy']} | {m['macro_f1']} | {m['weighted_f1']} |")
    print("Per-class (test)")
    print("| Class | Precision | Recall | F1 | Support |")
    print("|---|---|---|---|---|")
    for lab, c in metrics["test"]["per_class"].items():
        print(f"| {lab} | {c['precision']} | {c['recall']} | {c['f1']} | {c['support']} |")
    print(
        f"Train: {metrics['train_seconds']}s | "
        f"Inference: {metrics['inference']['predict_ms_per_example']} ms/example"
    )


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    results_dir = cfg["paths"]["results_dir"]
    labels = get_label_names(load_raw(cfg["dataset"]["name"]))

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    X_test, y_test = load_split("test")

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform(X_train)
    Xva, Xte = vec.transform(X_val), vec.transform(X_test)

    models = {
        "linear_svm": LinearSVC(random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
    }
    for name, model in models.items():
        logger.info("Training %s", name)
        print_result(
            run_model(name, model, Xtr, y_train, Xva, y_val, Xte, y_test, labels, results_dir)
        )


if __name__ == "__main__":
    main()