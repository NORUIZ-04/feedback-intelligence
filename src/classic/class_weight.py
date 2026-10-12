"""Class-weighting experiment: retrain the three TF-IDF models with balanced class weights.

Run from the repository root:  python -m src.classic.class_weight
Needs the unweighted results from logreg.py and svm_rf.py to be saved first.
"""

import json
import time
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.classic.evaluate import evaluate_predictions, load_split, time_predict
from src.data.loader import get_label_names, load_raw
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.results import save_metrics
from src.utils.seed import set_seed

logger = get_logger(__name__)

NAMES = {
    "logreg": "LogReg",
    "linear_svm": "SVM",
    "random_forest": "RF",
}


def load_unweighted(name: str, results_dir: str) -> dict:
    path = Path(results_dir) / f"baseline_{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


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
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=seed
        ),
    }

    before, after = {}, {}
    for name, model in models.items():
        logger.info("Training %s with balanced class weights", name)
        start = time.perf_counter()
        model.fit(Xtr, y_train)
        train_seconds = round(time.perf_counter() - start, 4)
        metrics = {
            "model": f"{name}_balanced",
            "features": int(Xtr.shape[1]),
            "train_seconds": train_seconds,
            "validation": evaluate_predictions(y_val, model.predict(Xva), labels),
            "test": evaluate_predictions(y_test, model.predict(Xte), labels),
            "inference": time_predict(model.predict, Xte),
        }
        save_metrics(f"baseline_{name}_balanced", metrics, results_dir=results_dir)
        after[name] = metrics
        before[name] = load_unweighted(name, results_dir)

    print()
    print("Effect of class_weight='balanced' (test split)")
    print("| Model | Setting | Accuracy | Macro-F1 | Love recall | Surprise recall | Train s |")
    print("|---|---|---|---|---|---|---|")
    for name in models:
        for setting, m in (("default", before[name]), ("balanced", after[name])):
            t = m["test"]
            love = t["per_class"]["love"]["recall"]
            surprise = t["per_class"]["surprise"]["recall"]
            print(
                f"| {NAMES[name]} | {setting} | {t['accuracy']} | {t['macro_f1']} | "
                f"{love} | {surprise} | {m['train_seconds']} |"
            )

    print()
    print("Per-class F1 (test)")
    header = ["Class"]
    for name in models:
        header += [NAMES[name], f"{NAMES[name]} bal"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for lab in labels:
        row = [lab]
        for name in models:
            row.append(str(before[name]["test"]["per_class"][lab]["f1"]))
            row.append(str(after[name]["test"]["per_class"][lab]["f1"]))
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()