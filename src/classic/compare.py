"""Build the baseline comparison table and chart from saved result files.

Run from the repository root:  python -m src.classic.compare
Needs the result files written by majority.py, logreg.py, svm_rf.py,
class_weight.py and logreg_sweep.py.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.results import save_metrics

plt.switch_backend("Agg")
logger = get_logger(__name__)

STANDARD = [
    ("Majority class", "baseline_majority"),
    ("LogReg (default)", "baseline_logreg"),
    ("LogReg (balanced)", "baseline_logreg_balanced"),
    ("Linear SVM (default)", "baseline_linear_svm"),
    ("Linear SVM (balanced)", "baseline_linear_svm_balanced"),
    ("Random Forest (default)", "baseline_random_forest"),
    ("Random Forest (balanced)", "baseline_random_forest_balanced"),
]


def read_metrics(results_dir: str, stem: str) -> dict:
    path = Path(results_dir) / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the earlier baseline scripts first.")
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def make_row(name: str, test: dict, train_seconds, inference_ms) -> dict:
    per_class = test["per_class"]
    return {
        "model": name,
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "weighted_f1": test["weighted_f1"],
        "love_f1": per_class["love"]["f1"],
        "surprise_f1": per_class["surprise"]["f1"],
        "train_seconds": train_seconds,
        "inference_ms": inference_ms,
    }


def build_rows(results_dir: str) -> list[dict]:
    rows = []
    for name, stem in STANDARD:
        m = read_metrics(results_dir, stem)
        rows.append(
            make_row(
                name,
                m["test"],
                m["train_seconds"],
                m["inference"]["predict_ms_per_example"],
            )
        )
    sweep = read_metrics(results_dir, "baseline_logreg_sweep")
    best = sweep["best"]
    label = f"LogReg (tuned: {best['class_weight']}, C={best['C']})"
    rows.append(make_row(label, sweep["best_test"], best["train_seconds"], None))
    return sorted(rows, key=lambda r: r["macro_f1"], reverse=True)


def fmt(value, digits: int = 4) -> str:
    return "n/a" if value is None else str(round(value, digits))


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "### Baseline comparison (test split)",
        "",
        "| Model | Accuracy | Macro-F1 | Love F1 | Surprise F1 | Train s | Inference ms/example |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['accuracy']} | {r['macro_f1']} | {r['love_f1']} | "
            f"{r['surprise_f1']} | {fmt(r['train_seconds'], 2)} | {fmt(r['inference_ms'])} |"
        )
    note = (
        "All models use the same TF-IDF features (1-2 word n-grams, 50,000 features, "
        "fitted on the train split only). Rows are sorted by test macro-F1. "
        "Inference time is not recorded for the tuned LogReg run."
    )
    lines += [
        "",
        note,
        "",
        "![Baseline comparison](results/plots/baseline_comparison.png)",
        "",
    ]
    return "\n".join(lines)


def plot_comparison(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = list(reversed(rows))
    positions = list(range(len(ordered)))
    height = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [p + height / 2 for p in positions],
        [r["accuracy"] for r in ordered],
        height=height,
        label="Accuracy",
        color="#4C72B0",
    )
    ax.barh(
        [p - height / 2 for p in positions],
        [r["macro_f1"] for r in ordered],
        height=height,
        label="Macro-F1",
        color="#55A868",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels([r["model"] for r in ordered])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score (test split)")
    ax.set_title("Classic ML baselines on dair-ai/emotion")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    results_dir = cfg["paths"]["results_dir"]
    rows = build_rows(results_dir)

    markdown = render_markdown(rows)
    Path(results_dir, "baseline_comparison.md").write_text(markdown, encoding="utf-8")
    plot_comparison(rows, Path(results_dir) / "plots" / "baseline_comparison.png")
    save_metrics("baseline_comparison", {"rows": rows}, results_dir=results_dir)
    logger.info("Saved comparison table and chart to %s", results_dir)

    print()
    print(markdown)


if __name__ == "__main__":
    main()