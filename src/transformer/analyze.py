"""Final model comparison, confusion matrices and error analysis.

Run from the repository root:  python -m src.transformer.analyze
Needs the classic results, compare.py output, both train.py runs and models/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.classic.compare import build_rows, fmt, read_metrics
from src.data.loader import get_label_names, load_raw
from src.transformer.data import TextDataset, collate_fn, load_frame
from src.transformer.train import predict
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.results import save_metrics

plt.switch_backend("Agg")
logger = get_logger(__name__)

RUNS = {
    "transformer_distilbert": "DistilBERT (unweighted)",
    "transformer_distilbert_balanced": "DistilBERT (class-weighted)",
}


def transformer_row(label: str, m: dict) -> dict:
    test = m["test"]
    per_class = test["per_class"]
    return {
        "model": label,
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "weighted_f1": test["weighted_f1"],
        "love_f1": per_class["love"]["f1"],
        "surprise_f1": per_class["surprise"]["f1"],
        "train_seconds": m["train_seconds"],
        "inference_ms": m["inference"]["predict_ms_per_example"],
        "hardware": "GPU, batch 64, full model",
    }


def predict_saved(model_dir: Path, frame, max_length: int, device, use_amp: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    dataset = TextDataset(
        frame["text"].tolist(), frame["label"].tolist(), tokenizer, max_length
    )
    loader = DataLoader(
        dataset, batch_size=64, collate_fn=collate_fn(tokenizer.pad_token_id)
    )
    return predict(model, loader, device, use_amp)


def top_confusions(cm: np.ndarray, labels: list[str], k: int = 5) -> list[dict]:
    pairs = [
        {"true": labels[i], "predicted": labels[j], "count": int(cm[i, j])}
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j
    ]
    return sorted(pairs, key=lambda p: p["count"], reverse=True)[:k]


def render_comparison(rows: list[dict]) -> str:
    lines = [
        "### Final model comparison (test split)",
        "",
        "| Model | Accuracy | Macro-F1 | Love F1 | Surprise F1 | Train s | Inference ms/example | Inference setup |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['accuracy']} | {r['macro_f1']} | {r['love_f1']} | "
            f"{r['surprise_f1']} | {fmt(r['train_seconds'], 2)} | {fmt(r['inference_ms'])} | "
            f"{r['hardware']} |"
        )
    note = (
        "Rows are sorted by test macro-F1. Classic models train on CPU and use the "
        "same TF-IDF features; DistilBERT trains on an RTX 4050 laptop GPU with fp16. "
        "Inference times are not comparable across the two groups: classic timings "
        "exclude TF-IDF vectorizing, transformer timings are GPU forward passes."
    )
    lines += ["", note, ""]
    return "\n".join(lines)


def plot_confusions(matrices: dict, labels: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(matrices), figsize=(7 * len(matrices), 6))
    for ax, (title, cm) in zip(axes, matrices.items()):
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        threshold = cm.max() / 2
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if cm[i, j] > threshold else "black"
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class(series: dict, labels: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width = 0.8 / len(series)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, (name, values) in enumerate(series.items()):
        offset = (k - (len(series) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 (test split)")
    ax.set_title("Per-class F1: best classic model vs DistilBERT")
    ax.legend(loc="upper center", ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    t = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(t["fp16"]) and device.type == "cuda"
    labels = get_label_names(load_raw(cfg["dataset"]["name"]))
    test_df = load_frame("test")
    y_true = np.asarray(test_df["label"])

    rows = build_rows(str(results_dir))
    for row in rows:
        row["hardware"] = (
            "n/a" if row["inference_ms"] is None else "CPU, pre-vectorized features"
        )

    preds, reproduced = {}, {}
    for stem, label in RUNS.items():
        saved = read_metrics(str(results_dir), stem)
        rows.append(transformer_row(label, saved))
        preds[stem] = predict_saved(models_dir / stem, test_df, t["max_length"], device, use_amp)
        acc = round(float((preds[stem] == y_true).mean()), 4)
        reproduced[stem] = {
            "saved_accuracy": saved["test"]["accuracy"],
            "recomputed_accuracy": acc,
            "match": acc == saved["test"]["accuracy"],
        }
    rows = sorted(rows, key=lambda r: r["macro_f1"], reverse=True)

    label_ids = list(range(len(labels)))
    matrices = {
        RUNS[stem]: confusion_matrix(y_true, p, labels=label_ids) for stem, p in preds.items()
    }
    confusions = {name: top_confusions(cm, labels) for name, cm in matrices.items()}
    plot_confusions(matrices, labels, results_dir / "plots" / "confusion_matrices.png")

    svm = read_metrics(str(results_dir), "baseline_linear_svm_balanced")
    series = {"Linear SVM (balanced)": [svm["test"]["per_class"][lab]["f1"] for lab in labels]}
    for stem, label in RUNS.items():
        m = read_metrics(str(results_dir), stem)
        series[label] = [m["test"]["per_class"][lab]["f1"] for lab in labels]
    plot_per_class(series, labels, results_dir / "plots" / "per_class_f1.png")

    unweighted = preds["transformer_distilbert"]
    errors = np.where(unweighted != y_true)[0]
    rng = np.random.default_rng(42)
    chosen = sorted(rng.choice(errors, size=min(15, len(errors)), replace=False).tolist())
    summary = (
        f"Total errors: {len(errors)} of {len(y_true)}. "
        "The 15 below are a random sample (seed 42), not hand-picked."
    )
    error_lines = [
        "### Misclassified test examples (DistilBERT unweighted)",
        "",
        summary,
        "",
        "| # | Text | True | Predicted |",
        "|---|---|---|---|",
    ]
    for n, idx in enumerate(chosen, 1):
        text = test_df["text"].iloc[idx].replace("|", "\\|")
        error_lines.append(
            f"| {n} | {text} | {labels[y_true[idx]]} | {labels[unweighted[idx]]} |"
        )
    error_md = "\n".join(error_lines) + "\n"

    comparison_md = render_comparison(rows)
    comparison_md += "\n![Confusion matrices](results/plots/confusion_matrices.png)\n"
    comparison_md += "\n![Per-class F1](results/plots/per_class_f1.png)\n"
    (results_dir / "final_comparison.md").write_text(comparison_md, encoding="utf-8")
    (results_dir / "error_examples.md").write_text(error_md, encoding="utf-8")
    save_metrics(
        "model_comparison",
        {
            "rows": rows,
            "reproduced": reproduced,
            "confusion_matrices": {k: v.tolist() for k, v in matrices.items()},
            "top_confusions": confusions,
            "error_count_unweighted": len(errors),
        },
        results_dir=str(results_dir),
    )

    print()
    print(comparison_md)
    print("Reloaded-model check (saved vs recomputed test accuracy)")
    for stem, r in reproduced.items():
        print(f"  {stem}: {r['saved_accuracy']} vs {r['recomputed_accuracy']} -> match {r['match']}")
    print()
    for name, pairs in confusions.items():
        print(f"Top confusions, {name} (true -> predicted: count)")
        for p in pairs:
            print(f"  {p['true']} -> {p['predicted']}: {p['count']}")
    print()
    print(error_md)


if __name__ == "__main__":
    main()