"""How stable is the score gate? Repeated random splits over the saved reranker scores.

Run from the repository root:  python -m src.rag.analyze_gate
Needs results/rag_rerank_gate.json from evaluate_rerank.py.
"""

import json
import random
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from src.rag.evaluate_rerank import choose_threshold
from src.utils.results import save_metrics

SEEDS = 200


def make_halves(answerable: list[dict], unanswerable: list[dict], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    originals = [q for q in answerable if "paraphrase_of" not in q]
    rng.shuffle(originals)
    dev = {q["id"] for q in originals[: len(originals) // 2]}
    half = {q["id"]: ("dev" if q["id"] in dev else "test") for q in originals}
    for q in answerable:
        if "paraphrase_of" in q:
            half[q["id"]] = half[q["paraphrase_of"]]
    shuffled = list(unanswerable)
    rng.shuffle(shuffled)
    for i, q in enumerate(shuffled):
        half[q["id"]] = "dev" if i < len(shuffled) // 2 else "test"
    return half


def auc(rows: list[dict], positive_kinds: set[str]) -> float:
    sel = [r for r in rows if r["kind"] in positive_kinds or r["kind"] == "unanswerable"]
    labels = [1 if r["kind"] in positive_kinds else 0 for r in sel]
    return float(roc_auc_score(labels, [r["top_score"] for r in sel]))


def spread(values: list[float]) -> str:
    a = np.array(values)
    return f"mean {a.mean():.3f}, 5th-95th pct {np.percentile(a, 5):.3f} to {np.percentile(a, 95):.3f}"


def main() -> None:
    saved = json.loads(Path("results/rag_rerank_gate.json").read_text(encoding="utf-8"))["metrics"]
    rows = saved["rows"]
    answerable = yaml.safe_load(Path("data/eval/questions.yaml").read_text(encoding="utf-8"))["questions"]
    unanswerable = yaml.safe_load(Path("data/eval/unanswerable.yaml").read_text(encoding="utf-8"))["questions"]

    kept_t, refused_t, thresholds = [], [], []
    for seed in range(SEEDS):
        half = make_halves(answerable, unanswerable, seed)
        dev = [r for r in rows if half[r["id"]] == "dev"]
        test = [r for r in rows if half[r["id"]] == "test"]
        thr = choose_threshold(dev)
        a = [r for r in test if r["kind"] != "unanswerable"]
        u = [r for r in test if r["kind"] == "unanswerable"]
        thresholds.append(thr)
        kept_t.append(sum(1 for r in a if r["top_score"] >= thr) / len(a))
        refused_t.append(sum(1 for r in u if r["top_score"] < thr) / len(u))

    print(f"\n{SEEDS} random dev/test splits, threshold chosen on dev, measured on test")
    print("Threshold chosen:               ", spread(thresholds))
    print("Answerable kept on test:        ", spread(kept_t))
    print("Unanswerable refused on test:   ", spread(refused_t))

    print("\nHow well does the top reranker score separate answerable from unanswerable? (AUC, 0.5 = chance)")
    result = {
        "originals vs unanswerable": auc(rows, {"original"}),
        "paraphrases vs unanswerable": auc(rows, {"paraphrase"}),
        "all answerable vs unanswerable": auc(rows, {"original", "paraphrase"}),
    }
    for name, value in result.items():
        print(f"  {name}: {value:.3f}")

    print("\nMedian top reranker score by group:")
    for kind in ("original", "paraphrase", "unanswerable"):
        vals = [r["top_score"] for r in rows if r["kind"] == kind]
        print(f"  {kind} (n={len(vals)}): {np.median(vals):.3f}")

    save_metrics("rag_gate_stability", {
        "seeds": SEEDS, "kept_test": kept_t, "refused_test": refused_t,
        "thresholds": thresholds, "auc": result})


if __name__ == "__main__":
    main()