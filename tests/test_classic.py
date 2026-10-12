import json

import pytest

from src.classic.compare import build_rows, fmt
from src.classic.evaluate import evaluate_predictions, time_predict


def test_evaluate_predictions_known_values():
    out = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b"])
    assert out["accuracy"] == 0.75
    assert out["per_class"]["a"]["precision"] == 1.0
    assert out["per_class"]["a"]["recall"] == 0.5
    assert out["per_class"]["b"]["precision"] == pytest.approx(0.6667, abs=1e-4)
    assert out["per_class"]["b"]["recall"] == 1.0
    assert out["macro_f1"] == pytest.approx(0.7333, abs=1e-4)


def test_time_predict_returns_positive_timings():
    def predict(x):
        return [0] * len(x)

    out = time_predict(predict, list(range(100)), repeats=2)
    assert out["predict_total_ms"] >= 0
    assert out["predict_ms_per_example"] >= 0


def test_fmt_handles_missing_values():
    assert fmt(None) == "n/a"
    assert fmt(1.23456, 2) == 1.23 or fmt(1.23456, 2) == "1.23"


def _fake_metrics(acc, macro):
    per_class = {c: {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 10}
                 for c in ("love", "surprise")}
    test = {"accuracy": acc, "macro_f1": macro, "weighted_f1": macro, "per_class": per_class}
    return {"train_seconds": 1.0, "inference": {"predict_ms_per_example": 0.1}, "test": test}


def test_build_rows_sorts_by_macro_f1(tmp_path):
    stems = [
        "baseline_majority", "baseline_logreg", "baseline_logreg_balanced",
        "baseline_linear_svm", "baseline_linear_svm_balanced",
        "baseline_random_forest", "baseline_random_forest_balanced",
    ]
    for i, stem in enumerate(stems):
        record = {"experiment": stem, "metrics": _fake_metrics(0.5, 0.1 * (i + 1))}
        (tmp_path / f"{stem}.json").write_text(json.dumps(record), encoding="utf-8")
    sweep = {
        "best": {"class_weight": "balanced", "C": 100, "train_seconds": 2.0},
        "best_test": _fake_metrics(0.9, 0.95)["test"],
    }
    (tmp_path / "baseline_logreg_sweep.json").write_text(
        json.dumps({"experiment": "sweep", "metrics": sweep}), encoding="utf-8"
    )
    rows = build_rows(str(tmp_path))
    assert len(rows) == 8
    assert rows[0]["macro_f1"] == 0.95
    assert rows[0]["inference_ms"] is None
    assert [r["macro_f1"] for r in rows] == sorted(
        (r["macro_f1"] for r in rows), reverse=True
    )