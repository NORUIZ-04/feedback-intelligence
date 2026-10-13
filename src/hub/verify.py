"""Load the published model from the Hub and check it against saved results.

Run from the repository root:  python -m src.hub.verify
"""

import numpy as np
import pandas as pd
import torch
from transformers import pipeline

from src.hub.model_card import REPO_ID
from src.utils.config import load_config
from src.utils.results import save_metrics

EXAMPLES = [
    "i feel so helpless right now",
    "i am so happy that i got the job",
    "i cannot believe this just happened, i did not see that coming",
    "i am furious about how they treated me",
    "i am terrified of what comes next",
    "i feel so close to her and care about her deeply",
]


def main() -> None:
    cfg = load_config()
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline(
        "text-classification", model=REPO_ID, top_k=None, device=device,
        truncation=True, max_length=cfg["training"]["max_length"],
    )
    labels = [clf.model.config.id2label[i] for i in range(clf.model.config.num_labels)]

    test = pd.read_parquet("data/processed/test.parquet")
    outputs = clf(test["text"].tolist(), batch_size=64)
    preds = np.array(
        [labels.index(max(o, key=lambda d: d["score"])["label"]) for o in outputs]
    )
    accuracy = round(float((preds == test["label"].to_numpy()).mean()), 4)
    saved = 0.9265

    print()
    print(f"Model loaded from the Hub: {REPO_ID}")
    print(f"Labels: {labels}")
    print(f"Test accuracy via Hub pipeline: {accuracy} (saved local result: {saved})")
    consistent = abs(accuracy - saved) <= 0.001
    print(f"Consistent (within 0.001, fp32 vs fp16): {consistent}")
    print()
    print("Example predictions (top 2 labels)")
    results = []
    for text, out in zip(EXAMPLES, clf(EXAMPLES)):
        top = sorted(out, key=lambda d: d["score"], reverse=True)[:2]
        line = ", ".join(f"{d['label']} {d['score']:.3f}" for d in top)
        print(f"  {text} -> {line}")
        results.append({"text": text, "top2": [{"label": d["label"], "score": round(d["score"], 4)} for d in top]})

    save_metrics(
        "hub_verification",
        {"repo": REPO_ID, "hub_test_accuracy": accuracy, "saved_test_accuracy": saved,
         "consistent_within_0_001": abs(accuracy - saved) <= 0.001, "examples": results},
        results_dir=cfg["paths"]["results_dir"],
    )


if __name__ == "__main__":
    main()