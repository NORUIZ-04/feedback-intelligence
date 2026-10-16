"""Write a blind labelling sheet of 20 sampled Phi-3.5-mini answers.

Run from the repository root:  python -m src.rag.make_label_sheet
"""

import csv
import json
import random
from pathlib import Path

import yaml

SOURCE = Path("results/rag_v2_phi35_4bit.json")
OUT = Path("results/human_labels.csv")
PLAN = {"original": 8, "paraphrase_formal": 2, "paraphrase_casual": 2, "unanswerable_on_topic": 4,
        "unanswerable_first_batch": 2, "unanswerable_adjacent": 2}


def main() -> None:
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))["metrics"]["rows"]
    gold = {q["id"]: q["gold"] for q in yaml.safe_load(
        Path("data/eval/questions.yaml").read_text(encoding="utf-8"))["questions"]}
    rng = random.Random(42)
    chosen = []
    for group, n in PLAN.items():
        pool = [r for r in rows if r["group"] == group]
        chosen += rng.sample(pool, n)
    rng.shuffle(chosen)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "question", "gold_phrase", "answer", "label"])
        for r in chosen:
            kind = "unanswerable" if r["group"].startswith("unanswerable") else "answerable"
            w.writerow([r["id"], kind, r["question"], gold.get(r["id"], ""), r["answer"], ""])
    print(f"wrote {len(chosen)} rows to {OUT}")


if __name__ == "__main__":
    main()