"""Compare human labels with the automatic scorer on the 20 sampled Phi answers.

Run from the repository root:  python -m src.rag.compare_labels
"""

import csv
import json
from collections import Counter
from pathlib import Path

from src.rag.scoring import is_abstain
from src.utils.results import save_metrics

ANSWERABLE = {"correct", "partial", "wrong"}
UNANSWERABLE = {"refused", "mixed", "answered"}


def read_labels(path: Path) -> list[dict]:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    first = text.splitlines()[0]
    delimiter = ";" if first.count(";") > first.count(",") else ","
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter))


def main() -> None:
    labels = read_labels(Path("results/human_labels.csv"))
    saved = json.loads(Path("results/rag_v2_phi35_4bit.json").read_text(encoding="utf-8"))["metrics"]["rows"]
    by_id = {r["id"]: r for r in saved}

    bad = []
    for row in labels:
        lab = row["label"].strip().lower()
        allowed = ANSWERABLE if row["kind"] == "answerable" else UNANSWERABLE
        if lab not in allowed:
            bad.append((row["id"], row["label"]))
    print(f"{len(labels)} rows read")
    if bad:
        print("Invalid or empty labels (fix these in the CSV, then rerun):")
        for qid, lab in bad:
            print(f"  {qid}: {lab!r}")
        return

    ans_pairs, una_pairs, disagreements = Counter(), Counter(), []
    for row in labels:
        r = by_id[row["id"]]
        human = row["label"].strip().lower()
        abstained = is_abstain(r["answer"])
        if row["kind"] == "answerable":
            scorer = "abstained" if abstained else ("correct" if r["overlap"] >= 0.6 else "not correct")
            ans_pairs[(scorer, human)] += 1
            agree = (scorer == "correct") == (human == "correct")
        else:
            scorer = "refused" if abstained else "answered"
            una_pairs[(scorer, human)] += 1
            agree = (scorer == "refused") == (human == "refused")
        if not agree:
            disagreements.append((row, scorer, human, r))

    print("\nAnswerable: scorer verdict (rows) vs your label (columns)")
    print("| Scorer \\ You | correct | partial | wrong |")
    print("|---|---|---|---|")
    for s in ("correct", "not correct", "abstained"):
        print(f"| {s} | " + " | ".join(str(ans_pairs[(s, h)]) for h in ("correct", "partial", "wrong")) + " |")

    print("\nUnanswerable: scorer verdict vs your label")
    print("| Scorer \\ You | refused | mixed | answered |")
    print("|---|---|---|---|")
    for s in ("refused", "answered"):
        print(f"| {s} | " + " | ".join(str(una_pairs[(s, h)]) for h in ("refused", "mixed", "answered")) + " |")

    na = sum(ans_pairs.values())
    nu = sum(una_pairs.values())
    agree_a = sum(c for (s, h), c in ans_pairs.items() if (s == "correct") == (h == "correct"))
    agree_u = sum(c for (s, h), c in una_pairs.items() if (s == "refused") == (h == "refused"))
    print(f"\nAgreement (strict: partial counts as not correct, mixed as not refused): "
          f"answerable {agree_a}/{na}, unanswerable {agree_u}/{nu}")
    print(f"Your labels: answerable correct {sum(c for (s, h), c in ans_pairs.items() if h == 'correct')}"
          f", partial {sum(c for (s, h), c in ans_pairs.items() if h == 'partial')}"
          f", wrong {sum(c for (s, h), c in ans_pairs.items() if h == 'wrong')}")

    print("\n--- Disagreements, full text ---")
    for row, scorer, human, r in disagreements:
        print(f"\n[{row['id']}] scorer: {scorer} (overlap {r.get('overlap', 'n/a')}) | you: {human}")
        print(f"Q: {row['question']}")
        if row["gold_phrase"]:
            print(f"Gold: {row['gold_phrase']}")
        print(f"A: {row['answer']}")

    save_metrics("human_label_comparison", {
        "n": len(labels), "answerable_pairs": {f"{s}|{h}": c for (s, h), c in ans_pairs.items()},
        "unanswerable_pairs": {f"{s}|{h}": c for (s, h), c in una_pairs.items()},
        "disagreement_ids": [d[0]["id"] for d in disagreements]})


if __name__ == "__main__":
    main()