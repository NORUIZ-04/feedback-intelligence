"""Chunk the corpus at several sizes and measure gold-phrase coverage.

Run from the repository root:  python -m src.rag.coverage
"""

from pathlib import Path

import yaml

from src.rag.chunking import build_chunks, normalize
from src.utils.results import save_metrics

SIZES = [200, 500, 1000]
OVERLAP_FRACTION = 0.15


def main() -> None:
    questions = yaml.safe_load(
        Path("data/eval/questions.yaml").read_text(encoding="utf-8")
    )["questions"]
    ids = [q["id"] for q in questions]
    print(f"Unique question ids: {len(set(ids))} of {len(ids)}")

    report = {}
    print()
    print("| Chunk size | Overlap | Chunks | Mean chars | Gold covered | Uncovered |")
    print("|---|---|---|---|---|---|")
    for size in SIZES:
        overlap = int(size * OVERLAP_FRACTION)
        chunks = build_chunks(size, overlap)
        by_doc: dict[str, list[str]] = {}
        for c in chunks:
            by_doc.setdefault(c["doc"], []).append(c["text"])
        uncovered = [
            q["id"]
            for q in questions
            if not any(normalize(q["gold"]) in t for t in by_doc.get(q["doc"], []))
        ]
        mean_chars = round(sum(len(c["text"]) for c in chunks) / len(chunks), 1)
        covered = len(questions) - len(uncovered)
        report[str(size)] = {
            "overlap": overlap,
            "chunks": len(chunks),
            "mean_chars": mean_chars,
            "gold_covered": covered,
            "uncovered": uncovered,
        }
        print(
            f"| {size} | {overlap} | {len(chunks)} | {mean_chars} | "
            f"{covered}/{len(questions)} | {', '.join(uncovered) or '-'} |"
        )
    save_metrics("chunk_coverage", report)


if __name__ == "__main__":
    main()