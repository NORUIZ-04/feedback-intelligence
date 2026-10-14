"""Check that no check_terms of an unanswerable question occur in the corpus.

Run from the repository root:  python -m src.rag.verify_unanswerable
"""

import re
from pathlib import Path

import yaml

CLEAN = Path("data/docs_clean")


def main() -> None:
    qs = yaml.safe_load(Path("data/eval/unanswerable.yaml").read_text(encoding="utf-8"))["questions"]
    docs = {p.stem: p.read_text(encoding="utf-8").lower() for p in CLEAN.glob("*.md")}
    flagged = 0
    print("| ID | Term found in corpus | Documents |")
    print("|---|---|---|")
    for q in qs:
        hits = {
            t: [n for n, text in docs.items() if re.search(r"(?<![a-z0-9])" + re.escape(t.lower()) + r"(?![a-z0-9])", text)] for t in q["check_terms"]
        }
        found = {t: d for t, d in hits.items() if d}
        if found:
            flagged += 1
        print(f"| {q['id']} | {', '.join(found) or '-'} | {'; '.join(','.join(d) for d in found.values()) or '-'} |")
    print(f"\n{len(qs)} questions, {flagged} with a term present in the corpus")


if __name__ == "__main__":
    main()