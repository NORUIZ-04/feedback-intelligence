"""Check that every gold phrase exists in its document (whitespace-normalized).

Run from the repository root:  python -m src.rag.verify_questions
"""

import sys
from pathlib import Path

import yaml

CLEAN = Path("data/docs_clean")


def normalize(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    cfg = yaml.safe_load(Path("data/eval/questions.yaml").read_text(encoding="utf-8"))
    docs = {p.stem: normalize(p.read_text(encoding="utf-8")) for p in CLEAN.glob("*.md")}
    problems = 0
    print("| ID | Document | Occurrences in doc | Also in other docs |")
    print("|---|---|---|---|")
    for q in cfg["questions"]:
        phrase = normalize(q["gold"])
        count = docs.get(q["doc"], "").count(phrase)
        others = [n for n, t in docs.items() if n != q["doc"] and phrase in t]
        print(f"| {q['id']} | {q['doc']} | {count} | {', '.join(others) or '-'} |")
        if count == 0:
            problems += 1
    print()
    print(f"{len(cfg['questions'])} questions, {problems} with a missing gold phrase")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()