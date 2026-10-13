"""Clean the downloaded markdown documents and report what changed.

Run from the repository root:  python -m src.rag.clean
"""

import json
import re
from collections import Counter
from pathlib import Path

from src.utils.results import save_metrics

RAW = Path("data/docs")
CLEAN = Path("data/docs_clean")


def clean_text(text: str) -> str:
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)  # front matter
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # html comments
    text = re.sub(r"^\s*(import|export)\s.*$", "", text, flags=re.MULTILINE)  # mdx imports
    text = re.sub(r"<[A-Za-z/][^>]*>", "", text, flags=re.DOTALL)  # html/jsx tags, multi-line
    text = re.sub(r"^\s*(class|href|style)=.*$", "", text, flags=re.MULTILINE)  # stray attribute lines
    text = re.sub(r"^```.*$", "", text, flags=re.MULTILINE)  # code fence lines
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links keep text
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def headings(text: str) -> list[str]:
    return re.findall(r"^#{1,4}\s+(.+)$", text, flags=re.MULTILINE)


def main() -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    report = {}
    for path in sorted(RAW.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        (CLEAN / path.name).write_text(cleaned, encoding="utf-8")
        lines = [ln.strip() for ln in cleaned.splitlines() if len(ln.strip()) > 20]
        common = Counter(lines).most_common(1)
        report[path.stem] = {
            "words_raw": len(raw.split()),
            "words_clean": len(cleaned.split()),
            "headings": len(headings(cleaned)),
            "most_repeated_line": common[0][0][:60] if common else "",
            "most_repeated_count": common[0][1] if common else 0,
        }

    save_metrics("rag_corpus_clean", report)
    print()
    print("| Document | Raw words | Clean words | Headings | Most repeated line (count) |")
    print("|---|---|---|---|---|")
    for name, r in report.items():
        print(
            f"| {name} | {r['words_raw']} | {r['words_clean']} | {r['headings']} | "
            f"{r['most_repeated_line']} ({r['most_repeated_count']}) |"
        )
    total = sum(r["words_clean"] for r in report.values())
    print(f"\nTotal clean words: {total}")
    print(json.dumps({"documents": len(report)}))


if __name__ == "__main__":
    main()