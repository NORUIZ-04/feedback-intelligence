"""Recall@k and MRR for a TF-IDF retriever at several chunk sizes.

Run from the repository root:  python -m src.rag.evaluate_retrieval
"""

import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from src.rag.chunking import build_chunks, normalize
from src.utils.results import save_metrics

SIZES = [200, 500, 1000]
KS = [1, 3, 5]
MAX_K = max(KS)


def first_hit_rank(ranked: list[dict], doc: str, gold: str) -> int | None:
    for rank, chunk in enumerate(ranked, 1):
        if chunk["doc"] == doc and gold in chunk["text"]:
            return rank
    return None


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 4) for k in KS}
    out["mrr"] = round(sum(1 / r for r in ranks if r) / n, 4)
    out["n"] = n
    return out


def main() -> None:
    questions = yaml.safe_load(
        Path("data/eval/questions.yaml").read_text(encoding="utf-8")
    )["questions"]
    groups = {
        "original": [q for q in questions if "paraphrase_of" not in q],
        "paraphrase": [q for q in questions if "paraphrase_of" in q],
        "all": questions,
    }
    report = {}
    rows = []
    for size in SIZES:
        chunks = build_chunks(size, int(size * 0.15))
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        matrix = vec.fit_transform([c["text"] for c in chunks])
        ranks = {}
        start = time.perf_counter()
        for q in questions:
            scores = (matrix @ vec.transform([q["question"]]).T).toarray().ravel()
            top = np.argsort(-scores)[:MAX_K]
            ranks[q["id"]] = first_hit_rank(
                [chunks[i] for i in top], q["doc"], normalize(q["gold"])
            )
        ms = round((time.perf_counter() - start) * 1000 / len(questions), 2)
        report[str(size)] = {"search_ms_per_query": ms, "chunks": len(chunks)}
        for name, qs in groups.items():
            s = summarize([ranks[q["id"]] for q in qs])
            report[str(size)][name] = s
            rows.append((size, name, s, ms))
        report[str(size)]["missed"] = [i for i, r in ranks.items() if r is None]

    save_metrics("rag_retrieval_tfidf", report)
    print()
    print("| Chunk | Group | n | Recall@1 | Recall@3 | Recall@5 | MRR | ms/query |")
    print("|---|---|---|---|---|---|---|---|")
    for size, name, s, ms in rows:
        print(
            f"| {size} | {name} | {s['n']} | {s['recall@1']} | {s['recall@3']} | "
            f"{s['recall@5']} | {s['mrr']} | {ms} |"
        )
    print()
    for size in SIZES:
        missed = report[str(size)]["missed"]
        print(f"Missed in top {MAX_K} at {size}: {len(missed)} -> {', '.join(missed)}")


if __name__ == "__main__":
    main()