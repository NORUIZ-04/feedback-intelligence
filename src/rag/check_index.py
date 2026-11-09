"""Build, save and reload the index; compare with the in-memory retriever.

Run from the repository root:  python -m src.rag.check_index
"""

import time
from pathlib import Path

import yaml

from src.rag.index_store import INDEX_DIR, StoredRetriever, build_index
from src.rag.retriever import HybridRetriever
from src.utils.results import save_metrics


def main() -> None:
    qs = [q["question"] for q in yaml.safe_load(
        Path("data/eval/questions.yaml").read_text(encoding="utf-8"))["questions"]]
    qs += [q["question"] for q in yaml.safe_load(
        Path("data/eval/unanswerable.yaml").read_text(encoding="utf-8"))["questions"]]

    meta = build_index()
    size_mb = sum(f.stat().st_size for f in INDEX_DIR.iterdir()) / 1024**2

    start = time.perf_counter()
    stored = StoredRetriever()
    stored_load = time.perf_counter() - start

    start = time.perf_counter()
    live = HybridRetriever(500)
    live_build = time.perf_counter() - start

    same_top1 = same_top5 = 0
    for q in qs:
        a = [c["id"] for c in stored.search(q, 5)]
        b = [c["id"] for c in live.search(q, 5)]
        same_top1 += a[0] == b[0]
        same_top5 += a == b

    start = time.perf_counter()
    for q in qs:
        stored.search(q, 15)
    ms = (time.perf_counter() - start) * 1000 / len(qs)

    result = {
        "chunks": meta["chunks"], "index_mb": round(size_mb, 2),
        "build_and_save_seconds": meta["build_seconds"],
        "stored_load_seconds": round(stored_load, 2),
        "live_build_seconds": round(live_build, 2),
        "queries": len(qs), "same_top1": same_top1, "same_top5": same_top5,
        "search_ms_per_query": round(ms, 2),
    }
    save_metrics("rag_index_check", result)
    print()
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()