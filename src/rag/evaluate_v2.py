"""Reranked pipeline with the v2 prompt on question set v2, split by question group.

Run from the repository root:
  python -m src.rag.evaluate_v2 --tag 1p5b
  python -m src.rag.evaluate_v2 --tag other --model <hub model name>
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sentence_transformers import CrossEncoder

from src.rag.chunking import normalize
from src.rag.evaluate_prompts import build
from src.rag.evaluate_rerank import CANDIDATES, RERANKER, K
from src.rag.generate import Generator
from src.rag.retriever import HybridRetriever
from src.rag.scoring import score_answerable, score_unanswerable
from src.utils.results import save_metrics


def answerable_group(q: dict) -> str:
    if "paraphrase_of" not in q:
        return "original"
    return "paraphrase_formal" if int(q["id"][1:]) <= 56 else "paraphrase_casual"


def count(rows: list[dict], key: str) -> int:
    return sum(1 for r in rows if r.get(key))


def mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="1p5b")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    args = parser.parse_args()

    answerable = yaml.safe_load(Path("data/eval/questions.yaml").read_text(encoding="utf-8"))["questions"]
    unanswerable = yaml.safe_load(Path("data/eval/unanswerable.yaml").read_text(encoding="utf-8"))["questions"]
    retriever = HybridRetriever(500)
    reranker = CrossEncoder(RERANKER, device="cuda")
    gen = Generator(args.model, four_bit=args.four_bit)
    reranker.predict([("warm up", "warm up")])

    rows = []
    for q in answerable + unanswerable:
        t0 = time.perf_counter()
        cands = retriever.search(q["question"], CANDIDATES)
        t1 = time.perf_counter()
        scores = reranker.predict([(q["question"], c["text"]) for c in cands])
        t2 = time.perf_counter()
        top = [cands[i] for i in np.argsort(-scores)[:K]]
        answer, gen_seconds = gen.answer(build(q["question"], top))
        row = {
            "id": q["id"],
            "question": q["question"],
            "answer": answer,
            "chunk_ids": [c["id"] for c in top],
            "top_score": float(np.max(scores)),
            "retrieve_ms": round((t1 - t0) * 1000, 1),
            "rerank_ms": round((t2 - t1) * 1000, 1),
            "generate_s": round(gen_seconds, 2),
        }
        if "gold" in q:
            row["group"] = answerable_group(q)
            row.update(score_answerable(answer, normalize(q["gold"]), top, q["doc"]))
        else:
            row["group"] = "unanswerable_" + q.get("group", "first_batch")
            row.update(score_unanswerable(answer))
        rows.append(row)
    print("generation done", flush=True)

    ans = [r for r in rows if not r["group"].startswith("unanswerable")]
    una = [r for r in rows if r["group"].startswith("unanswerable")]
    peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    summary = {
        "retrieve_ms": mean([r["retrieve_ms"] for r in rows]),
        "rerank_ms": mean([r["rerank_ms"] for r in rows]),
        "generate_s": mean([r["generate_s"] for r in rows]),
        "peak_gpu_gb": peak,
    }
    save_metrics(f"rag_v2_{args.tag}", {"model": args.model, "reranker": RERANKER, "k": K,
                                        "candidates": CANDIDATES, "summary": summary, "rows": rows})

    print(f"\nModel {args.model} | reranker {RERANKER} | top {K} of {CANDIDATES} | peak GPU {peak} GB")
    print("\nAnswerable questions (counts)")
    print("| Group | n | Gold in top 4 | Correct | Number check ok | Has citation | Cited the gold passage | Wrongly abstained |")
    print("|---|---|---|---|---|---|---|---|")
    groups = ["original", "paraphrase_formal", "paraphrase_casual"]
    for name in groups + ["all"]:
        sub = ans if name == "all" else [r for r in ans if r["group"] == name]
        nums = [r for r in sub if r["number_ok"] is not None]
        ok = sum(1 for r in nums if r["number_ok"])
        print(f"| {name} | {len(sub)} | {count(sub, 'retrieval_hit')} | {count(sub, 'correct')} | "
              f"{ok}/{len(nums)} | {count(sub, 'has_citation')} | {count(sub, 'cited_relevant')} | "
              f"{count(sub, 'abstained')} |")

    print("\nUnanswerable questions (counts)")
    print("| Group | n | Refused (good) | Answered anyway |")
    print("|---|---|---|---|")
    names = sorted({r["group"] for r in una})
    for name in names + ["all"]:
        sub = una if name == "all" else [r for r in una if r["group"] == name]
        print(f"| {name.replace('unanswerable_', '')} | {len(sub)} | {count(sub, 'abstained')} | "
              f"{count(sub, 'hallucinated')} |")

    print(f"\nMean per question: retrieve {summary['retrieve_ms']} ms, rerank {summary['rerank_ms']} ms, "
          f"generate {summary['generate_s']} s")
    print("\nControl questions (should be refused):")
    for r in una:
        if r["group"] == "unanswerable_control":
            print(f"  {r['id']} abstained={r['abstained']} | {r['answer'][:100]!r}")


if __name__ == "__main__":
    main()