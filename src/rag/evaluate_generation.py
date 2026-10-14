"""Evaluate answer generation: baseline prompt vs grounded prompt.

Run from the repository root:  python -m src.rag.evaluate_generation
"""

import random
import sys
from pathlib import Path

import torch
import yaml

from src.rag.chunking import normalize
from src.rag.generate import Generator, build_messages
from src.rag.retriever import HybridRetriever
from src.rag.scoring import score_answerable, score_unanswerable
from src.utils.results import save_metrics

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
K = 4
BASELINE_SYSTEM = "Answer the question using the passages provided."


def messages_for(prompt: str, question: str, chunks: list[dict]) -> list[dict]:
    msgs = build_messages(question, chunks)
    if prompt == "baseline":
        msgs[0] = {"role": "system", "content": BASELINE_SYSTEM}
    return msgs


def rate(items: list[dict], key: str) -> float:
    vals = [i[key] for i in items if i[key] is not None]
    return round(sum(1 for v in vals if v) / len(vals), 4) if vals else 0.0


def main() -> None:
    answerable = yaml.safe_load(Path("data/eval/questions.yaml").read_text(encoding="utf-8"))["questions"]
    unanswerable = yaml.safe_load(Path("data/eval/unanswerable.yaml").read_text(encoding="utf-8"))["questions"]
    print("Building retriever...", flush=True)
    retriever = HybridRetriever(500)
    print("Loading generator...", flush=True)
    gen = Generator(MODEL)

    results = {}
    for prompt in ("baseline", "grounded"):
        rows, times = [], []
        for n, q in enumerate(answerable + unanswerable, 1):
            chunks = retriever.search(q["question"], K)
            answer, seconds = gen.answer(messages_for(prompt, q["question"], chunks))
            times.append(seconds)
            if "gold" in q:
                score = score_answerable(answer, normalize(q["gold"]), chunks, q["doc"])
                kind = "paraphrase" if "paraphrase_of" in q else "original"
            else:
                score = score_unanswerable(answer)
                kind = "unanswerable"
            rows.append({"id": q["id"], "kind": kind, "question": q["question"],
                         "answer": answer, "chunk_ids": [c["id"] for c in chunks], **score})
            if n % 17 == 0:
                print(f"  {prompt}: {n}/{len(answerable) + len(unanswerable)}", flush=True)
        results[prompt] = {"rows": rows, "mean_seconds": round(sum(times) / len(times), 2)}

    summary = {}
    for prompt, data in results.items():
        rows = data["rows"]
        ans = [r for r in rows if r["kind"] != "unanswerable"]
        una = [r for r in rows if r["kind"] == "unanswerable"]
        hit = [r for r in ans if r["retrieval_hit"]]
        summary[prompt] = {
            "answerable_n": len(ans),
            "correct": rate(ans, "correct"),
            "correct_when_retrieved": rate(hit, "correct"),
            "number_ok": rate(ans, "number_ok"),
            "abstained_on_answerable": rate(ans, "abstained"),
            "has_citation": rate(ans, "has_citation"),
            "citation_valid": rate(ans, "citation_valid"),
            "cited_relevant": rate(ans, "cited_relevant"),
            "retrieval_hit": rate(ans, "retrieval_hit"),
            "unanswerable_n": len(una),
            "abstained_correctly": rate(una, "abstained"),
            "hallucinated": rate(una, "hallucinated"),
            "mean_seconds": data["mean_seconds"],
        }
        for kind in ("original", "paraphrase"):
            sub = [r for r in ans if r["kind"] == kind]
            summary[prompt][f"correct_{kind}"] = rate(sub, "correct")

    peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    save_metrics("rag_generation_1p5b", {"model": MODEL, "k": K, "chunk_size": 500,
                                         "peak_gpu_gb": peak, "summary": summary, "results": results})

    labels = [
        ("Retrieval hit (gold in top 4)", "retrieval_hit"),
        ("Correct (keyword overlap >= 0.6)", "correct"),
        ("Correct, when retrieval hit", "correct_when_retrieved"),
        ("Correct, originals", "correct_original"),
        ("Correct, paraphrases", "correct_paraphrase"),
        ("Number check passed", "number_ok"),
        ("Abstained on answerable (wrongly)", "abstained_on_answerable"),
        ("Has citation", "has_citation"),
        ("Citation valid", "citation_valid"),
        ("Cited a chunk holding the gold", "cited_relevant"),
        ("Abstained on unanswerable (good)", "abstained_correctly"),
        ("Hallucinated on unanswerable", "hallucinated"),
        ("Mean seconds per answer", "mean_seconds"),
    ]
    print()
    print(f"Model {MODEL}, hybrid retriever, 500-char chunks, top {K}, peak GPU {peak} GB")
    print("| Metric | Baseline prompt | Grounded prompt |")
    print("|---|---|---|")
    for name, key in labels:
        print(f"| {name} | {summary['baseline'][key]} | {summary['grounded'][key]} |")

    random.seed(42)
    sample = random.sample(range(len(results["grounded"]["rows"])), 10)
    print("\n--- 10 random grounded answers for you to check by eye ---")
    for i in sample:
        r = results["grounded"]["rows"][i]
        print(f"\n[{r['id']} | {r['kind']}] {r['question']}")
        print(f"Answer: {r['answer'][:300]}")
        keys = ("correct", "overlap", "citation_valid", "cited_relevant", "abstained", "hallucinated")
        print("Score:", {k: r[k] for k in keys if k in r})


if __name__ == "__main__":
    sys.exit(main())