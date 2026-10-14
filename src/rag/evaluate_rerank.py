"""Reranker plus a score-threshold refusal gate, tuned on a dev half and reported on both halves.

Run from the repository root:  python -m src.rag.evaluate_rerank
Needs results/rag_prompt_variants_1p5b.json from evaluate_prompts.py.
"""

import json
import random
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import CrossEncoder

from src.rag.chunking import normalize
from src.rag.evaluate_prompts import MODEL, build
from src.rag.generate import Generator
from src.rag.retriever import HybridRetriever
from src.rag.scoring import score_answerable, score_unanswerable
from src.utils.results import save_metrics

RERANKER = "BAAI/bge-reranker-base"
CANDIDATES = 15
K = 4
ABSTAIN_TEXT = "I could not find this in the documents."


def split_halves(answerable: list[dict], unanswerable: list[dict]) -> dict[str, str]:
    rng = random.Random(42)
    originals = [q for q in answerable if "paraphrase_of" not in q]
    rng.shuffle(originals)
    dev_orig = {q["id"] for q in originals[: len(originals) // 2]}
    half = {q["id"]: ("dev" if q["id"] in dev_orig else "test") for q in originals}
    for q in answerable:
        if "paraphrase_of" in q:
            half[q["id"]] = half[q["paraphrase_of"]]
    shuffled = list(unanswerable)
    rng.shuffle(shuffled)
    for i, q in enumerate(shuffled):
        half[q["id"]] = "dev" if i < len(shuffled) // 2 else "test"
    return half


def choose_threshold(dev_rows: list[dict]) -> float:
    """Pick the cutoff that best separates answerable from unanswerable on dev only."""
    best_thr, best_score = None, -1.0
    ans = [r for r in dev_rows if r["kind"] != "unanswerable"]
    una = [r for r in dev_rows if r["kind"] == "unanswerable"]
    for thr in sorted({r["top_score"] for r in dev_rows}):
        kept = sum(1 for r in ans if r["top_score"] >= thr) / len(ans)
        refused = sum(1 for r in una if r["top_score"] < thr) / len(una)
        balanced = (kept + refused) / 2
        if balanced > best_score:
            best_thr, best_score = thr, balanced
    return float(best_thr)


def counts(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["kind"] != "unanswerable"]
    una = [r for r in rows if r["kind"] == "unanswerable"]
    return {
        "answerable": len(ans),
        "retrieval_hit": sum(1 for r in ans if r["retrieval_hit"]),
        "correct": sum(1 for r in ans if r["correct"]),
        "wrong_abstentions": sum(1 for r in ans if r["abstained"]),
        "cited_relevant": sum(1 for r in ans if r["cited_relevant"]),
        "unanswerable": len(una),
        "refused": sum(1 for r in una if r["abstained"]),
        "hallucinated": sum(1 for r in una if r["hallucinated"]),
    }


def cell(c: dict, key: str, total_key: str) -> str:
    return f"{c[key]}/{c[total_key]}"


def main() -> None:
    answerable = yaml.safe_load(Path("data/eval/questions.yaml").read_text(encoding="utf-8"))["questions"]
    unanswerable = yaml.safe_load(Path("data/eval/unanswerable.yaml").read_text(encoding="utf-8"))["questions"]
    saved = json.loads(Path("results/rag_prompt_variants_1p5b.json").read_text(encoding="utf-8"))["metrics"]
    half = split_halves(answerable, unanswerable)

    retriever = HybridRetriever(500)
    reranker = CrossEncoder(RERANKER, device="cuda")
    gen = Generator(MODEL)

    rows = []
    for q in answerable + unanswerable:
        cands = retriever.search(q["question"], CANDIDATES)
        scores = reranker.predict([(q["question"], c["text"]) for c in cands])
        order = np.argsort(-scores)
        top = [cands[i] for i in order[:K]]
        raw, _ = gen.answer(build(q["question"], top))
        row = {"id": q["id"], "half": half[q["id"]], "question": q["question"],
               "top_score": float(scores[order[0]]), "chunk_ids": [c["id"] for c in top]}
        if "gold" in q:
            gold = normalize(q["gold"])
            row["kind"] = "paraphrase" if "paraphrase_of" in q else "original"
            row["hit15"] = any(c["doc"] == q["doc"] and gold in c["text"] for c in cands)
            row["hit4_hybrid"] = any(c["doc"] == q["doc"] and gold in c["text"] for c in cands[:K])
            row["raw"] = {"answer": raw, **score_answerable(raw, gold, top, q["doc"])}
            row["gold"], row["doc"] = gold, q["doc"]
        else:
            row["kind"] = "unanswerable"
            row["raw"] = {"answer": raw, **score_unanswerable(raw)}
        row["top"] = top
        rows.append(row)
    print("generation done", flush=True)

    thr = choose_threshold([r for r in rows if r["half"] == "dev"])
    for r in rows:
        gated = r["top_score"] < thr
        ans = ABSTAIN_TEXT if gated else r["raw"]["answer"]
        if r["kind"] == "unanswerable":
            r["gated"] = {"answer": ans, **score_unanswerable(ans)}
        else:
            r["gated"] = {"answer": ans, **score_answerable(ans, r["gold"], r["top"], r["doc"])}

    saved_rows = {r["id"]: r for r in saved["rows"]["grounded_v2"]}
    variants = {
        "v2 hybrid top4 (saved)": lambda r: saved_rows[r["id"]],
        "v2 reranked top4": lambda r: r["raw"],
        "v2 reranked + gate": lambda r: r["gated"],
    }
    summary = {}
    print(f"\nReranker {RERANKER}; gate threshold chosen on dev only: {thr:.4f}")
    for half_name in ("dev", "test"):
        subset = [r for r in rows if r["half"] == half_name]
        a = [r for r in subset if r["kind"] != "unanswerable"]
        u = [r for r in subset if r["kind"] == "unanswerable"]
        kept = sum(1 for r in a if r["top_score"] >= thr)
        refused = sum(1 for r in u if r["top_score"] < thr)
        print(f"\n=== {half_name} half: {len(a)} answerable, {len(u)} unanswerable ===")
        print(f"Gate keeps {kept}/{len(a)} answerable and refuses {refused}/{len(u)} unanswerable")
        print(f"Gold in hybrid top 4: {sum(1 for r in a if r['hit4_hybrid'])}/{len(a)} | "
              f"gold in candidates top 15: {sum(1 for r in a if r['hit15'])}/{len(a)} | "
              f"gold in reranked top 4: {sum(1 for r in a if r['raw']['retrieval_hit'])}/{len(a)}")
        table = {}
        for name, getter in variants.items():
            table[name] = counts([{**getter(r), "kind": r["kind"]} for r in subset])
        summary[half_name] = {"gate_keeps": kept, "gate_refuses": refused, "table": table}
        names = list(variants)
        print("| Metric | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        spec = [("Correct", "correct", "answerable"),
                ("Wrong abstentions", "wrong_abstentions", "answerable"),
                ("Cited a chunk with the gold", "cited_relevant", "answerable"),
                ("Unanswerable refused", "refused", "unanswerable"),
                ("Unanswerable hallucinated", "hallucinated", "unanswerable")]
        for label, key, total in spec:
            print(f"| {label} | " + " | ".join(cell(table[n], key, total) for n in names) + " |")

    def avg(subset: list[dict]) -> str:
        return f"{np.mean([r['top_score'] for r in subset]):.3f} (n={len(subset)})" if subset else "n/a"

    print("\nMean top reranker score by group (all questions):")
    print("  answerable, gold in reranked top 4:", avg([r for r in rows if r["kind"] != "unanswerable" and r["raw"]["retrieval_hit"]]))
    print("  answerable, gold missed:           ", avg([r for r in rows if r["kind"] != "unanswerable" and not r["raw"]["retrieval_hit"]]))
    print("  unanswerable:                      ", avg([r for r in rows if r["kind"] == "unanswerable"]))

    slim = [{k: v for k, v in r.items() if k not in ("top",)} for r in rows]
    save_metrics("rag_rerank_gate", {"reranker": RERANKER, "threshold": thr, "k": K,
                                     "candidates": CANDIDATES, "summary": summary, "rows": slim})
    print("\nUnanswerable questions that still got an answer after gating (test and dev):")
    for r in rows:
        if r["kind"] == "unanswerable" and r["gated"]["hallucinated"]:
            print(f"  {r['id']} [{r['half']}] score {r['top_score']:.3f} | {r['gated']['answer'][:90]!r}")
    print("Answerable questions the gate refused:")
    for r in rows:
        if r["kind"] != "unanswerable" and r["gated"]["abstained"]:
            print(f"  {r['id']} [{r['half']}] score {r['top_score']:.3f} hit={r['raw']['retrieval_hit']}")


if __name__ == "__main__":
    main()