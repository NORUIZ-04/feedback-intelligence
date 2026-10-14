"""Prompt and top-k variants, compared with the saved baseline and strict-prompt results.

Run from the repository root:  python -m src.rag.evaluate_prompts
Needs results/rag_generation_1p5b.json from evaluate_generation.py.
"""

import json
from pathlib import Path

import torch
import yaml

from src.rag.chunking import normalize
from src.rag.evaluate_generation import MODEL, rate
from src.rag.generate import Generator
from src.rag.retriever import HybridRetriever
from src.rag.scoring import score_answerable, score_unanswerable
from src.utils.results import save_metrics

SYSTEM_V2 = (
    "You answer questions about Hugging Face documentation using the numbered passages "
    "provided. Answer directly and briefly whenever the passages support an answer, and "
    "end each sentence with the number of the passage it came from, like [2]. Only if "
    "none of the passages is relevant to the question, reply exactly: "
    "I could not find this in the documents."
)
EXAMPLE_USER = (
    "Passages:\n[1] (demo) The tool exposes a --verbose flag that prints extra details.\n\n"
    "Question: How do I get more output?\nAnswer:"
)
EXAMPLE_ASSISTANT = "Use the --verbose flag to print extra details [1]."

VARIANTS = {"grounded_v2": 4, "grounded_v2_k6": 6}


def build(question: str, chunks: list[dict]) -> list[dict]:
    passages = "\n\n".join(f"[{i}] ({c['doc']}) {c['text']}" for i, c in enumerate(chunks, 1))
    user = f"Passages:\n{passages}\n\nQuestion: {question}\nAnswer:"
    return [
        {"role": "system", "content": SYSTEM_V2},
        {"role": "user", "content": EXAMPLE_USER},
        {"role": "assistant", "content": EXAMPLE_ASSISTANT},
        {"role": "user", "content": user},
    ]


def summarize(rows: list[dict], seconds: float) -> dict:
    ans = [r for r in rows if r["kind"] != "unanswerable"]
    una = [r for r in rows if r["kind"] == "unanswerable"]
    hit = [r for r in ans if r["retrieval_hit"]]
    out = {
        "retrieval_hit": rate(ans, "retrieval_hit"),
        "correct": rate(ans, "correct"),
        "correct_when_retrieved": rate(hit, "correct"),
        "correct_original": rate([r for r in ans if r["kind"] == "original"], "correct"),
        "correct_paraphrase": rate([r for r in ans if r["kind"] == "paraphrase"], "correct"),
        "number_ok": rate(ans, "number_ok"),
        "abstained_on_answerable": rate(ans, "abstained"),
        "has_citation": rate(ans, "has_citation"),
        "citation_valid": rate(ans, "citation_valid"),
        "cited_relevant": rate(ans, "cited_relevant"),
        "abstained_correctly": rate(una, "abstained"),
        "hallucinated": rate(una, "hallucinated"),
        "mean_seconds": seconds,
    }
    wrong = [r for r in ans if r["abstained"]]
    out["wrong_abstentions"] = len(wrong)
    out["wrong_abstentions_with_retrieval_hit"] = sum(1 for r in wrong if r["retrieval_hit"])
    return out


def main() -> None:
    saved = json.loads(Path("results/rag_generation_1p5b.json").read_text(encoding="utf-8"))["metrics"]
    answerable = yaml.safe_load(Path("data/eval/questions.yaml").read_text(encoding="utf-8"))["questions"]
    unanswerable = yaml.safe_load(Path("data/eval/unanswerable.yaml").read_text(encoding="utf-8"))["questions"]
    retriever = HybridRetriever(500)
    gen = Generator(MODEL)

    summaries = {name: saved["summary"][name] for name in ("baseline", "grounded")}
    for name in list(summaries):
        rows = saved["results"][name]["rows"]
        extra = summarize(rows, saved["results"][name]["mean_seconds"])
        summaries[name].update(
            wrong_abstentions=extra["wrong_abstentions"],
            wrong_abstentions_with_retrieval_hit=extra["wrong_abstentions_with_retrieval_hit"],
        )
    all_rows = {}
    for name, k in VARIANTS.items():
        rows, times = [], []
        for q in answerable + unanswerable:
            chunks = retriever.search(q["question"], k)
            answer, seconds = gen.answer(build(q["question"], chunks))
            times.append(seconds)
            if "gold" in q:
                score = score_answerable(answer, normalize(q["gold"]), chunks, q["doc"])
                kind = "paraphrase" if "paraphrase_of" in q else "original"
            else:
                score = score_unanswerable(answer)
                kind = "unanswerable"
            rows.append({"id": q["id"], "kind": kind, "question": q["question"], "answer": answer,
                         "chunk_ids": [c["id"] for c in chunks], **score})
        all_rows[name] = rows
        summaries[name] = summarize(rows, round(sum(times) / len(times), 2))
        print(f"finished {name}", flush=True)

    save_metrics("rag_prompt_variants_1p5b",
                 {"model": MODEL, "summary": summaries, "rows": all_rows},)
    cols = ["baseline", "grounded", "grounded_v2", "grounded_v2_k6"]
    labels = [
        ("Retrieval hit (gold in top k)", "retrieval_hit"),
        ("Correct", "correct"),
        ("Correct, when retrieval hit", "correct_when_retrieved"),
        ("Correct, originals", "correct_original"),
        ("Correct, paraphrases", "correct_paraphrase"),
        ("Number check passed", "number_ok"),
        ("Wrong abstentions (of 56)", "wrong_abstentions"),
        ("  ...of those, retrieval had hit", "wrong_abstentions_with_retrieval_hit"),
        ("Has citation", "has_citation"),
        ("Citation valid", "citation_valid"),
        ("Cited a chunk holding the gold", "cited_relevant"),
        ("Abstained on unanswerable (good)", "abstained_correctly"),
        ("Hallucinated on unanswerable", "hallucinated"),
        ("Mean seconds per answer", "mean_seconds"),
    ]
    print()
    print(f"Model {MODEL}, hybrid retriever, 500-char chunks, peak GPU "
          f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("| Metric | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, key in labels:
        print(f"| {name} | " + " | ".join(str(summaries[c].get(key, "n/a")) for c in cols) + " |")

    print("\n--- Wrong abstentions in grounded_v2 (retrieval hit?) ---")
    for r in all_rows["grounded_v2"]:
        if r["kind"] != "unanswerable" and r["abstained"]:
            print(f"{r['id']} | retrieval_hit={r['retrieval_hit']} | {r['question']}")
    print("\n--- Unanswerable questions answered by grounded_v2 ---")
    for r in all_rows["grounded_v2"]:
        if r["kind"] == "unanswerable" and r["hallucinated"]:
            print(f"{r['id']} | {r['question']} | {r['answer'][:110]!r}")


if __name__ == "__main__":
    main()