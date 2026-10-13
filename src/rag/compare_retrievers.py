"""Compare TF-IDF, embedding and hybrid retrievers across chunk sizes.

Run from the repository root:  python -m src.rag.compare_retrievers
"""

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.rag.chunking import build_chunks, normalize
from src.utils.logging_utils import get_logger
from src.utils.results import save_metrics

try:
    import faiss
except ImportError:
    faiss = None

plt.switch_backend("Agg")
logger = get_logger(__name__)

SIZES = [200, 500, 1000]
KS = [1, 3, 5]
BUDGETS = [1000, 2500]
MAX_K = 15
OVERLAP_FRACTION = 0.15
MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "BGE-small": "BAAI/bge-small-en-v1.5",
}
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
HYBRID_PARTNER = "BGE-small"
HYBRID = "Hybrid (TF-IDF + BGE-small)"
RETRIEVERS = ["TF-IDF", "MiniLM", "BGE-small", HYBRID]
RRF_K = 60


def first_hit_rank(ranked: list[dict], doc: str, gold: str) -> int | None:
    for rank, chunk in enumerate(ranked, 1):
        if chunk["doc"] == doc and gold in chunk["text"]:
            return rank
    return None


def summarize(ranks: list[int | None], size: int) -> dict:
    n = len(ranks)
    out = {"n": n}
    for k in KS:
        out[f"recall@{k}"] = round(sum(1 for r in ranks if r and r <= k) / n, 4)
    for budget in BUDGETS:
        k = math.ceil(budget / size)
        out[f"recall@{budget}chars"] = round(sum(1 for r in ranks if r and r <= k) / n, 4)
    out["mrr"] = round(sum(1 / r for r in ranks if r) / n, 4)
    return out


def fuse(scores_a: np.ndarray, scores_b: np.ndarray, k: int = RRF_K) -> np.ndarray:
    """Reciprocal rank fusion of two score vectors over the same items."""
    fused = np.zeros(len(scores_a))
    for scores in (scores_a, scores_b):
        order = np.argsort(-scores)
        ranks = np.empty(len(scores), dtype=int)
        ranks[order] = np.arange(1, len(scores) + 1)
        fused += 1.0 / (k + ranks)
    return fused


def faiss_check(chunk_emb: np.ndarray, q_emb: np.ndarray, k: int = 5) -> dict | None:
    if faiss is None:
        return None
    chunk_emb = np.ascontiguousarray(chunk_emb, dtype="float32")
    q_emb = np.ascontiguousarray(q_emb, dtype="float32")
    index = faiss.IndexFlatIP(chunk_emb.shape[1])
    index.add(chunk_emb)
    start = time.perf_counter()
    _, ids = index.search(q_emb, k)
    ms = (time.perf_counter() - start) * 1000 / len(q_emb)
    exact = np.argsort(-(q_emb @ chunk_emb.T), axis=1)[:, :k]
    return {
        "vectors": int(index.ntotal),
        "faiss_ms_per_query": round(ms, 4),
        "top5_identical_share": round(float((ids == exact).all(axis=1).mean()), 4),
    }


def index_rows(rows: list[dict]) -> dict:
    return {(r["size"], r["retriever"], r["group"]): r for r in rows}


def fmt(value, digits: int = 2) -> str:
    return "n/a" if value is None else str(round(value, digits))


def render_tables(rows: list[dict]) -> str:
    look = index_rows(rows)
    lines = [
        "### Retrieval comparison (56 questions: 44 original, 12 paraphrase)",
        "",
        "| Chunk | Retriever | R@1 | R@3 | R@5 | MRR | Orig R@5 | Para R@5 | Para MRR | ms/query | Embed s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for size in SIZES:
        for retr in RETRIEVERS:
            a = look[(size, retr, "all")]
            o = look[(size, retr, "original")]
            p = look[(size, retr, "paraphrase")]
            lines.append(
                f"| {size} | {retr} | {a['recall@1']} | {a['recall@3']} | {a['recall@5']} | "
                f"{a['mrr']} | {o['recall@5']} | {p['recall@5']} | {p['mrr']} | "
                f"{fmt(a['search_ms'])} | {fmt(a['embed_seconds'])} |"
            )
    lines += [
        "",
        (
            "**Equal context budget** (Recall over all 56 questions when the retriever may return "
            "at most this many characters, so chunk sizes compare fairly):"
        ),
        "",
        "| Retriever | Budget | 200-char chunks | 500-char chunks | 1000-char chunks |",
        "|---|---|---|---|---|",
    ]
    for retr in RETRIEVERS:
        for budget in BUDGETS:
            vals = [look[(s, retr, "all")][f"recall@{budget}chars"] for s in SIZES]
            lines.append(f"| {retr} | {budget} chars | {vals[0]} | {vals[1]} | {vals[2]} |")
    note = (
        "Search is exact (NumPy dot product, or TF-IDF cosine). Hybrid is reciprocal rank "
        "fusion of TF-IDF and BGE-small. Hybrid ms/query is the sum of its two parts."
    )
    lines += ["", note, "", "![Retrieval comparison](results/plots/rag_retrieval_mrr.png)", ""]
    return "\n".join(lines)


def render_findings(rows: list[dict], faiss_info: dict | None) -> str:
    look = index_rows(rows)
    all_rows = [r for r in rows if r["group"] == "all"]
    para_rows = [r for r in rows if r["group"] == "paraphrase"]
    best = max(all_rows, key=lambda r: r["mrr"])
    best_p = max(para_rows, key=lambda r: r["mrr"])
    lines = [
        "### Findings (computed from the tables, not written by hand)",
        "",
        (
            f"- Best overall MRR: {best['retriever']} at {best['size']} characters "
            f"(MRR {best['mrr']}, Recall@5 {best['recall@5']})."
        ),
        (
            f"- Best paraphrase MRR: {best_p['retriever']} at {best_p['size']} characters "
            f"(MRR {best_p['mrr']}, Recall@5 {best_p['recall@5']})."
        ),
        (
            "- Paraphrase Recall@5, TF-IDF vs the better embedding model at the same chunk size "
            "(n = 12, so one question is 0.083):"
        ),
    ]
    for size in SIZES:
        base = look[(size, "TF-IDF", "paraphrase")]["recall@5"]
        name = max(("MiniLM", "BGE-small"), key=lambda r: look[(size, r, "paraphrase")]["recall@5"])
        val = look[(size, name, "paraphrase")]["recall@5"]
        verdict = "higher" if val > base else "not higher"
        lines.append(f"  - {size} chars: TF-IDF {base}, {name} {val} ({verdict})")
    lines.append("- Overall MRR, hybrid vs BGE-small alone:")
    for size in SIZES:
        h = look[(size, HYBRID, "all")]["mrr"]
        b = look[(size, "BGE-small", "all")]["mrr"]
        lines.append(f"  - {size} chars: hybrid {h}, BGE-small {b}")
    if faiss_info is None:
        lines.append("- FAISS: not installed in this environment; all numbers use exact NumPy search.")
    else:
        lines.append(
            f"- FAISS (IndexFlatIP, BGE-small, 500-char chunks, {faiss_info['vectors']} vectors): "
            f"{faiss_info['faiss_ms_per_query']} ms/query, top-5 identical to NumPy search for "
            f"{faiss_info['top5_identical_share']} of queries."
        )
    return "\n".join(lines) + "\n"


def plot_results(rows: list[dict], out_path: Path) -> None:
    look = index_rows(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("all", "mrr", "MRR, all 56 questions"),
        ("paraphrase", "recall@5", "Recall@5, 12 paraphrase questions"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.8 / len(RETRIEVERS)
    x = np.arange(len(SIZES))
    for ax, (group, metric, title) in zip(axes, panels):
        for k, retr in enumerate(RETRIEVERS):
            vals = [look[(s, retr, group)][metric] for s in SIZES]
            ax.bar(x + (k - (len(RETRIEVERS) - 1) / 2) * width, vals, width, label=retr)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s} chars" for s in SIZES])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    questions = yaml.safe_load(
        Path("data/eval/questions.yaml").read_text(encoding="utf-8")
    )["questions"]
    texts = [q["question"] for q in questions]
    groups = {
        "all": questions,
        "original": [q for q in questions if "paraphrase_of" not in q],
        "paraphrase": [q for q in questions if "paraphrase_of" in q],
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {label: SentenceTransformer(name, device=device) for label, name in MODELS.items()}
    for model in models.values():
        model.encode(["warm up"], show_progress_bar=False)

    rows = []
    faiss_info = None
    for size in SIZES:
        chunks = build_chunks(size, int(size * OVERLAP_FRACTION))
        chunk_texts = [c["text"] for c in chunks]
        scores, timing = {}, {}

        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        matrix = vec.fit_transform(chunk_texts)
        start = time.perf_counter()
        scores["TF-IDF"] = (vec.transform(texts) @ matrix.T).toarray()
        timing["TF-IDF"] = (None, (time.perf_counter() - start) * 1000 / len(texts))

        for label, model in models.items():
            start = time.perf_counter()
            chunk_emb = model.encode(
                chunk_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
            )
            embed_seconds = time.perf_counter() - start
            prefix = BGE_QUERY_PREFIX if label == "BGE-small" else ""
            start = time.perf_counter()
            q_emb = model.encode(
                [prefix + t for t in texts],
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            scores[label] = q_emb @ chunk_emb.T
            timing[label] = (embed_seconds, (time.perf_counter() - start) * 1000 / len(texts))
            if label == HYBRID_PARTNER and size == 500:
                faiss_info = faiss_check(chunk_emb, q_emb)

        scores[HYBRID] = np.vstack(
            [fuse(scores["TF-IDF"][i], scores[HYBRID_PARTNER][i]) for i in range(len(texts))]
        )
        timing[HYBRID] = (
            timing[HYBRID_PARTNER][0],
            timing["TF-IDF"][1] + timing[HYBRID_PARTNER][1],
        )

        for retr in RETRIEVERS:
            ranks = {}
            for i, q in enumerate(questions):
                top = np.argsort(-scores[retr][i])[:MAX_K]
                ranks[q["id"]] = first_hit_rank(
                    [chunks[j] for j in top], q["doc"], normalize(q["gold"])
                )
            for group, qs in groups.items():
                summary = summarize([ranks[q["id"]] for q in qs], size)
                rows.append(
                    {
                        "size": size,
                        "retriever": retr,
                        "group": group,
                        **summary,
                        "chunks": len(chunks),
                        "embed_seconds": timing[retr][0],
                        "search_ms": timing[retr][1],
                        "missed_top15": [i for i, r in ranks.items() if r is None]
                        if group == "all"
                        else None,
                    }
                )
        logger.info("Finished chunk size %s", size)

    tables = render_tables(rows)
    findings = render_findings(rows, faiss_info)
    Path("results/rag_retrieval_comparison.md").write_text(tables, encoding="utf-8")
    Path("results/rag_findings.md").write_text(findings, encoding="utf-8")
    plot_results(rows, Path("results/plots/rag_retrieval_mrr.png"))
    save_metrics(
        "rag_retrieval_comparison",
        {"device": device, "models": MODELS, "rows": rows, "faiss": faiss_info},
    )
    print()
    print(tables)
    print(findings)


if __name__ == "__main__":
    main()