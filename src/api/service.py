"""Loads the classifier, retriever, reranker and generator once and exposes two calls."""

import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from transformers import pipeline

from src.rag.evaluate_prompts import build
from src.rag.generate import Generator
from src.rag.index_store import INDEX_DIR, StoredRetriever, build_index
from src.rag.scoring import is_abstain

RERANKER = "BAAI/bge-reranker-base"
CANDIDATES = 15
TOP_K = 4
MAX_LENGTH = 64
LOCAL_CLASSIFIER = Path("models/transformer_distilbert")
HUB_CLASSIFIER = "BHARATHKUMAR0409/emotion-distilbert"


def gpu_gb() -> float:
    return round(torch.cuda.memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else 0.0


class Services:
    def __init__(self) -> None:
        self.generator_name = os.environ.get("RAG_MODEL", "microsoft/Phi-3.5-mini-instruct")
        self.four_bit = os.environ.get("RAG_4BIT", "1") == "1"
        self.classifier_name = os.environ.get(
            "CLASSIFIER_MODEL", str(LOCAL_CLASSIFIER) if LOCAL_CLASSIFIER.exists() else HUB_CLASSIFIER
        )
        self.lock = threading.Lock()
        self.classifier = None
        self.retriever = None
        self.reranker = None
        self.generator = None
        self.load_report: dict[str, dict] = {}

    def _timed(self, name: str, fn):
        before, start = gpu_gb(), time.perf_counter()
        result = fn()
        self.load_report[name] = {
            "seconds": round(time.perf_counter() - start, 2),
            "gpu_gb_added": round(gpu_gb() - before, 2),
        }
        return result

    def load(self) -> None:
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = self._timed("classifier", lambda: pipeline(
            "text-classification", model=self.classifier_name, top_k=None, device=device,
            truncation=True, max_length=MAX_LENGTH))
        if not (INDEX_DIR / "meta.json").exists():
            build_index()
        self.retriever = self._timed("retriever", StoredRetriever)
        self.reranker = self._timed("reranker", lambda: CrossEncoder(RERANKER, device="cuda"))
        self.generator = self._timed(
            "generator", lambda: Generator(self.generator_name, four_bit=self.four_bit))
        self.reranker.predict([("warm up", "warm up")])

    def classify(self, texts: list[str]) -> list[dict]:
        with self.lock:
            outputs = self.classifier(texts, batch_size=32)
        results = []
        for out in outputs:
            ranked = sorted(out, key=lambda d: d["score"], reverse=True)
            results.append({
                "label": ranked[0]["label"],
                "score": round(ranked[0]["score"], 4),
                "scores": {d["label"]: round(d["score"], 4) for d in ranked},
            })
        return results

    def ask(self, question: str) -> dict:
        with self.lock:
            t0 = time.perf_counter()
            candidates = self.retriever.search(question, CANDIDATES)
            t1 = time.perf_counter()
            scores = self.reranker.predict([(question, c["text"]) for c in candidates])
            t2 = time.perf_counter()
            order = np.argsort(-scores)[:TOP_K]
            top = [candidates[i] for i in order]
            answer, gen_seconds = self.generator.answer(build(question, top))
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(top)})
        return {
            "answer": answer,
            "refused": is_abstain(answer),
            "cited": cited,
            "sources": [
                {"n": i + 1, "id": c["id"], "doc": c["doc"], "rerank_score": round(float(scores[j]), 4),
                 "text": c["text"]}
                for i, (c, j) in enumerate(zip(top, order))
            ],
            "timings": {
                "retrieve_ms": round((t1 - t0) * 1000, 1),
                "rerank_ms": round((t2 - t1) * 1000, 1),
                "generate_s": round(gen_seconds, 2),
            },
        }

    def status(self) -> dict:
        return {
            "classifier": self.classifier_name,
            "generator": self.generator_name,
            "four_bit": self.four_bit,
            "reranker": RERANKER,
            "index_chunks": self.retriever.meta["chunks"] if self.retriever else None,
            "load_report": self.load_report,
            "gpu_gb_now": gpu_gb(),
        }