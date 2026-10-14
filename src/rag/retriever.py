"""Hybrid retriever: TF-IDF + BGE-small fused with reciprocal rank fusion."""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.rag.chunking import build_chunks
from src.rag.compare_retrievers import BGE_QUERY_PREFIX, fuse


class HybridRetriever:
    def __init__(self, size: int = 500, device: str = "cuda"):
        self.chunks = build_chunks(size, int(size * 0.15))
        texts = [c["text"] for c in self.chunks]
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.tfidf.fit_transform(texts)
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
        self.emb = self.embedder.encode(texts, batch_size=64, normalize_embeddings=True)

    def search(self, query: str, k: int = 5) -> list[dict]:
        t = (self.matrix @ self.tfidf.transform([query]).T).toarray().ravel()
        q = self.embedder.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True)[0]
        fused = fuse(t, self.emb @ q)
        return [self.chunks[i] for i in np.argsort(-fused)[:k]]