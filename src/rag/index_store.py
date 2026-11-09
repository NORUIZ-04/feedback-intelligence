"""Build the retrieval index once, save it to disk, and load it quickly."""

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.rag.chunking import CLEAN, build_chunks
from src.rag.compare_retrievers import BGE_QUERY_PREFIX, fuse

INDEX_DIR = Path("models/rag_index")
EMBEDDER = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 500


def corpus_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(CLEAN.glob("*.md")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_index(size: int = CHUNK_SIZE, device: str = "cuda", out: Path = INDEX_DIR) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    chunks = build_chunks(size, int(size * 0.15))
    texts = [c["text"] for c in chunks]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    matrix = tfidf.fit_transform(texts)
    embedder = SentenceTransformer(EMBEDDER, device=device)
    emb = embedder.encode(texts, batch_size=64, normalize_embeddings=True).astype("float32")
    (out / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    joblib.dump(tfidf, out / "tfidf.joblib")
    sparse.save_npz(out / "tfidf_matrix.npz", matrix)
    np.save(out / "embeddings.npy", emb)
    meta = {
        "chunk_size": size,
        "chunks": len(chunks),
        "embedder": EMBEDDER,
        "corpus_sha256": corpus_hash(),
        "build_seconds": round(time.perf_counter() - start, 2),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class StoredRetriever:
    """Same search() as HybridRetriever, but loads a saved index instead of rebuilding."""

    def __init__(self, index_dir: Path = INDEX_DIR, device: str = "cuda", check_corpus: bool = True):
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        if check_corpus and meta["corpus_sha256"] != corpus_hash():
            raise RuntimeError("The saved index does not match the current corpus. Rebuild it.")
        self.meta = meta
        self.chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.tfidf = joblib.load(index_dir / "tfidf.joblib")
        self.matrix = sparse.load_npz(index_dir / "tfidf_matrix.npz")
        self.emb = np.load(index_dir / "embeddings.npy")
        self.embedder = SentenceTransformer(meta["embedder"], device=device)

    def search(self, query: str, k: int = 5) -> list[dict]:
        t = (self.matrix @ self.tfidf.transform([query]).T).toarray().ravel()
        q = self.embedder.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True)[0]
        fused = fuse(t, self.emb @ q)
        return [self.chunks[i] for i in np.argsort(-fused)[:k]]