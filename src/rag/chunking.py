"""Split cleaned documents into overlapping character chunks that break at spaces."""

from pathlib import Path

CLEAN = Path("data/docs_clean")


def normalize(text: str) -> str:
    return " ".join(text.split())


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = normalize(text)
    n = len(text)
    chunks = []
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        space = text.find(" ", nxt)
        start = space + 1 if space != -1 else n
    return chunks


def build_chunks(size: int, overlap: int) -> list[dict]:
    out = []
    for path in sorted(CLEAN.glob("*.md")):
        pieces = chunk_text(path.read_text(encoding="utf-8"), size, overlap)
        for i, piece in enumerate(pieces):
            out.append({"id": f"{path.stem}-{i}", "doc": path.stem, "text": piece})
    return out