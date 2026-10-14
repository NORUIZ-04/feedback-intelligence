"""One grounded answer end to end. Not a result.

Run from the repository root:  python -m src.rag.smoke
"""

import torch

from src.rag.generate import Generator, build_messages
from src.rag.retriever import HybridRetriever

QUESTION = "What is the default batch size when using map with batched processing?"


def main() -> None:
    retriever = HybridRetriever(500)
    chunks = retriever.search(QUESTION, 4)
    gen = Generator()
    answer, seconds = gen.answer(build_messages(QUESTION, chunks))
    print()
    print("Question:", QUESTION)
    print("Retrieved:", [c["id"] for c in chunks])
    print("Answer:", answer)
    print(f"Generation: {seconds:.1f}s | peak GPU {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()