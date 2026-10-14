"""Automatic answer scoring. Proxies, not human judgement."""

import re

ABSTAIN = "i could not find this in the documents"
STOP = {
    "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for", "on", "with",
    "that", "this", "it", "as", "be", "by", "at", "from", "you", "can", "your", "if",
}


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9_.]+", text.lower()) if w not in STOP]


def is_abstain(answer: str) -> bool:
    return ABSTAIN in " ".join(answer.lower().split())


def keyword_overlap(answer: str, gold: str) -> float:
    gold_words = set(words(gold))
    if not gold_words:
        return 0.0
    return len(gold_words & set(words(answer))) / len(gold_words)


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def score_answerable(answer: str, gold: str, chunks: list[dict], doc: str) -> dict:
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    valid = bool(cited) and all(1 <= n <= len(chunks) for n in cited)
    relevant = any(
        1 <= n <= len(chunks) and chunks[n - 1]["doc"] == doc and gold in chunks[n - 1]["text"]
        for n in cited
    )
    gold_nums = numbers(gold)
    return {
        "abstained": is_abstain(answer),
        "overlap": round(keyword_overlap(answer, gold), 3),
        "correct": keyword_overlap(answer, gold) >= 0.6 and not is_abstain(answer),
        "number_ok": None if not gold_nums else gold_nums <= numbers(answer),
        "has_citation": bool(cited),
        "citation_valid": valid,
        "cited_relevant": relevant,
        "retrieval_hit": any(c["doc"] == doc and gold in c["text"] for c in chunks),
    }


def score_unanswerable(answer: str) -> dict:
    return {"abstained": is_abstain(answer), "hallucinated": not is_abstain(answer)}