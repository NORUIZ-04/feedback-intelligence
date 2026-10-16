from pathlib import Path

import numpy as np
import yaml

from src.rag.chunking import build_chunks, chunk_text, normalize
from src.rag.compare_retrievers import first_hit_rank, fuse, summarize


def test_chunk_text_respects_size_and_covers_every_word():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, 100, 15)
    assert all(len(c) <= 100 for c in chunks)
    assert set(" ".join(chunks).split()) == set(text.split())


def test_chunk_text_overlaps_between_neighbours():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, 100, 15)
    assert len(chunks) > 2
    assert set(chunks[0].split()) & set(chunks[1].split())


def test_first_hit_rank_needs_right_document_and_phrase():
    ranked = [
        {"doc": "a", "text": "hello world"},
        {"doc": "b", "text": "the gold phrase here"},
        {"doc": "a", "text": "the gold phrase here"},
    ]
    assert first_hit_rank(ranked, "a", "gold phrase") == 3
    assert first_hit_rank(ranked, "c", "gold phrase") is None
    assert first_hit_rank(ranked, "a", "missing") is None


def test_summarize_recall_budget_and_mrr():
    out = summarize([1, None, 3, 6], size=500)
    assert out["recall@1"] == 0.25
    assert out["recall@3"] == 0.5
    assert out["recall@5"] == 0.5
    assert out["recall@1000chars"] == 0.25
    assert out["recall@2500chars"] == 0.5
    assert out["mrr"] == 0.375


def test_fuse_prefers_items_ranked_high_by_both():
    a = np.array([0.9, 0.8, 0.1, 0.0])
    b = np.array([0.9, 0.1, 0.8, 0.0])
    fused = fuse(a, b)
    assert int(np.argmax(fused)) == 0
    assert fused[3] == fused.min()


def _questions():
    path = Path("data/eval/questions.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]


def test_question_ids_unique_and_gold_phrases_exist():
    questions = _questions()
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    docs = {
        p.stem: normalize(p.read_text(encoding="utf-8"))
        for p in Path("data/docs_clean").glob("*.md")
    }
    for q in questions:
        assert normalize(q["gold"]) in docs[q["doc"]], q["id"]


def test_every_gold_phrase_fits_inside_a_chunk_at_size_1000():
    by_doc: dict[str, list[str]] = {}
    for c in build_chunks(1000, 150):
        by_doc.setdefault(c["doc"], []).append(c["text"])
    for q in _questions():
        assert any(normalize(q["gold"]) in t for t in by_doc[q["doc"]]), q["id"]

def test_unanswerable_questions_well_formed():
    path = Path("data/eval/unanswerable.yaml")
    qs = yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids))
    assert all(q["question"] and q["check_terms"] for q in qs)
    answerable_ids = {q["id"] for q in _questions()}
    assert not set(ids) & answerable_ids


def test_paraphrases_point_at_existing_originals_with_same_gold():
    questions = {q["id"]: q for q in _questions()}
    for q in questions.values():
        if "paraphrase_of" in q:
            original = questions[q["paraphrase_of"]]
            assert original["gold"] == q["gold"] and original["doc"] == q["doc"], q["id"]


def test_split_halves_keeps_paraphrases_with_their_original():
    from src.rag.evaluate_rerank import split_halves

    answerable = [
        {"id": "a", "question": "q", "doc": "d", "gold": "g"},
        {"id": "b", "question": "q", "doc": "d", "gold": "g"},
        {"id": "a2", "paraphrase_of": "a", "question": "q", "doc": "d", "gold": "g"},
        {"id": "b2", "paraphrase_of": "b", "question": "q", "doc": "d", "gold": "g"},
    ]
    unanswerable = [{"id": f"u{i}", "question": "q", "check_terms": ["x"]} for i in range(4)]
    half = split_halves(answerable, unanswerable)
    assert half["a"] == half["a2"] and half["b"] == half["b2"]
    assert sorted(half[f"u{i}"] for i in range(4)).count("dev") == 2


def test_choose_threshold_separates_clean_groups():
    from src.rag.evaluate_rerank import choose_threshold

    rows = [
        {"kind": "original", "top_score": 0.9},
        {"kind": "original", "top_score": 0.8},
        {"kind": "unanswerable", "top_score": 0.1},
        {"kind": "unanswerable", "top_score": 0.2},
    ]
    thr = choose_threshold(rows)
    assert 0.2 < thr <= 0.8
