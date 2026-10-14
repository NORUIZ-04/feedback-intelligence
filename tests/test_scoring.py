from src.rag.scoring import (
    is_abstain,
    keyword_overlap,
    score_answerable,
    score_unanswerable,
)

CHUNKS = [
    {"doc": "d1", "text": "other text"},
    {"doc": "d2", "text": "the default batch size is 1000 rows"},
]


def test_abstain_detection_ignores_case_and_spacing():
    assert is_abstain("I could not  find this in the documents.")
    assert not is_abstain("The answer is 1000.")


def test_keyword_overlap_bounds():
    assert keyword_overlap("the default batch size is 1000", "default batch size is 1000") == 1.0
    assert keyword_overlap("something else entirely", "default batch size is 1000") == 0.0


def test_answerable_correct_cited_and_relevant():
    s = score_answerable("The default batch size is 1000 [2].", "default batch size is 1000", CHUNKS, "d2")
    assert s["correct"] and s["number_ok"] and s["citation_valid"] and s["cited_relevant"]
    assert s["retrieval_hit"]


def test_answerable_wrong_number_and_bad_citation():
    s = score_answerable("The default batch size is 500 [7].", "default batch size is 1000", CHUNKS, "d2")
    assert s["number_ok"] is False
    assert not s["citation_valid"]
    assert not s["cited_relevant"]


def test_abstaining_on_answerable_is_not_correct():
    s = score_answerable("I could not find this in the documents.", "default batch size is 1000", CHUNKS, "d2")
    assert s["abstained"] and not s["correct"]


def test_unanswerable_scoring():
    assert score_unanswerable("I could not find this in the documents.")["abstained"]
    assert score_unanswerable("Use LoRA with PEFT.")["hallucinated"]