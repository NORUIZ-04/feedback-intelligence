import json
from pathlib import Path

from src.hub.model_card import REPO_ID, build_card


def test_card_has_valid_front_matter_and_no_placeholders():
    card = build_card()
    assert card.startswith("---\n")
    assert card.count("\n---\n") >= 1
    assert "license: other" in card
    assert "educational and research purposes" in card
    assert "D18-1404" in card
    assert "dair-ai/emotion" in card
    assert REPO_ID in card
    assert "{t[" not in card and "{m[" not in card and "{wt[" not in card


def test_card_numbers_match_saved_results():
    saved = json.loads(
        Path("results/transformer_distilbert.json").read_text(encoding="utf-8")
    )["metrics"]["test"]
    card = build_card()
    assert f"value: {saved['accuracy']}" in card
    assert f"value: {saved['macro_f1']}" in card
    for label in ("sadness", "joy", "love", "anger", "fear", "surprise"):
        assert f"| {label} |" in card


def test_card_states_limitations():
    card = build_card()
    assert "## Limitations" in card
    assert "Single run" in card