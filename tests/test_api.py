import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.service import Services


class FakeServices:
    def __init__(self):
        self.seen = []

    def classify(self, texts):
        self.seen.append(texts)
        return [{"label": "joy", "score": 0.9, "scores": {"joy": 0.9, "fear": 0.1}} for _ in texts]

    def ask(self, question):
        return {
            "answer": "The default batch size is 1000 [1].",
            "refused": False,
            "cited": [1],
            "sources": [{"n": 1, "id": "d-1", "doc": "d", "rerank_score": 0.9, "text": "size is 1000"}],
            "timings": {"retrieve_ms": 1.0, "rerank_ms": 2.0, "generate_s": 0.5},
        }

    def status(self):
        return {"generator": "fake", "four_bit": False, "index_chunks": 3, "gpu_gb_now": 0.0}


@pytest.fixture()
def client():
    app.state.services = FakeServices()
    with TestClient(app) as c:
        yield c
    del app.state.services


def test_health_and_status(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/status").json()["generator"] == "fake"


def test_classify_returns_label_and_scores_and_timing_header(client):
    r = client.post("/classify", json={"texts": ["i am happy", "so scared"]})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 2 and results[0]["label"] == "joy"
    assert float(r.headers["x-process-time-ms"]) >= 0


def test_classify_collapses_whitespace_before_the_model(client):
    client.post("/classify", json={"texts": ["  too   much \n space  "]})
    assert app.state.services.seen[-1] == ["too much space"]


@pytest.mark.parametrize("payload", [
    {"texts": []},
    {"texts": ["   "]},
    {"texts": ["x" * 1001]},
    {"texts": ["ok"] * 33},
    {},
])
def test_classify_rejects_bad_input(client, payload):
    assert client.post("/classify", json=payload).status_code == 422


@pytest.mark.parametrize("payload", [{"question": "hi"}, {"question": "q" * 501}, {}])
def test_ask_rejects_bad_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_response_has_answer_sources_and_timings(client):
    body = client.post("/ask", json={"question": "What is the default batch size?"}).json()
    assert body["answer"] and body["refused"] is False and body["cited"] == [1]
    assert body["sources"][0]["text"] == "size is 1000"
    assert set(body["timings"]) == {"retrieve_ms", "rerank_ms", "generate_s"}


class FakeRetriever:
    def __init__(self):
        self.meta = {"chunks": 5}

    def search(self, question, k):
        return [{"id": f"c{i}", "doc": "d", "text": f"text {i}"} for i in range(5)]


class FakeReranker:
    def predict(self, pairs):
        return np.array([0.1, 0.9, 0.5, 0.2, 0.05])


class FakeGenerator:
    def __init__(self, answer):
        self.answer_text = answer

    def answer(self, messages):
        return self.answer_text, 0.01


def make_services(answer):
    svc = Services()
    svc.retriever, svc.reranker, svc.generator = FakeRetriever(), FakeReranker(), FakeGenerator(answer)
    return svc


def test_ask_orders_sources_by_rerank_score_and_keeps_four():
    result = make_services("Answer [2].").ask("some question")
    assert [s["id"] for s in result["sources"]] == ["c1", "c2", "c3", "c0"]
    assert [s["rerank_score"] for s in result["sources"]] == [0.9, 0.5, 0.2, 0.1]


def test_ask_cited_only_counts_valid_passage_numbers():
    result = make_services("It is so [1] [2] and also [9] [0].").ask("some question")
    assert result["cited"] == [1, 2]


def test_ask_sets_the_refused_flag():
    refused = make_services("I could not find this in the documents.").ask("some question")
    answered = make_services("Use the flag [1].").ask("some question")
    assert refused["refused"] is True and answered["refused"] is False