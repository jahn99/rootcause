"""Tests for text chunking, rank fusion, and the ask endpoint."""

from dataclasses import dataclass
from datetime import date

from fastapi.testclient import TestClient

from app.api.ask import get_llm_provider, get_retriever
from app.ingestion.chunking import chunk_text
from app.main import app
from app.retrieval.search import reciprocal_rank_fusion


@dataclass
class Item:
    id: str


def test_chunking_preserves_text_with_overlap() -> None:
    text = "one two three four five six seven"
    chunks = chunk_text(text, chunk_size=15, overlap=4)

    assert len(chunks) > 1
    assert chunks[0].split()[-1] in chunks[1].split()
    assert " ".join(chunks).replace("  ", " ").startswith("one two three")


def test_reciprocal_rank_fusion_combines_rankings_and_deduplicates() -> None:
    first = [Item("a"), Item("b")]
    second = [Item("b"), Item("c")]

    merged = reciprocal_rank_fusion([first, second])

    assert [item.id for item in merged] == ["b", "a", "c"]


class FakeRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[object]:
        return [
            type(
                "Chunk",
                (),
                {
                    "source_type": "complaint",
                    "source_id": "123456",
                    "content": "The owner reported an electrical short near the door wiring.",
                    "component": "electrical",
                    "record_date": date(2024, 1, 2),
                },
            )()
        ]


class FakeLLM:
    def __init__(self) -> None:
        self.prompt = ""

    async def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return "The report describes an electrical short near the door wiring. [complaint:123456]"


def test_ask_uses_fake_llm_and_returns_cited_source() -> None:
    fake_llm = FakeLLM()
    app.dependency_overrides[get_retriever] = FakeRetriever
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm
    try:
        response = TestClient(app).post(
            "/ask", json={"question": "What electrical issue was reported?"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The report describes an electrical short near the door wiring. [complaint:123456]",
        "sources": [
            {
                "id": "complaint:123456",
                "snippet": "The owner reported an electrical short near the door wiring.",
                "source_type": "complaint",
                "component": "electrical",
                "date": "2024-01-02",
            }
        ],
    }
    assert "only the supplied NHTSA document excerpts" in fake_llm.prompt


def test_ask_says_not_found_when_retrieval_is_empty() -> None:
    app.dependency_overrides[get_retriever] = lambda: type("EmptyRetriever", (), {"retrieve": lambda *_args, **_kwargs: []})()
    app.dependency_overrides[get_llm_provider] = lambda: FakeLLM()
    try:
        response = TestClient(app).post("/ask", json={"question": "Why did the brakes fail?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"answer": "not found in the documents", "sources": []}


def test_ask_does_not_guess_internal_root_cause_information() -> None:
    class MustNotRetrieve:
        def retrieve(self, query: str, top_k: int = 5) -> list[object]:
            raise AssertionError("Unavailable internal information must short-circuit retrieval")

    app.dependency_overrides[get_retriever] = MustNotRetrieve
    app.dependency_overrides[get_llm_provider] = lambda: FakeLLM()
    try:
        response = TestClient(app).post(
            "/ask",
            json={"question": "What is Mercedes-Benz's internal root cause analysis for door-seal failures?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"answer": "not found in the documents", "sources": []}
