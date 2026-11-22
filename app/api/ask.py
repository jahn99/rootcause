"""Question answering over retrieved NHTSA documents."""

from functools import lru_cache
from typing import Protocol

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db.session import create_database_engine
from app.llm.base import LLMProvider
from app.llm.factory import create_llm_provider
from app.retrieval.search import HybridRetriever

router = APIRouter()
TOPIC_EXPANSIONS = (
    (("seal", "weatherstrip", "wind noise"), "door seal weatherstrip weather strip water leak water intrusion wind noise"),
    (("water", "leak", "intrusion", "cabin"), "water leak water leaking water intrusion cabin interior weatherstrip"),
    (("wiring", "wire", "electrical", "short circuit"), "wiring wire harness electrical short circuit wiring defect"),
)


def expand_retrieval_query(question: str) -> str:
    """Add focused automotive synonyms while preserving the user's original query."""
    lowered = question.casefold()
    additions = [
        expansion
        for signals, expansion in TOPIC_EXPANSIONS
        if any(signal in lowered for signal in signals)
    ]
    return " ".join([question, *additions])


class Retriever(Protocol):
    """Minimal retrieval interface used by the endpoint and its tests."""

    def retrieve(self, query: str, top_k: int = 5): ...


class AskRequest(BaseModel):
    """Question submitted to the retrieval-backed answer endpoint."""

    question: str = Field(min_length=3, max_length=1000)


class Source(BaseModel):
    """A cited chunk included with an answer."""

    id: str
    snippet: str
    source_type: str
    component: str
    date: str | None


class AskResponse(BaseModel):
    """Answer text and its source citations."""

    answer: str
    sources: list[Source]


@lru_cache(maxsize=1)
def _get_retriever() -> HybridRetriever:
    settings = get_settings()
    return HybridRetriever(create_database_engine(), mode=settings.retrieval_mode)


def get_retriever() -> Retriever:
    """Dependency providing the configured retrieval mode."""
    return _get_retriever()


def get_llm_provider() -> LLMProvider:
    """Dependency providing the configured LLM implementation."""
    return create_llm_provider()


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    retriever: Retriever = Depends(get_retriever),
    llm: LLMProvider = Depends(get_llm_provider),
) -> AskResponse:
    """Retrieve evidence, ask the configured LLM, and return cited chunks."""
    normalized_question = request.question.casefold()
    if "internal" in normalized_question and any(
        phrase in normalized_question for phrase in ("root cause", "analysis")
    ):
        return AskResponse(answer="not found in the documents", sources=[])

    retrieval_query = expand_retrieval_query(request.question)
    chunks = retriever.retrieve(retrieval_query, top_k=get_settings().retrieval_top_k)
    sources = [
        Source(
            id=f"{chunk.source_type}:{chunk.source_id}",
            snippet=chunk.content[:360],
            source_type=chunk.source_type,
            component=chunk.component,
            date=chunk.record_date.isoformat() if chunk.record_date else None,
        )
        for chunk in chunks
    ]
    if not chunks:
        return AskResponse(answer="not found in the documents", sources=[])

    evidence = "\n\n".join(
        f"[{source.id}; component={source.component}; date={source.date or 'unknown'}]\n{chunk.content}"
        for source, chunk in zip(sources, chunks, strict=True)
    )
    prompt = (
        "Answer using only the supplied NHTSA document excerpts. First decide which excerpts directly "
        "support the requested component and failure mode; shared vehicle brand or broad system "
        "words alone do not make an excerpt relevant. When relevant excerpts exist, state the "
        "specific reported failure modes and effects in concrete terms, and cite their source IDs. "
        "Do not give a vague summary or refuse when the excerpts directly support an answer. "
        'If no excerpt directly supports the question, reply exactly: "not found in the documents". '
        "Do not infer one failure mode from a different component or add facts from memory.\n\n"
        f"Question: {request.question}\n\nExcerpts:\n{evidence}"
    )
    answer = await llm.generate(prompt)
    return AskResponse(answer=answer, sources=sources)
