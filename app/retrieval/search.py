"""BM25, pgvector, and reciprocal-rank-fusion retrieval."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import TypeVar

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.knowledge_models import KnowledgeChunk
from app.ingestion.embeddings import embed_query

T = TypeVar("T")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer shared by BM25 and questions."""
    return TOKEN_PATTERN.findall(text.lower())


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[T]],
    id_getter: Callable[[T], object] = lambda item: getattr(item, "id"),
    rank_constant: int = 60,
) -> list[T]:
    """Merge ranked lists using standard reciprocal rank fusion scoring."""
    scores: dict[object, float] = {}
    items: dict[object, T] = {}
    for ranking in rankings:
        seen_in_list: set[object] = set()
        for rank, item in enumerate(ranking, start=1):
            item_id = id_getter(item)
            if item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)
            items[item_id] = item
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rank_constant + rank)
    return [items[item_id] for item_id in sorted(scores, key=lambda key: (-scores[key], str(key)))]


class HybridRetriever:
    """Search knowledge chunks using BM25, pgvector, or reciprocal-rank fusion."""

    def __init__(self, engine, mode: str = "hybrid") -> None:
        if mode not in {"hybrid", "bm25", "vector"}:
            raise ValueError("mode must be one of: hybrid, bm25, vector")
        self.engine = engine
        self.mode = mode

    def retrieve(self, query: str, top_k: int = 5) -> list[KnowledgeChunk]:
        """Return the top matching chunks for a question."""
        if top_k < 1:
            return []
        with Session(self.engine) as session:
            documents = list(session.scalars(select(KnowledgeChunk)).all())
            if not documents:
                return []

            bm25_results: list[KnowledgeChunk] = []
            if self.mode in {"hybrid", "bm25"}:
                tokenized_corpus = [tokenize(document.content) for document in documents]
                if any(tokenized_corpus):
                    scorer = BM25Okapi(tokenized_corpus)
                    scores = scorer.get_scores(tokenize(query))
                    bm25_results = [
                        document
                        for _, document in sorted(
                            zip(scores, documents, strict=True), key=lambda pair: pair[0], reverse=True
                        )[:top_k]
                    ]
            if self.mode == "bm25":
                return bm25_results

            query_vector = embed_query(query)
            vector_results = list(
                session.scalars(
                    select(KnowledgeChunk)
                    .order_by(KnowledgeChunk.embedding.cosine_distance(query_vector))
                    .limit(top_k)
                ).all()
            )
            if self.mode == "vector":
                return vector_results
            return reciprocal_rank_fusion([bm25_results, vector_results])[:top_k]
