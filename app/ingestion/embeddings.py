"""Lazy local FastEmbed model loading and vectorization."""

from functools import lru_cache
import os

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    """Load the small 384-dimensional CPU model on first use."""
    return TextEmbedding(MODEL_NAME, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed source chunks as retrieval passages."""
    if not texts:
        return []
    model = get_embedding_model()
    return [vector.tolist() for vector in model.embed([f"passage: {text}" for text in texts])]


def embed_query(query: str) -> list[float]:
    """Embed a query using the local model's retrieval query prefix."""
    model = get_embedding_model()
    return next(model.embed([f"query: {query}"])).tolist()
