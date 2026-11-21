"""Small deterministic text chunker for source documents."""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into overlapping chunks while keeping all non-whitespace text."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        next_start = max(end - overlap, start + 1)
        previous_boundary = normalized.rfind(" ", start, next_start)
        if previous_boundary >= start:
            next_start = previous_boundary + 1
        else:
            following_boundary = normalized.find(" ", next_start, end)
            if following_boundary >= 0:
                next_start = following_boundary + 1
        start = max(next_start, start + 1)
    return chunks
