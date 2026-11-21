"""Chunk, embed, and persist NHTSA source records."""

from sqlalchemy import delete, text
from sqlalchemy.orm import Session

from app.db.knowledge_models import Base, KnowledgeChunk
from app.db.session import create_database_engine
from app.ingestion.chunking import chunk_text
from app.ingestion.embeddings import embed_passages
from app.ingestion.nhtsa import NHTSARecord


def ingest_records(records: list[NHTSARecord]) -> int:
    """Replace the knowledge chunk table contents with embedded records."""
    engine = create_database_engine()
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(engine)

        pending: list[dict[str, object]] = []
        for record in records:
            for index, chunk in enumerate(chunk_text(record.text)):
                pending.append(
                    {
                        "source_type": record.source_type,
                        "source_id": record.source_id,
                        "component": record.component,
                        "record_date": record.record_date,
                        "chunk_index": index,
                        "content": chunk,
                    }
                )
        embeddings = embed_passages([str(row["content"]) for row in pending])
        for row, vector in zip(pending, embeddings, strict=True):
            row["embedding"] = vector

        with Session(engine) as session, session.begin():
            session.execute(delete(KnowledgeChunk))
            if pending:
                session.execute(KnowledgeChunk.__table__.insert(), pending)
        return len(pending)
    finally:
        engine.dispose()
