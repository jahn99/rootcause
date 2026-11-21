"""SQLAlchemy model for NHTSA document chunks and their embeddings."""

from datetime import date as DateValue

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models import Base

EMBEDDING_DIMENSIONS = 384


class KnowledgeChunk(Base):
    """A searchable text chunk from an NHTSA recall or complaint."""

    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        UniqueConstraint("source_type", "source_id", "chunk_index", name="uq_knowledge_source_chunk"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    component: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    record_date: Mapped[DateValue | None] = mapped_column(Date, nullable=True, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
