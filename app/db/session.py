"""Database engine construction helpers."""

from sqlalchemy import Engine, create_engine

from app.config import get_settings


def create_database_engine(database_url: str | None = None) -> Engine:
    """Create a synchronous SQLAlchemy engine from settings or an explicit URL."""
    return create_engine(database_url or get_settings().database_url, pool_pre_ping=True)
