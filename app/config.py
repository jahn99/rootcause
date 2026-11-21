"""Environment-backed application settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    app_name: str = "RootCause"
    llm_provider: str = "groq"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-20b"
    database_url: str = "postgresql+psycopg://rootcause:rootcause@db:5432/rootcause"
    readonly_db_password: str = "rootcause_readonly"
    readonly_database_url: str = "postgresql+psycopg://rootcause_readonly:rootcause_readonly@db:5432/rootcause"
    sql_agent_row_limit: int = 100
    sql_agent_timeout_ms: int = 3000
    retrieval_mode: str = "hybrid"
    retrieval_top_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return cached settings from the process environment."""
    return Settings()
