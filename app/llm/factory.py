"""Factory for selecting the configured LLM provider."""

from app.config import Settings, get_settings
from app.llm.base import LLMProvider
from app.llm.groq_provider import GroqProvider


def create_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Create the provider selected by the ``LLM_PROVIDER`` environment variable."""
    active_settings = settings or get_settings()
    provider = active_settings.llm_provider.lower()
    if provider == "groq":
        return GroqProvider(active_settings)
    raise ValueError(f"Unsupported LLM_PROVIDER: {active_settings.llm_provider}")
