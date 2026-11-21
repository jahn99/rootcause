"""Groq implementation of the shared LLM interface."""

from groq import AsyncGroq

from app.config import Settings
from app.llm.base import LLMProvider


class GroqProvider(LLMProvider):
    """Generate text using Groq's asynchronous chat completion API."""

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY must be set when LLM_PROVIDER=groq")
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self._model = settings.groq_model

    async def generate(self, prompt: str) -> str:
        """Send a prompt to Groq and return the generated message text."""
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = completion.choices[0].message.content
        return content or ""
