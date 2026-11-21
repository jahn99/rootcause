"""Abstract interface implemented by all language model providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Common async interface for text generation providers."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate a text response for a prompt."""
