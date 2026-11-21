"""Abstract interface implemented by all language model providers."""

from abc import ABC, abstractmethod
import json


class LLMProvider(ABC):
    """Common async interface for text generation providers."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate a text response for a prompt."""

    async def generate_json(self, prompt: str, schema: dict[str, object]) -> str:
        """Generate JSON matching a schema, using the provider's structured mode if available."""
        schema_text = json.dumps(schema, separators=(",", ":"))
        return await self.generate(
            f"{prompt}\nReturn one valid JSON object matching this JSON schema: {schema_text}"
        )
