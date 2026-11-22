"""Groq implementation of the shared LLM interface."""

import asyncio
import json

from groq import APIStatusError, AsyncGroq

from app.config import Settings
from app.llm.base import LLMProvider

MAX_COMPLETION_TOKENS = 900


class GroqProviderError(RuntimeError):
    """Safe, user-facing error raised for failed Groq requests."""


class GroqRateLimitError(GroqProviderError):
    """Groq continued to rate limit a request after its retry."""


class GroqRequestTooLargeError(GroqProviderError):
    """Groq rejected the original and reduced evidence prompts as too large."""


def _smaller_evidence_prompt(prompt: str) -> str | None:
    """Return a reduced report prompt after a context-size rejection."""
    marker = "Gathered evidence:"
    if marker not in prompt:
        return None
    prefix, raw_evidence = prompt.split(marker, maxsplit=1)
    try:
        evidence = json.loads(raw_evidence.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(evidence, dict):
        return None

    triage = evidence.get("triage", {})
    evidence["triage"] = {
        key: triage[key]
        for key in ("part", "plant", "line", "start_date", "end_date")
        if isinstance(triage, dict) and key in triage
    }
    evidence["sql_results"] = [
        {"id": item.get("id"), "rows": item.get("rows", [])[:5]}
        for item in evidence.get("sql_results", [])[:3]
        if isinstance(item, dict)
    ]
    evidence["documents"] = [
        {"id": item.get("id"), "snippet": str(item.get("snippet", ""))[:250]}
        for item in evidence.get("documents", [])[:2]
        if isinstance(item, dict)
    ]
    evidence["nhtsa_documents"] = [
        {"id": item.get("id"), "snippet": str(item.get("snippet", ""))[:250]}
        for item in evidence.get("nhtsa_documents", [])[:2]
        if isinstance(item, dict)
    ]
    evidence["limitations"] = [str(item)[:160] for item in evidence.get("limitations", [])[:3]]

    allowed_ids = [
        item["id"]
        for item in evidence.get("sql_results", []) + evidence.get("documents", []) + evidence.get("nhtsa_documents", [])
        if item.get("id")
    ]
    prefix_lines = [
        f"Allowed evidence IDs: {allowed_ids}" if line.startswith("Allowed evidence IDs:") else line
        for line in prefix.splitlines()
    ]
    return "\n".join(prefix_lines).rstrip() + f"\n{marker} " + json.dumps(
        evidence, ensure_ascii=False, separators=(",", ":")
    )


class GroqProvider(LLMProvider):
    """Generate text using Groq's asynchronous chat completion API."""

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY must be set when LLM_PROVIDER=groq")
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self._model = settings.groq_model

    async def _completion(self, prompt: str, response_format: dict[str, object] | None = None):
        """Create a completion with one retry for rate limits and a 413 fallback."""
        active_prompt = prompt
        reduced = False
        while True:
            for attempt in range(2):
                try:
                    params: dict[str, object] = {
                        "model": self._model,
                        "messages": [{"role": "user", "content": active_prompt}],
                        "max_completion_tokens": MAX_COMPLETION_TOKENS,
                    }
                    if response_format is not None:
                        params["response_format"] = response_format
                    return await self._client.chat.completions.create(**params)
                except APIStatusError as error:
                    if error.status_code == 429:
                        if attempt == 0:
                            await asyncio.sleep(1)
                            continue
                        raise GroqRateLimitError(
                            "Groq is rate limiting requests. Please retry shortly."
                        ) from error
                    if error.status_code == 413:
                        break
                    raise GroqProviderError(
                        f"Groq request failed with HTTP {error.status_code}."
                    ) from error
            else:
                # The inner retry loop only completes normally when a request succeeds.
                raise AssertionError("unreachable Groq retry state")

            if reduced:
                raise GroqRequestTooLargeError(
                    "Groq rejected the reduced report evidence as too large. Try a narrower issue."
                )
            active_prompt = _smaller_evidence_prompt(active_prompt) or ""
            if not active_prompt:
                raise GroqRequestTooLargeError(
                    "Groq rejected the prompt as too large. Try a narrower issue."
                )
            reduced = True

    async def generate(self, prompt: str) -> str:
        """Send a prompt to Groq and return the generated message text."""
        completion = await self._completion(prompt)
        content = completion.choices[0].message.content
        return content or ""

    async def generate_json(self, prompt: str, schema: dict[str, object]) -> str:
        """Use Groq JSON-schema output mode and return its JSON response."""
        completion = await self._completion(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "data_agent_sql", "strict": True, "schema": schema},
            },
        )
        content = completion.choices[0].message.content
        return content or ""
