"""Groq implementation of the shared LLM interface."""

import asyncio
import json
import logging
import math
import time
from collections import deque
from typing import Any

from groq import APIStatusError, AsyncGroq

from app.config import Settings
from app.llm.base import LLMProvider

DEFAULT_COMPLETION_TOKENS = 600
TPM_TOKEN_LIMIT = 8000
TPM_WINDOW_SECONDS = 60.0
logger = logging.getLogger(__name__)


class GroqProviderError(RuntimeError):
    """Safe, user-facing error raised for failed Groq requests."""


class GroqRateLimitError(GroqProviderError):
    """Groq continued to rate limit a request after its retry."""


class GroqRequestTooLargeError(GroqProviderError):
    """Groq rejected the original and reduced evidence prompts as too large."""


class GroqStructuredJSONError(GroqProviderError):
    """Groq rejected a structured response and supplied no valid JSON output."""


class TokenPacer:
    """Reserve estimated prompt and completion tokens against a rolling TPM budget."""

    def __init__(self, token_limit: int = TPM_TOKEN_LIMIT, window_seconds: float = TPM_WINDOW_SECONDS):
        self.token_limit = token_limit
        self.window_seconds = window_seconds
        self._reservations: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, token_count: int) -> None:
        """Wait until a request's estimated token reservation fits the window."""
        if token_count > self.token_limit:
            raise GroqRequestTooLargeError(
                "The estimated prompt and output exceed the configured per-minute token budget."
            )
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._reservations and now - self._reservations[0][0] >= self.window_seconds:
                    self._reservations.popleft()
                used = sum(tokens for _, tokens in self._reservations)
                if used + token_count <= self.token_limit:
                    self._reservations.append((now, token_count))
                    return
                wait_seconds = max(
                    0.01,
                    self._reservations[0][0] + self.window_seconds - now,
                )
            logger.info(
                "Groq TPM pacing: waiting %.1f seconds (reserved=%d requested=%d limit=%d)",
                wait_seconds,
                used,
                token_count,
                self.token_limit,
            )
            await asyncio.sleep(wait_seconds)


def estimate_request_tokens(prompt: str, response_format: dict[str, object] | None, output_cap: int) -> int:
    """Estimate input, JSON schema, and reserved output tokens for Groq TPM pacing."""
    schema_text = json.dumps(response_format, ensure_ascii=False, separators=(",", ":")) if response_format else ""
    return math.ceil((len(prompt) + len(schema_text)) / 3.5) + output_cap


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
        self._api_key = settings.groq_api_key
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self._model = settings.groq_model
        self._pacer = TokenPacer()
        self.last_raw_output: str | None = None
        self.last_finish_reason: str | None = None

    def _safe_output(self, value: str) -> str:
        """Redact the credential defensively before retaining or logging model text."""
        return value.replace(self._api_key, "[REDACTED]") if self._api_key else value

    async def _completion(
        self,
        prompt: str,
        response_format: dict[str, object] | None = None,
        max_output_tokens: int = DEFAULT_COMPLETION_TOKENS,
    ):
        """Create a completion with one retry for rate limits and a 413 fallback."""
        active_prompt = prompt
        reduced = False
        while True:
            reservation = estimate_request_tokens(active_prompt, response_format, max_output_tokens)
            if reservation > self._pacer.token_limit:
                if reduced:
                    raise GroqRequestTooLargeError(
                        "The reduced prompt still exceeds the per-minute token budget."
                    )
                active_prompt = _smaller_evidence_prompt(active_prompt) or ""
                if not active_prompt:
                    raise GroqRequestTooLargeError(
                        "The prompt exceeds the per-minute token budget and has no reducible evidence."
                    )
                reduced = True
                continue
            for attempt in range(2):
                try:
                    await self._pacer.acquire(reservation)
                    params: dict[str, object] = {
                        "model": self._model,
                        "messages": [{"role": "user", "content": active_prompt}],
                        "max_completion_tokens": max_output_tokens,
                    }
                    if self._model.startswith("openai/gpt-oss-"):
                        params["reasoning_effort"] = "low"
                        params["reasoning_format"] = "hidden"
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
                    if error.status_code == 400 and self._is_json_validation_error(error):
                        failed_generation = self._failed_generation(error)
                        self.last_raw_output = self._safe_output(failed_generation)
                        self.last_finish_reason = None
                        logger.warning(
                            "Groq structured JSON rejected; finish_reason=%s raw_output=%r",
                            self.last_finish_reason,
                            self.last_raw_output,
                        )
                        raise GroqStructuredJSONError(
                            "Groq could not produce a valid structured JSON response."
                        ) from error
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

    @staticmethod
    def _is_json_validation_error(error: APIStatusError) -> bool:
        body = error.body if isinstance(error.body, dict) else {}
        details = body.get("error", {}) if isinstance(body, dict) else {}
        return isinstance(details, dict) and details.get("code") == "json_validate_failed"

    @staticmethod
    def _failed_generation(error: APIStatusError) -> str:
        body = error.body if isinstance(error.body, dict) else {}
        details = body.get("error", {}) if isinstance(body, dict) else {}
        value = details.get("failed_generation", "") if isinstance(details, dict) else ""
        return str(value or "")

    async def generate(self, prompt: str) -> str:
        """Send a prompt to Groq and return the generated message text."""
        completion = await self._completion(prompt, max_output_tokens=DEFAULT_COMPLETION_TOKENS)
        content = completion.choices[0].message.content
        output = content or ""
        self.last_raw_output = self._safe_output(output)
        self.last_finish_reason = completion.choices[0].finish_reason
        return output

    async def generate_json(
        self,
        prompt: str,
        schema: dict[str, object],
        *,
        max_output_tokens: int = DEFAULT_COMPLETION_TOKENS,
    ) -> str:
        """Use Groq JSON-schema output mode and return its JSON response."""
        try:
            completion = await self._completion(
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "data_agent_sql", "strict": True, "schema": schema},
                },
                max_output_tokens=max_output_tokens,
            )
        except GroqStructuredJSONError:
            return self.last_raw_output or ""
        content = completion.choices[0].message.content
        output = content or ""
        self.last_raw_output = self._safe_output(output)
        self.last_finish_reason = completion.choices[0].finish_reason
        return output
