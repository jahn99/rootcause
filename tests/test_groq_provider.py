"""Groq provider retry and oversized-evidence fallback tests."""

from __future__ import annotations

import json
import asyncio
from types import SimpleNamespace

import httpx
import pytest
from groq import APIStatusError

from app.config import Settings
from app.llm.groq_provider import (
    GroqProvider,
    GroqRateLimitError,
    GroqRequestTooLargeError,
    TokenPacer,
    _smaller_evidence_prompt,
)


def status_error(code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(code, request=request)
    return APIStatusError("provider error", response=response, body={"error": "limited"})


def fake_completion(content: str = "{}") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")]
    )


def build_provider(completions: object) -> GroqProvider:
    settings = Settings.model_construct(groq_api_key="test-key", groq_model="openai/gpt-oss-20b")
    provider = GroqProvider(settings)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider


def test_groq_retries_429_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def create(**kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise status_error(429)
        return fake_completion("ok")

    async def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("app.llm.groq_provider.asyncio.sleep", no_sleep)
    provider = build_provider(SimpleNamespace(create=create))

    async def exercise() -> str:
        return await provider.generate("hello")

    assert asyncio.run(exercise()) == "ok"
    assert calls == 2


def test_groq_413_retries_with_smaller_evidence_prompt() -> None:
    prompts: list[str] = []
    completion_limits: list[int] = []
    reasoning_parameters: list[tuple[str, str]] = []

    async def create(**kwargs: object) -> SimpleNamespace:
        prompt = kwargs["messages"][0]["content"]
        prompts.append(prompt)
        completion_limits.append(kwargs["max_completion_tokens"])
        reasoning_parameters.append((kwargs["reasoning_effort"], kwargs["reasoning_format"]))
        if len(prompts) == 1:
            raise status_error(413)
        return fake_completion()

    evidence = {
        "triage": {"part": "door seal", "unused": "metadata"},
        "sql_results": [{"id": "sql:1", "rows": [{"value": index} for index in range(20)]}],
        "documents": [
            {"id": f"complaint:{index}", "snippet": "wind noise " * 100}
            for index in range(4)
        ],
    }
    prompt = "Allowed evidence IDs: []\nGathered evidence: " + json.dumps(evidence)
    provider = build_provider(SimpleNamespace(create=create))

    async def exercise() -> str:
        return await provider.generate_json(prompt, {"type": "object"})

    assert asyncio.run(exercise()) == "{}"
    assert len(prompts) == 2
    reduced = _smaller_evidence_prompt(prompts[0])
    assert reduced is not None
    assert len(json.loads(reduced.split("Gathered evidence:", 1)[1])["documents"]) <= 2
    assert completion_limits == [600, 600]
    assert reasoning_parameters == [("low", "hidden"), ("low", "hidden")]


def test_413_fallback_is_unavailable_for_non_evidence_prompts() -> None:
    assert _smaller_evidence_prompt("plain prompt") is None


def test_repeated_413_raises_safe_custom_error() -> None:
    async def create(**kwargs: object) -> SimpleNamespace:
        raise status_error(413)

    provider = build_provider(SimpleNamespace(create=create))

    async def exercise() -> str:
        return await provider.generate_json(
            'Gathered evidence: {"documents":[],"sql_results":[],"triage":{}}',
            {"type": "object"},
        )

    with pytest.raises(GroqRequestTooLargeError, match="reduced report evidence"):
        asyncio.run(exercise())


def test_repeated_429_raises_safe_custom_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def create(**kwargs: object) -> SimpleNamespace:
        raise status_error(429)

    async def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("app.llm.groq_provider.asyncio.sleep", no_sleep)
    provider = build_provider(SimpleNamespace(create=create))

    async def exercise() -> str:
        return await provider.generate("hello")

    with pytest.raises(GroqRateLimitError, match="rate limiting"):
        asyncio.run(exercise())


def test_json_validate_failed_returns_empty_for_agent_retry() -> None:
    async def create(**kwargs: object) -> SimpleNamespace:
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(400, request=request)
        raise APIStatusError(
            "json validation failed",
            response=response,
            body={"error": {"code": "json_validate_failed", "failed_generation": ""}},
        )

    provider = build_provider(SimpleNamespace(create=create))

    async def exercise() -> str:
        return await provider.generate_json("generate JSON", {"type": "object"})

    assert asyncio.run(exercise()) == ""
    assert provider.last_raw_output == ""
    assert provider.last_finish_reason is None


def test_token_pacer_waits_for_rolling_window(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 0.0
    waits: list[float] = []

    def monotonic() -> float:
        return now

    async def sleep(seconds: float) -> None:
        nonlocal now
        waits.append(seconds)
        now += seconds

    monkeypatch.setattr("app.llm.groq_provider.time.monotonic", monotonic)
    monkeypatch.setattr("app.llm.groq_provider.asyncio.sleep", sleep)
    pacer = TokenPacer(token_limit=100, window_seconds=60)

    async def exercise() -> None:
        await pacer.acquire(80)
        await pacer.acquire(30)

    asyncio.run(exercise())

    assert waits == [60]
