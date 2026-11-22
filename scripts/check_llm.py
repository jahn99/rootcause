"""Send a short prompt through the configured application LLM provider."""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.llm.factory import create_llm_provider  # noqa: E402


def friendly_error(error: Exception) -> str:
    """Translate provider errors into safe, actionable messages."""
    message = str(error).lower()
    if "groq_api_key" in message or "api key" in message or "api_key" in message:
        return "GROQ_API_KEY is missing. Add your Groq key to .env and try again."

    status_code = getattr(error, "status_code", None)
    if status_code in (400, 404):
        return "The request was rejected. Check that GROQ_MODEL is a valid model name."
    if status_code in (401, 403):
        return "Groq rejected the credentials. Check that GROQ_API_KEY is valid."

    if "connection" in error.__class__.__name__.lower() or "timeout" in error.__class__.__name__.lower():
        return "Could not reach Groq. Check your network connection and try again."

    return "The LLM check failed. Check your provider settings and network, then try again."


async def main() -> int:
    """Generate one response using the same provider factory as the app."""
    try:
        provider = create_llm_provider()
        reply = await provider.generate("Reply with the single word: ready")
    except Exception as error:
        print(f"LLM check failed: {friendly_error(error)}")
        return 1

    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
