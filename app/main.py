"""FastAPI application entry point."""

from fastapi import FastAPI

from app.config import get_settings

app = FastAPI(title=get_settings().app_name)


@app.get("/health")
async def health() -> dict[str, str]:
    """Report that the API process is ready."""
    return {"status": "ok"}
