"""FastAPI application entry point."""

from fastapi import FastAPI

from app.config import get_settings
from app.api.ask import router as ask_router

app = FastAPI(title=get_settings().app_name)
app.include_router(ask_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Report that the API process is ready."""
    return {"status": "ok"}
