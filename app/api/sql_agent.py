"""HTTP interface for guarded read-only data questions."""

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
import logging

from app.agents.data_agent import DataAgent, DataAgentResult
from app.agents.sql_validator import SQLValidationError
from app.config import get_settings
from app.db.session import create_database_engine
from app.llm.base import LLMProvider
from app.llm.factory import create_llm_provider

router = APIRouter()
logger = logging.getLogger(__name__)


class SQLAgentRequest(BaseModel):
    """A natural-language question about the synthetic manufacturing tables."""

    question: str = Field(min_length=3, max_length=1000)


class SQLAgentResponse(BaseModel):
    """Executed SQL, returned rows, and a short answer."""

    sql: str | None
    rows: list[dict[str, object]]
    answer: str
    refused: bool = False


class _LazyLLMProvider(LLMProvider):
    """Delay provider construction until a non-refused question needs an LLM."""

    def __init__(self) -> None:
        self._provider: LLMProvider | None = None

    def _get_provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = create_llm_provider()
        return self._provider

    async def generate(self, prompt: str) -> str:
        return await self._get_provider().generate(prompt)

    async def generate_json(
        self,
        prompt: str,
        schema: dict[str, object],
        *,
        max_output_tokens: int = 600,
    ) -> str:
        return await self._get_provider().generate_json(
            prompt, schema, max_output_tokens=max_output_tokens
        )


@lru_cache(maxsize=1)
def _readonly_engine() -> Engine:
    """Create the data-agent engine from the dedicated read-only URL."""
    return create_database_engine(get_settings().readonly_database_url)


def get_data_agent() -> DataAgent:
    """Dependency providing an LLM-backed agent connected as read-only."""
    settings = get_settings()
    return DataAgent(
        llm=_LazyLLMProvider(),
        engine=_readonly_engine(),
        row_limit=settings.sql_agent_row_limit,
        timeout_ms=settings.sql_agent_timeout_ms,
    )


@router.post("/sql-agent", response_model=SQLAgentResponse)
async def sql_agent(
    request: SQLAgentRequest,
    agent: DataAgent = Depends(get_data_agent),
) -> SQLAgentResponse:
    """Answer a question from the allow-listed manufacturing tables."""
    try:
        result: DataAgentResult = await agent.ask(request.question)
    except SQLAlchemyError:
        logger.exception("Read-only SQL agent database operation failed")
        return SQLAgentResponse(
            sql=None,
            rows=[],
            answer="The database could not complete that read request. No results were returned.",
            refused=True,
        )
    except SQLValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return SQLAgentResponse(**result.model_dump())
