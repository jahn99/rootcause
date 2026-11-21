"""HTTP endpoint for generating validated evidence-linked 8D report drafts."""

from functools import lru_cache
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agents.data_agent import DataAgent
from app.api.ask import get_retriever
from app.api.sql_agent import get_data_agent
from app.llm.factory import create_llm_provider
from app.reports.schema import EightDReport
from app.reports.workflow import InvestigationWorkflow, ReportValidationError

router = APIRouter()
logger = logging.getLogger(__name__)


class InvestigateRequest(BaseModel):
    """Manufacturing issue to triage and investigate."""

    issue: str = Field(min_length=5, max_length=2000)


@lru_cache(maxsize=1)
def _workflow() -> InvestigationWorkflow:
    """Create the configured data, retrieval, and report workflow once per process."""
    return InvestigationWorkflow(
        data_agent=get_data_agent(),
        knowledge_retriever=get_retriever(),
        llm=create_llm_provider(),
    )


def get_investigation_workflow() -> InvestigationWorkflow:
    """FastAPI dependency for the configured workflow."""
    return _workflow()


@router.post("/investigate", response_model=EightDReport)
async def investigate(
    request: InvestigateRequest,
    workflow: InvestigationWorkflow = Depends(get_investigation_workflow),
) -> EightDReport:
    """Return an 8D report draft, validated against the gathered evidence."""
    try:
        return await workflow.run(request.issue)
    except ReportValidationError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception as error:
        logger.exception("Unexpected error while investigating issue")
        error_text = str(error).casefold()
        if any(term in error_text for term in ("request too large", "context length", "tokens per minute", "413")):
            raise HTTPException(
                status_code=502,
                detail="The LLM provider could not process the report context. Try again with a narrower issue.",
            ) from error
        if "429" in error_text or "rate limit" in error_text:
            raise HTTPException(
                status_code=503,
                detail="The LLM provider is rate limited. Please retry shortly.",
            ) from error
        raise HTTPException(
            status_code=500,
            detail="Investigation failed unexpectedly. The error was logged; please retry.",
        ) from error
