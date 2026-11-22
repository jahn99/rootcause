"""Compact, evidence-linked Pydantic schema for an 8D draft report."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReportModel(BaseModel):
    """Base settings shared by report models."""

    model_config = ConfigDict(extra="forbid")


class RootCauseHypothesis(ReportModel):
    """A possible cause with an explicit confidence and gathered evidence IDs."""

    hypothesis: str = Field(min_length=1, max_length=240)
    confidence_level: Literal["low", "medium", "high"]
    evidence: list[str] = Field(min_length=1, max_length=4)


class SupportingData(ReportModel):
    """The validated SQL and result rows gathered for the report."""

    id: str
    sql: str
    rows: list[dict[str, Any]]


class SupportingDocument(ReportModel):
    """A retrieved public document excerpt and its citation ID."""

    id: str
    source_type: str
    snippet: str
    date: str | None


class ReportDraft(ReportModel):
    """LLM-authored report fields that do not include evidence payloads."""

    problem_description: str = Field(min_length=1, max_length=350)
    containment_actions: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(max_length=3)
    root_cause_hypotheses: list[RootCauseHypothesis] = Field(max_length=3)
    corrective_actions: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(max_length=3)
    preventive_actions: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(max_length=3)
    limitations: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(max_length=5)


class EightDReport(ReportDraft):
    """Complete 8D report with evidence gathered outside the LLM draft."""

    supporting_data: list[SupportingData]
    supporting_documents: list[SupportingDocument]
    human_review_note: Literal["Draft for human review."] = "Draft for human review."

    @model_validator(mode="after")
    def validate_evidence_citations(self) -> EightDReport:
        """Require each hypothesis to cite gathered, non-empty SQL or document evidence."""
        allowed_ids = {
            item.id for item in self.supporting_data if item.rows
        } | {item.id for item in self.supporting_documents}
        for hypothesis in self.root_cause_hypotheses:
            unknown = set(hypothesis.evidence) - allowed_ids
            if unknown:
                raise ValueError("Each root-cause hypothesis must cite gathered evidence IDs")
        if not allowed_ids:
            if self.root_cause_hypotheses:
                raise ValueError("Hypotheses require supporting data or document evidence")
            if not any("insufficient evidence" in item.lower() for item in self.limitations):
                raise ValueError("Limitations must say 'insufficient evidence' when evidence is absent")
        return self
