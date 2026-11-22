"""Conservative rule-based checks for numeric and cited-source report claims."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.reports.workflow import REPORT_SOURCE_ID_PATTERN, _report_number_tokens

ClaimStatus = Literal["supported", "unsupported", "not_checkable"]
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+|\s*;\s*")
_SOURCE_TERMS = re.compile(r"\b(?:recall|complaint|NHTSA)\b", re.IGNORECASE)


class VerifiedClaim(BaseModel):
    """A single report statement and the outcome of its deterministic checks."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    status: ClaimStatus
    reason: str


class ClaimVerification(BaseModel):
    """Claim results and totals for a report draft."""

    model_config = ConfigDict(extra="forbid")

    supported: int = Field(ge=0)
    unsupported: int = Field(ge=0)
    not_checkable: int = Field(ge=0)
    claims: list[VerifiedClaim]


def _report_claims(report: Any) -> list[str]:
    """Collect short text statements from report fields without treating metadata as claims."""
    payload = report.model_dump(mode="python") if hasattr(report, "model_dump") else report
    if not isinstance(payload, dict):
        return []

    values: list[Any] = []
    for field in (
        "problem_description", "containment_actions", "corrective_actions",
        "preventive_actions", "limitations", "answer", "key_findings",
        "human_review_note",
    ):
        values.append(payload.get(field))
    hypotheses = payload.get("root_cause_hypotheses", [])
    if isinstance(hypotheses, list):
        for item in hypotheses:
            if not isinstance(item, dict):
                continue
            values.append(item.get("hypothesis"))
            citations = item.get("evidence", [])
            if isinstance(citations, list):
                values.extend(
                    f"Evidence citation {citation}"
                    for citation in citations
                    if isinstance(citation, str)
                    and REPORT_SOURCE_ID_PATTERN.fullmatch(citation)
                )

    claims: list[str] = []

    def add_text(value: Any) -> None:
        if isinstance(value, str):
            for piece in _SENTENCE_BOUNDARY.split(value):
                cleaned = piece.strip().lstrip("-* \u2022").strip()
                if cleaned:
                    claims.append(cleaned)
        elif isinstance(value, list):
            for item in value:
                add_text(item)

    for value in values:
        add_text(value)
    return claims


def verify_report_claims(
    report: Any,
    *,
    key_findings: list[str],
    sql_rows: list[dict[str, Any]],
    retrieved_sources: list[dict[str, Any]],
) -> ClaimVerification:
    """Check numeric claims against Python findings and cited IDs against retrieval.

    SQL rows are accepted as part of the evidence input for an explicit, stable
    interface, but raw rows are not used to establish numeric support.
    """
    del sql_rows
    supported_numbers = {
        number
        for finding in key_findings
        for number in _report_number_tokens(finding)
    }
    source_ids = {
        str(source.get("id"))
        for source in retrieved_sources
        if isinstance(source, dict) and source.get("id") is not None
    }

    verified: list[VerifiedClaim] = []
    for claim in _report_claims(report):
        numbers = _report_number_tokens(claim)
        cited_ids = REPORT_SOURCE_ID_PATTERN.findall(claim)
        if numbers and any(number not in supported_numbers for number in numbers):
            missing = sorted(number for number in numbers if number not in supported_numbers)
            status: ClaimStatus = "unsupported"
            reason = f"Numbers {', '.join(missing)} are absent from key_findings."
        elif cited_ids and any(source_id not in source_ids for source_id in cited_ids):
            missing_ids = [source_id for source_id in cited_ids if source_id not in source_ids]
            status = "unsupported"
            reason = f"Cited source ID(s) not retrieved: {', '.join(missing_ids)}."
        elif numbers or cited_ids:
            status = "supported"
            reason = (
                "Numbers appear in key_findings and cited source IDs were retrieved."
                if numbers and cited_ids
                else "Numbers appear in key_findings."
                if numbers
                else "Cited source ID was retrieved."
            )
        elif _SOURCE_TERMS.search(claim):
            status = "not_checkable"
            reason = "Source-related statement has no verifiable cited ID."
        else:
            status = "not_checkable"
            reason = "No numeric or cited-ID rule applies to this statement."
        verified.append(VerifiedClaim(claim=claim, status=status, reason=reason))

    counts = {status: sum(item.status == status for item in verified) for status in (
        "supported", "unsupported", "not_checkable"
    )}
    return ClaimVerification(**counts, claims=verified)
