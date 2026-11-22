"""Conservative rule-based checks for numeric and cited-source report claims."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.reports.workflow import REPORT_SOURCE_ID_PATTERN, _report_number_tokens

ClaimStatus = Literal["supported", "unsupported", "not_checkable"]
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+|\s*;\s*")
_SOURCE_TERMS = re.compile(r"\b(?:recall|complaint|NHTSA)\b", re.IGNORECASE)
_NUMBER_TOKEN = re.compile(r"(?<![A-Za-z0-9])\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z0-9])")
_PERCENT_SUFFIX = re.compile(r"^\s*(?:%|percent(?:age)?\b)", re.IGNORECASE)
_UNIT_SUFFIX = re.compile(
    r"^\s*(?:[A-Za-z-]+\s+){0,3}"
    r"(?:defects?|counts?|totals?|complaints?|records?|parts?|minutes?|hours?|"
    r"weeks?|days?|vehicles?|cases?|incidents?)\b",
    re.IGNORECASE,
)
_METRIC_PREFIX = re.compile(r"(?:total|count|rate|percentage|percent)\D{0,12}$", re.IGNORECASE)
_LOCATION_IDENTIFIER = re.compile(r"\b(?:line|plant)\s*#?\s*\d+\b", re.IGNORECASE)


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


def _report_claims(report: Any) -> list[tuple[str, str]]:
    """Collect short statements with their origin for field-specific treatment."""
    payload = report.model_dump(mode="python") if hasattr(report, "model_dump") else report
    if not isinstance(payload, dict):
        return []

    values: list[tuple[Any, str]] = []
    for field in (
        "containment_actions", "corrective_actions", "preventive_actions",
        "limitations", "answer", "human_review_note",
    ):
        values.append((payload.get(field), field))
    # Problem descriptions repeat the user-provided issue; they are not independent evidence.
    values.append((payload.get("problem_description"), "problem_description"))
    hypotheses = payload.get("root_cause_hypotheses", [])
    if isinstance(hypotheses, list):
        for item in hypotheses:
            if not isinstance(item, dict):
                continue
            values.append((item.get("hypothesis"), "root_cause_hypothesis"))
            citations = item.get("evidence", [])
            if isinstance(citations, list):
                values.extend(
                    (f"Evidence citation {citation}", "evidence_citation")
                    for citation in citations
                    if isinstance(citation, str)
                    and REPORT_SOURCE_ID_PATTERN.fullmatch(citation)
                )

    unique: dict[str, tuple[str, str]] = {}

    def add_text(value: Any, origin: str) -> None:
        if isinstance(value, str):
            for piece in _SENTENCE_BOUNDARY.split(value):
                cleaned = piece.strip().lstrip("-* \u2022").strip()
                if cleaned:
                    marker = " ".join(cleaned.split()).casefold()
                    previous = unique.get(marker)
                    if previous is None or origin == "problem_description":
                        unique[marker] = (cleaned, origin)
        elif isinstance(value, list):
            for item in value:
                add_text(item, origin)

    for value, origin in values:
        add_text(value, origin)
    return list(unique.values())


def _metric_numbers(text: str) -> set[str]:
    """Return only numbers attached to a count, total, or percentage-like metric."""
    normalized = _LOCATION_IDENTIFIER.sub(" ", text)
    candidate_numbers = _report_number_tokens(normalized)
    metrics: set[str] = set()
    for match in _NUMBER_TOKEN.finditer(normalized):
        number = match.group().replace(",", "")
        if number not in candidate_numbers:
            continue
        suffix = normalized[match.end():match.end() + 48]
        prefix = normalized[max(0, match.start() - 24):match.start()]
        if (
            _PERCENT_SUFFIX.match(suffix)
            or _UNIT_SUFFIX.match(suffix)
            or _METRIC_PREFIX.search(prefix)
        ):
            metrics.add(number)
    return metrics


def _remove_retrieved_excerpts(text: str, retrieved_sources: list[dict[str, Any]]) -> str:
    """Remove quoted or directly included retrieved snippets before metric checks."""
    cleaned = text
    snippets = [
        str(source.get("snippet", "")).strip()
        for source in retrieved_sources
        if isinstance(source, dict) and source.get("snippet")
    ]
    quote_pattern = re.compile(r"[\"“](.+?)[\"”]")
    for match in reversed(list(quote_pattern.finditer(cleaned))):
        quoted = match.group(1)
        if any(quoted.casefold() in snippet.casefold() for snippet in snippets):
            cleaned = cleaned[:match.start()] + " " + cleaned[match.end():]
    for snippet in snippets:
        excerpt = snippet[:180].strip()
        if excerpt:
            cleaned = re.sub(re.escape(excerpt), " ", cleaned, flags=re.IGNORECASE)
    return cleaned


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
        for number in _metric_numbers(finding)
    }
    source_ids = {
        str(source.get("id"))
        for source in retrieved_sources
        if isinstance(source, dict) and source.get("id") is not None
    }
    derived_finding_claims = {
        " ".join(piece.split()).casefold()
        for finding in key_findings
        for piece in _SENTENCE_BOUNDARY.split(finding)
        if piece.strip()
    }

    verified: list[VerifiedClaim] = []
    for claim, origin in _report_claims(report):
        if " ".join(claim.split()).casefold() in derived_finding_claims:
            continue
        if origin == "problem_description":
            verified.append(VerifiedClaim(
                claim=claim, status="not_checkable", reason="user-provided"
            ))
            continue

        text_for_metrics = _remove_retrieved_excerpts(claim, retrieved_sources)
        cited_ids = REPORT_SOURCE_ID_PATTERN.findall(text_for_metrics)
        numbers = _metric_numbers(text_for_metrics)
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
