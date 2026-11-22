"""LangGraph triage, evidence gathering, and 8D report drafting workflow."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import date, datetime, timedelta
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, ValidationError

from app.agents.data_agent import DataAgent
from app.agents.investigation_queries import InvestigationQueryResult, InvestigationQueryRunner
from app.db.knowledge_models import KnowledgeChunk
from app.llm.base import LLMProvider
from app.reports.schema import (
    EightDReport,
    ReportDraft,
    SupportingData,
    SupportingDocument,
)

REPORT_DRAFT_SCHEMA = ReportDraft.model_json_schema()
REPORT_EVIDENCE_TOKEN_BUDGET = 3500
logger = logging.getLogger(__name__)
PLANTS = ("Cedar Falls", "Lakeview", "Pine Ridge", "Stuttgart")
PART_PATTERNS = (
    (r"door[\s-]*seals?", "door seal"),
    (r"brake(?:s| pads?| calipers?)?", "brake system"),
    (r"air ?bags?", "airbag"),
    (r"seat ?belts?", "seat belt"),
    (r"fuel (?:pump|system|line|leak)", "fuel system"),
    (r"steering", "steering system"),
    (r"electrical|wiring|short circuit", "electrical system"),
)
ISO_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NAMED_DATE_PATTERN = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}\s+"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+\d{4}\b",
    re.IGNORECASE,
)
MONTH_YEAR_PATTERN = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+(\d{4})\b",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
DAY_RANGE_PATTERN = re.compile(
    r"\bbetween\s+(\d{1,2})\s+(?:and|to|through)\s+(\d{1,2})\s+"
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+(20\d{2})\b",
    re.IGNORECASE,
)

PART_RETRIEVAL_TERMS = {
    "door seal": "door seal weatherstrip water leak wind noise",
    "brake system": "brakes braking system brake fluid stopping distance",
    "airbag": "airbag inflator deployment warning light",
    "seat belt": "seat belt buckle retractor restraint",
    "fuel system": "fuel leak fuel line pump fire",
    "steering system": "steering loss control power steering",
    "electrical system": "electrical wiring short circuit fire",
}
DOCUMENT_METADATA_KEYS = {
    "vehicle", "vin", "manufacturer", "make", "model", "modelyear", "model year",
    "product", "component", "components", "odi number", "odinumber", "nhtsa campaign number",
    "nhtsaid", "crash", "fire", "numberofinjuries", "numberofdeaths",
    "dateofincident", "datecomplaintfiled", "reportreceiveddate", "reportdate",
}
REPORT_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z0-9])")
REPORT_SOURCE_ID_PATTERN = re.compile(
    r"\b(?:complaint|recall):[A-Za-z0-9][A-Za-z0-9-]*\b", re.IGNORECASE
)
REPORT_NHTSA_ID_PATTERN = re.compile(
    r"\b(?:ODI\s*Number|NHTSA\s*Campaign\s*Number|NHTSAID)\b"
    r"\s*[:#]?\s*[A-Za-z0-9][A-Za-z0-9-]*",
    re.IGNORECASE,
)
REPORT_MONTH_PATTERN = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
REPORT_MONTH_DAY_PATTERN = re.compile(
    rf"\b(?:{REPORT_MONTH_PATTERN})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?"
    rf"(?:,?\s+20\d{{2}})?\b|\b\d{{1,2}}(?:st|nd|rd|th)?\s+"
    rf"(?:{REPORT_MONTH_PATTERN})\.?(?:,?\s+20\d{{2}})?\b",
    re.IGNORECASE,
)


class TriageContext(BaseModel):
    """Issue details extracted before agents gather evidence."""

    model_config = ConfigDict(extra="forbid")

    part: str | None = None
    plant: str | None = None
    line: int | None = None
    start_date: date | None = None
    end_date: date | None = None
    top_part_query: bool = False


class KnowledgeRetriever(Protocol):
    """Required retrieval interface for the knowledge agent node."""

    def retrieve(self, query: str, top_k: int = 5) -> list[KnowledgeChunk]: ...


class WorkflowState(TypedDict, total=False):
    """Shared state passed between the LangGraph nodes."""

    issue: str
    triage: dict[str, Any]
    supporting_data: list[dict[str, Any]]
    supporting_documents: list[dict[str, Any]]
    data_limitation: str
    knowledge_limitation: str
    report: dict[str, Any]


class ReportValidationError(ValueError):
    """Raised when the report remains invalid after one retry."""


REPORT_LIST_LIMITS = {
    "containment_actions": (3, 200),
    "corrective_actions": (3, 200),
    "preventive_actions": (3, 200),
    "limitations": (5, 200),
}


def normalize_report_payload(payload: Any) -> dict[str, Any]:
    """Repair common list-formatting noise before validating a report draft."""
    if not isinstance(payload, dict):
        raise ValueError("The report JSON must be an object")
    normalized = dict(payload)

    def text_items(value: Any, field_name: str, max_item_length: int) -> list[str]:
        items: list[str] = []

        def collect(item: Any) -> None:
            if isinstance(item, str):
                cleaned = item.strip().strip("\\\"' ")
                folded = cleaned.casefold()
                if not cleaned or folded in {field_name, ":", "-", "none", "null"}:
                    return
                if cleaned.endswith(":") and cleaned[:-1].casefold() == field_name:
                    return
                items.append(cleaned[:max_item_length])
            elif isinstance(item, (list, tuple)):
                for child in item:
                    collect(child)
            elif isinstance(item, dict):
                for child in item.values():
                    collect(child)

        collect(value)
        unique: list[str] = []
        seen: set[str] = set()
        for item in items:
            marker = item.casefold()
            if marker not in seen:
                seen.add(marker)
                unique.append(item)
        max_items = REPORT_LIST_LIMITS.get(field_name, (4, max_item_length))[0]
        return unique[:max_items]

    for field_name, (_, max_item_length) in REPORT_LIST_LIMITS.items():
        if field_name in normalized:
            normalized[field_name] = text_items(
                normalized[field_name], field_name, max_item_length
            )

    hypotheses = normalized.get("root_cause_hypotheses")
    if isinstance(hypotheses, list):
        cleaned_hypotheses: list[dict[str, Any]] = []
        seen_hypotheses: set[str] = set()
        for item in hypotheses:
            if not isinstance(item, dict):
                continue
            hypothesis = str(item.get("hypothesis", "")).strip()
            evidence = text_items(item.get("evidence", []), "evidence", 120)[:4]
            if not hypothesis or not evidence or hypothesis.casefold() in seen_hypotheses:
                continue
            seen_hypotheses.add(hypothesis.casefold())
            cleaned_hypotheses.append(
                {**item, "hypothesis": hypothesis[:240], "evidence": evidence}
            )
        normalized["root_cause_hypotheses"] = cleaned_hypotheses[:3]
    return normalized


def deterministic_fallback_report(
    issue: str,
    supporting_data: list[SupportingData],
    supporting_documents: list[SupportingDocument],
    limitations: list[str],
    key_findings: list[str] | None = None,
) -> EightDReport:
    """Build a conservative report from gathered evidence after invalid LLM JSON."""
    fallback_limitations = [
        "The LLM draft could not be validated after one retry; a deterministic fallback was used.",
        "insufficient evidence to confirm a root cause without a validated evidence-linked draft.",
        *limitations,
    ]
    return EightDReport(
        problem_description=issue.strip()[:350] or "An issue was submitted for investigation.",
        answer=_direct_answer(key_findings or [], supporting_documents),
        key_findings=key_findings or [],
        containment_actions=["Review the affected output and contain suspected parts pending human review."],
        root_cause_hypotheses=[],
        corrective_actions=["Review the gathered SQL results and public documents before selecting corrective action."],
        preventive_actions=["Confirm corrective actions with a human reviewer before standardizing them."],
        limitations=list(dict.fromkeys(item[:200] for item in fallback_limitations))[:5],
        supporting_data=supporting_data,
        supporting_documents=supporting_documents,
        human_review_note="Draft for human review.",
    )


def _direct_answer(
    key_findings: list[str], documents: list[SupportingDocument]
) -> str:
    """Build a direct evidence-led answer from Python-computed findings and citations."""
    parts = list(key_findings)
    if documents:
        document_facts = [
            f"[{item.id}] {item.snippet[:180].strip()}"
            for item in documents[:3]
            if item.snippet.strip()
        ]
        if document_facts:
            parts.append("Related public reports: " + " ".join(document_facts))
    else:
        parts.append("No relevant NHTSA document evidence was retrieved.")
    return " ".join(parts) or "No matching data or document evidence was found."


def build_key_findings(
    triage: dict[str, Any], supporting_data: list[SupportingData]
) -> list[str]:
    """Compute concise, numeric findings directly from the fixed SQL result rows."""
    result_by_name = {
        item.id.removeprefix("sql:"): item.rows
        for item in supporting_data
    }
    findings: list[str] = []
    weekly_defects = result_by_name.get("weekly_defects", [])
    if weekly_defects:
        total = sum(int(row.get("defect_count", 0)) for row in weekly_defects)
        part = triage.get("part") or "the selected part"
        location = " ".join(
            value
            for value in (
                str(triage["plant"]) if triage.get("plant") else "",
                f"Line {triage['line']}" if triage.get("line") is not None else "",
            )
            if value
        )
        bounds = " through ".join(
            str(triage[key]) for key in ("start_date", "end_date") if triage.get(key)
        )
        scope = " ".join(value for value in (location, f"during {bounds}" if bounds else "") if value)
        findings.append(f"Defect total: {total} {part} defects {scope}.".strip())
        peak = max(weekly_defects, key=lambda row: int(row.get("defect_count", 0)))
        findings.append(
            f"Peak week starting {str(peak.get('week', 'unknown'))[:10]}: "
            f"{int(peak.get('defect_count', 0))} defects."
        )

    downtime_rows = result_by_name.get("weekly_downtime", [])
    if downtime_rows and triage.get("start_date") and triage.get("end_date"):
        start = date.fromisoformat(str(triage["start_date"]))
        end = date.fromisoformat(str(triage["end_date"]))
        first_week = start - timedelta(days=start.weekday())
        last_week = end - timedelta(days=end.weekday())
        dated_rows: list[tuple[date, int]] = []
        for row in downtime_rows:
            try:
                week = date.fromisoformat(str(row["week"])[:10])
            except (KeyError, ValueError):
                continue
            dated_rows.append((week, int(row.get("downtime_minutes", 0))))
        in_window = [minutes for week, minutes in dated_rows if first_week <= week <= last_week]
        surrounding = [
            minutes for week, minutes in dated_rows if week < first_week or week > last_week
        ]
        if in_window and surrounding:
            window_total = sum(in_window)
            window_average = round(window_total / len(in_window), 1)
            surrounding_average = round(sum(surrounding) / len(surrounding), 1)
            findings.append(
                f"Downtime comparison: {window_total} minutes total across {len(in_window)} "
                f"window weeks ({window_average} per week), versus {surrounding_average} "
                f"minutes per week across {len(surrounding)} surrounding weeks."
            )

    top_parts = result_by_name.get("top_parts", [])
    if top_parts:
        winner = top_parts[0]
        findings.append(
            f"Top part by defect count: {winner.get('part')} with "
            f"{int(winner.get('defect_count', 0))} defects."
        )
    return findings[:6]


def _unsupported_report_numbers(
    report_fields: dict[str, Any], key_findings: list[str]
) -> list[str]:
    """Find numeric claims in prose that are absent from Python-computed findings."""
    allowed = {
        number
        for finding in key_findings
        for number in _report_number_tokens(finding)
    }
    prose: list[str] = []
    for key in (
        "problem_description", "containment_actions", "corrective_actions",
        "preventive_actions", "limitations",
    ):
        value = report_fields.get(key)
        if isinstance(value, str):
            prose.append(value)
        elif isinstance(value, list):
            prose.extend(item for item in value if isinstance(item, str))
    hypotheses = report_fields.get("root_cause_hypotheses", [])
    if isinstance(hypotheses, list):
        prose.extend(
            str(item.get("hypothesis", ""))
            for item in hypotheses
            if isinstance(item, dict)
        )
    unsupported = {
        number
        for item in prose
        for number in _report_number_tokens(item)
        if number not in allowed
    }
    return sorted(unsupported)


def _report_number_tokens(text: str) -> set[str]:
    """Extract metric numbers while excluding dates, years, and known NHTSA IDs."""
    normalized = REPORT_SOURCE_ID_PATTERN.sub(" ", text)
    normalized = REPORT_NHTSA_ID_PATTERN.sub(" ", normalized)
    normalized = ISO_DATE_PATTERN.sub(" ", normalized)
    normalized = DAY_RANGE_PATTERN.sub(" ", normalized)
    normalized = REPORT_MONTH_DAY_PATTERN.sub(" ", normalized)
    normalized = YEAR_PATTERN.sub(" ", normalized)
    return {
        value.replace(",", "")
        for value in REPORT_NUMBER_PATTERN.findall(normalized)
    }


def _parse_date(raw: str) -> date | None:
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(raw.replace(".", ""), fmt).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def triage_issue(issue: str) -> TriageContext:
    """Extract supported part, plant, line, and date details without inventing them."""
    lowered = issue.lower()
    part = next((name for pattern, name in PART_PATTERNS if re.search(pattern, lowered)), None)

    plant = next((name for name in PLANTS if name.lower() in lowered), None)
    if plant is None:
        plant_match = re.search(r"\b(?:at|in)\s+([A-Z][a-z]+(?:[ -]+[A-Z][a-z]+){0,2})", issue)
        if plant_match and plant_match.group(1).lower() not in {"line", "the"}:
            plant = plant_match.group(1)

    line_match = re.search(r"\bline\s*(\d+)\b", issue, flags=re.IGNORECASE)
    dates = [_parse_date(item) for item in ISO_DATE_PATTERN.findall(issue)]
    dates.extend(_parse_date(item) for item in NAMED_DATE_PATTERN.findall(issue))
    parsed_dates = sorted(item for item in dates if item is not None)
    day_range = DAY_RANGE_PATTERN.search(issue)
    if day_range:
        month = datetime.strptime(day_range.group(3)[:3].title(), "%b").month
        year = int(day_range.group(4))
        parsed_dates = sorted(
            (date(year, month, int(day_range.group(1))), date(year, month, int(day_range.group(2))))
        )
    month_year = MONTH_YEAR_PATTERN.search(issue)
    if month_year:
        month_date = datetime.strptime(month_year.group(1)[:3].title(), "%b").date()
        first_day = month_date.replace(year=int(month_year.group(2)), day=1)
        if month_date.month == 12:
            next_month = first_day.replace(year=first_day.year + 1, month=1)
        else:
            next_month = first_day.replace(month=first_day.month + 1)
        month_start, month_end = first_day, next_month - timedelta(days=1)
        if not parsed_dates:
            parsed_dates = [month_start, month_end]
    year_match = YEAR_PATTERN.search(issue)
    if year_match and not parsed_dates:
        year = int(year_match.group(1))
        parsed_dates = [date(year, 1, 1), date(year, 12, 31)]
    wants_top_part = bool(
        re.search(r"\b(?:which|what) part\b", lowered)
        and re.search(r"\b(?:highest|most|top)\b", lowered)
        and re.search(r"\bdefect", lowered)
    )
    return TriageContext(
        part=part,
        plant=plant,
        line=int(line_match.group(1)) if line_match else None,
        start_date=parsed_dates[0] if parsed_dates else None,
        end_date=parsed_dates[-1] if parsed_dates else None,
        top_part_query=wants_top_part,
    )


def _retrieval_query(context: TriageContext) -> str:
    """Build a topical retrieval query from extracted issue attributes."""
    terms = PART_RETRIEVAL_TERMS.get(context.part or "", context.part or "automotive defect")
    return " ".join(
        item
        for item in (
            terms,
            context.plant,
            f"line {context.line}" if context.line is not None else None,
            f"{context.start_date} {context.end_date}" if context.start_date else None,
        )
        if item
    )


def clean_document_snippet(content: str, max_characters: int = 500) -> str:
    """Remove record metadata and raw product dictionaries from a document excerpt."""
    try:
        raw = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        raw = None
    if isinstance(raw, dict):
        content = "\n".join(
            f"{key}: {value}"
            for key, value in raw.items()
            if str(key).casefold().replace("_", " ") not in DOCUMENT_METADATA_KEYS
            and isinstance(value, (str, int, float))
        )

    useful_lines: list[str] = []
    for line in content.splitlines():
        key, separator, _ = line.partition(":")
        normalized_key = key.strip().casefold().replace("_", " ")
        if separator and normalized_key in DOCUMENT_METADATA_KEYS:
            continue
        if line.strip().startswith(("{", "}", "[", "]")):
            continue
        if line.strip():
            useful_lines.append(line.strip())
    return " ".join(useful_lines)[:max_characters]


def estimate_tokens(value: Any) -> int:
    """Estimate token count conservatively from compact JSON or text length."""
    serialized = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    )
    return math.ceil(len(serialized) / 3.5)


def _compact_row_value(value: Any) -> Any:
    """Bound unusual nested SQL values before sending them to the report model."""
    if isinstance(value, str):
        return value[:200]
    if isinstance(value, (dict, list, tuple)):
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        return serialized[:200]
    return value


def _trim_query_rows(query_id: str, rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Trim result rows while retaining every maximum week in the defect trend."""
    if len(rows) <= limit:
        return rows
    if query_id == "sql:weekly_defects" and rows:
        peak = max(float(row.get("defect_count", 0)) for row in rows)
        peak_rows = [row for row in rows if float(row.get("defect_count", 0)) == peak]
        other_rows = [row for row in rows if float(row.get("defect_count", 0)) != peak]
        selected = peak_rows + other_rows[: max(0, limit - len(peak_rows))]
        return sorted(selected, key=lambda row: str(row.get("week", "")))
    return rows[:limit]


def prepare_report_evidence(
    triage: dict[str, Any],
    sql_data: list[SupportingData],
    documents: list[SupportingDocument],
    limitations: list[str],
    token_budget: int = REPORT_EVIDENCE_TOKEN_BUDGET,
    key_findings: list[str] | None = None,
) -> tuple[dict[str, Any], int]:
    """Trim and normalize evidence to the report agent's prompt budget."""
    evidence: dict[str, Any] = {
        "triage": {
            key: triage.get(key)
            for key in ("part", "plant", "line", "start_date", "end_date")
            if triage.get(key) is not None
        },
        "sql_results": [
            {"id": item.id, "rows": _trim_query_rows(item.id, item.rows, 20)}
            for item in sql_data[:3]
        ],
        "documents": [
            {"id": item.id, "snippet": clean_document_snippet(item.snippet, 500)}
            for item in documents[:4]
        ],
        "limitations": [item[:200] for item in limitations[:5]],
        "key_findings": (key_findings or [])[:6],
    }
    for rowset in evidence["sql_results"]:
        rowset["rows"] = [
            {
                str(key)[:80]: _compact_row_value(value)
                for key, value in row.items()
            }
            for row in rowset["rows"]
        ]

    token_count = estimate_tokens(evidence)
    while token_count > token_budget:
        snippets = [item["snippet"] for item in evidence["documents"] if item["snippet"]]
        if snippets and max(map(len, snippets)) > 100:
            for item in evidence["documents"]:
                item["snippet"] = item["snippet"][: max(100, len(item["snippet"]) - 100)]
        elif len(evidence["documents"]) > 1:
            evidence["documents"].pop()
        elif any(len(rowset["rows"]) > 1 for rowset in evidence["sql_results"]):
            for rowset in evidence["sql_results"]:
                if len(rowset["rows"]) > 1:
                    rowset["rows"] = _trim_query_rows(
                        rowset["id"], rowset["rows"], max(1, len(rowset["rows"]) - 5)
                    )
        elif evidence["limitations"]:
            evidence["limitations"].pop()
        else:
            raise ValueError("The required report context exceeds its evidence token budget")
        token_count = estimate_tokens(evidence)
    return evidence, token_count


class InvestigationWorkflow:
    """Run parallel SQL and document retrieval after triage, then draft an 8D report."""

    def __init__(
        self,
        data_agent: DataAgent,
        knowledge_retriever: KnowledgeRetriever,
        llm: LLMProvider,
        top_k_documents: int = 4,
        query_runner: InvestigationQueryRunner | None = None,
    ) -> None:
        self.data_agent = data_agent
        self.knowledge_retriever = knowledge_retriever
        self.llm = llm
        self.top_k_documents = top_k_documents
        engine = getattr(data_agent, "engine", None)
        self.query_runner = query_runner or (
            InvestigationQueryRunner(
                engine,
                row_limit=getattr(data_agent, "row_limit", 100),
                timeout_ms=getattr(data_agent, "timeout_ms", 3000),
            )
            if engine is not None
            else None
        )
        builder = StateGraph(WorkflowState)
        builder.add_node("triage", self._triage_node)
        builder.add_node("data_agent", self._data_agent_node)
        builder.add_node("knowledge_agent", self._knowledge_agent_node)
        builder.add_node("report_agent", self._report_agent_node)
        builder.add_edge(START, "triage")
        builder.add_edge("triage", "data_agent")
        builder.add_edge("data_agent", "knowledge_agent")
        builder.add_edge("knowledge_agent", "report_agent")
        builder.add_edge("report_agent", END)
        self._graph = builder.compile()

    async def run(self, issue: str) -> EightDReport:
        """Run the compiled graph and return its validated 8D report."""
        result = await self._graph.ainvoke({"issue": issue})
        return EightDReport.model_validate(result["report"])

    async def _triage_node(self, state: WorkflowState) -> dict[str, Any]:
        context = triage_issue(state["issue"])
        if context.part and self.query_runner is not None:
            context.part = self.query_runner.resolve_part(context.part)
        triage = context.model_dump(mode="json")
        logger.debug("8D triage output: %s", triage)
        return {"triage": triage}

    async def _data_agent_node(self, state: WorkflowState) -> dict[str, Any]:
        context = TriageContext.model_validate(state["triage"])
        if context.part is None and not context.top_part_query:
            return {
                "supporting_data": [],
                "data_limitation": "No matching part value was found in the production data.",
            }
        if self.query_runner is None:
            return {
                "supporting_data": [],
                "data_limitation": "Fixed investigation SQL is unavailable.",
            }

        result_rows: list[dict[str, Any]] = []
        try:
            query_context = context.model_dump(mode="json")
            query_context["top_part_query"] = context.top_part_query
            results: list[InvestigationQueryResult] = self.query_runner.run_overviews(
                query_context
            )
        except Exception:
            logger.exception("Fixed investigation SQL failed")
            raise
        for result in results:
            logger.debug(
                "Investigation SQL %s: %s (row_count=%d)",
                result.name,
                result.sql,
                len(result.rows),
            )
            result_rows.append(
                SupportingData(
                    id=f"sql:{result.name}",
                    sql=result.sql,
                    parameters=result.parameters,
                    rows=result.rows,
                ).model_dump(mode="json")
            )
        triage = context.model_dump(mode="json")
        top_parts = next((item.rows for item in results if item.name == "top_parts"), [])
        if top_parts:
            triage["part"] = str(top_parts[0]["part"])
            triage["top_part_query"] = False
        limitation = "" if any(item["rows"] for item in result_rows) else "The fixed investigation queries returned no rows."
        return {"supporting_data": result_rows, "data_limitation": limitation, "triage": triage}

    async def _knowledge_agent_node(self, state: WorkflowState) -> dict[str, Any]:
        context = TriageContext.model_validate(state["triage"])
        question = _retrieval_query(context)
        try:
            chunks = self.knowledge_retriever.retrieve(question, top_k=self.top_k_documents)
        except Exception:
            logger.exception("8D knowledge retrieval failed")
            return {
                "supporting_documents": [],
                "knowledge_limitation": "NHTSA documents could not be retrieved.",
            }
        if context.part == "door seal":
            seal_terms = (
                "weatherstrip", "weather strip", "wind noise", "water leak", "water leaks",
                "water entering", "door seal", "door-seal", "seal gap",
            )
            chunks = [
                chunk
                for chunk in chunks
                if any(term in chunk.content.casefold() for term in seal_terms)
                or (
                    "water intrusion" in chunk.content.casefold()
                    and any(
                        term in chunk.content.casefold()
                        for term in ("door", "window", "cabin")
                    )
                    and "tail light" not in chunk.content.casefold()
                )
            ]
        documents = []
        for chunk in chunks:
            snippet = clean_document_snippet(chunk.content, 500)
            if not snippet:
                continue
            documents.append(
                SupportingDocument(
                    id=f"{chunk.source_type}:{chunk.source_id}",
                    source_type=chunk.source_type,
                    snippet=snippet,
                    date=chunk.record_date.isoformat() if chunk.record_date else None,
                ).model_dump(mode="json")
            )
        logger.debug(
            "8D retrieval query=%r top_chunk_ids=%s",
            question,
            [item["id"] for item in documents],
        )
        return {
            "supporting_documents": documents,
            "knowledge_limitation": "" if documents else "No supporting NHTSA documents were retrieved.",
        }

    async def _report_agent_node(self, state: WorkflowState) -> dict[str, Any]:
        raw_sql_data = [SupportingData.model_validate(item) for item in state.get("supporting_data", [])]
        documents = [
            SupportingDocument.model_validate(item)
            for item in state.get("supporting_documents", [])[:4]
        ]
        limitations = [
            item
            for item in (state.get("data_limitation", ""), state.get("knowledge_limitation", ""))
            if item
        ]
        if not any(item.rows for item in raw_sql_data) and not documents:
            limitations.append("insufficient evidence to identify or confirm a root cause.")
        if not raw_sql_data or not any(item.rows for item in raw_sql_data):
            limitations.append("No manufacturing data rows were found for the issue.")
        key_findings = build_key_findings(state.get("triage", {}), raw_sql_data)
        selected_ids = {"sql:weekly_defects", "sql:weekly_downtime", "sql:top_parts"}
        selected_sql_data = [item for item in raw_sql_data if item.id in selected_ids]
        selected_sql_data.extend(item for item in raw_sql_data if item not in selected_sql_data)
        selected_sql_data = selected_sql_data[:3]
        evidence, evidence_tokens = prepare_report_evidence(
            state.get("triage", {}), selected_sql_data, documents, limitations,
            key_findings=key_findings,
        )
        logger.debug("8D estimated report prompt evidence tokens=%d", evidence_tokens)

        sql_by_id = {item.id: item for item in raw_sql_data}
        sql_data = [
            sql_by_id[item["id"]].model_copy(update={"rows": item["rows"]})
            for item in evidence["sql_results"]
        ]
        document_by_id = {item.id: item for item in documents}
        documents = [
            document_by_id[item["id"]].model_copy(update={"snippet": item["snippet"]})
            for item in evidence["documents"]
        ]
        evidence_ids = sorted(
            {item.id for item in sql_data if item.rows} | {item.id for item in documents}
        )

        base_prompt = (
            "Draft a concise automotive 8D report using ONLY the gathered evidence JSON below. "
            "Do not invent causes, measurements, facts, or recall numbers. Every root-cause "
            "hypothesis must cite one or more exact IDs from the allowed evidence IDs. "
            "The Python-generated key_findings are the only source from which you may quote "
            "numbers; do not repeat or derive numbers from raw SQL rows, and do not change them. "
            "If evidence does not support a cause, provide no hypothesis and include the exact "
            "phrase 'insufficient evidence' in limitations. Use at most three short items in each "
            "action section and at most three hypotheses. Keep each item to one short sentence. "
            "This is a draft for human review. Return only JSON matching the provided schema.\n\n"
            f"Issue: {state['issue']}\nAllowed evidence IDs: {evidence_ids}\n"
            f"Gathered evidence: {json.dumps(evidence, ensure_ascii=False, default=str)}"
        )

        validation_error = ""
        for attempt in range(2):
            prompt = base_prompt
            if validation_error:
                prompt += f"\n\nThe prior report failed validation: {validation_error}. Correct it."
            try:
                logger.debug(
                    "8D estimated prompt tokens=%d (evidence=%d)",
                    estimate_tokens(prompt) + estimate_tokens(REPORT_DRAFT_SCHEMA),
                    evidence_tokens,
                )
                raw = await self.llm.generate_json(
                    prompt, REPORT_DRAFT_SCHEMA, max_output_tokens=1800
                )
                payload = normalize_report_payload(json.loads(raw))
                draft = ReportDraft.model_validate(payload)
                report_fields = draft.model_dump()
                unsupported_numbers = _unsupported_report_numbers(report_fields, key_findings)
                if unsupported_numbers:
                    raise ValueError(
                        "The draft contains numbers absent from Python key_findings: "
                        f"{unsupported_numbers}"
                    )
                combined_limitations = list(dict.fromkeys(draft.limitations + limitations))
                combined_limitations = [item[:200] for item in combined_limitations]
                if not evidence_ids and not any("insufficient evidence" in item.lower() for item in combined_limitations):
                    combined_limitations.insert(0, "insufficient evidence to identify or confirm a root cause.")
                if len(combined_limitations) > 5:
                    priority = [
                        item for item in combined_limitations
                        if "insufficient evidence" in item.lower()
                        or "no manufacturing data rows" in item.lower()
                    ]
                    combined_limitations = list(dict.fromkeys(priority + combined_limitations))[:5]
                report_fields["limitations"] = combined_limitations
                report = EightDReport(
                    **report_fields,
                    answer=_direct_answer(key_findings, documents),
                    key_findings=key_findings,
                    supporting_data=sql_data,
                    supporting_documents=documents,
                    human_review_note="Draft for human review.",
                )
                # Validate once more so every hypothesis maps to gathered evidence IDs.
                if any(set(hypothesis.evidence) - set(evidence_ids) for hypothesis in report.root_cause_hypotheses):
                    raise ValueError("A root-cause hypothesis cited an unknown evidence ID")
                return {"report": report.model_dump(mode="json")}
            except (ValueError, ValidationError) as error:
                validation_error = str(error)
                logger.warning(
                    "Report JSON validation failed (attempt %d); finish_reason=%s raw_output=%r error=%s",
                    attempt + 1,
                    getattr(self.llm, "last_finish_reason", None),
                    getattr(self.llm, "last_raw_output", None)
                    if getattr(self.llm, "last_raw_output", None) is not None
                    else raw,
                    error,
                )

        logger.warning(
            "Report validation retries exhausted; returning deterministic fallback from gathered evidence"
        )
        fallback = deterministic_fallback_report(
            state["issue"], sql_data, documents, limitations, key_findings
        )
        return {"report": fallback.model_dump(mode="json")}
