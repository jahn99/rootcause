"""LangGraph triage, evidence gathering, and 8D report drafting workflow."""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, ValidationError

from app.agents.data_agent import DataAgent, DataAgentResult
from app.db.knowledge_models import KnowledgeChunk
from app.llm.base import LLMProvider
from app.reports.schema import (
    EightDReport,
    ReportDraft,
    SupportingData,
    SupportingDocument,
)

REPORT_DRAFT_SCHEMA = ReportDraft.model_json_schema()
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

PART_RETRIEVAL_TERMS = {
    "door seal": "door seal weatherstrip water leak wind noise",
    "brake system": "brakes braking system brake fluid stopping distance",
    "airbag": "airbag inflator deployment warning light",
    "seat belt": "seat belt buckle retractor restraint",
    "fuel system": "fuel leak fuel line pump fire",
    "steering system": "steering loss control power steering",
    "electrical system": "electrical wiring short circuit fire",
}


class TriageContext(BaseModel):
    """Issue details extracted before agents gather evidence."""

    model_config = ConfigDict(extra="forbid")

    part: str | None = None
    plant: str | None = None
    line: int | None = None
    start_date: date | None = None
    end_date: date | None = None


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
    return TriageContext(
        part=part,
        plant=plant,
        line=int(line_match.group(1)) if line_match else None,
        start_date=parsed_dates[0] if parsed_dates else None,
        end_date=parsed_dates[-1] if parsed_dates else None,
    )


def _triaged_question(issue: str, context: TriageContext) -> str:
    constraints = [
        f"part={context.part}" if context.part else None,
        f"plant={context.plant}" if context.plant else None,
        f"line={context.line}" if context.line is not None else None,
        f"start_date={context.start_date}" if context.start_date else None,
        f"end_date={context.end_date}" if context.end_date else None,
    ]
    details = ", ".join(item for item in constraints if item) or "no structured filters extracted"
    return f"Issue: {issue}. Triage filters: {details}. Find relevant defect and downtime evidence."


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


class InvestigationWorkflow:
    """Run parallel SQL and document retrieval after triage, then draft an 8D report."""

    def __init__(
        self,
        data_agent: DataAgent,
        knowledge_retriever: KnowledgeRetriever,
        llm: LLMProvider,
        top_k_documents: int = 3,
    ) -> None:
        self.data_agent = data_agent
        self.knowledge_retriever = knowledge_retriever
        self.llm = llm
        self.top_k_documents = top_k_documents
        builder = StateGraph(WorkflowState)
        builder.add_node("triage", self._triage_node)
        builder.add_node("data_agent", self._data_agent_node)
        builder.add_node("knowledge_agent", self._knowledge_agent_node)
        builder.add_node("report_agent", self._report_agent_node)
        builder.add_edge(START, "triage")
        builder.add_edge("triage", "data_agent")
        builder.add_edge("triage", "knowledge_agent")
        builder.add_edge(["data_agent", "knowledge_agent"], "report_agent")
        builder.add_edge("report_agent", END)
        self._graph = builder.compile()

    async def run(self, issue: str) -> EightDReport:
        """Run the compiled graph and return its validated 8D report."""
        result = await self._graph.ainvoke({"issue": issue})
        return EightDReport.model_validate(result["report"])

    @staticmethod
    def _triage_node(state: WorkflowState) -> dict[str, Any]:
        triage = triage_issue(state["issue"]).model_dump(mode="json")
        logger.debug("8D triage output: %s", triage)
        return {"triage": triage}

    async def _data_agent_node(self, state: WorkflowState) -> dict[str, Any]:
        context = TriageContext.model_validate(state["triage"])
        questions = [_triaged_question(state["issue"], context)]
        if context.part and (context.plant is None or context.start_date is None):
            filters = f"part {context.part}"
            questions = [
                f"For {filters}, count production defects grouped by plant and line across the full date range available in the database. Include all dates and order by count descending.",
                f"For {filters} on line {context.line if context.line is not None else 'all lines'}, show weekly defect counts across the full date range available in the database. Include all dates.",
                f"For {filters}, calculate average severity and defect count by plant and line across the full date range available in the database. Include all dates.",
                f"Show downtime minutes by plant, line, and reason related to {filters} across the full date range available in the database. Include all dates.",
            ]

        gathered: list[dict[str, Any]] = []
        limitations: list[str] = []
        for index, question in enumerate(questions, start=1):
            try:
                result: DataAgentResult = await self.data_agent.ask(question)
            except Exception:
                logger.exception("8D SQL overview query %d failed", index)
                limitations.append(f"SQL overview query {index} could not be completed.")
                continue
            if result.refused or result.sql is None:
                limitations.append(f"SQL overview query {index} was unavailable.")
                continue
            logger.debug("8D SQL query %d: %s (row_count=%d)", index, result.sql, len(result.rows))
            if len(result.rows) > 20:
                limitations.append(f"SQL overview query {index} was limited to the first 20 rows for the report.")
            gathered.append(
                SupportingData(
                    id=f"sql:{index}", sql=result.sql, rows=result.rows[:20]
                ).model_dump(mode="json")
            )
            if not result.rows:
                limitations.append(f"SQL overview query {index} returned no results.")

        if not gathered:
            limitations.append("No SQL results were available for this issue.")
        return {"supporting_data": gathered, "data_limitation": " ".join(limitations)}

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
        documents = [
            SupportingDocument(
                id=f"{chunk.source_type}:{chunk.source_id}",
                source_type=chunk.source_type,
                snippet=chunk.content[:300],
                date=chunk.record_date.isoformat() if chunk.record_date else None,
            ).model_dump(mode="json")
            for chunk in chunks
        ]
        logger.debug(
            "8D retrieval query=%r top_chunks=%s",
            question,
            [
                {"id": item["id"], "snippet": item["snippet"][:180]}
                for item in documents
            ],
        )
        return {
            "supporting_documents": documents,
            "knowledge_limitation": "" if documents else "No supporting NHTSA documents were retrieved.",
        }

    async def _report_agent_node(self, state: WorkflowState) -> dict[str, Any]:
        sql_data = [SupportingData.model_validate(item) for item in state.get("supporting_data", [])]
        documents = [
            SupportingDocument.model_validate(item)
            for item in state.get("supporting_documents", [])
        ]
        evidence_ids = sorted(
            {item.id for item in sql_data if item.rows} | {item.id for item in documents}
        )
        evidence = {
            "triage": state.get("triage", {}),
            "sql_results": [
                {"id": item.id, "rows": item.rows}
                for item in sql_data
            ],
            "nhtsa_documents": [item.model_dump(mode="json") for item in documents],
        }
        limitations = [
            item
            for item in (state.get("data_limitation", ""), state.get("knowledge_limitation", ""))
            if item
        ]
        if not evidence_ids:
            limitations.append("insufficient evidence to identify or confirm a root cause.")
        if not sql_data or not any(item.rows for item in sql_data):
            limitations.append("No manufacturing data rows were found for the issue.")
        evidence["limitations"] = limitations

        base_prompt = (
            "Draft a concise automotive 8D report using ONLY the gathered evidence JSON below. "
            "Do not invent causes, measurements, facts, or recall numbers. Every root-cause "
            "hypothesis must cite one or more exact IDs from the allowed evidence IDs. "
            "If evidence does not support a cause, provide no hypothesis and include the exact "
            "phrase 'insufficient evidence' in limitations. Keep actions evidence-based. "
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
                raw = await self.llm.generate_json(prompt, REPORT_DRAFT_SCHEMA)
                draft = ReportDraft.model_validate_json(raw)
                report_fields = draft.model_dump()
                report_fields["limitations"] = list(dict.fromkeys(draft.limitations + limitations))
                report = EightDReport(
                    **report_fields,
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

        raise ReportValidationError(
            "The 8D report did not pass validation after one retry. Please try again."
        )
