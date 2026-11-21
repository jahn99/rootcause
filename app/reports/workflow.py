"""LangGraph triage, evidence gathering, and 8D report drafting workflow."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
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


class InvestigationWorkflow:
    """Run parallel SQL and document retrieval after triage, then draft an 8D report."""

    def __init__(
        self,
        data_agent: DataAgent,
        knowledge_retriever: KnowledgeRetriever,
        llm: LLMProvider,
        top_k_documents: int = 5,
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
        return {"triage": triage_issue(state["issue"]).model_dump(mode="json")}

    async def _data_agent_node(self, state: WorkflowState) -> dict[str, Any]:
        context = TriageContext.model_validate(state["triage"])
        question = _triaged_question(state["issue"], context)
        try:
            result: DataAgentResult = await self.data_agent.ask(question)
        except Exception:
            return {"supporting_data": [], "data_limitation": "SQL data could not be gathered."}
        if result.refused or result.sql is None:
            return {"supporting_data": [], "data_limitation": "SQL data was unavailable for this issue."}
        return {
            "supporting_data": [
                SupportingData(id="sql:1", sql=result.sql, rows=result.rows).model_dump(mode="json")
            ],
            "data_limitation": "" if result.rows else "The SQL query returned no supporting rows.",
        }

    async def _knowledge_agent_node(self, state: WorkflowState) -> dict[str, Any]:
        context = TriageContext.model_validate(state["triage"])
        question = _triaged_question(state["issue"], context)
        try:
            chunks = self.knowledge_retriever.retrieve(question, top_k=self.top_k_documents)
        except Exception:
            return {
                "supporting_documents": [],
                "knowledge_limitation": "NHTSA documents could not be retrieved.",
            }
        documents = [
            SupportingDocument(
                id=f"{chunk.source_type}:{chunk.source_id}",
                source_type=chunk.source_type,
                snippet=chunk.content[:500],
                date=chunk.record_date.isoformat() if chunk.record_date else None,
            ).model_dump(mode="json")
            for chunk in chunks
        ]
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
            "sql_results": [item.model_dump(mode="json") for item in sql_data],
            "nhtsa_documents": [item.model_dump(mode="json") for item in documents],
        }
        limitations = [
            item
            for item in (state.get("data_limitation", ""), state.get("knowledge_limitation", ""))
            if item
        ]
        if not evidence_ids:
            limitations.append("insufficient evidence to identify or confirm a root cause.")

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
                report = EightDReport(
                    **draft.model_dump(),
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
