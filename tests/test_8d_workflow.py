"""Schema, workflow, retry, and endpoint tests for Slice 5."""

import asyncio
import json
from datetime import date

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.agents.data_agent import DataAgentResult, SQLGenerationError
from app.api.investigate import get_investigation_workflow
from app.main import app
from app.reports.schema import EightDReport, SupportingData, SupportingDocument
from app.reports.workflow import (
    REPORT_EVIDENCE_TOKEN_BUDGET,
    InvestigationWorkflow,
    ReportValidationError,
    estimate_tokens,
    prepare_report_evidence,
    triage_issue,
)


def sample_draft() -> dict[str, object]:
    return {
        "problem_description": "Door-seal defects increased on Line 3.",
        "containment_actions": ["Inspect affected door seals."],
        "root_cause_hypotheses": [
            {
                "hypothesis": "Seal fit variation may be contributing to the reported defects.",
                "confidence_level": "low",
                "evidence": ["sql:1", "complaint:123"],
            }
        ],
        "corrective_actions": ["Verify seal installation settings."],
        "preventive_actions": ["Monitor the defect trend after adjustments."],
        "limitations": ["The available evidence does not establish a confirmed root cause."],
    }


def report_payload(**overrides: object) -> dict[str, object]:
    payload = {
        **sample_draft(),
        "supporting_data": [
            {
                "id": "sql:1",
                "sql": "SELECT COUNT(*) FROM production_defects LIMIT 100",
                "rows": [{"count": 5}],
            }
        ],
        "supporting_documents": [
            {
                "id": "complaint:123",
                "source_type": "complaint",
                "snippet": "Owner reported a door seal issue.",
                "date": "2024-05-01",
            }
        ],
        "human_review_note": "Draft for human review.",
    }
    payload.update(overrides)
    return payload


class FakeDataAgent:
    def __init__(self) -> None:
        self.question = ""
        self.questions: list[str] = []

    async def ask(self, question: str) -> DataAgentResult:
        self.question = question
        self.questions.append(question)
        return DataAgentResult(
            sql="SELECT COUNT(*) AS count FROM production_defects LIMIT 100",
            rows=[{"count": 5}],
            answer="Five matching defects were found.",
        )


class FakeChunk:
    source_type = "complaint"
    source_id = "123"
    content = "Owner reported a door seal issue with wind noise."
    record_date = date(2024, 5, 1)


class IrrelevantLockChunk:
    source_type = "complaint"
    source_id = "lock-1"
    content = "Door lock actuator clicks and fails to unlock."
    record_date = date(2024, 5, 1)


class IrrelevantWiringChunk:
    source_type = "complaint"
    source_id = "wiring-1"
    content = "Camera wiring intermittently loses power."
    record_date = date(2024, 5, 1)


class FakeRetriever:
    def __init__(self) -> None:
        self.query = ""

    def retrieve(self, query: str, top_k: int = 5) -> list[FakeChunk]:
        self.query = query
        return [FakeChunk(), IrrelevantLockChunk(), IrrelevantWiringChunk()]


class FakeLLM:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or [json.dumps(sample_draft())]
        self.prompts: list[str] = []
        self.last_raw_output: str | None = None
        self.last_finish_reason: str | None = "stop"

    async def generate_json(
        self, prompt: str, schema: dict[str, object], *, max_output_tokens: int = 600
    ) -> str:
        self.prompts.append(prompt)
        assert max_output_tokens == 1800
        assert schema["type"] == "object"
        self.last_raw_output = self.responses.pop(0)
        return self.last_raw_output

    async def generate(self, prompt: str) -> str:
        raise AssertionError("The report workflow should use structured JSON output only")


def build_workflow(
    llm: FakeLLM | None = None,
) -> tuple[InvestigationWorkflow, FakeDataAgent, FakeRetriever, FakeLLM]:
    data_agent = FakeDataAgent()
    retriever = FakeRetriever()
    provider = llm or FakeLLM()
    return InvestigationWorkflow(data_agent, retriever, provider), data_agent, retriever, provider


def test_8d_schema_requires_citations_to_gathered_evidence() -> None:
    report = EightDReport.model_validate(report_payload())

    assert report.human_review_note == "Draft for human review."
    assert report.root_cause_hypotheses[0].confidence_level == "low"

    unsupported = report_payload()
    unsupported["root_cause_hypotheses"] = [
        {
            "hypothesis": "Unsupported cause",
            "confidence_level": "high",
            "evidence": ["recall:invented"],
        }
    ]
    with pytest.raises(ValidationError, match="gathered evidence IDs"):
        EightDReport.model_validate(unsupported)


def test_8d_schema_requires_insufficient_evidence_limitation_when_empty() -> None:
    payload = report_payload(
        root_cause_hypotheses=[], supporting_data=[], supporting_documents=[], limitations=[]
    )

    with pytest.raises(ValidationError, match="insufficient evidence"):
        EightDReport.model_validate(payload)


def test_triage_extracts_part_line_plant_and_date_range() -> None:
    result = triage_issue(
        "Door-seal defects increased at Cedar Falls Line 3 between 2025-05-05 and 2025-05-18."
    )

    assert result.part == "door seal"
    assert result.plant == "Cedar Falls"
    assert result.line == 3
    assert result.start_date == date(2025, 5, 5)
    assert result.end_date == date(2025, 5, 18)


def test_workflow_gathers_both_sources_and_uses_only_evidence() -> None:
    workflow, data_agent, retriever, llm = build_workflow()
    report = asyncio.run(workflow.run("Door-seal defects are spiking on Line 3"))

    assert "part door seal" in data_agent.question
    assert "line 3" in retriever.query
    assert report.supporting_data[0].id == "sql:1"
    assert report.supporting_documents[0].id == "complaint:123"
    assert report.root_cause_hypotheses[0].evidence == ["sql:1", "complaint:123"]
    assert "ONLY the gathered evidence" in llm.prompts[0]
    assert "Do not invent causes" in llm.prompts[0]


def test_workflow_without_plant_or_dates_runs_part_overviews() -> None:
    workflow, data_agent, retriever, _ = build_workflow()
    report = asyncio.run(workflow.run("Door-seal defects are spiking on Line 3"))

    assert len(data_agent.questions) == 3
    assert any("grouped by plant and line" in item for item in data_agent.questions)
    assert any("weekly defect counts" in item for item in data_agent.questions)
    assert any("average severity" in item for item in data_agent.questions)
    assert any("downtime minutes" in item for item in data_agent.questions)
    assert "door seal weatherstrip water leak wind noise" in retriever.query
    assert len(report.supporting_data) == 3
    assert report.supporting_documents[0].id == "complaint:123"


def test_month_only_triage_and_investigation_input() -> None:
    workflow, data_agent, retriever, _ = build_workflow()
    issue = "Door-seal defects are spiking on Line 3 at Cedar Falls in May 2025"
    triage = triage_issue(issue)
    report = asyncio.run(workflow.run(issue))

    assert triage.plant == "Cedar Falls"
    assert triage.start_date == date(2025, 5, 1)
    assert triage.end_date == date(2025, 5, 31)
    assert len(data_agent.questions) >= 1
    assert "start_date=2025-05-01" in data_agent.question
    assert "end_date=2025-05-31" in data_agent.question
    assert "door seal weatherstrip water leak wind noise" in retriever.query
    assert report.supporting_data


def test_no_results_are_reported_as_limitations() -> None:
    class EmptyDataAgent:
        async def ask(self, question: str) -> DataAgentResult:
            return DataAgentResult(sql="SELECT COUNT(*) FROM production_defects", rows=[], answer="No records.")

    class EmptyRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[FakeChunk]:
            return []

    draft = sample_draft()
    draft["root_cause_hypotheses"] = []
    draft["limitations"] = []
    workflow = InvestigationWorkflow(
        EmptyDataAgent(), EmptyRetriever(), FakeLLM([json.dumps(draft)])  # type: ignore[arg-type]
    )
    report = asyncio.run(workflow.run("Door-seal defects are spiking on Line 3"))

    assert not report.supporting_data or not any(item.rows for item in report.supporting_data)
    assert any("no manufacturing data rows" in item.lower() for item in report.limitations)
    assert any("insufficient evidence" in item.lower() for item in report.limitations)


def test_oversized_evidence_is_compacted_to_token_budget() -> None:
    sql_data = [
        SupportingData(
            id=f"sql:{query_index}",
            sql="SELECT ...",
            rows=[{"plant": "Cedar Falls", "details": "x" * 200} for _ in range(30)],
        )
        for query_index in range(5)
    ]
    documents = [
        SupportingDocument(
            id=f"complaint:{index}",
            source_type="complaint",
            snippet=("Vehicle: large raw product dictionary " + "x" * 1000 + "\nSummary: seal wind noise " + "y" * 800),
            date="2025-01-01",
        )
        for index in range(7)
    ]

    evidence, token_count = prepare_report_evidence(
        {"part": "door seal", "plant": "Cedar Falls", "unused": "z" * 1000},
        sql_data,
        documents,
        ["limitation"] * 8,
    )

    assert token_count == estimate_tokens(evidence)
    assert token_count <= REPORT_EVIDENCE_TOKEN_BUDGET
    assert len(evidence["sql_results"]) <= 3
    assert all(len(item["rows"]) <= 20 for item in evidence["sql_results"])
    assert len(evidence["documents"]) <= 4
    assert all(len(item["snippet"]) <= 500 for item in evidence["documents"])
    assert "source_type" not in evidence["documents"][0]
    assert "date" not in evidence["documents"][0]
    assert "unused" not in evidence["triage"]


def test_report_validation_retries_once_then_returns_validated_report(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = FakeLLM(responses=["", json.dumps(sample_draft())])
    workflow, _, _, _ = build_workflow(llm)

    report = asyncio.run(workflow.run("Door-seal defects are spiking on Line 3"))

    assert report.problem_description
    assert len(llm.prompts) == 2
    assert "failed validation" in llm.prompts[1]
    assert "finish_reason=stop" in caplog.text
    assert "raw_output=''" in caplog.text


def test_report_validation_stops_after_one_retry() -> None:
    llm = FakeLLM(responses=["{}", "{}"])
    workflow, _, _, _ = build_workflow(llm)

    with pytest.raises(ReportValidationError, match="after one retry"):
        asyncio.run(workflow.run("Door-seal defects are spiking on Line 3"))
    assert len(llm.prompts) == 2


def test_investigate_endpoint_end_to_end_with_fake_agents_and_llm() -> None:
    workflow, _, _, _ = build_workflow()
    app.dependency_overrides[get_investigation_workflow] = lambda: workflow
    try:
        response = TestClient(app).post(
            "/investigate", json={"issue": "Door-seal defects are spiking on Line 3"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["problem_description"] == "Door-seal defects increased on Line 3."
    assert body["supporting_data"][0]["rows"] == [{"count": 5}]
    assert body["supporting_documents"][0]["id"] == "complaint:123"
    assert body["human_review_note"] == "Draft for human review."


def test_investigate_unexpected_error_returns_clear_json() -> None:
    class BrokenWorkflow:
        async def run(self, issue: str) -> EightDReport:
            raise RuntimeError("simulated unexpected failure")

    app.dependency_overrides[get_investigation_workflow] = lambda: BrokenWorkflow()
    try:
        response = TestClient(app).post(
            "/investigate", json={"issue": "Door-seal defects at Cedar Falls in May 2025"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Investigation failed unexpectedly. The error was logged; please retry."
    }


def test_investigate_provider_context_error_returns_clear_json() -> None:
    class LargeContextWorkflow:
        async def run(self, issue: str) -> EightDReport:
            raise RuntimeError("Error 413: request too large for model")

    app.dependency_overrides[get_investigation_workflow] = lambda: LargeContextWorkflow()
    try:
        response = TestClient(app).post(
            "/investigate", json={"issue": "Door-seal defects at Cedar Falls in May 2025"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert "could not process the report context" in response.json()["detail"]


def test_investigate_sql_retry_exhaustion_returns_clear_json() -> None:
    class InvalidSQLWorkflow:
        async def run(self, issue: str) -> EightDReport:
            raise SQLGenerationError("The SQL agent could not produce a valid query after one retry.")

    app.dependency_overrides[get_investigation_workflow] = lambda: InvalidSQLWorkflow()
    try:
        response = TestClient(app).post(
            "/investigate", json={"issue": "Door-seal defects are spiking on Line 3"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json()["detail"].endswith("after one retry.")
