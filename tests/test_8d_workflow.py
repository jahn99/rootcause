"""Schema, workflow, retry, and endpoint tests for Slice 5."""

import asyncio
import json
from datetime import date

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.agents.data_agent import DataAgentResult
from app.api.investigate import get_investigation_workflow
from app.main import app
from app.reports.schema import EightDReport
from app.reports.workflow import InvestigationWorkflow, ReportValidationError, triage_issue


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

    async def ask(self, question: str) -> DataAgentResult:
        self.question = question
        return DataAgentResult(
            sql="SELECT COUNT(*) AS count FROM production_defects LIMIT 100",
            rows=[{"count": 5}],
            answer="Five matching defects were found.",
        )


class FakeChunk:
    source_type = "complaint"
    source_id = "123"
    content = "Owner reported a door seal issue."
    record_date = date(2024, 5, 1)


class FakeRetriever:
    def __init__(self) -> None:
        self.query = ""

    def retrieve(self, query: str, top_k: int = 5) -> list[FakeChunk]:
        self.query = query
        return [FakeChunk()]


class FakeLLM:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or [json.dumps(sample_draft())]
        self.prompts: list[str] = []

    async def generate_json(self, prompt: str, schema: dict[str, object]) -> str:
        self.prompts.append(prompt)
        assert schema["type"] == "object"
        return self.responses.pop(0)

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

    assert "part=door seal" in data_agent.question
    assert "line=3" in retriever.query
    assert report.supporting_data[0].id == "sql:1"
    assert report.supporting_documents[0].id == "complaint:123"
    assert report.root_cause_hypotheses[0].evidence == ["sql:1", "complaint:123"]
    assert "ONLY the gathered evidence" in llm.prompts[0]
    assert "Do not invent causes" in llm.prompts[0]


def test_report_validation_retries_once_then_returns_validated_report() -> None:
    llm = FakeLLM(responses=["{}", json.dumps(sample_draft())])
    workflow, _, _, _ = build_workflow(llm)

    report = asyncio.run(workflow.run("Door-seal defects are spiking on Line 3"))

    assert report.problem_description
    assert len(llm.prompts) == 2
    assert "failed validation" in llm.prompts[1]


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
