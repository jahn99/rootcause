"""Schema, workflow, retry, and endpoint tests for Slice 5."""

import asyncio
import json
from datetime import date

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.agents.data_agent import DataAgentResult, SQLGenerationError
from app.agents.investigation_queries import InvestigationQueryResult
from app.api.investigate import get_investigation_workflow
from app.main import app
from app.reports.schema import EightDReport, SupportingData, SupportingDocument
from app.reports.workflow import (
    REPORT_EVIDENCE_TOKEN_BUDGET,
    InvestigationWorkflow,
    build_key_findings,
    deterministic_fallback_report,
    estimate_tokens,
    normalize_report_payload,
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
                "evidence": ["sql:weekly_defects", "complaint:123"],
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
                "id": "sql:weekly_defects",
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


class FakeInvestigationQueryRunner:
    def __init__(self, empty: bool = False) -> None:
        self.context: dict[str, object] = {}
        self.empty = empty

    def resolve_part(self, requested: str) -> str:
        assert requested == "door seal"
        return "door seal"

    def run_overviews(self, context: dict[str, object]) -> list[InvestigationQueryResult]:
        self.context = context
        if context.get("top_part_query"):
            return [
                InvestigationQueryResult(
                    "top_parts",
                    "SELECT part, COUNT(*) FROM production_defects GROUP BY part",
                    {},
                    [] if self.empty else [{"part": "door seal", "defect_count": 457}],
                )
            ]
        rows = [] if self.empty else [{"count": 5}]
        return [
            InvestigationQueryResult(
                name="weekly_defects",
                sql="SELECT date_trunc('week', date) FROM production_defects WHERE part = %(part)s",
                parameters={"part": "door seal"},
                rows=[] if self.empty else [
                    {"week": "2025-05-05", "defect_count": 42},
                    {"week": "2025-05-12", "defect_count": 91},
                ],
            ),
            InvestigationQueryResult("defects_by_plant_line", "SELECT plant FROM production_defects", {"part": "door seal"}, rows),
            InvestigationQueryResult("severity_comparison", "SELECT AVG(severity) FROM production_defects", {"part": "door seal"}, rows),
            InvestigationQueryResult("weekly_downtime", "SELECT date_trunc('week', date) FROM downtime_events", {}, rows),
            InvestigationQueryResult(
                "top_parts",
                "SELECT part, COUNT(*) FROM production_defects GROUP BY part",
                {},
                [] if self.empty else [{"part": "door seal", "defect_count": 457}],
            ),
        ]


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
) -> tuple[InvestigationWorkflow, FakeInvestigationQueryRunner, FakeRetriever, FakeLLM]:
    data_agent = FakeDataAgent()
    query_runner = FakeInvestigationQueryRunner()
    retriever = FakeRetriever()
    provider = llm or FakeLLM()
    return (
        InvestigationWorkflow(data_agent, retriever, provider, query_runner=query_runner),
        query_runner,
        retriever,
        provider,
    )


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


def test_triage_extracts_natural_language_day_range() -> None:
    result = triage_issue(
        "Door-seal defects increased at Cedar Falls Line 3 between 5 and 18 May 2025."
    )

    assert result.start_date == date(2025, 5, 5)
    assert result.end_date == date(2025, 5, 18)


def test_workflow_gathers_both_sources_and_uses_only_evidence() -> None:
    workflow, query_runner, retriever, llm = build_workflow()
    report = asyncio.run(workflow.run("Door-seal defects are spiking on Line 3"))

    assert query_runner.context["part"] == "door seal"
    assert "line 3" in retriever.query
    assert report.supporting_data[0].id == "sql:weekly_defects"
    assert report.supporting_documents[0].id == "complaint:123"
    assert any("133 door seal defects" in item for item in report.key_findings)
    assert "133" in report.answer
    assert "key_findings" in llm.prompts[0]
    assert report.root_cause_hypotheses[0].evidence == ["sql:weekly_defects", "complaint:123"]
    assert "ONLY the gathered evidence" in llm.prompts[0]
    assert "Do not invent causes" in llm.prompts[0]


def test_workflow_without_plant_or_dates_runs_part_overviews() -> None:
    workflow, query_runner, retriever, _ = build_workflow()
    report = asyncio.run(workflow.run("Door-seal defects are spiking on Line 3"))

    assert query_runner.context == {
        "part": "door seal", "plant": None, "line": 3,
        "start_date": None, "end_date": None,
        "top_part_query": False,
    }
    assert "door seal weatherstrip water leak wind noise" in retriever.query
    assert len(report.supporting_data) == 3
    assert report.supporting_documents[0].id == "complaint:123"


def test_month_only_triage_and_investigation_input() -> None:
    workflow, query_runner, retriever, _ = build_workflow()
    issue = "Door-seal defects are spiking on Line 3 at Cedar Falls in May 2025"
    triage = triage_issue(issue)
    report = asyncio.run(workflow.run(issue))

    assert triage.plant == "Cedar Falls"
    assert triage.start_date == date(2025, 5, 1)
    assert triage.end_date == date(2025, 5, 31)
    assert query_runner.context["start_date"] == "2025-05-01"
    assert query_runner.context["end_date"] == "2025-05-31"
    assert "door seal weatherstrip water leak wind noise" in retriever.query
    assert report.supporting_data


def test_top_part_query_wins_triage_part_for_document_retrieval() -> None:
    draft = sample_draft()
    draft["problem_description"] = "The highest-count production part was identified."
    draft["root_cause_hypotheses"][0]["evidence"] = ["sql:top_parts", "complaint:123"]
    workflow, query_runner, retriever, _ = build_workflow(FakeLLM([json.dumps(draft)]))
    report = asyncio.run(
        workflow.run(
            "Which part had the highest production defect count across all plants and lines during 2025?"
        )
    )

    assert query_runner.context["top_part_query"] is True
    assert report.supporting_data[0].id == "sql:top_parts"
    assert "door seal weatherstrip water leak wind noise" in retriever.query


def test_key_findings_cover_window_total_peak_downtime_and_top_part() -> None:
    sql_data = [
        SupportingData(
            id="sql:weekly_defects",
            sql="SELECT week, defect_count FROM production_defects",
            rows=[
                {"week": "2025-05-05", "defect_count": 42},
                {"week": "2025-05-12", "defect_count": 91},
            ],
        ),
        SupportingData(
            id="sql:weekly_downtime",
            sql="SELECT week, downtime_minutes FROM downtime_events",
            rows=[
                {"week": "2025-04-21", "downtime_minutes": 100},
                {"week": "2025-04-28", "downtime_minutes": 120},
                {"week": "2025-05-05", "downtime_minutes": 259},
                {"week": "2025-05-12", "downtime_minutes": 455},
                {"week": "2025-05-19", "downtime_minutes": 300},
                {"week": "2025-05-26", "downtime_minutes": 200},
            ],
        ),
        SupportingData(
            id="sql:top_parts",
            sql="SELECT part, COUNT(*) FROM production_defects GROUP BY part",
            rows=[{"part": "door seal", "defect_count": 457}],
        ),
    ]
    findings = build_key_findings(
        {
            "part": "door seal", "plant": "Cedar Falls", "line": 3,
            "start_date": "2025-05-05", "end_date": "2025-05-18",
        },
        sql_data,
    )

    assert any("133 door seal defects" in item for item in findings)
    assert any("2025-05-12: 91 defects" in item for item in findings)
    assert any("714 minutes total" in item and "180.0 minutes per week" in item for item in findings)
    assert any("door seal with 457 defects" in item for item in findings)


def test_deterministic_fallback_keeps_key_findings_and_relevant_documents() -> None:
    findings = ["Peak week starting 2025-05-12: 91 defects."]
    document = SupportingDocument(
        id="complaint:public-1", source_type="complaint",
        snippet="The owner reported water leaking past the door seal.", date=None,
    )
    report = deterministic_fallback_report("Door seal failures", [], [document], [], findings)

    assert report.key_findings == findings
    assert "91" in report.answer
    assert "complaint:public-1" in report.answer
    assert report.supporting_documents == [document]


def test_report_retries_unsupported_numbers_then_fallback_keeps_findings_and_sources() -> None:
    draft = sample_draft()
    draft["problem_description"] = "The line logged 999 defects."
    llm = FakeLLM([json.dumps(draft), json.dumps(draft)])
    workflow, _, _, _ = build_workflow(llm)

    report = asyncio.run(workflow.run("Door-seal defects are spiking on Line 3"))

    assert len(llm.prompts) == 2
    assert report.key_findings
    assert "133" in report.answer
    assert report.supporting_documents[0].id == "complaint:123"
    assert any("LLM draft could not be validated" in item for item in report.limitations)


def test_no_results_are_reported_as_limitations() -> None:
    class EmptyRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[FakeChunk]:
            return []

    draft = sample_draft()
    draft["problem_description"] = "No matching manufacturing rows were found."
    draft["root_cause_hypotheses"] = []
    draft["limitations"] = []
    workflow = InvestigationWorkflow(
        FakeDataAgent(),
        EmptyRetriever(),
        FakeLLM([json.dumps(draft)]),
        query_runner=FakeInvestigationQueryRunner(empty=True),
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
    assert len(evidence["sql_results"]) <= 4
    assert all(len(item["rows"]) <= 20 for item in evidence["sql_results"])
    assert len(evidence["documents"]) <= 4
    assert all(len(item["snippet"]) <= 500 for item in evidence["documents"])
    assert "source_type" not in evidence["documents"][0]
    assert "date" not in evidence["documents"][0]
    assert "unused" not in evidence["triage"]


def test_weekly_trend_compaction_keeps_peak_week() -> None:
    weekly_rows = [
        {"week": f"2025-{index:02d}-01", "defect_count": index}
        for index in range(1, 27)
    ]
    weekly = SupportingData(
        id="sql:weekly_defects", sql="SELECT ...", rows=weekly_rows
    )

    evidence, _ = prepare_report_evidence({}, [weekly], [], [], token_budget=300)

    retained = evidence["sql_results"][0]["rows"]
    assert len(retained) == 20
    assert {item["defect_count"] for item in retained} >= {26}


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

    report = asyncio.run(workflow.run("Door-seal defects are spiking on Line 3"))

    assert len(llm.prompts) == 2
    assert report.supporting_data
    assert report.supporting_documents
    assert report.root_cause_hypotheses == []
    assert any("LLM draft could not be validated" in item for item in report.limitations)
    assert any("insufficient evidence" in item for item in report.limitations)


def test_report_list_normalization_repairs_noise_and_bounds_items() -> None:
    payload = sample_draft()
    payload["preventive_actions"] = [
        "preventive_actions",
        ":",
        ["  Monitor the trend.  ", "", "Monitor the trend."],
        "x" * 250,
        "discard after maximum length",
    ]

    normalized = normalize_report_payload(payload)

    assert normalized["preventive_actions"] == [
        "Monitor the trend.",
        "x" * 200,
        "discard after maximum length",
    ]


def test_report_workflow_accepts_normalized_lists_before_validation() -> None:
    draft = sample_draft()
    draft["preventive_actions"] = ["preventive_actions", ":", ["Monitor the trend.", ""]]
    workflow, _, _, llm = build_workflow(FakeLLM(responses=[json.dumps(draft)]))

    report = asyncio.run(workflow.run("Door-seal defects are spiking on Line 3"))

    assert report.preventive_actions == ["Monitor the trend."]
    assert len(llm.prompts) == 1


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
    assert body["supporting_data"][0]["rows"] == [
        {"week": "2025-05-05", "defect_count": 42},
        {"week": "2025-05-12", "defect_count": 91},
    ]
    assert len(body["supporting_data"]) == 3
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
