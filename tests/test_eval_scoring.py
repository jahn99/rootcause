"""Tests for deterministic evaluation scoring and failure explanations."""

from eval.scoring import sql_expected_values_match
from scripts.run_eval import _failure_reason, _render_failure_report, _score, _summary


def test_sql_expected_values_normalize_commas_and_accept_two_decimal_rounding() -> None:
    score = sql_expected_values_match(
        [1332, 3.206, 2.697],
        [{"count": "1,332", "first_avg": 3.21, "second_avg": 2.70}],
    )

    assert score["number_match"] is True
    assert score["missing_numbers"] == []


def test_sql_explicit_values_still_fail_for_wrong_answer() -> None:
    question = {
        "category": "sql",
        "check": {"expected_values": [133]},
    }

    score = _score(question, {"rows": [{"defect_count": 0}]})

    assert score["number_match"] is False
    assert score["missing_numbers"] == [133]
    assert "133" in _failure_reason({"score": score})


def test_failure_report_explains_answer_vs_source_keyword_results() -> None:
    questions = [{
        "id": "Q03",
        "category": "doc",
        "question": "What complaints describe water leaking into the vehicle cabin?",
        "check": {"expected_keywords": ["water", "leak"]},
    }]
    rows = [{
        "id": "Q03",
        "category": "doc",
        "status": "completed",
        "response": {
            "answer": "not found in the documents",
            "sources": [{"id": "complaint:Q3", "snippet": "Water leaked into the vehicle cabin."}],
        },
        "score": {
            "status": "scored",
            "keyword_hit_rate": 0.0,
            "source_keyword_hit": True,
            "matched_keywords": [],
        },
    }]

    report = _render_failure_report(rows, questions)

    assert "answer keyword hit rate was 0%" in report.casefold()
    assert "Cited-source keyword hit: True" in report
    assert "not found in the documents" in report
    assert "Root cause category:** LLM output" in report


def test_eval_summary_counts_scored_failures_not_only_request_errors() -> None:
    rows = [
        {"status": "completed", "score": {"status": "scored", "number_match": False}},
        {"status": "completed", "score": {"status": "scored", "number_match": True}},
    ]

    assert _summary(rows, {})["failure_count"] == 1
