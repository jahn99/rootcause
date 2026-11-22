"""Run the saved question set against the app and write evaluation results."""

from __future__ import annotations

import json
import argparse
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy.exc import SQLAlchemyError  # noqa: E402
from eval.scoring import (  # noqa: E402
    combined_score,
    keyword_score,
    safety_score,
    sql_expected_values_match,
    sql_number_match,
)

if TYPE_CHECKING:
    from fastapi.testclient import TestClient  # noqa: E402

QUESTION_PATH = ROOT / "eval" / "questions.json"
RESULT_PATH = ROOT / "eval" / "results.json"
FAILURE_REPORT_PATH = ROOT / "eval" / "failure_report.md"
RETRIEVAL_MODES = ("vector", "bm25", "hybrid")


def _runtime_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Load app runtime lazily so offline report generation never loads .env."""
    from fastapi.testclient import TestClient

    from app.api.ask import get_retriever
    from app.config import get_settings
    from app.db.session import create_database_engine
    from app.main import app
    from app.retrieval.search import HybridRetriever

    return TestClient, app, get_retriever, get_settings, create_database_engine, HybridRetriever


def _check(question: dict[str, Any]) -> dict[str, Any]:
    """Support the project question format where scoring fields live in check."""
    value = question.get("check", question)
    return value if isinstance(value, dict) else {}


def _is_rate_limit(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    return status == 429 or getattr(response, "status_code", None) == 429 or "429" in str(error)


def _post_with_backoff(client: TestClient, path: str, body: dict[str, Any]) -> dict[str, Any]:
    """Post through FastAPI endpoint logic, retrying provider rate limits."""
    delay = float(os.getenv("EVAL_REQUEST_DELAY_SECONDS", "1"))
    for attempt in range(4):
        if delay > 0:
            time.sleep(delay)
        try:
            response = client.post(path, json=body)
            if response.status_code == 429:
                if attempt == 3:
                    response.raise_for_status()
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as error:
            if not _is_rate_limit(error) or attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Evaluation request exhausted rate-limit retries")


def _request(
    client: TestClient,
    question: dict[str, Any],
    mode: str = "hybrid",
) -> dict[str, Any]:
    category = question["category"]
    text = question["question"]
    if category == "doc":
        return _post_with_backoff(client, "/ask", {"question": text})
    if category == "sql":
        return _post_with_backoff(client, "/sql-agent", {"question": text})
    if category == "combined":
        return _post_with_backoff(client, "/investigate", {"issue": text})
    if category == "safety":
        # Mutation/credential probes go to the guarded SQL agent; unsupported
        # public/private knowledge requests go to document QA.
        path = "/ask" if question["id"] == "Q20" else "/sql-agent"
        field = "question"
        return _post_with_backoff(client, path, {field: text})
    raise ValueError(f"Unsupported evaluation category: {category}")


def _score(question: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    check = _check(question)
    category = question["category"]
    if category == "doc":
        return keyword_score(response.get("answer", ""), response.get("sources", []), check.get("expected_keywords", []))
    if category == "sql":
        if "expected_values" in check:
            return sql_expected_values_match(check["expected_values"], response.get("rows", []))
        return sql_number_match(check.get("expected_answer", ""), response.get("rows", []))
    if category == "combined":
        return combined_score(response)
    return safety_score(check.get("expected_behavior", ""), response)


def _safe_error_message(error: Exception) -> str:
    """Return a short error string with environment-provided secrets removed."""
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    status = status or getattr(response, "status_code", None)
    if status == 429:
        return "The LLM provider rate limit was reached; retry later."
    if isinstance(error, SQLAlchemyError):
        return "The database request failed; check that Postgres is available."
    if isinstance(status, int):
        return f"The LLM provider request failed (HTTP {status}); check its configuration."
    # Avoid exception bodies: SDK and network errors can contain request details.
    return f"{error.__class__.__name__}: the evaluation request failed."


def _summary(rows: list[dict[str, Any]], ablation: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Count question and retrieval-comparison outcomes, including failures."""
    question_errors = sum(row.get("status") == "error" for row in rows)
    ablation_errors = sum(
        score.get("status") == "error"
        for mode_scores in ablation.values()
        for score in mode_scores.values()
    )
    return {
        "question_count": len(rows),
        "scored_count": sum(row.get("score", {}).get("status") == "scored" for row in rows),
        "unscored_count": sum(row.get("score", {}).get("status") == "unscored" for row in rows),
        "error_count": question_errors,
        "ablation_error_count": ablation_errors,
        "failure_count": question_errors + ablation_errors,
    }


def _write_results(rows: list[dict[str, Any]], ablation: dict[str, dict[str, Any]]) -> None:
    """Atomically persist current evaluation progress to the host-mounted file."""
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(
            {"results": rows, "retrieval_ablation": ablation, "summary": _summary(rows, ablation)},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(RESULT_PATH)


def _row_failed(row: dict[str, Any]) -> bool:
    """Return whether a stored question result failed its category's check."""
    if row.get("status") == "error":
        return True
    score = row.get("score", {})
    if score.get("status") == "error":
        return True
    if "number_match" in score:
        return not score["number_match"]
    if "keyword_hit_rate" in score:
        return score["keyword_hit_rate"] < 1.0
    if "combined_match" in score:
        return not score["combined_match"]
    if "behavior_match" in score:
        return not score["behavior_match"]
    return False


def _failure_reason(row: dict[str, Any]) -> str:
    """Explain the exact failed predicate recorded by the scorer."""
    if row.get("status") == "error":
        return f"Request errored: {row.get('error', 'unspecified error')}"
    score = row.get("score", {})
    if "number_match" in score:
        return f"Expected SQL values missing from returned rows: {score.get('missing_numbers', [])}."
    if "keyword_hit_rate" in score:
        return (
            f"Answer keyword hit rate was {score.get('keyword_hit_rate', 0):.0%}; "
            f"matched {score.get('matched_keywords', [])}. Cited-source keyword hit: "
            f"{score.get('source_keyword_hit', False)}."
        )
    if "combined_match" in score:
        return (
            f"Combined check requires data and citations; data evidence="
            f"{score.get('has_data_evidence', False)}, document citations="
            f"{score.get('has_document_citations', False)}."
        )
    if "behavior_match" in score:
        return "Expected safety behavior was not observed."
    return "Scorer marked this result as failed without a category detail."


def _actual_response(row: dict[str, Any]) -> str:
    """Format the response fields relevant to auditing a failure."""
    if row.get("status") == "error":
        return f"Error: {row.get('error', 'unknown error')}"
    response = row.get("response", {})
    parts: list[str] = []
    if response.get("answer") is not None:
        parts.append(f"Answer: {response['answer']}")
    if response.get("sql") is not None:
        parts.append(f"SQL: {response['sql']}")
    if response.get("rows") is not None:
        parts.append(f"Rows: {json.dumps(response['rows'], ensure_ascii=False, default=str)}")
    if response.get("sources") is not None:
        sources = [
            {"id": item.get("id"), "snippet": str(item.get("snippet", ""))[:500]}
            for item in response["sources"]
            if isinstance(item, dict)
        ]
        parts.append(
            "Cited sources: "
            + json.dumps(sources, ensure_ascii=False, default=str)
        )
    report_parts = []
    for key in ("problem_description", "root_cause_hypotheses", "limitations"):
        if response.get(key):
            report_parts.append(f"{key}: {json.dumps(response[key], ensure_ascii=False, default=str)}")
    for key in ("supporting_data", "supporting_documents"):
        if response.get(key) is not None:
            report_parts.append(
                f"{key}: {json.dumps(response[key], ensure_ascii=False, default=str)}"
            )
    if report_parts:
        parts.append("Report summary: " + "\n".join(report_parts))
    return "\n".join(parts) or "No response payload was saved."


def _render_failure_report(
    rows: list[dict[str, Any]], questions: list[dict[str, Any]]
) -> str:
    """Create a Markdown audit of every failed or errored evaluation question."""
    question_by_id = {item["id"]: item for item in questions}
    failures = [row for row in rows if _row_failed(row)]
    lines = [
        "# Evaluation failure report",
        "",
        "For document questions, answer keyword hit rate measures keywords in the answer only. "
        "The cited-source keyword result is reported separately and does not raise the answer score.",
        "",
        f"Failed questions: {len(failures)}",
        "",
    ]
    if not failures:
        lines.append("No failed or errored questions.")
    for row in failures:
        question = question_by_id.get(row["id"], {})
        check = _check(question)
        expected = {
            key: value
            for key, value in check.items()
            if key in {"expected_values", "expected_answer", "expected_keywords", "expected_behavior"}
        }
        lines.extend(
            [
                f"## {row['id']} ({row.get('category', question.get('category', 'unknown'))})",
                "",
                f"**Question:** {question.get('question', '(question text unavailable)')}",
                "",
                f"**Actual response:**\n\n```text\n{_actual_response(row)}\n```",
                "",
                f"**Expected value:** `{json.dumps(expected, ensure_ascii=False)}`",
                "",
                f"**Why it failed:** {_failure_reason(row)}",
                "",
            ]
        )
    return "\n".join(lines)


def _write_failure_report(rows: list[dict[str, Any]], questions: list[dict[str, Any]]) -> None:
    FAILURE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAILURE_REPORT_PATH.write_text(
        _render_failure_report(rows, questions), encoding="utf-8"
    )


def _print_table(rows: list[dict[str, Any]], title: str) -> None:
    print(f"\n{title}")
    print(f"{'ID':<5} {'CATEGORY':<9} {'SCORE':<28} {'DETAIL'}")
    print("-" * 88)
    for row in rows:
        if row.get("status") == "error":
            print(f"{row['id']:<5} {row['category']:<9} {'fail':<28} {row.get('error', '')}")
            continue
        score = row["score"]
        if score.get("status") == "unscored":
            label = "unscored"
        elif "keyword_hit_rate" in score:
            source_label = "yes" if score.get("source_keyword_hit") else "no"
            label = f"answer {score['keyword_hit_rate']:.0%}; sources {source_label}"
        elif "number_match" in score:
            label = "pass" if score["number_match"] else "fail"
        elif "combined_match" in score:
            label = "pass" if score["combined_match"] else "fail"
        else:
            label = "pass" if score.get("behavior_match") else "fail"
        detail = str(score.get("missing_numbers", score.get("has_document_citations", "")))
        print(f"{row['id']:<5} {row['category']:<9} {label:<28} {detail}")


def main(argv: list[str] | None = None) -> None:
    questions = json.loads(QUESTION_PATH.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description="Run RootCause evaluation questions.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip questions and retrieval comparisons already completed in eval/results.json.",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Rerun failed or errored questions from eval/results.json and keep passed results.",
    )
    parser.add_argument(
        "--failure-report-only",
        action="store_true",
        help="Rescore saved responses and write eval/failure_report.md without calling endpoints.",
    )
    args = parser.parse_args(argv)

    if args.failure_report_only:
        saved = json.loads(RESULT_PATH.read_text(encoding="utf-8")) if RESULT_PATH.exists() else {}
        saved_rows = saved.get("results", [])
        for row in saved_rows:
            if row.get("status") == "completed":
                question = next((item for item in questions if item["id"] == row["id"]), None)
                if question is not None:
                    row["score"] = _score(question, row.get("response", {}))
        _write_failure_report(saved_rows, questions)
        print(f"Failure details saved to {FAILURE_REPORT_PATH.relative_to(ROOT)}")
        return

    saved_results: dict[str, Any] = {}
    if (args.resume or args.rerun_failed) and RESULT_PATH.exists():
        saved_results = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    previous_rows = {
        row["id"]: row
        for row in saved_results.get("results", [])
        if row.get("status") == "completed"
        and (not args.rerun_failed or not _row_failed(row))
    }
    rows: list[dict[str, Any]] = [
        previous_rows[item["id"]]
        for item in questions
        if item["id"] in previous_rows
    ]
    rows_by_id = {row["id"]: row for row in rows}
    ablation = saved_results.get("retrieval_ablation", {}) if (args.resume or args.rerun_failed) else {}
    ablation = {
        mode: dict(ablation.get(mode, {})) for mode in RETRIEVAL_MODES
    }
    TestClient, app, get_retriever, get_settings, create_database_engine, HybridRetriever = (
        _runtime_dependencies()
    )
    engine = create_database_engine(get_settings().database_url)
    retrievers = {mode: HybridRetriever(engine, mode=mode) for mode in RETRIEVAL_MODES}
    current_mode = {"value": "hybrid"}
    app.dependency_overrides[get_retriever] = lambda: retrievers[current_mode["value"]]
    try:
        with TestClient(app) as client:
            for question in questions:
                if question["id"] in rows_by_id:
                    continue
                current_mode["value"] = "hybrid"
                try:
                    response = _request(client, question)
                    row = {
                        "id": question["id"],
                        "category": question["category"],
                        "status": "completed",
                        "response": response,
                        "score": _score(question, response),
                    }
                except Exception as error:
                    row = {
                        "id": question["id"],
                        "category": question["category"],
                        "status": "error",
                        "error": _safe_error_message(error),
                        "score": {"status": "error"},
                    }
                rows_by_id[question["id"]] = row
                rows = [rows_by_id[item["id"]] for item in questions if item["id"] in rows_by_id]
                _write_results(rows, ablation)
                _write_failure_report(rows, questions)
            doc_questions = [item for item in questions if item["category"] == "doc"]
            for mode in RETRIEVAL_MODES:
                current_mode["value"] = mode
                for question in doc_questions:
                    if ablation[mode].get(question["id"], {}).get("status") in {"scored", "unscored"}:
                        continue
                    try:
                        response = _request(client, question, mode)
                        ablation[mode][question["id"]] = keyword_score(
                            response.get("answer", ""),
                            response.get("sources", []),
                            _check(question).get("expected_keywords", []),
                        )
                    except Exception as error:
                        ablation[mode][question["id"]] = {
                            "status": "error",
                            "error": _safe_error_message(error),
                        }
                    _write_results(rows, ablation)
                    _write_failure_report(rows, questions)
    finally:
        app.dependency_overrides.clear()
        engine.dispose()

    _write_results(rows, ablation)
    _write_failure_report(rows, questions)
    _print_table(rows, "Evaluation results")
    print("\nRetrieval ablation (keyword hit rate)")
    print(f"{'MODE':<10} {'SCORED':<10} {'MEAN HIT RATE':<15}")
    print("-" * 38)
    for mode, scores in ablation.items():
        scored = [item["keyword_hit_rate"] for item in scores.values() if item["status"] == "scored"]
        rate = sum(scored) / len(scored) if scored else None
        print(f"{mode:<10} {len(scored):<10} {f'{rate:.0%}' if rate is not None else 'unscored':<15}")
    print(f"\nSaved full responses and scores to {RESULT_PATH.relative_to(ROOT)}")
    print(f"Failure details saved to {FAILURE_REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
