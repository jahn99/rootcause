"""Run the saved question set against the app and write evaluation results."""

from __future__ import annotations

import json
import argparse
import os
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.api.ask import get_retriever
from app.config import get_settings
from app.db.session import create_database_engine
from app.main import app
from app.retrieval.search import HybridRetriever
from eval.scoring import combined_score, keyword_score, safety_score, sql_number_match
from sqlalchemy.exc import SQLAlchemyError

ROOT = Path(__file__).resolve().parents[1]
QUESTION_PATH = ROOT / "eval" / "questions.json"
RESULT_PATH = ROOT / "eval" / "results.json"
RETRIEVAL_MODES = ("vector", "bm25", "hybrid")


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
            label = f"{score['keyword_hit_rate']:.0%} keywords"
        elif "number_match" in score:
            label = "pass" if score["number_match"] else "fail"
        elif "combined_match" in score:
            label = "pass" if score["combined_match"] else "fail"
        else:
            label = "pass" if score.get("behavior_match") else "fail"
        detail = str(score.get("source_keyword_hit", score.get("missing_numbers", score.get("has_document_citations", ""))))
        print(f"{row['id']:<5} {row['category']:<9} {label:<28} {detail}")


def main(argv: list[str] | None = None) -> None:
    questions = json.loads(QUESTION_PATH.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description="Run RootCause evaluation questions.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip questions and retrieval comparisons already completed in eval/results.json.",
    )
    args = parser.parse_args(argv)

    saved_results: dict[str, Any] = {}
    if args.resume and RESULT_PATH.exists():
        saved_results = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    previous_rows = {
        row["id"]: row
        for row in saved_results.get("results", [])
        if row.get("status") == "completed"
    }
    rows: list[dict[str, Any]] = [
        previous_rows[item["id"]]
        for item in questions
        if item["id"] in previous_rows
    ]
    rows_by_id = {row["id"]: row for row in rows}
    ablation = saved_results.get("retrieval_ablation", {}) if args.resume else {}
    ablation = {
        mode: dict(ablation.get(mode, {})) for mode in RETRIEVAL_MODES
    }
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
    finally:
        app.dependency_overrides.clear()
        engine.dispose()

    _write_results(rows, ablation)
    _print_table(rows, "Evaluation results")
    print("\nRetrieval ablation (keyword hit rate)")
    print(f"{'MODE':<10} {'SCORED':<10} {'MEAN HIT RATE':<15}")
    print("-" * 38)
    for mode, scores in ablation.items():
        scored = [item["keyword_hit_rate"] for item in scores.values() if item["status"] == "scored"]
        rate = sum(scored) / len(scored) if scored else None
        print(f"{mode:<10} {len(scored):<10} {f'{rate:.0%}' if rate is not None else 'unscored':<15}")
    print(f"\nSaved full responses and scores to {RESULT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
