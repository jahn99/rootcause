"""Run the saved question set against the app and write evaluation results."""

from __future__ import annotations

import json
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


def _print_table(rows: list[dict[str, Any]], title: str) -> None:
    print(f"\n{title}")
    print(f"{'ID':<5} {'CATEGORY':<9} {'SCORE':<28} {'DETAIL'}")
    print("-" * 88)
    for row in rows:
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


def main() -> None:
    questions = json.loads(QUESTION_PATH.read_text(encoding="utf-8"))
    engine = create_database_engine(get_settings().database_url)
    retrievers = {mode: HybridRetriever(engine, mode=mode) for mode in RETRIEVAL_MODES}
    current_mode = {"value": "hybrid"}
    app.dependency_overrides[get_retriever] = lambda: retrievers[current_mode["value"]]
    rows: list[dict[str, Any]] = []
    ablation: dict[str, dict[str, Any]] = {mode: {} for mode in RETRIEVAL_MODES}
    try:
        with TestClient(app) as client:
            for question in questions:
                current_mode["value"] = "hybrid"
                response = _request(client, question)
                rows.append(
                    {
                        "id": question["id"],
                        "category": question["category"],
                        "response": response,
                        "score": _score(question, response),
                    }
                )
            doc_questions = [item for item in questions if item["category"] == "doc"]
            for mode in RETRIEVAL_MODES:
                current_mode["value"] = mode
                for question in doc_questions:
                    response = _request(client, question, mode)
                    ablation[mode][question["id"]] = keyword_score(
                        response.get("answer", ""),
                        response.get("sources", []),
                        _check(question).get("expected_keywords", []),
                    )
    finally:
        app.dependency_overrides.clear()
        engine.dispose()

    RESULT_PATH.write_text(
        json.dumps(
            {
                "results": rows,
                "retrieval_ablation": ablation,
                "summary": {
                    "question_count": len(rows),
                    "scored_count": sum(row["score"].get("status") == "scored" for row in rows),
                    "unscored_count": sum(row["score"].get("status") == "unscored" for row in rows),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
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
