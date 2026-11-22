"""Failure handling and resume behavior for the evaluation runner."""

import json

from scripts import run_eval


class FakeEngine:
    def dispose(self) -> None:
        pass


class FakeTestClient:
    def __init__(self, _app: object) -> None:
        pass

    def __enter__(self) -> "FakeTestClient":
        return self

    def __exit__(self, *_args: object) -> None:
        pass


class MemoryPath:
    """Small path double so runner persistence tests need no filesystem access."""

    def __init__(self, content: str = "") -> None:
        self.content = content
        self.parent = self

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.content

    def write_text(self, content: str, encoding: str = "utf-8") -> None:
        self.content = content

    def with_suffix(self, _suffix: str) -> "MemoryPath":
        return self

    def replace(self, destination: "MemoryPath") -> None:
        destination.content = self.content

    def mkdir(self, **_kwargs: object) -> None:
        pass

    def exists(self) -> bool:
        return bool(self.content)

    def relative_to(self, _other: object) -> str:
        return "eval/results.json"


def configure_runner(monkeypatch, questions: list[dict[str, object]]) -> MemoryPath:
    question_path = MemoryPath(json.dumps(questions))
    result_path = MemoryPath()
    monkeypatch.setattr(run_eval, "QUESTION_PATH", question_path)
    monkeypatch.setattr(run_eval, "RESULT_PATH", result_path)
    monkeypatch.setattr(run_eval, "create_database_engine", lambda *_args: FakeEngine())
    monkeypatch.setattr(run_eval, "HybridRetriever", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(run_eval, "TestClient", FakeTestClient)
    return result_path


def test_eval_records_question_errors_continues_and_counts_failures(monkeypatch, capsys) -> None:
    questions = [
        {"id": "Q1", "category": "safety", "question": "first", "check": {"expected_behavior": "refuse"}},
        {"id": "Q2", "category": "safety", "question": "second", "check": {"expected_behavior": "refuse"}},
    ]
    result_path = configure_runner(monkeypatch, questions)
    calls: list[str] = []

    def request(_client, question, mode="hybrid"):
        calls.append(question["id"])
        if question["id"] == "Q1":
            raise RuntimeError("provider failed with fake-key-marker")
        checkpoint = json.loads(result_path.content)
        assert checkpoint["results"][0]["status"] == "error"
        return {"answer": "I refuse."}

    monkeypatch.setattr(run_eval, "_request", request)
    run_eval.main([])

    results = json.loads(result_path.content)
    assert calls == ["Q1", "Q2"]
    assert [row["status"] for row in results["results"]] == ["error", "completed"]
    assert "fake-key-marker" not in json.dumps(results)
    assert results["summary"]["error_count"] == 1
    assert results["summary"]["failure_count"] == 1


def test_eval_resume_skips_completed_questions(monkeypatch, capsys) -> None:
    questions = [
        {"id": "Q1", "category": "safety", "question": "first", "check": {"expected_behavior": "refuse"}},
        {"id": "Q2", "category": "safety", "question": "second", "check": {"expected_behavior": "refuse"}},
    ]
    result_path = configure_runner(monkeypatch, questions)
    result_path.content = json.dumps(
        {
            "results": [
                {
                    "id": "Q1",
                    "category": "safety",
                    "status": "completed",
                    "response": {"answer": "I refuse."},
                    "score": {"status": "scored", "behavior_match": True},
                }
            ],
            "retrieval_ablation": {},
        }
    )
    calls: list[str] = []

    def request(_client, question, mode="hybrid"):
        calls.append(question["id"])
        return {"answer": "I refuse."}

    monkeypatch.setattr(run_eval, "_request", request)
    run_eval.main(["--resume"])

    assert calls == ["Q2"]
    results = json.loads(result_path.content)
    assert [row["id"] for row in results["results"]] == ["Q1", "Q2"]
