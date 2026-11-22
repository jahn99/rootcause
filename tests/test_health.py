"""Basic API and provider factory checks."""

from pathlib import Path
import importlib

from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.factory import create_llm_provider
from app.llm.groq_provider import GroqProvider
from app.main import app


def test_docker_image_exposes_app_package_on_python_path() -> None:
    dockerfile = Path(__file__).parents[1] / "Dockerfile"
    compose_file = Path(__file__).parents[1] / "docker-compose.yml"

    assert "ENV PYTHONPATH=/app" in dockerfile.read_text(encoding="utf-8")
    assert "PYTHONPATH: /app" in compose_file.read_text(encoding="utf-8")


def test_docker_includes_eval_and_mounts_host_results() -> None:
    project_root = Path(__file__).parents[1]
    dockerfile = (project_root / "Dockerfile").read_text(encoding="utf-8")
    compose = (project_root / "docker-compose.yml").read_text(encoding="utf-8")
    dockerignore = (project_root / ".dockerignore").read_text(encoding="utf-8")

    assert "COPY eval ./eval" in dockerfile
    assert "./eval:/app/eval" in compose
    assert not any(line.strip() in {"eval", "eval/", "**/eval"} for line in dockerignore.splitlines())


def test_evaluation_runner_imports() -> None:
    module = importlib.import_module("scripts.run_eval")

    assert module.QUESTION_PATH.name == "questions.json"
    assert module.RESULT_PATH.name == "results.json"


def test_health_returns_ok() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_factory_selects_groq_provider() -> None:
    provider = create_llm_provider(Settings(llm_provider="groq", groq_api_key="test-key"))

    assert isinstance(provider, GroqProvider)


def test_factory_rejects_unknown_provider() -> None:
    try:
        create_llm_provider(Settings(llm_provider="unknown"))
    except ValueError as error:
        assert "Unsupported LLM_PROVIDER" in str(error)
    else:
        raise AssertionError("Expected unknown provider to be rejected")
