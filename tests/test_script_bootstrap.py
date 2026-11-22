"""Ensure maintenance scripts work when invoked directly from the repository."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
SCRIPTS = ("generate_data.py", "ingest.py", "run_eval.py", "check_llm.py")


def test_direct_scripts_add_the_repository_root_to_python_path() -> None:
    for script_name in SCRIPTS:
        source = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")

        assert "Path(__file__).resolve().parents[1]" in source, script_name
        assert "sys.path.insert(0, str(" in source, script_name


def test_ci_job_sets_pythonpath_for_generation_and_pytest() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "PYTHONPATH: ${{ github.workspace }}" in workflow
    assert "run: python scripts/generate_data.py" in workflow
    assert "run: python -m pytest -q" in workflow
