# RootCause: project spec

You are my senior engineer. We are building "RootCause", a multi-agent GenAI
service for automotive quality engineering that investigates a manufacturing
defect and drafts an 8D report. This is for a job application (AI Engineer,
FastAPI/RAG/agents focus), so code quality, tests and measured evaluation matter
more than feature count. I am a student and still learning, so explain things
simply and keep each step small.

## Working rules
1. FIRST produce a short plan and folder layout. Wait for my approval before coding.
2. Work in vertical slices, one at a time. After each slice: run the tests, show
   me the results, and make a git commit. Never start the next slice until I say so.
3. Do not invent library APIs. Check the installed package version and its docs
   before using LangGraph, LangChain, pgvector or any LLM SDK. Pin versions
   in requirements.txt.
4. All LLM calls go through llm/base.py (an abstract interface) with
   implementations for Azure OpenAI, Gemini, Groq and Ollama, chosen by env var.
   No SDK calls anywhere else.
5. Secrets only via environment variables. Provide .env.example. Never commit keys.
6. Use only public data (NHTSA API, OBD-II code list) and synthetic data.
   No Mercedes-Benz material or branding.
7. Write clean, modular, typed Python with Pydantic models and docstrings.
8. Each slice must include pytest tests. Do not write tests that only mock
   everything and prove nothing.
9. Tell me honestly when something is unfinished or untested. Do not claim it works
   unless you have run it.
10. I am on Windows with PowerShell. Give commands that work there.

## Stack
Python 3.11, FastAPI, Pydantic v2, Postgres + pgvector, SQLAlchemy,
rank_bm25 (or Postgres full-text), LangGraph for orchestration, pytest, Docker
Compose, GitHub Actions, ruff.

## Slices (Phase 1)
1. Skeleton: repo layout, docker-compose (app + Postgres/pgvector), FastAPI app
   with /health, config, llm/base.py interface with one working provider,
   a basic test, README stub.
2. Synthetic data: script generating production_defects and downtime_events
   (plant, line, part, defect_type, severity, date) with ONE injected defect
   spike (e.g. door-seal defects on Line 3) so anomaly detection has something to find.
3. Ingestion + retrieval: fetch NHTSA recalls/complaints via REST, chunk,
   embed, store in pgvector. Hybrid retrieval (BM25 + vector). POST /ask returns
   an answer with source citations.
4. Data agent: tool-calling agent with a guarded SQL tool. Read-only DB role,
   table allow-list, validator that allows SELECT only and blocks everything
   else. Tests including SQL injection and prompt-injection cases.
5. 8D report: Pydantic 8D schema, LangGraph workflow
   (triage -> data agent + knowledge agent -> report agent), POST /investigate
   returns validated JSON.
6. Eval: script that runs an eval set from eval/questions.json (I will write the
   ground truth myself), reports retrieval hit rate and answer correctness.
   Save results to a file.
7. Engineering: Dockerfile, GitHub Actions running ruff, pytest, pip-audit,
   gitleaks and a Trivy image scan.
8. README: architecture diagram, setup steps, eval results, limitations.