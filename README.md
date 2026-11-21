# RootCause

RootCause is a learning project for investigating automotive manufacturing
quality defects and drafting 8D reports with retrieval and agent workflows.
The project uses public NHTSA data and synthetic manufacturing data.

## Slice 1: API skeleton

### Run locally (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Set GROQ_API_KEY in .env before making any LLM calls.
uvicorn app.main:app --reload
```

Visit <http://127.0.0.1:8000/health> for the health response.

### Run with Docker Compose (PowerShell)

```powershell
Copy-Item .env.example .env
# Set GROQ_API_KEY in .env before making any LLM calls.
docker compose up --build
```

The API listens on port 8000 and Postgres with pgvector listens on port 5432.
The health endpoint does not call the LLM or database.

### Check the configured LLM with Docker Compose

With `GROQ_API_KEY` set in `.env`, run this in another PowerShell window while
the Compose services are available:

```powershell
docker compose run --rm app python scripts/check_llm.py
```

The script uses the same LLM provider factory as the application and prints
the provider reply. On failure it reports a safe hint without showing secrets.
Groq retires models periodically. If the check reports that the model was
rejected, verify `GROQ_MODEL` against the [Groq models list](https://console.groq.com/docs/models).

## Slice 2: synthetic manufacturing data

Generate (or reset and regenerate) the synthetic PostgreSQL tables with:

```powershell
docker compose run --rm app python scripts/generate_data.py
```

The script creates `production_defects` and `downtime_events`, then replaces
their contents with reproducible data spanning January 1 through June 30, 2025.
It uses seed `2025`, so each run produces the same rows and IDs.

There is exactly one deliberately injected defect spike: Cedar Falls, Line 3,
door seal part, `seal gap` defects from May 5 through May 18, 2025. Daily spike
counts rise from 3 to 16, with higher severity and one related downtime event
per day for door-seal rework during containment.

### Run tests

```powershell
python -m pytest
```
