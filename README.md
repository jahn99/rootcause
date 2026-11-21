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

## Slice 3: NHTSA retrieval and question answering

Fetch a small topic-focused sample of public NHTSA recalls and complaints, then
chunk, embed locally, and store it in PostgreSQL with pgvector:

```powershell
docker compose run --rm app python scripts/ingest.py
```

The first run downloads the FastEmbed `BAAI/bge-small-en-v1.5` model and caches
the NHTSA API responses plus model files under `data/`. Later runs reuse those
files. Ingestion handles at most 2,000 topic-matching records across Ford F-150,
Toyota Camry, Honda Accord, and Mercedes-Benz C-Class (2021 model year). It
queries NHTSA's documented [recalls-by-vehicle](https://www.nhtsa.gov/nhtsa-datasets-and-apis)
and [complaints-by-vehicle](https://www.nhtsa.gov/nhtsa-datasets-and-apis) REST
endpoints and focuses on door/seal, brakes, airbags, fuel, steering, and
electrical terms.

After ingestion, ask a question from PowerShell:

```powershell
$body = @{ question = "What electrical issues are reported?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType "application/json" -Body $body
```

The response includes an answer and source IDs with snippets. If retrieval has
no matching documents, the endpoint responds `not found in the documents`.
Choose `RETRIEVAL_MODE=hybrid`, `bm25`, or `vector` in `.env` to compare search
modes. Hybrid retrieval merges BM25 and pgvector rankings with reciprocal rank
fusion. The local embedding model is CPU-based and does not need an API key.

### Run tests

```powershell
python -m pytest
```
