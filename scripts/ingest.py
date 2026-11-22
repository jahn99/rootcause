"""Fetch, chunk, embed, and store a bounded NHTSA sample."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.nhtsa import fetch_nhtsa_records  # noqa: E402
from app.ingestion.store import ingest_records  # noqa: E402
from app.ingestion.chunking import chunk_text  # noqa: E402

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "nhtsa_cache"
MAX_RECORDS = 2000
COVERAGE_TERMS = ("door seal", "weatherstrip", "water leak", "wind noise", "wiring", "short circuit")


def main() -> None:
    """Run the end-to-end NHTSA ingestion job."""
    records = fetch_nhtsa_records(CACHE_DIR, max_records=MAX_RECORDS)
    if not records:
        raise SystemExit("No NHTSA records matched the selected safety topics.")
    chunk_count = ingest_records(records)
    print(f"Ingested {len(records)} NHTSA records as {chunk_count} chunks.")
    chunks = [chunk for record in records for chunk in chunk_text(record.text)]
    for term in COVERAGE_TERMS:
        count = sum(term in chunk.casefold() for chunk in chunks)
        print(f"Chunks mentioning {term!r}: {count}")
    print(f"Raw API responses are cached in {CACHE_DIR}.")


if __name__ == "__main__":
    main()
