"""Fetch, chunk, embed, and store a bounded NHTSA sample."""

from pathlib import Path

from app.ingestion.nhtsa import fetch_nhtsa_records
from app.ingestion.store import ingest_records

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "nhtsa_cache"
MAX_RECORDS = 2000


def main() -> None:
    """Run the end-to-end NHTSA ingestion job."""
    records = fetch_nhtsa_records(CACHE_DIR, max_records=MAX_RECORDS)
    if not records:
        raise SystemExit("No NHTSA records matched the selected safety topics.")
    chunk_count = ingest_records(records)
    print(f"Ingested {len(records)} NHTSA records as {chunk_count} chunks.")
    print(f"Raw API responses are cached in {CACHE_DIR}.")


if __name__ == "__main__":
    main()
