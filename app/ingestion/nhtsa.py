"""Fetch and normalize a bounded sample of public NHTSA safety records."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import httpx

NHTSA_BASE_URL = "https://api.nhtsa.gov"
MODEL_YEAR = 2021
VEHICLES = (
    ("FORD", "F-150"),
    ("TOYOTA", "CAMRY"),
    ("HONDA", "ACCORD"),
    ("CHEVROLET", "SILVERADO"),
    ("NISSAN", "ALTIMA"),
    ("SUBARU", "OUTBACK"),
)
MODEL_YEARS = (2025, 2024, 2023, 2022, 2021, 2020)
TOPIC_TERMS = (
    "door seal", "door seals", "seal gap", "brake", "airbag", "air bag", "fuel", "steer",
    "electrical", "electric", "wiring", "battery", "short circuit",
    "weatherstrip", "weather strip", "water leak", "water leaks", "water intrusion",
    "water leaking", "water entering", "wind noise", "wire harness",
)
SOURCE_ENDPOINTS = {
    "recall": "/recalls/recallsByVehicle",
    "complaint": "/complaints/complaintsByVehicle",
}
MAX_RECORDS_PER_QUERY = 120


@dataclass(frozen=True)
class NHTSARecord:
    """Normalized source record ready for chunking and storage."""

    source_type: str
    source_id: str
    component: str
    record_date: date | None
    text: str


def _cache_file(cache_dir: Path, source_type: str, make: str, model: str, model_year: int) -> Path:
    key = f"{source_type}-{make}-{model}-{model_year}"
    safe_key = re.sub(r"[^a-z0-9-]+", "-", key.lower()).strip("-")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"{safe_key}-{digest}.json"


def fetch_raw_response(
    source_type: str,
    make: str,
    model: str,
    model_year: int,
    cache_dir: Path,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Read a cached raw response or fetch the documented NHTSA vehicle endpoint."""
    if source_type not in SOURCE_ENDPOINTS:
        raise ValueError(f"Unsupported NHTSA source type: {source_type}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_file(cache_dir, source_type, make, model, model_year)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    endpoint = SOURCE_ENDPOINTS[source_type]
    owns_client = client is None
    api_client = client or httpx.Client(timeout=30.0)
    try:
        response = api_client.get(
            f"{NHTSA_BASE_URL}{endpoint}",
            params={"make": make, "model": model, "modelYear": model_year},
        )
        try:
            response.raise_for_status()
            raw = response.json()
        except httpx.HTTPStatusError as error:
            if error.response.status_code not in {400, 404}:
                raise
            # Some make/model-year combinations may not be in the NHTSA catalog.
            raw = {"results": []}
    finally:
        if owns_client:
            api_client.close()
    cache_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return raw


def _record_text(record: dict[str, Any]) -> str:
    """Flatten public text fields into a stable, searchable record string."""
    parts: list[str] = []
    for key, value in record.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float)):
            text = str(value).strip()
            if text:
                parts.append(f"{key}: {text}")
        elif isinstance(value, list):
            values = [str(item).strip() for item in value if item is not None and str(item).strip()]
            if values:
                parts.append(f"{key}: {', '.join(values)}")
    return "\n".join(parts)


def _first(record: dict[str, Any], *keys: str) -> str:
    lowered = {str(key).lower(): value for key, value in record.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _component(record: dict[str, Any], text: str) -> str:
    component = _first(record, "Component", "components", "Product")
    if component:
        return component[:120]
    lowered = text.lower()
    for topic in TOPIC_TERMS:
        if topic in lowered:
            return topic
    return "general safety"


def normalize_records(
    raw: dict[str, Any], source_type: str, make: str, model: str, model_year: int
) -> list[NHTSARecord]:
    """Keep only topic-focused result rows and normalize their searchable fields."""
    results = raw.get("results", [])
    if not isinstance(results, list):
        return []
    normalized: list[NHTSARecord] = []
    for index, record in enumerate(results, start=1):
        if not isinstance(record, dict):
            continue
        text = _record_text(record)
        lowered = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
        if not any(
            re.sub(r"[^a-z0-9]+", " ", topic.casefold()).strip() in lowered
            for topic in TOPIC_TERMS
        ):
            continue
        source_id = _first(
            record,
            "ODI Number", "odiNumber", "NHTSA Campaign Number", "NHTSAID", "nhtsaId",
        ) or f"{make}-{model}-{model_year}-{index}"
        date_value = _first(
            record, "dateOfIncident", "dateComplaintFiled", "ReportReceivedDate", "ReportDate"
        )
        normalized.append(
            NHTSARecord(
                source_type=source_type,
                source_id=source_id,
                component=_component(record, text),
                record_date=_parse_date(date_value),
                text=f"Vehicle: {make} {model} {model_year}\n{text}",
            )
        )
    return normalized


def fetch_nhtsa_records(
    cache_dir: Path, max_records: int = 2000, client: httpx.Client | None = None
) -> list[NHTSARecord]:
    """Fetch up to ``max_records`` topic-focused records from NHTSA, using cache."""
    if max_records < 1:
        raise ValueError("max_records must be greater than zero")
    owns_client = client is None
    api_client = client or httpx.Client(timeout=30.0)
    records: list[NHTSARecord] = []
    try:
        per_source_limit = max(1, max_records // len(SOURCE_ENDPOINTS))
        for source_type in SOURCE_ENDPOINTS:
            source_record_count = 0
            for model_year in MODEL_YEARS:
                for make, model in VEHICLES:
                    if len(records) >= max_records or source_record_count >= per_source_limit:
                        break
                    raw = fetch_raw_response(
                        source_type, make, model, model_year, cache_dir, api_client
                    )
                    batch = normalize_records(raw, source_type, make, model, model_year)
                    take = min(
                        MAX_RECORDS_PER_QUERY,
                        per_source_limit - source_record_count,
                        max_records - len(records),
                    )
                    selected = batch[:take]
                    records.extend(selected)
                    source_record_count += len(selected)
                if len(records) >= max_records or source_record_count >= per_source_limit:
                    break
    finally:
        if owns_client:
            api_client.close()
    return records
