"""Regression tests for door-seal-focused NHTSA ingestion topics."""

from app.ingestion.nhtsa import normalize_records


def test_normalizer_keeps_door_seal_related_complaint_topics() -> None:
    raw = {
        "results": [
            {"ODI Number": "A1", "Component": "Weatherstrip", "Summary": "Water leaks into cabin."},
            {"ODI Number": "A2", "Component": "Door", "Summary": "Wind noise at highway speed."},
            {"ODI Number": "A3", "Component": "Door Lock", "Summary": "Lock actuator clicks."},
        ]
    }

    records = normalize_records(raw, "complaint", "TEST", "CAR", 2025)

    assert [record.source_id for record in records] == ["A1", "A2", "A3"]
    assert "Water leaks into cabin" in records[0].text
    assert "Wind noise" in records[1].text
