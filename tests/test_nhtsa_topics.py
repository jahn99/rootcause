"""Regression tests for door-seal-focused NHTSA ingestion topics."""

from pathlib import Path

from app.ingestion.nhtsa import MODEL_YEARS, VEHICLES, fetch_nhtsa_records, normalize_records


def test_ingestion_uses_only_the_approved_public_vehicle_sample() -> None:
    approved_makes = {"FORD", "TOYOTA", "HONDA", "CHEVROLET", "NISSAN", "SUBARU"}
    assert all(make in approved_makes for make, _ in VEHICLES)


def test_normalizer_keeps_door_seal_related_complaint_topics() -> None:
    raw = {
        "results": [
            {"ODI Number": "A1", "Component": "Weatherstrip", "Summary": "Water leaks into cabin."},
            {"ODI Number": "A2", "Component": "Door", "Summary": "Wind noise at highway speed."},
            {"ODI Number": "A3", "Component": "Door Lock", "Summary": "Lock actuator clicks."},
            {"ODI Number": "A4", "Component": "ELECTRICAL SYSTEM", "Summary": "A short-circuit can disable wiring."},
        ]
    }

    records = normalize_records(raw, "complaint", "TEST", "CAR", 2025)

    assert [record.source_id for record in records] == ["A1", "A2", "A4"]
    assert "Water leaks into cabin" in records[0].text
    assert "Wind noise" in records[1].text
    assert "short-circuit" in records[2].text


def test_expanded_ingestion_samples_multiple_models_and_model_years(monkeypatch) -> None:
    calls: list[tuple[str, str, str, int]] = []

    def fake_fetch(source_type, make, model, model_year, _cache_dir, _client):
        calls.append((source_type, make, model, model_year))
        return {
            "results": [
                {
                    "ODI Number": f"{source_type}-{make}-{model_year}",
                    "Component": "WIRING",
                    "Summary": "Electrical wiring defect.",
                }
            ]
        }

    monkeypatch.setattr("app.ingestion.nhtsa.fetch_raw_response", fake_fetch)
    records = fetch_nhtsa_records(Path("unused-cache"), max_records=20, client=object())

    assert len(records) == 20
    assert len({(source, make, year) for source, make, _, year in calls}) == 20
    assert {year for _, _, _, year in calls} == {MODEL_YEARS[0], MODEL_YEARS[1]}
    assert {source for source, *_ in calls} == {"recall", "complaint"}
