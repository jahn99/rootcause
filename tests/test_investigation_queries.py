"""Real SQL-template tests using the synthetic injected spike in SQLite."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, event, text

from app.agents.investigation_queries import InvestigationQueryRunner, map_part_name
from scripts.generate_data import generate_synthetic_data


@pytest.fixture
def query_runner() -> InvestigationQueryRunner:
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def register_week_start(dbapi_connection, _connection_record) -> None:
        def week_start(_unit: str, value: str) -> str:
            current = date.fromisoformat(str(value)[:10])
            return (current - timedelta(days=current.weekday())).isoformat()

        dbapi_connection.create_function("DATE_TRUNC", 2, week_start)

    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE production_defects (id INTEGER, plant TEXT, line INTEGER, "
                "part TEXT, defect_type TEXT, severity INTEGER, date DATE)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE downtime_events (id INTEGER, plant TEXT, line INTEGER, "
                "reason TEXT, minutes INTEGER, date DATE)"
            )
        )
        all_defects, all_downtime = generate_synthetic_data()
        defects = [
            {**row, "date": row["date"].isoformat()}
            for row in all_defects
            if row["plant"] == "Cedar Falls"
            and row["line"] == 3
            and row["part"] == "door seal"
            and date(2025, 5, 5) <= row["date"] <= date(2025, 5, 18)
        ]
        defects.extend(
            [
                {
                    "id": 20000,
                    "plant": "Lakeview",
                    "line": 2,
                    "part": "brake caliper",
                    "defect_type": "surface blemish",
                    "severity": 3,
                    "date": "2025-05-08",
                },
                {
                    "id": 20001,
                    "plant": "Lakeview",
                    "line": 2,
                    "part": "brake caliper",
                    "defect_type": "surface blemish",
                    "severity": 4,
                    "date": "2025-05-09",
                },
            ]
        )
        downtime = [
            {**row, "date": row["date"].isoformat()}
            for row in all_downtime
            if row["plant"] == "Cedar Falls"
            and row["line"] == 3
            and date(2025, 5, 5) <= row["date"] <= date(2025, 5, 18)
        ]
        connection.execute(
            text(
                "INSERT INTO production_defects VALUES "
                "(:id, :plant, :line, :part, :defect_type, :severity, :date)"
            ),
            defects,
        )
        connection.execute(
            text("INSERT INTO downtime_events VALUES (:id, :plant, :line, :reason, :minutes, :date)"),
            downtime,
        )

    runner = InvestigationQueryRunner(engine)
    try:
        yield runner
    finally:
        engine.dispose()


def test_templates_return_spike_peak_weeks(query_runner: InvestigationQueryRunner) -> None:
    context = {
        "part": query_runner.resolve_part("door seal"),
        "plant": "Cedar Falls",
        "line": 3,
        "start_date": "2025-05-05",
        "end_date": "2025-05-18",
    }

    results = query_runner.run_overviews(context)
    weekly = results[0]

    assert [item.name for item in results] == [
        "weekly_defects",
        "defects_by_plant_line",
        "severity_comparison",
        "weekly_downtime",
    ]
    assert weekly.rows == [
        {"week": "2025-05-05", "defect_count": 42},
        {"week": "2025-05-12", "defect_count": 91},
    ]
    assert "%(part)s" in weekly.sql
    assert weekly.parameters["part"] == "door seal"


def test_part_name_maps_to_actual_database_value(query_runner: InvestigationQueryRunner) -> None:
    assert query_runner.resolve_part("Door-Seal") == "door seal"
    assert map_part_name("brakes", ["brake caliper", "wire harness"]) == "brake caliper"


def test_injection_looking_part_name_is_rejected(query_runner: InvestigationQueryRunner) -> None:
    with pytest.raises(ValueError, match="unsupported characters"):
        query_runner.resolve_part("door seal'; DROP TABLE production_defects; --")
