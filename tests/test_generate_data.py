"""Tests for deterministic synthetic manufacturing data generation."""

from collections import Counter
from datetime import date

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.db.models import Base, DowntimeEvent, ProductionDefect
from scripts.generate_data import (
    SPIKE_END,
    SPIKE_PART,
    SPIKE_PLANT,
    SPIKE_START,
    generate_synthetic_data,
    populate_database,
)


def test_generated_data_has_expected_volume_and_coverage() -> None:
    defects, downtime = generate_synthetic_data()

    assert len(defects) > 1_000
    assert len(downtime) > 80
    assert {row["plant"] for row in defects} == {"Cedar Falls", "Lakeview", "Pine Ridge"}
    assert min(row["date"] for row in defects) == date(2025, 1, 1)
    assert max(row["date"] for row in defects) == date(2025, 6, 30)
    assert all(1 <= row["severity"] <= 5 for row in defects)


def test_database_reset_and_spike_with_related_downtime() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    try:
        first_counts = populate_database(engine)
        second_counts = populate_database(engine)
        assert first_counts == second_counts

        with Session(engine) as session:
            assert session.scalar(select(func.count()).select_from(ProductionDefect)) == first_counts[0]
            assert session.scalar(select(func.count()).select_from(DowntimeEvent)) == first_counts[1]

            spike_rows = session.scalars(
                select(ProductionDefect).where(
                    ProductionDefect.plant == SPIKE_PLANT,
                    ProductionDefect.line == 3,
                    ProductionDefect.part == SPIKE_PART,
                    ProductionDefect.date.between(SPIKE_START, SPIKE_END),
                )
            ).all()
            daily_spike_counts = Counter(row.date for row in spike_rows)
            assert daily_spike_counts[SPIKE_START] == 3
            assert daily_spike_counts[SPIKE_END] == 16
            assert len(daily_spike_counts) == 14
            assert sum(row.severity >= 4 for row in spike_rows) > len(spike_rows) // 2

            related_downtime = session.scalars(
                select(DowntimeEvent).where(
                    DowntimeEvent.plant == SPIKE_PLANT,
                    DowntimeEvent.line == 3,
                    DowntimeEvent.reason == "door-seal rework during containment",
                    DowntimeEvent.date.between(SPIKE_START, SPIKE_END),
                )
            ).all()
            assert len(related_downtime) == 14
            assert all(event.minutes >= 25 for event in related_downtime)
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_generation_is_reproducible_for_same_seed() -> None:
    assert generate_synthetic_data(seed=77) == generate_synthetic_data(seed=77)
