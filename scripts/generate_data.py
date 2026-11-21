"""Generate deterministic synthetic defects and downtime in PostgreSQL."""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any

from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from app.db.models import Base, DowntimeEvent, ProductionDefect
from app.db.session import create_database_engine

SEED = 2025
START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 6, 30)
SPIKE_PLANT = "Cedar Falls"
SPIKE_LINE = 3
SPIKE_PART = "door seal"
SPIKE_START = date(2025, 5, 5)
SPIKE_END = date(2025, 5, 18)

PLANTS = ("Cedar Falls", "Lakeview", "Pine Ridge")
LINES = (1, 2, 3, 4)
PART_DEFECTS = (
    ("door seal", "seal gap"),
    ("brake caliper", "surface blemish"),
    ("wire harness", "connector misalignment"),
    ("instrument panel", "fit variation"),
    ("seat frame", "weld porosity"),
)


def generate_synthetic_data(
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return deterministic defect and downtime rows for the six-month window."""
    rng = random.Random(seed)
    defects: list[dict[str, Any]] = []
    downtime: list[dict[str, Any]] = []

    current_date = START_DATE
    while current_date <= END_DATE:
        for plant in PLANTS:
            for line in LINES:
                # Ordinary variation stays low; only the specified combination
                # receives a deliberately increasing two-week defect count.
                ordinary_count = rng.choices((0, 1, 2), weights=(0.35, 0.55, 0.10))[0]
                for _ in range(ordinary_count):
                    ordinary_part_defects = PART_DEFECTS
                    if (
                        plant == SPIKE_PLANT
                        and line == SPIKE_LINE
                        and SPIKE_START <= current_date <= SPIKE_END
                    ):
                        ordinary_part_defects = PART_DEFECTS[1:]
                    part, defect_type = rng.choice(ordinary_part_defects)
                    severity = rng.choices((1, 2, 3, 4, 5), weights=(0.10, 0.35, 0.35, 0.17, 0.03))[0]
                    defects.append(
                        {
                            "id": len(defects) + 1,
                            "plant": plant,
                            "line": line,
                            "part": part,
                            "defect_type": defect_type,
                            "severity": severity,
                            "date": current_date,
                        }
                    )

                if (
                    plant == SPIKE_PLANT
                    and line == SPIKE_LINE
                    and SPIKE_START <= current_date <= SPIKE_END
                ):
                    day_number = (current_date - SPIKE_START).days
                    spike_count = 3 + day_number
                    for _ in range(spike_count):
                        defects.append(
                            {
                                "id": len(defects) + 1,
                                "plant": SPIKE_PLANT,
                                "line": SPIKE_LINE,
                                "part": SPIKE_PART,
                                "defect_type": "seal gap",
                                "severity": rng.choices((3, 4, 5), weights=(0.15, 0.55, 0.30))[0],
                                "date": current_date,
                            }
                        )
                    downtime.append(
                        {
                            "id": len(downtime) + 1,
                            "plant": SPIKE_PLANT,
                            "line": SPIKE_LINE,
                            "reason": "door-seal rework during containment",
                            "minutes": 25 + day_number * 4,
                            "date": current_date,
                        }
                    )

                # Other lines occasionally experience routine downtime.
                if rng.random() < 0.035:
                    downtime.append(
                        {
                            "id": len(downtime) + 1,
                            "plant": plant,
                            "line": line,
                            "reason": rng.choice(("equipment adjustment", "material changeover", "quality check")),
                            "minutes": rng.randint(8, 45),
                            "date": current_date,
                        }
                    )
        current_date += timedelta(days=1)

    return defects, downtime


def populate_database(engine: Engine, seed: int = SEED) -> tuple[int, int]:
    """Create tables, clear previous generated rows, and insert reproducible data."""
    Base.metadata.create_all(engine)
    defects, downtime = generate_synthetic_data(seed)
    with Session(engine) as session, session.begin():
        session.execute(delete(ProductionDefect))
        session.execute(delete(DowntimeEvent))
        session.execute(ProductionDefect.__table__.insert(), defects)
        session.execute(DowntimeEvent.__table__.insert(), downtime)
    return len(defects), len(downtime)


def main() -> None:
    """Populate the configured database and report the number of inserted rows."""
    engine = create_database_engine()
    try:
        defect_count, downtime_count = populate_database(engine)
        print(f"Loaded {defect_count} production defects and {downtime_count} downtime events.")
        print(
            f"Injected door-seal spike: {SPIKE_PLANT}, Line {SPIKE_LINE}, {SPIKE_PART}, "
            f"{SPIKE_START} through {SPIKE_END}."
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
