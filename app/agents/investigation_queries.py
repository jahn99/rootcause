"""Fixed, parameterized SQL templates used by the investigation workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from sqlalchemy import Engine

from app.agents.data_agent import execute_readonly_sql
from app.agents.sql_validator import validate_sql
from app.db.readonly_role import READONLY_ROLE

PART_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,79}\Z")
PART_ALIASES = {
    "door seal": ("door seal", "door seals", "door-seal", "weatherstrip"),
    "brakes": ("brake caliper", "brakes", "brake"),
    "brake": ("brake caliper", "brake", "brakes"),
    "brake system": ("brake caliper", "brakes", "brake"),
    "electrical system": ("wire harness", "electrical", "wiring"),
    "airbag": ("airbag", "air bag", "inflator"),
    "seat belt": ("seat belt", "belt"),
    "fuel system": ("fuel", "fuel line"),
    "steering system": ("steering",),
}


@dataclass(frozen=True)
class InvestigationQueryResult:
    """One validated template query and its result rows."""

    name: str
    sql: str
    parameters: dict[str, Any]
    rows: list[dict[str, Any]]


def map_part_name(requested: str, actual_values: list[str]) -> str | None:
    """Resolve a safe part phrase to a real database value using aliases and fuzzy matching."""
    if not PART_NAME_PATTERN.fullmatch(requested.strip()):
        raise ValueError("The triaged part name contains unsupported characters")
    normalized = " ".join(requested.casefold().replace("-", " ").split())
    values_by_normalized = {
        " ".join(value.casefold().replace("-", " ").split()): value
        for value in actual_values
    }
    if normalized in values_by_normalized:
        return values_by_normalized[normalized]

    aliases = PART_ALIASES.get(normalized, (requested,))
    for alias in aliases:
        alias_normalized = " ".join(alias.casefold().replace("-", " ").split())
        if alias_normalized in values_by_normalized:
            return values_by_normalized[alias_normalized]

    best = max(
        actual_values,
        key=lambda value: max(
            SequenceMatcher(
                None,
                " ".join(alias.casefold().replace("-", " ").split()),
                " ".join(value.casefold().replace("-", " ").split()),
            ).ratio()
            for alias in aliases
        ),
        default=None,
    )
    if best is not None:
        similarity = max(
            SequenceMatcher(
                None,
                " ".join(alias.casefold().replace("-", " ").split()),
                " ".join(best.casefold().replace("-", " ").split()),
            ).ratio()
            for alias in aliases
        )
        if similarity >= 0.72:
            return best
    return None


def _date_filter(context: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    clauses: list[str] = []
    parameters: dict[str, Any] = {}
    if context.get("start_date"):
        clauses.append("date >= :start_date")
        parameters["start_date"] = context["start_date"]
    if context.get("end_date"):
        clauses.append("date <= :end_date")
        parameters["end_date"] = context["end_date"]
    return clauses, parameters


def _template_queries(context: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Build fixed query text and bound parameter dictionaries from triage values."""
    part = context.get("part")
    dates, date_params = _date_filter(context)
    if not part:
        if not context.get("top_part_query"):
            return []
        top_parts_sql = (
            "SELECT part, COUNT(*) AS defect_count FROM production_defects"
            + (" WHERE " + " AND ".join(dates) if dates else "")
            + " GROUP BY part ORDER BY defect_count DESC, part"
        )
        return [("top_parts", top_parts_sql, dict(date_params))]

    weekly_filters = ["LOWER(part) = LOWER(:part)"]
    weekly_params: dict[str, Any] = {"part": part, **date_params}
    for key, column in (("plant", "plant"), ("line", "line")):
        if context.get(key) is not None:
            weekly_filters.append(f"{column} = :{key}")
            weekly_params[key] = context[key]
    weekly_filters.extend(dates)
    weekly_sql = (
        "SELECT DATE_TRUNC('week', date) AS week, COUNT(*) AS defect_count "
        "FROM production_defects WHERE "
        + " AND ".join(weekly_filters)
        + " GROUP BY DATE_TRUNC('week', date) ORDER BY week"
    )

    part_filters = ["LOWER(part) = LOWER(:part)", *dates]
    count_sql = (
        "SELECT plant, line, COUNT(*) AS defect_count "
        "FROM production_defects WHERE "
        + " AND ".join(part_filters)
        + " GROUP BY plant, line ORDER BY defect_count DESC, plant, line"
    )

    average_sql = (
        "SELECT part, AVG(severity) AS average_severity, COUNT(*) AS defect_count "
        "FROM production_defects WHERE "
        + " AND ".join(dates or ["1 = 1"])
        + " GROUP BY part ORDER BY part"
    )
    average_params = dict(date_params)

    downtime_filters: list[str] = []
    downtime_params: dict[str, Any] = dict(date_params)
    for key, column in (("plant", "plant"), ("line", "line")):
        if context.get(key) is not None:
            downtime_filters.append(f"{column} = :{key}")
            downtime_params[key] = context[key]
    downtime_filters.extend(dates)
    downtime_sql = (
        "SELECT DATE_TRUNC('week', date) AS week, SUM(minutes) AS downtime_minutes "
        "FROM downtime_events"
        + (" WHERE " + " AND ".join(downtime_filters) if downtime_filters else "")
        + " GROUP BY DATE_TRUNC('week', date) ORDER BY week"
    )

    return [
        ("weekly_defects", weekly_sql, weekly_params),
        ("defects_by_plant_line", count_sql, {"part": part, **date_params}),
        ("severity_comparison", average_sql, average_params),
        ("weekly_downtime", downtime_sql, downtime_params),
    ]


class InvestigationQueryRunner:
    """Execute only the fixed validated templates using the read-only SQL-agent engine."""

    def __init__(self, engine: Engine, row_limit: int = 100, timeout_ms: int = 3000) -> None:
        if engine.dialect.name == "postgresql" and engine.url.username != READONLY_ROLE:
            raise ValueError(f"Investigation SQL must use the {READONLY_ROLE} database role")
        if engine.dialect.name not in {"postgresql", "sqlite"}:
            raise ValueError("Investigation query runner requires PostgreSQL")
        self.engine = engine
        self.row_limit = row_limit
        self.timeout_ms = timeout_ms

    def _execute(self, sql: str, parameters: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        validated = validate_sql(sql, row_limit=self.row_limit)
        # SQLGlot renders named placeholders in psycopg format; restore SQLAlchemy's
        # portable named bind syntax before executing with the explicit parameter map.
        sqlalchemy_sql = re.sub(r"%\(([A-Za-z_][A-Za-z0-9_]*)\)s", r":\1", validated)
        rows = execute_readonly_sql(
            self.engine, sqlalchemy_sql, self.timeout_ms, parameters=parameters
        )
        return validated, rows

    def distinct_part_values(self) -> list[str]:
        """Read actual part values with a validated SELECT on the read-only connection."""
        _, rows = self._execute(
            "SELECT DISTINCT part FROM production_defects ORDER BY part", {}
        )
        return [str(row["part"]) for row in rows if row.get("part") is not None]

    def resolve_part(self, requested: str) -> str | None:
        """Map triage's canonical part phrase to an actual stored part value."""
        if not PART_NAME_PATTERN.fullmatch(requested.strip()):
            raise ValueError("The triaged part name contains unsupported characters")
        return map_part_name(requested, self.distinct_part_values())

    def run_overviews(self, context: dict[str, Any]) -> list[InvestigationQueryResult]:
        """Run the four fixed investigation query templates independently."""
        results: list[InvestigationQueryResult] = []
        for name, sql, parameters in _template_queries(context):
            validated_sql, rows = self._execute(sql, parameters)
            if name == "severity_comparison":
                rows = self._combine_severity_rows(rows, str(context["part"]))
            results.append(InvestigationQueryResult(name, validated_sql, parameters, rows))
        return results

    @staticmethod
    def _combine_severity_rows(
        rows: list[dict[str, Any]], requested_part: str
    ) -> list[dict[str, Any]]:
        """Combine per-part SQL aggregates into the requested-versus-other comparison."""
        requested = " ".join(requested_part.casefold().replace("-", " ").split())
        matching = [
            row for row in rows
            if " ".join(str(row.get("part", "")).casefold().replace("-", " ").split()) == requested
        ]
        others = [row for row in rows if row not in matching]
        combined: list[dict[str, Any]] = []
        if matching:
            count = sum(int(row["defect_count"]) for row in matching)
            total = sum(float(row["average_severity"]) * int(row["defect_count"]) for row in matching)
            combined.append(
                {"part_group": "requested part", "average_severity": round(total / count, 2), "defect_count": count}
            )
        if others:
            count = sum(int(row["defect_count"]) for row in others)
            total = sum(float(row["average_severity"]) * int(row["defect_count"]) for row in others)
            combined.append(
                {"part_group": "all other parts", "average_severity": round(total / count, 2), "defect_count": count}
            )
        return combined
