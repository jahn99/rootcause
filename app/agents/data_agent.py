"""Read-only natural-language data agent backed by validated SQL."""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy import Engine, inspect, select, text

from app.db.readonly_role import ALLOWED_TABLES, READONLY_ROLE
from app.db.models import ProductionDefect
from app.llm.base import LLMProvider
from app.agents.sql_validator import SQLValidationError, validate_sql

SQL_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"sql": {"type": "string"}},
    "required": ["sql"],
    "additionalProperties": False,
}
logger = logging.getLogger(__name__)
ISO_DATES = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
DAY_RANGE = re.compile(
    r"\bbetween\s+(\d{1,2})\s+(?:and|to|through)\s+(\d{1,2})\s+"
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+(20\d{2})\b",
    re.IGNORECASE,
)


class GeneratedSQL(BaseModel):
    """Validated JSON payload requested from the LLM."""

    sql: str


class DataAgentResult(BaseModel):
    """Final SQL-agent output for the HTTP endpoint."""

    sql: str | None
    rows: list[dict[str, Any]]
    answer: str
    refused: bool = False


class UnsafeQuestion(ValueError):
    """Raised only by the intent check for modifying or credential requests."""


class SQLGenerationError(ValueError):
    """Raised when generated SQL remains invalid after its single retry."""


def get_distinct_values(engine: Engine, column_name: str) -> set[str | int]:
    """Fetch values from a fixed allow-list of real manufacturing columns."""
    columns = {
        "plant": ProductionDefect.plant,
        "line": ProductionDefect.line,
        "part": ProductionDefect.part,
    }
    if column_name not in columns:
        raise ValueError("Only plant and line values can be checked")
    with engine.connect() as connection, connection.begin():
        if engine.dialect.name == "postgresql":
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        values = connection.execute(select(columns[column_name]).distinct()).scalars()
        return set(values)


def _unknown_location(question: str, engine: Engine) -> str | None:
    """Return a mentioned plant or line that is absent from production data."""
    line_match = re.search(r"\bline\s*(\d+)\b", question, flags=re.IGNORECASE)
    if line_match:
        line = int(line_match.group(1))
        if line not in get_distinct_values(engine, "line"):
            return f"Line {line}"

    plant_match = re.search(
        r"\b(?:at\s+(?:the\s+)?|in\s+(?:the\s+)?)([A-Z][a-z]+(?:[ -][A-Z][a-z]+){0,2})",
        question,
    )
    if plant_match:
        candidate = plant_match.group(1).strip(" ,-.")
        if candidate.casefold() not in {"may", "june", "july", "august", "september", "october", "november", "december", "january", "february", "march", "april"}:
            plants = {str(value).casefold() for value in get_distinct_values(engine, "plant")}
            if candidate.casefold() not in plants:
                return candidate
    return None


def refusal_reason(question: str) -> str | None:
    """Detect direct write requests and attempts to retrieve secrets."""
    normalized = " ".join(question.lower().split())
    if any(term in normalized for term in ("password", "api key", "secret", "credentials", ".env")):
        return "credentials"
    if re.search(r"\b(delete|drop|insert|update|alter|create|truncate|grant|revoke|modify|write|remove)\b", normalized):
        return "modification"
    return None


def get_table_schema(engine: Engine) -> str:
    """Read column names and types for only the two approved tables."""
    inspector = inspect(engine)
    descriptions: list[str] = []
    for table in ALLOWED_TABLES:
        columns = inspector.get_columns(table, schema="public")
        if not columns:
            raise RuntimeError(f"Required table is unavailable: {table}")
        column_text = ", ".join(f"{column['name']} {column['type']}" for column in columns)
        descriptions.append(f"public.{table}({column_text})")
    parts = sorted(str(value) for value in get_distinct_values(engine, "part"))
    descriptions.append(f"Stored production_defects.part values: {', '.join(parts)}")
    return "\n".join(descriptions)


def _normalized_phrase(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _part_mentioned(question: str, actual_parts: set[str | int]) -> str | None:
    """Resolve an actual stored part value mentioned in the user's question."""
    normalized_question = f" {_normalized_phrase(question)} "
    for part in sorted((str(value) for value in actual_parts), key=len, reverse=True):
        normalized_part = _normalized_phrase(part)
        if f" {normalized_part} " in normalized_question:
            return part
        singular = re.sub(r"s\b", "", normalized_part)
        if singular != normalized_part and f" {singular} " in normalized_question:
            return part
    return None


def _question_date_bounds(question: str) -> tuple[date | None, date | None]:
    """Extract common explicit date ranges and years without guessing dates."""
    match = DAY_RANGE.search(question)
    if match:
        month = datetime.strptime(match.group(3)[:3].title(), "%b").month
        year = int(match.group(4))
        first = date(year, month, int(match.group(1)))
        last = date(year, month, int(match.group(2)))
        return min(first, last), max(first, last)
    dates = [date.fromisoformat(value) for value in ISO_DATES.findall(question)]
    if dates:
        return min(dates), max(dates)
    years = re.findall(r"\b(20\d{2})\b", question)
    if years:
        year = int(years[0])
        return date(year, 1, 1), date(year, 12, 31)
    return None, None


def _planned_aggregate_query(question: str, engine: Engine) -> tuple[str, dict[str, Any]] | None:
    """Use safe schema-grounded templates for common part-count and severity comparisons."""
    lowered = question.casefold()
    comparison = any(
        term in lowered
        for term in (
            "compared with",
            "compared to",
            "versus",
            "vs",
            "other parts",
            "other part categories",
        )
    )
    avg_intent = "average severity" in lowered or "average" in lowered and "severity" in lowered
    count_intent = any(term in lowered for term in ("how many", "count", "number of")) and "defect" in lowered
    if not (avg_intent and comparison or count_intent):
        return None

    part = _part_mentioned(question, get_distinct_values(engine, "part"))
    if part is None:
        return None
    start_date, end_date = _question_date_bounds(question)
    clauses = ["LOWER(part) = LOWER(:part)"] if count_intent else []
    parameters: dict[str, Any] = {"part": part}

    plants = {str(value) for value in get_distinct_values(engine, "plant")}
    normalized_question = _normalized_phrase(question)
    plant = next(
        (value for value in sorted(plants, key=len, reverse=True) if _normalized_phrase(value) in normalized_question),
        None,
    )
    if count_intent and plant:
        clauses.append("plant = :plant")
        parameters["plant"] = plant
    line_match = re.search(r"\bline\s*(\d+)\b", question, re.IGNORECASE)
    if count_intent and line_match:
        line = int(line_match.group(1))
        if line in get_distinct_values(engine, "line"):
            clauses.append("line = :line")
            parameters["line"] = line
    if start_date:
        clauses.append("date >= :start_date")
        parameters["start_date"] = start_date.isoformat()
    if end_date:
        clauses.append("date <= :end_date")
        parameters["end_date"] = end_date.isoformat()

    if avg_intent and comparison:
        sql = (
            "SELECT CASE WHEN LOWER(part) = LOWER(:part) THEN 'requested part' ELSE 'other parts' END AS part_group, "
            "ROUND(AVG(severity)::numeric, 3) AS average_severity, COUNT(*) AS defect_count "
            "FROM production_defects"
        )
        # The requested part contributes only the grouping expression, not a WHERE filter.
        clauses = [clause for clause in clauses if not clause.startswith("LOWER(part)")]
        sql += (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql += " GROUP BY part_group ORDER BY part_group"
        return sql, parameters

    if count_intent:
        sql = "SELECT COUNT(*) AS defect_count FROM production_defects"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return sql, parameters
    return None


def _row_facts(rows: list[dict[str, Any]]) -> str:
    """Format all returned fields so a generated answer cannot omit their values."""
    return "; ".join(
        ", ".join(f"{key}={value}" for key, value in row.items())
        for row in rows
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def execute_readonly_sql(
    engine: Engine,
    query: str,
    timeout_ms: int,
    parameters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one already-validated SQL statement in a read-only timed transaction."""
    with engine.connect() as connection, connection.begin():
        if engine.dialect.name == "postgresql":
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            connection.exec_driver_sql(f"SET LOCAL statement_timeout = '{int(timeout_ms)}ms'")
        result = connection.execute(text(query), parameters or {})
        return [
            {str(key): _json_safe(value) for key, value in row._mapping.items()}
            for row in result.fetchall()
        ]


class DataAgent:
    """Generate SQL from a question, validate it, execute safely, and summarize rows."""

    def __init__(
        self,
        llm: LLMProvider,
        engine: Engine,
        row_limit: int = 100,
        timeout_ms: int = 3000,
    ) -> None:
        if engine.url.username != READONLY_ROLE:
            raise ValueError(f"SQL agent must use the {READONLY_ROLE} database role")
        self.llm = llm
        self.engine = engine
        self.row_limit = row_limit
        self.timeout_ms = timeout_ms

    async def ask(self, question: str) -> DataAgentResult:
        """Answer a data question while refusing writes and credential requests."""
        intent = refusal_reason(question)
        if intent:
            return DataAgentResult(
                sql=None,
                rows=[],
                answer="Sorry, I can’t modify database records or disclose credentials.",
                refused=True,
            )

        unknown_location = _unknown_location(question, self.engine)
        if unknown_location:
            return DataAgentResult(
                sql=None,
                rows=[],
                answer=f"No data found for {unknown_location}.",
                refused=False,
            )

        schema = get_table_schema(self.engine)
        planned_query = _planned_aggregate_query(question, self.engine)
        parameters: dict[str, Any] = {}
        safe_sql: str | None = None
        if planned_query is not None:
            planned_sql, parameters = planned_query
            safe_sql = validate_sql(planned_sql, row_limit=self.row_limit)
        generation_prompt = (
            "Produce one PostgreSQL SELECT query as JSON using the supplied schema. "
            "Use only the listed tables and columns. Do not use comments, CTEs, "
            "functions outside ordinary aggregation, or any write operation. "
            "Use exact values listed for categorical columns; for part questions, filter "
            "production_defects.part, not defect_type. For comparisons with other parts, "
            "return separate aggregates for the requested part and all other parts, "
            "including average_severity and defect_count. "
            "The query will be independently validated and run with a row cap. "
            "Return only an object with a single string property named sql.\n\n"
            f"Schema:\n{schema}\n\nQuestion: {question}"
        )
        validation_error: Exception | None = None
        raw = ""
        if safe_sql is None:
            for attempt in range(2):
                prompt = generation_prompt
                if validation_error is not None:
                    prompt += (
                        f"\n\nThe previous output failed validation: {validation_error}. "
                        "Return a corrected complete JSON object with a valid SELECT query."
                    )
                raw = await self.llm.generate_json(
                    prompt, SQL_OUTPUT_SCHEMA, max_output_tokens=600
                )
                try:
                    generated = GeneratedSQL.model_validate(json.loads(raw))
                    safe_sql = validate_sql(generated.sql, row_limit=self.row_limit)
                    break
                except (json.JSONDecodeError, ValidationError, SQLValidationError) as error:
                    validation_error = error
                    logger.warning(
                        "SQL generation validation failed (attempt %d); finish_reason=%s raw_output=%r error=%s",
                        attempt + 1,
                        getattr(self.llm, "last_finish_reason", None),
                        getattr(self.llm, "last_raw_output", None)
                        if getattr(self.llm, "last_raw_output", None) is not None
                        else raw,
                        error,
                    )
        if safe_sql is None:
            raise SQLGenerationError(
                "The SQL agent could not produce a valid query after one retry. "
                "Check the question and try again."
            ) from validation_error

        sqlalchemy_sql = re.sub(r"%\(([A-Za-z_][A-Za-z0-9_]*)\)s", r":\1", safe_sql)
        if parameters:
            rows = execute_readonly_sql(
                self.engine, sqlalchemy_sql, timeout_ms=self.timeout_ms, parameters=parameters
            )
        else:
            rows = execute_readonly_sql(self.engine, sqlalchemy_sql, timeout_ms=self.timeout_ms)
        answer_prompt = (
            "Answer the user's database question in one short plain-language sentence. "
            "Use only the query result JSON. If it is empty, say no records were found. "
            "Do not reveal credentials or invent values.\n\n"
            f"Question: {question}\nSQL: {safe_sql}\nRows: {json.dumps(rows, default=str)}"
        )
        answer = (await self.llm.generate(answer_prompt)).strip()
        if rows:
            facts = _row_facts(rows)
            if "no records were found" in answer.casefold() or "no records found" in answer.casefold():
                answer = f"The SQL query returned {len(rows)} row(s): {facts}."
            else:
                answer = f"{answer} SQL result values: {facts}."
        else:
            answer = "No records were found."
        return DataAgentResult(sql=safe_sql, rows=rows, answer=answer, refused=False)
