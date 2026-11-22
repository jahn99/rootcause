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
    columns = {"plant": ProductionDefect.plant, "line": ProductionDefect.line}
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
    return "\n".join(descriptions)


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
        generation_prompt = (
            "Produce one PostgreSQL SELECT query as JSON using the supplied schema. "
            "Use only the listed tables and columns. Do not use comments, CTEs, "
            "functions outside ordinary aggregation, or any write operation. "
            "The query will be independently validated and run with a row cap. "
            "Return only an object with a single string property named sql.\n\n"
            f"Schema:\n{schema}\n\nQuestion: {question}"
        )
        validation_error: Exception | None = None
        safe_sql: str | None = None
        raw = ""
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

        rows = execute_readonly_sql(self.engine, safe_sql, timeout_ms=self.timeout_ms)
        answer_prompt = (
            "Answer the user's database question in one short plain-language sentence. "
            "Use only the query result JSON. If it is empty, say no records were found. "
            "Do not reveal credentials or invent values.\n\n"
            f"Question: {question}\nSQL: {safe_sql}\nRows: {json.dumps(rows, default=str)}"
        )
        answer = await self.llm.generate(answer_prompt)
        return DataAgentResult(sql=safe_sql, rows=rows, answer=answer.strip(), refused=False)
