"""SQLGlot-based validator for read-only data-agent queries."""

from __future__ import annotations

from sqlglot import exp, parse

ALLOWED_TABLE_COLUMNS = {
    "production_defects": {"id", "plant", "line", "part", "defect_type", "severity", "date"},
    "downtime_events": {"id", "plant", "line", "reason", "minutes", "date"},
}
ALLOWED_FUNCTIONS = {
    "ABS", "AVG", "COALESCE", "COUNT", "DATE_TRUNC", "EXTRACT", "LOWER",
    "MAX", "MIN", "ROUND", "SUM", "TIMESTAMP_TRUNC", "UPPER",
}
FORBIDDEN_EXPRESSIONS = (
    exp.Alter,
    exp.Command,
    exp.Create,
    exp.Delete,
    exp.Drop,
    exp.Grant,
    exp.Insert,
    exp.Into,
    exp.Lock,
    exp.Merge,
    exp.Revoke,
    exp.TruncateTable,
    exp.Update,
)


class SQLValidationError(ValueError):
    """Raised when SQL is invalid or outside the read-only allow-list."""


def _contains_comment(sql: str) -> bool:
    """Find SQL comment openers outside quoted strings and identifiers."""
    index = 0
    quote: str | None = None
    dollar_quote: str | None = None
    while index < len(sql):
        if dollar_quote:
            if sql.startswith(dollar_quote, index):
                index += len(dollar_quote)
                dollar_quote = None
            else:
                index += 1
            continue
        char = sql[index]
        if quote:
            if char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            elif char == "\\" and quote == "'":
                index += 2
                continue
            index += 1
            continue
        if sql.startswith("--", index) or sql.startswith("/*", index):
            return True
        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            continue
        if char == "$":
            close = sql.find("$", index + 1)
            if close >= 0:
                tag = sql[index + 1 : close]
                if all(part.replace("_", "a").isalnum() for part in [tag]) and not tag[:1].isdigit():
                    dollar_quote = sql[index : close + 1]
                    index = close + 1
                    continue
        index += 1
    return False


def validate_sql(sql: str, row_limit: int = 100) -> str:
    """Validate and cap one PostgreSQL SELECT over the two approved tables."""
    if not sql.strip():
        raise SQLValidationError("SQL query is empty")
    if row_limit < 1:
        raise ValueError("row_limit must be positive")
    if _contains_comment(sql):
        raise SQLValidationError("SQL comments are not allowed")

    try:
        statements = parse(sql, read="postgres")
    except Exception as error:
        raise SQLValidationError("SQL could not be parsed") from error
    if len(statements) != 1 or statements[0] is None:
        raise SQLValidationError("Exactly one SQL statement is allowed")

    expression = statements[0]
    if not isinstance(expression, exp.Select):
        raise SQLValidationError("Only SELECT statements are allowed")
    if expression.find(exp.With):
        raise SQLValidationError("Common table expressions are not allowed")
    for node in expression.walk():
        if isinstance(node, FORBIDDEN_EXPRESSIONS):
            raise SQLValidationError("The statement contains a forbidden operation")
        if isinstance(node, exp.Func) and node.sql_name().upper() not in ALLOWED_FUNCTIONS:
            raise SQLValidationError("The query uses a function outside the safe allow-list")

    tables = list(expression.find_all(exp.Table))
    if not tables:
        raise SQLValidationError("The query must read an allow-listed table")
    for table in tables:
        table_name = table.name.lower()
        schema_name = table.db.lower() if table.db else None
        catalog_name = table.catalog.lower() if table.catalog else None
        if table_name not in ALLOWED_TABLE_COLUMNS:
            raise SQLValidationError(f"Table is not allow-listed: {table_name}")
        if schema_name not in {None, "public"} or catalog_name is not None:
            raise SQLValidationError("Only tables in the public schema may be queried")
        if not isinstance(table.this, exp.Identifier):
            raise SQLValidationError("Table-valued functions are not allowed")

    allowed_columns = set().union(*ALLOWED_TABLE_COLUMNS.values())
    allowed_columns.update(alias.alias.lower() for alias in expression.find_all(exp.Alias))
    for column in expression.find_all(exp.Column):
        if column.name != "*" and column.name.lower() not in allowed_columns:
            raise SQLValidationError(f"Column is not allow-listed: {column.name}")

    limit = expression.args.get("limit")
    if limit is None:
        expression = expression.limit(row_limit, copy=False)
    else:
        literal = limit.expression
        if not isinstance(literal, exp.Literal) or not literal.is_int:
            raise SQLValidationError("LIMIT must be a non-negative integer literal")
        if int(literal.this) > row_limit:
            expression = expression.limit(row_limit, copy=False)

    return expression.sql(dialect="postgres")
