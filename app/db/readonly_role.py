"""Provision the least-privilege database role used by the SQL agent."""

from psycopg import sql
from sqlalchemy import Engine

READONLY_ROLE = "rootcause_readonly"
ALLOWED_TABLES = ("production_defects", "downtime_events")


def ensure_readonly_role(engine: Engine, password: str) -> None:
    """Create or reset a login role with SELECT on the two approved tables only."""
    if not password:
        raise ValueError("READONLY_DB_PASSWORD must not be empty")

    connection = engine.raw_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (READONLY_ROLE,))
            role_exists = cursor.fetchone() is not None
            verb = "ALTER ROLE" if role_exists else "CREATE ROLE"
            cursor.execute(
                sql.SQL("{} {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS")
                .format(sql.SQL(verb), sql.Identifier(READONLY_ROLE), sql.Literal(password))
            )
            cursor.execute(
                sql.SQL("REVOKE ALL PRIVILEGES ON DATABASE {} FROM {}")
                .format(sql.Identifier(engine.url.database), sql.Identifier(READONLY_ROLE))
            )
            cursor.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                    sql.Identifier(engine.url.database), sql.Identifier(READONLY_ROLE)
                )
            )
            cursor.execute(
                sql.SQL("REVOKE TEMPORARY ON DATABASE {} FROM PUBLIC").format(
                    sql.Identifier(engine.url.database)
                )
            )
            cursor.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
            cursor.execute(
                sql.SQL("REVOKE ALL PRIVILEGES ON SCHEMA public FROM {}")
                .format(sql.Identifier(READONLY_ROLE))
            )
            cursor.execute(
                sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(READONLY_ROLE))
            )
            cursor.execute("REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM PUBLIC")
            cursor.execute(
                sql.SQL("REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM {}")
                .format(sql.Identifier(READONLY_ROLE))
            )
            for table_name in ALLOWED_TABLES:
                cursor.execute(
                    "SELECT to_regclass(%s)", (f"public.{table_name}",)
                )
                if cursor.fetchone()[0] is not None:
                    cursor.execute(
                        sql.SQL("GRANT SELECT ON TABLE public.{} TO {}").format(
                            sql.Identifier(table_name), sql.Identifier(READONLY_ROLE)
                        )
                    )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
