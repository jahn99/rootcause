"""Integration test proving the configured Postgres role cannot write."""

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError, OperationalError

from app.agents.data_agent import execute_readonly_sql, get_table_schema
from app.db.readonly_role import ensure_readonly_role

ADMIN_URL = os.getenv(
    "ROOTCAUSE_TEST_ADMIN_DATABASE_URL",
    "postgresql+psycopg://rootcause:rootcause@127.0.0.1:5432/rootcause",
)
READONLY_PASSWORD = os.getenv("READONLY_DB_PASSWORD", "rootcause_readonly")
READONLY_URL = os.getenv(
    "ROOTCAUSE_TEST_READONLY_DATABASE_URL",
    f"postgresql+psycopg://rootcause_readonly:{READONLY_PASSWORD}@127.0.0.1:5432/rootcause",
)


def test_readonly_role_can_select_but_cannot_write() -> None:
    admin_engine = create_engine(ADMIN_URL, pool_pre_ping=True)
    readonly_engine = create_engine(READONLY_URL, pool_pre_ping=True)
    try:
        try:
            with admin_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except OperationalError as error:
            pytest.skip(f"PostgreSQL is not available for the role integration test: {error}")

        ensure_readonly_role(admin_engine, READONLY_PASSWORD)
        with admin_engine.connect() as connection:
            public_tables = connection.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            ).scalars()
            granted_tables = {
                table
                for table in public_tables
                if connection.execute(
                    text("SELECT has_table_privilege('rootcause_readonly', :table, 'SELECT')"),
                    {"table": f"public.{table}"},
                ).scalar_one()
            }
        assert granted_tables == {"production_defects", "downtime_events"}

        with readonly_engine.connect() as connection:
            connection.execute(text("SELECT COUNT(*) FROM production_defects")).scalar_one()
        schema = get_table_schema(readonly_engine)
        assert "public.production_defects" in schema
        assert "public.downtime_events" in schema
        assert execute_readonly_sql(
            readonly_engine, "SELECT COUNT(*) AS row_count FROM production_defects LIMIT 100", 3000
        )

        with pytest.raises(DBAPIError):
            with readonly_engine.begin() as connection:
                connection.execute(
                    text("UPDATE production_defects SET severity = severity WHERE false")
                )
    finally:
        readonly_engine.dispose()
        admin_engine.dispose()
