"""Tests for the shared seed-migration downgrade guard helper."""

import sqlalchemy as sa
from sqlalchemy import create_engine, text

from xagent.migrations.seed_helpers import delete_unmodified_seeded_rows

PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("name", sa.String),
    sa.column("description", sa.Text),
    sa.column("transport", sa.String),
    sa.column("provider_name", sa.String),
    sa.column("category", sa.String),
)

SEED_ROWS = [
    {
        "app_id": "widget",
        "name": "Widget",
        "description": "A seeded widget connector.",
        "transport": "oauth",
        "provider_name": "widget-co",
        "category": "Productivity",
    },
]


def _create_table(connection):
    connection.execute(
        text(
            """
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                transport VARCHAR(50) NOT NULL,
                provider_name VARCHAR(50),
                category VARCHAR(100)
            )
            """
        )
    )


def _app_ids(connection):
    return set(connection.execute(text("SELECT app_id FROM public_mcp_apps")).scalars())


def test_deletes_row_matching_the_seed_snapshot(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, description, transport,"
                " provider_name, category)"
                " VALUES ('widget', 'Widget', 'A seeded widget connector.', 'oauth',"
                " 'widget-co', 'Productivity')"
            )
        )
        delete_unmodified_seeded_rows(connection, PUBLIC_MCP_APPS_TABLE, SEED_ROWS)
        assert "widget" not in _app_ids(connection)


def test_preserves_row_with_same_id_but_different_fields(tmp_path):
    """A row that reuses the seed's app_id but was created or edited by an
    operator (any field differs from the seed snapshot) must not be deleted."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, description, transport,"
                " provider_name, category)"
                " VALUES ('widget', 'Custom Widget', 'An operator-created connector.',"
                " 'stdio', NULL, 'Custom')"
            )
        )
        delete_unmodified_seeded_rows(connection, PUBLIC_MCP_APPS_TABLE, SEED_ROWS)
        assert "widget" in _app_ids(connection)


def test_noop_when_table_missing(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
        delete_unmodified_seeded_rows(connection, PUBLIC_MCP_APPS_TABLE, SEED_ROWS)


def test_skips_missing_match_columns(tmp_path):
    """Columns absent from the current schema (e.g. dropped by a later
    migration) are skipped rather than blocking the delete."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    app_id VARCHAR(100) NOT NULL UNIQUE,
                    name VARCHAR(200) NOT NULL,
                    transport VARCHAR(50) NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, transport)"
                " VALUES ('widget', 'Widget', 'oauth')"
            )
        )
        delete_unmodified_seeded_rows(connection, PUBLIC_MCP_APPS_TABLE, SEED_ROWS)
        assert "widget" not in _app_ids(connection)
