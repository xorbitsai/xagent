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


def test_handles_table_clause_missing_a_matched_column(tmp_path):
    """A caller-supplied sa.table() that declares fewer columns than the real
    schema (a common lightweight pattern) must not raise KeyError; the column
    is still matched by name via sa.column()."""
    narrow_table = sa.table(
        "public_mcp_apps",
        sa.column("app_id", sa.String),
        sa.column("name", sa.String),
        # "category" exists in the DB and in match_columns, but is not
        # declared on this TableClause.
    )
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
        delete_unmodified_seeded_rows(connection, narrow_table, SEED_ROWS)
        assert "widget" not in _app_ids(connection)


def test_skips_row_missing_the_id_column(tmp_path):
    """A seed row without the id_column key is skipped rather than raising
    KeyError, and other rows are still processed normally."""
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
        rows_without_id = [{"name": "Widget", "transport": "oauth"}]
        delete_unmodified_seeded_rows(
            connection, PUBLIC_MCP_APPS_TABLE, rows_without_id
        )
        assert "widget" in _app_ids(connection)


def test_preserves_row_when_only_icon_or_visibility_differs(tmp_path):
    """An admin can edit a builtin row's icon/is_visible_in_connector without
    touching name/description/transport/provider_name/category (see
    admin_mcp.py's _BUILTIN_PROTECTED_FIELDS). That edit alone must still be
    enough to preserve the row on downgrade."""
    full_table = sa.table(
        "public_mcp_apps",
        sa.column("app_id", sa.String),
        sa.column("name", sa.String),
        sa.column("description", sa.Text),
        sa.column("transport", sa.String),
        sa.column("provider_name", sa.String),
        sa.column("category", sa.String),
        sa.column("icon", sa.String),
        sa.column("is_visible_in_connector", sa.Boolean),
    )
    seed_rows = [
        {
            **SEED_ROWS[0],
            "icon": "https://example.com/widget.png",
            "is_visible_in_connector": True,
        }
    ]
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
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
                    category VARCHAR(100),
                    icon VARCHAR(1000),
                    is_visible_in_connector BOOLEAN NOT NULL DEFAULT 1
                )
                """
            )
        )
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, description, transport,"
                " provider_name, category, icon, is_visible_in_connector)"
                " VALUES ('widget', 'Widget', 'A seeded widget connector.', 'oauth',"
                " 'widget-co', 'Productivity', 'https://example.com/widget.png', 0)"
            )
        )
        delete_unmodified_seeded_rows(connection, full_table, seed_rows)
        assert "widget" in _app_ids(connection)


def test_noop_when_no_match_columns_exist_in_schema(tmp_path):
    """If none of match_columns exist in the current schema, provenance can't
    be verified at all, so nothing is deleted (matching on id_column alone
    would reproduce the exact bug this helper prevents)."""
    narrow_table = sa.table(
        "public_mcp_apps",
        sa.column("app_id", sa.String),
        sa.column("notes", sa.String),
    )
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    app_id VARCHAR(100) NOT NULL UNIQUE,
                    notes VARCHAR(200)
                )
                """
            )
        )
        connection.execute(
            text("INSERT INTO public_mcp_apps (app_id, notes) VALUES ('widget', 'x')")
        )
        delete_unmodified_seeded_rows(connection, narrow_table, SEED_ROWS)
        assert "widget" in _app_ids(connection)


def test_matches_on_id_only_when_match_columns_explicitly_empty(tmp_path):
    """An explicit empty match_columns is a deliberate opt-out, distinct from
    the schema-drift no-op case above."""
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
        delete_unmodified_seeded_rows(
            connection, PUBLIC_MCP_APPS_TABLE, SEED_ROWS, match_columns=()
        )
        assert "widget" not in _app_ids(connection)


def test_skips_row_missing_a_match_column_key(tmp_path):
    """A seed row missing one of the match_columns keys still matches on the
    remaining fields instead of raising KeyError."""
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
        rows_without_category = [
            {
                "app_id": "widget",
                "name": "Widget",
                "description": "A seeded widget connector.",
                "transport": "oauth",
                "provider_name": "widget-co",
            }
        ]
        delete_unmodified_seeded_rows(
            connection, PUBLIC_MCP_APPS_TABLE, rows_without_category
        )
        assert "widget" not in _app_ids(connection)
