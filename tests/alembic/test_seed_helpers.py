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


def test_handles_table_clause_declaring_no_columns_at_all(tmp_path):
    """A bare sa.table("name") with zero declared columns (id_column falls
    back to sa.column() too, not just the match columns) must not produce a
    FROM-less SELECT: every column reference resolves via sa.column(), so the
    query needs an explicit select_from(table) to still target the right
    table instead of raising "no such column"."""
    bare_table = sa.table("public_mcp_apps")
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
        delete_unmodified_seeded_rows(connection, bare_table, SEED_ROWS)
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


def test_preserves_row_when_only_launch_config_or_oauth_scopes_differ(tmp_path):
    """A row that predates this app_id being seeded (or predates the admin
    API's built-in protections) can carry arbitrary oauth_scopes/launch_config
    even if every scalar field happens to match the seed. That JSON-only
    difference alone must be enough to preserve the row on downgrade."""
    full_table = sa.table(
        "public_mcp_apps",
        sa.column("app_id", sa.String),
        sa.column("name", sa.String),
        sa.column("description", sa.Text),
        sa.column("transport", sa.String),
        sa.column("provider_name", sa.String),
        sa.column("category", sa.String),
        sa.column("oauth_scopes", sa.JSON),
        sa.column("launch_config", sa.JSON),
    )
    seed_rows = [
        {
            **SEED_ROWS[0],
            "oauth_scopes": ["widget.read"],
            "launch_config": {"command": "python", "args": ["-m", "widget"]},
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
                    oauth_scopes JSON,
                    launch_config JSON
                )
                """
            )
        )
        connection.execute(
            sa.insert(full_table),
            [
                {
                    "app_id": "widget",
                    "name": "Widget",
                    "description": "A seeded widget connector.",
                    "transport": "oauth",
                    "provider_name": "widget-co",
                    "category": "Productivity",
                    "oauth_scopes": ["widget.read", "widget.write"],
                    "launch_config": {"command": "custom-runner"},
                }
            ],
        )
        delete_unmodified_seeded_rows(connection, full_table, seed_rows)
        assert "widget" in _app_ids(connection)


def test_deletes_row_when_json_columns_match_with_different_key_order(tmp_path):
    """JSON-typed columns compare by value in Python, not by raw SQL/text
    equality, so a stored value that serializes with a different key order
    than the seed's dict still matches and the row is deleted."""
    full_table = sa.table(
        "public_mcp_apps",
        sa.column("app_id", sa.String),
        sa.column("name", sa.String),
        sa.column("description", sa.Text),
        sa.column("transport", sa.String),
        sa.column("provider_name", sa.String),
        sa.column("category", sa.String),
        sa.column("launch_config", sa.JSON),
    )
    seed_rows = [
        {
            **SEED_ROWS[0],
            "launch_config": {"command": "python", "args": ["-m", "widget"]},
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
                    launch_config JSON
                )
                """
            )
        )
        # Raw JSON text with keys in the opposite order from the seed dict
        # above; a raw SQL/text equality comparison would not match this.
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, description, transport,"
                " provider_name, category, launch_config)"
                " VALUES ('widget', 'Widget', 'A seeded widget connector.', 'oauth',"
                " 'widget-co', 'Productivity',"
                ' \'{"args": ["-m", "widget"], "command": "python"}\')'
            )
        )
        delete_unmodified_seeded_rows(connection, full_table, seed_rows)
        assert "widget" not in _app_ids(connection)


def test_preserves_row_when_json_column_untyped_on_table_clause(tmp_path):
    """If the caller's TableClause leaves a JSON-typed column untyped (the
    "lightweight subset of columns" pattern this module otherwise supports),
    that column comes back as a raw driver value instead of a deserialized
    Python object, so it can never compare equal to the seed's Python value.
    This must fail safe (row preserved), not raise and not delete."""
    untyped_launch_config_table = sa.table(
        "public_mcp_apps",
        sa.column("app_id", sa.String),
        sa.column("name", sa.String),
        sa.column("description", sa.Text),
        sa.column("transport", sa.String),
        sa.column("provider_name", sa.String),
        sa.column("category", sa.String),
        sa.column("launch_config"),  # no sa.JSON type declared
    )
    seed_rows = [
        {
            **SEED_ROWS[0],
            "launch_config": {"command": "python", "args": ["-m", "widget"]},
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
                    launch_config JSON
                )
                """
            )
        )
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, description, transport,"
                " provider_name, category, launch_config)"
                " VALUES ('widget', 'Widget', 'A seeded widget connector.', 'oauth',"
                " 'widget-co', 'Productivity',"
                ' \'{"command": "python", "args": ["-m", "widget"]}\')'
            )
        )
        delete_unmodified_seeded_rows(
            connection, untyped_launch_config_table, seed_rows
        )
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
