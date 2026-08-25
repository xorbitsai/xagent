"""Tests for the Google Search Console MCP connector seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260824_seed_google_search_console_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_google_search_console_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_table(connection):
    connection.execute(
        text(
            """
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                icon VARCHAR(1000),
                transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                provider_name VARCHAR(50),
                category VARCHAR(100),
                oauth_scopes JSON,
                is_visible_in_connector BOOLEAN NOT NULL DEFAULT 1,
                launch_config JSON
            )
            """
        )
    )


def _app_ids(connection):
    return set(connection.execute(text("SELECT app_id FROM public_mcp_apps")).scalars())


def test_upgrade_inserts_google_search_console(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "google-search-console" in _app_ids(connection)
        row = connection.execute(
            text(
                "SELECT transport, provider_name, launch_config FROM public_mcp_apps"
                " WHERE app_id='google-search-console'"
            )
        ).first()
        assert row[0] == "oauth"
        assert row[1] == "google"
        assert "xagent.web.tools.mcp.google_search_console" in str(row[2])


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()  # second run must not raise or duplicate
        rows = connection.execute(
            text(
                "SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='google-search-console'"
            )
        ).scalar()
        assert rows == 1


def test_seed_row_matches_registry():
    """The migration snapshot and the runtime registry must define the same
    google-search-console row (the migration is a frozen copy; this catches
    drift)."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r
        for r in get_builtin_public_mcp_app_rows()
        if r["app_id"] == "google-search-console"
    )
    assert migration.ROW == registry_row


def test_app_id_does_not_collide_with_other_builtin_apps():
    """A duplicate app_id across the registry would violate the DB's UNIQUE
    constraint at seed time for whichever migration runs second; catch it
    directly here instead of relying on that indirect signal."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    app_ids = [row["app_id"] for row in get_builtin_public_mcp_app_rows()]
    assert app_ids.count("google-search-console") == 1


def test_upgrade_inserts_only_columns_present_on_older_schema(tmp_path):
    """A pre-existing deployment's public_mcp_apps table may predate a
    column ROW defines. upgrade() must insert only the columns that
    actually exist rather than failing or inserting into a nonexistent
    column.

    This fixture mirrors the table's actual migration history rather than
    an arbitrary subset: oauth_scopes was added at table creation
    (f1427c3a7261, 2026-04-21) while is_visible_in_connector was added over
    a month later (20260519_add_connector_visibility_to_public_mcp_apps) —
    so a real pre-2026-05-19 deployment has oauth_scopes but not
    is_visible_in_connector, never the other way around.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    app_id VARCHAR(100) NOT NULL UNIQUE,
                    name VARCHAR(200) NOT NULL,
                    description TEXT,
                    icon VARCHAR(1000),
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                    provider_name VARCHAR(50),
                    category VARCHAR(100),
                    oauth_scopes JSON,
                    launch_config JSON
                )
                """
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "google-search-console" in _app_ids(connection)
        row = connection.execute(
            text(
                "SELECT category, oauth_scopes, launch_config FROM public_mcp_apps"
                " WHERE app_id='google-search-console'"
            )
        ).first()
        assert row[0] == "Analytics"
        assert "webmasters.readonly" in str(row[1])
        assert "xagent.web.tools.mcp.google_search_console" in str(row[2])


def test_downgrade_removes_google_search_console(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "google-search-console" not in _app_ids(connection)


def test_downgrade_is_a_no_op_when_row_already_absent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()  # never upgraded; row was never inserted
        assert "google-search-console" not in _app_ids(connection)


def test_upgrade_and_downgrade_no_op_without_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        table_names = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).scalars()
        )
        assert "public_mcp_apps" not in table_names
