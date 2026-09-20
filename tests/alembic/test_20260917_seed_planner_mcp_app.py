"""Tests for the Planner MCP connector seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260917_seed_planner_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_planner_migration", migration_file
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


def test_upgrade_inserts_planner(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "planner" in _app_ids(connection)
        row = connection.execute(
            text(
                "SELECT transport, provider_name, launch_config FROM public_mcp_apps"
                " WHERE app_id='planner'"
            )
        ).first()
        assert row[0] == "oauth"
        assert row[1] == "microsoft"
        assert "xagent.web.tools.mcp.planner" in str(row[2])


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()  # second run must not raise or duplicate
        rows = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='planner'")
        ).scalar()
        assert rows == 1


def test_seed_row_matches_registry(tmp_path):
    """The migration snapshot and the runtime registry must define the same
    planner row (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "planner"
    )
    assert migration.ROW == registry_row


def test_downgrade_removes_planner(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "planner" not in _app_ids(connection)


def test_downgrade_preserves_operator_modified_row(tmp_path):
    """downgrade() must not delete a row that no longer matches the
    snapshot this migration seeded -- an operator could have hand-edited
    it (or hand-created a different app under the same app_id before
    upgrade() ever ran, since upgrade() no-ops on that collision)."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "UPDATE public_mcp_apps SET name='Custom Planner' "
                    "WHERE app_id='planner'"
                )
            )
            migration.downgrade()
        assert "planner" in _app_ids(connection)


def test_downgrade_skips_delete_when_snapshot_columns_missing(tmp_path):
    """sa.delete(table).where(*conditions) with an empty conditions list
    compiles to an unconditional DELETE FROM public_mcp_apps -- if none of
    the snapshot columns (app_id/name/description/transport) exist on the
    live table, downgrade() must skip the delete entirely rather than
    wiping every row in the shared catalog table."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    icon VARCHAR(1000)
                )
                """
            )
        )
        connection.execute(
            text("INSERT INTO public_mcp_apps (id, icon) VALUES (1, 'unrelated-row')")
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        remaining = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps")
        ).scalar()
        assert remaining == 1


def test_upgrade_warns_and_skips_missing_columns(tmp_path):
    """upgrade() must still insert a row when the live table is missing a
    column ROW defines (e.g. an out-of-order migration state), omitting
    only that column rather than failing the whole seed."""
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
                    is_visible_in_connector BOOLEAN NOT NULL DEFAULT 1,
                    launch_config JSON
                )
                """
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "planner" in _app_ids(connection)


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
