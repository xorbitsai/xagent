"""Tests for the Planner MCP connector seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
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
        assert "builtin_provenance" in str(row[2])


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


def test_upgrade_rejects_unmarked_existing_planner_row(tmp_path):
    """A custom row cannot be silently reclassified as the builtin Planner."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, launch_config) VALUES "
                "('planner', 'Custom Planner', 'stdio', '{}')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="no matching builtin_provenance"):
                migration.upgrade()
        assert "planner" in _app_ids(connection)


@pytest.mark.parametrize("colliding_app_id", ["Planner", " planner ", "PLANNER"])
def test_upgrade_rejects_normalized_planner_app_id_collision(
    tmp_path, colliding_app_id
):
    """Case/whitespace variants must fail closed instead of leaving two
    catalog rows that runtime lookup later rejects as ambiguous."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, launch_config) VALUES "
                "(:app_id, 'Custom Planner', 'stdio', '{}')"
            ),
            {"app_id": colliding_app_id},
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="normalized builtin identity"):
                migration.upgrade()
        assert _app_ids(connection) == {colliding_app_id}


def test_downgrade_preserves_row_without_planner_provenance(tmp_path):
    """Rollback deletes only the row carrying this migration's ownership marker."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, launch_config) VALUES "
                "('planner', 'Planner', 'oauth', "
                '\'{"builtin_provenance": {"registry": "custom", '
                '"app_id": "planner", "version": 1}}\')'
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert "planner" in _app_ids(connection)


def test_downgrade_skips_delete_when_snapshot_columns_missing(tmp_path):
    """downgrade() must skip deletion without ownership columns."""
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


def test_downgrade_skips_delete_when_only_app_id_column_present(tmp_path):
    """A table with app_id but no provenance storage must fail closed."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    app_id VARCHAR(100) NOT NULL UNIQUE
                )
                """
            )
        )
        connection.execute(
            text("INSERT INTO public_mcp_apps (id, app_id) VALUES (1, 'planner')")
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert "planner" in _app_ids(connection)


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
