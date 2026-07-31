"""Tests for the Granola remote-MCP connector seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260731_seed_granola_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_granola_migration", migration_file
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


def test_upgrade_inserts_granola(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "granola" in _app_ids(connection)
        row = connection.execute(
            text(
                "SELECT transport, provider_name, launch_config FROM public_mcp_apps WHERE app_id='granola'"
            )
        ).first()
        assert row[0] == "streamable_http"
        assert row[1] is None
        assert "https://mcp.granola.ai/mcp" in str(row[2])
        assert "mcp_oauth" in str(row[2])


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()  # second run must not raise or duplicate
        rows = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='granola'")
        ).scalar()
        assert rows == 1


def test_seed_row_matches_registry(tmp_path):
    """The migration snapshot and the runtime registry must define the same
    granola row (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "granola"
    )
    assert migration.ROW == registry_row


def test_seed_row_classifies_as_mcp_oauth(tmp_path):
    """The seeded shape must classify as a remote-MCP OAuth connector — an
    "unconnectable" classification would make the catalog entry dead on
    arrival (no connect endpoint accepts it)."""
    from xagent.web.mcp_apps import classify_app_auth

    migration = _load_migration_module()
    assert (
        classify_app_auth(migration.ROW["transport"], migration.ROW["launch_config"])
        == "mcp_oauth"
    )


def test_downgrade_removes_granola(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "granola" not in _app_ids(connection)
