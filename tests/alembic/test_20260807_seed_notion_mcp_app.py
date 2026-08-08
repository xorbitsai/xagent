"""Tests for the Notion remote-MCP connector seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, select, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260807_seed_notion_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_notion_migration", migration_file
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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


def test_upgrade_inserts_notion(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        # Full-row comparison through the migration's own typed table object:
        # JSON columns deserialize and booleans come back as real bools, so
        # the persisted row compares to ROW directly — no field is left
        # unchecked at the DB level. ROW's content is itself pinned against
        # the registry by test_seed_row_matches_registry, so deriving the
        # expectation from it keeps identical coverage without a third
        # hand-typed copy of the seed data.
        row = (
            connection.execute(
                select(migration.PUBLIC_MCP_APPS_TABLE).where(
                    migration.PUBLIC_MCP_APPS_TABLE.c.app_id == "notion"
                )
            )
            .mappings()
            .first()
        )
        assert row is not None
        assert dict(row) == migration.ROW


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()  # second run must not raise or duplicate
        rows = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='notion'")
        ).scalar()
        assert rows == 1


def test_seed_row_matches_registry():
    """The migration snapshot and the runtime registry must define the same
    notion row (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "notion"
    )
    assert migration.ROW == registry_row


def test_seed_row_classifies_as_mcp_oauth():
    """The seeded shape must classify as a remote-MCP OAuth connector — an
    "unconnectable" classification would make the catalog entry dead on
    arrival (no connect endpoint accepts it)."""
    from xagent.web.mcp_apps import classify_app_auth

    migration = _load_migration_module()
    assert (
        classify_app_auth(migration.ROW["transport"], migration.ROW["launch_config"])
        == "mcp_oauth"
    )


def test_downgrade_removes_notion(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        # A sentinel row unrelated to this migration must survive the
        # downgrade — otherwise "notion" missing from _app_ids could
        # equally mean the whole table was wiped, not just its own row.
        # transport is set explicitly: the real column is NOT NULL with no
        # server default, so the insert must not lean on the default this
        # test's own _create_table happens to declare.
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, transport)"
                " VALUES ('sentinel', 'Sentinel', 'oauth')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert _app_ids(connection) == {"sentinel"}
