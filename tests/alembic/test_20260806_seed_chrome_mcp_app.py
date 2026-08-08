"""Tests for the Chrome MCP connector seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260806_seed_chrome_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_chrome_migration", migration_file
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


def test_upgrade_inserts_chrome(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "chrome" in _app_ids(connection)
        row = connection.execute(
            text(
                "SELECT transport, launch_config, is_visible_in_connector "
                "FROM public_mcp_apps WHERE app_id='chrome'"
            )
        ).first()
        assert row[0] == "stdio"
        assert "chrome-devtools-mcp" in str(row[1])
        # Assert the persisted column value, not just dict agreement
        # (test_seed_row_matches_registry): the table DDL defaults this
        # column to 1, so a dropped/mistyped key would ship the connector
        # visible — and it must stay hidden until persistent stdio MCP
        # sessions land.
        assert row[2] == 0


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()  # second run must not raise or duplicate
        rows = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='chrome'")
        ).scalar()
        assert rows == 1


def test_seed_row_matches_registry():
    """The migration snapshot and the runtime registry must define the same
    chrome row (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "chrome"
    )
    assert migration.ROW == registry_row


def test_seed_row_classifies_keyless():
    """The Chrome entry must classify as "keyless" — an "unconnectable"
    classification would make the catalog entry dead on arrival in the
    connector UI."""
    from xagent.web.mcp_apps import classify_app_auth

    migration = _load_migration_module()
    assert (
        classify_app_auth(migration.ROW["transport"], migration.ROW["launch_config"])
        == "keyless"
    )


def test_downgrade_removes_chrome(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "chrome" not in _app_ids(connection)


def test_dockerfile_npx_cache_pin_matches_registry():
    """The chrome-devtools-mcp version is pinned in three places: the
    registry launch args, the migration snapshot (tied to the registry by
    test_seed_row_matches_registry), and the Dockerfile's npx cache-warm
    line. The first two are covered by dict equality; this closes the third
    side of the triangle — an unsynchronized bump would silently reopen the
    per-launch registry fetch the cache warm exists to prevent (the warmed
    version would no longer match the version the launch args request).
    """
    import re

    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "chrome"
    )
    registry_specs = {
        arg
        for arg in registry_row["launch_config"]["args"]
        if arg.startswith("chrome-devtools-mcp@")
    }
    assert len(registry_specs) == 1

    dockerfile = (
        Path(__file__).parent.parent.parent / "docker/Dockerfile.backend"
    ).read_text()
    dockerfile_specs = set(re.findall(r"chrome-devtools-mcp@[\w.\-]+", dockerfile))
    assert dockerfile_specs == registry_specs
