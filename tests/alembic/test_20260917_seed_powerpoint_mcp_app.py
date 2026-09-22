"""Tests for the PowerPoint MCP connector seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260917_seed_powerpoint_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_powerpoint_migration", migration_file
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


def test_upgrade_inserts_powerpoint(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "powerpoint" in _app_ids(connection)
        row = connection.execute(
            text(
                "SELECT transport, provider_name, launch_config FROM public_mcp_apps"
                " WHERE app_id='powerpoint'"
            )
        ).first()
        assert row[0] == "oauth"
        assert row[1] == "microsoft"
        assert "xagent.web.tools.mcp.powerpoint" in str(row[2])
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
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='powerpoint'")
        ).scalar()
        assert rows == 1


def test_upgrade_rejects_unprovenanced_app_id_collision(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, launch_config) "
                "VALUES ('powerpoint', 'Operator app', 'stdio', '{}')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="no matching builtin_provenance"):
                migration.upgrade()


def test_downgrade_preserves_unprovenanced_app_id_collision(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, launch_config) "
                "VALUES ('powerpoint', 'Operator app', 'stdio', '{}')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()

        assert "powerpoint" in _app_ids(connection)


def test_seed_row_matches_registry(tmp_path):
    """The migration snapshot and the runtime registry must define the same
    powerpoint row (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "powerpoint"
    )
    assert migration.ROW == registry_row


def test_downgrade_removes_powerpoint(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "powerpoint" not in _app_ids(connection)


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


def test_fresh_registry_seed_rejects_normalized_custom_server_collision(tmp_path):
    """seed_builtin_oauth_and_public_mcp_apps's protected_server_identities
    guard now covers powerpoint -- pin it the same way excel's is already
    pinned, so a future edit can't silently reopen the fresh-install
    collision hole for this connector specifically."""
    from xagent.web.builtin_mcp_registry import seed_builtin_oauth_and_public_mcp_apps
    from xagent.web.models.database import Base
    from xagent.web.models.mcp import MCPServer

    engine = create_engine(f"sqlite:///{tmp_path / 'fresh-seed.sqlite'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    db = session_factory()
    db.add(
        MCPServer(
            name=" PowerPoint ",
            managed="external",
            transport="stdio",
            command="custom",
        )
    )
    db.commit()
    db.close()

    with engine.begin() as connection:
        with pytest.raises(RuntimeError, match="custom mcp_servers identity"):
            seed_builtin_oauth_and_public_mcp_apps(connection)
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='powerpoint'")
        ).scalar_one()

    assert count == 0
