"""Tests for the Excel MCP connector seed migration."""

import importlib.util
import json
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
        / "src/xagent/migrations/versions/20260917_seed_excel_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_excel_migration", migration_file
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


def _create_mcp_servers_table(connection):
    connection.execute(
        text(
            """
            CREATE TABLE mcp_servers (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                auth JSON
            )
            """
        )
    )


def _app_ids(connection):
    return set(connection.execute(text("SELECT app_id FROM public_mcp_apps")).scalars())


def test_upgrade_inserts_excel(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "excel" in _app_ids(connection)
        row = connection.execute(
            text(
                "SELECT transport, provider_name, launch_config FROM public_mcp_apps"
                " WHERE app_id='excel'"
            )
        ).first()
        assert row[0] == "oauth"
        assert row[1] == "microsoft"
        assert "xagent.web.tools.mcp.excel" in str(row[2])
        assert '"builtin_provenance"' in str(row[2])


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()  # second run must not raise or duplicate
        rows = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='excel'")
        ).scalar()
        assert rows == 1


def test_upgrade_accepts_provenance_owned_row(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        launch_config = dict(migration.ROW["launch_config"])
        marker = dict(migration.BUILTIN_PROVENANCE)
        marker["version"] = 0
        launch_config["builtin_provenance"] = marker
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, launch_config) "
                "VALUES ('excel', 'Excel', 'oauth', :launch_config)"
            ),
            {"launch_config": json.dumps(launch_config)},
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "excel" in _app_ids(connection)


def test_upgrade_refuses_unowned_custom_excel_row(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, launch_config) "
                "VALUES ('excel', 'Operator Excel', 'custom', 'stdio', NULL)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="reserved Excel identity"):
                migration.upgrade()
        row = connection.execute(
            text("SELECT name FROM public_mcp_apps WHERE app_id='excel'")
        ).scalar_one()
        assert row == "Operator Excel"


@pytest.mark.parametrize(
    ("app_id", "name"),
    [
        ("Excel", "Custom connector"),
        (" excel ", "Custom connector"),
        ("custom-excel", "Excel"),
        ("custom-excel", " EXCEL "),
    ],
)
def test_upgrade_refuses_normalized_excel_identity_collisions(tmp_path, app_id, name):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, launch_config) "
                "VALUES (:app_id, :name, 'custom', 'stdio', NULL)"
            ),
            {"app_id": app_id, "name": name},
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="reserved Excel identity"):
                migration.upgrade()
        assert "excel" not in _app_ids(connection)
        assert app_id in _app_ids(connection)


@pytest.mark.parametrize("name", ["Excel", "excel", " EXCEL "])
def test_upgrade_refuses_normalized_mcp_server_collisions(tmp_path, name):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _create_mcp_servers_table(connection)
        connection.execute(
            text("INSERT INTO mcp_servers (name, auth) VALUES (:name, '{}')"),
            {"name": name},
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="custom mcp_servers"):
                migration.upgrade()
        assert "excel" not in _app_ids(connection)


def test_upgrade_accepts_provenance_owned_server_after_downgrade(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _create_mcp_servers_table(connection)
        connection.execute(
            text("INSERT INTO mcp_servers (name, auth) VALUES ('Excel', :auth)"),
            {
                "auth": json.dumps(
                    {
                        "app_id": "excel",
                        "provider": "microsoft",
                        "builtin_provenance": migration.BUILTIN_PROVENANCE,
                    }
                )
            },
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
            migration.upgrade()
        assert "excel" in _app_ids(connection)


def test_upgrade_rechecks_server_namespace_when_catalog_row_is_owned(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        _create_mcp_servers_table(connection)
        connection.execute(
            text("INSERT INTO mcp_servers (name, auth) VALUES ('Excel', '{}')")
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="custom mcp_servers"):
                migration.upgrade()


def test_fresh_registry_seed_rejects_normalized_custom_server_collision(tmp_path):
    from xagent.web.builtin_mcp_registry import seed_builtin_oauth_and_public_mcp_apps
    from xagent.web.models.database import Base
    from xagent.web.models.mcp import MCPServer

    engine = create_engine(f"sqlite:///{tmp_path / 'fresh-seed.sqlite'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    db = session_factory()
    db.add(
        MCPServer(
            name=" Excel ",
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
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='excel'")
        ).scalar_one()

    assert count == 0


def test_fresh_registry_seed_accepts_owned_server_from_older_version(tmp_path):
    from xagent.web.builtin_mcp_registry import seed_builtin_oauth_and_public_mcp_apps
    from xagent.web.models.database import Base
    from xagent.web.models.mcp import MCPServer

    migration = _load_migration_module()
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh-owned.sqlite'}")
    Base.metadata.create_all(engine)
    old_marker = dict(migration.BUILTIN_PROVENANCE)
    old_marker["version"] = 0
    session_factory = sessionmaker(bind=engine)
    db = session_factory()
    db.add(
        MCPServer(
            name="Excel",
            managed="external",
            transport="oauth",
            auth={"builtin_provenance": old_marker},
        )
    )
    db.commit()
    db.close()

    with engine.begin() as connection:
        seed_builtin_oauth_and_public_mcp_apps(connection)
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='excel'")
        ).scalar_one()

    assert count == 1


def test_seed_row_matches_registry(tmp_path):
    """The migration snapshot and the runtime registry must define the same
    excel row (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "excel"
    )
    assert migration.ROW == registry_row


def test_downgrade_removes_excel(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "excel" not in _app_ids(connection)


def test_downgrade_preserves_unowned_custom_excel_row(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, launch_config) "
                "VALUES ('excel', 'Operator Excel', 'custom', 'stdio', '{}')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        row = connection.execute(
            text("SELECT name FROM public_mcp_apps WHERE app_id='excel'")
        ).scalar_one()
        assert row == "Operator Excel"


def test_upgrade_requires_launch_config_for_provenance(tmp_path):
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
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth'
                )
                """
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="launch_config"):
                migration.upgrade()


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
