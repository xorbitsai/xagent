import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
)


def _load_migration():
    path = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20261002_migrate_hubspot_to_remote_mcp.py"
    )
    spec = importlib.util.spec_from_file_location("hubspot_remote_mcp_migration", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _schema(metadata):
    apps = Table(
        "public_mcp_apps",
        metadata,
        Column("app_id", String, primary_key=True),
        Column("name", String),
        Column("description", Text),
        Column("transport", String),
        Column("provider_name", String),
        Column("oauth_scopes", JSON),
        Column("launch_config", JSON),
        Column("is_visible_in_connector", Boolean, default=True),
    )
    servers = Table(
        "mcp_servers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String),
        Column("transport", String),
        Column("url", String),
        Column("auth", JSON),
    )
    associations = Table(
        "user_mcpservers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("mcpserver_id", Integer),
    )
    return apps, servers, associations


def _insert_legacy_app(connection, apps, migration, *, description=None):
    connection.execute(
        apps.insert().values(
            app_id="hubspot",
            name="HubSpot",
            description=description or migration.LEGACY_DESCRIPTION,
            transport="oauth",
            provider_name="hubspot",
            oauth_scopes=migration.LEGACY_SCOPES,
            launch_config=migration.LEGACY_LAUNCH_CONFIG,
            is_visible_in_connector=True,
        )
    )


def test_upgrade_switches_catalog_and_removes_legacy_server(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = MetaData()
    apps, servers, associations = _schema(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        _insert_legacy_app(connection, apps, migration)
        connection.execute(
            servers.insert().values(
                id=7,
                name="HubSpot",
                transport="oauth",
                auth={"app_id": "hubspot", "provider": "hubspot"},
            )
        )
        connection.execute(associations.insert().values(id=9, mcpserver_id=7))

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        row = (
            connection.execute(select(apps).where(apps.c.app_id == "hubspot"))
            .mappings()
            .one()
        )
        assert row["transport"] == "streamable_http"
        assert row["provider_name"] is None
        assert row["oauth_scopes"] is None
        assert row["description"] == migration.REMOTE_DESCRIPTION
        assert row["launch_config"] == migration.REMOTE_LAUNCH_CONFIG
        assert connection.execute(select(servers)).all() == []
        assert connection.execute(select(associations)).all() == []


def test_upgrade_preserves_custom_description(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = MetaData()
    apps, _, _ = _schema(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        _insert_legacy_app(
            connection, apps, migration, description="Our CRM connection"
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert (
            connection.execute(
                select(apps.c.description).where(apps.c.app_id == "hubspot")
            ).scalar_one()
            == "Our CRM connection"
        )


def test_upgrade_rejects_unowned_catalog_collision(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = MetaData()
    apps, _, _ = _schema(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            apps.insert().values(
                app_id="hubspot",
                name="Custom HubSpot",
                description="Operator connector",
                transport="streamable_http",
                provider_name=None,
                oauth_scopes=None,
                launch_config={"url": "https://custom.example/mcp"},
                is_visible_in_connector=True,
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="does not match"):
                migration.upgrade()


def test_downgrade_restores_legacy_catalog_and_removes_remote_server(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = MetaData()
    apps, servers, associations = _schema(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        _insert_legacy_app(connection, apps, migration)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        connection.execute(
            servers.insert().values(
                id=8,
                name="hubspot",
                transport="streamable_http",
                url="https://mcp.hubspot.com",
                auth={"type": "mcp_oauth", "client_id": "encrypted"},
            )
        )
        connection.execute(associations.insert().values(id=10, mcpserver_id=8))

        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()

        row = (
            connection.execute(select(apps).where(apps.c.app_id == "hubspot"))
            .mappings()
            .one()
        )
        assert row["transport"] == "oauth"
        assert row["provider_name"] == "hubspot"
        assert row["oauth_scopes"] == migration.LEGACY_SCOPES
        assert row["description"] == migration.LEGACY_DESCRIPTION
        assert row["launch_config"] == migration.LEGACY_LAUNCH_CONFIG
        assert connection.execute(select(servers)).all() == []
        assert connection.execute(select(associations)).all() == []


def test_remote_row_matches_builtin_registry():
    migration = _load_migration()
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app

    row = get_builtin_public_mcp_app("hubspot")
    assert row is not None
    assert row["description"] == migration.REMOTE_DESCRIPTION
    assert row["transport"] == "streamable_http"
    assert row["provider_name"] is None
    assert row["oauth_scopes"] is None
    assert row["launch_config"] == migration.REMOTE_LAUNCH_CONFIG
