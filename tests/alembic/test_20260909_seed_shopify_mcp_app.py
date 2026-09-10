"""Tests for the provenance-safe Shopify connector seed."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def _load_migration():
    path = (
        Path(__file__).parents[2]
        / "src/xagent/migrations/versions/20260909_seed_shopify_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location("seed_shopify_migration", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_tables(connection):
    connection.execute(
        text(
            """
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                icon VARCHAR(1000),
                transport VARCHAR(50) NOT NULL,
                provider_name VARCHAR(50),
                category VARCHAR(100),
                oauth_scopes JSON,
                is_visible_in_connector BOOLEAN NOT NULL DEFAULT 1,
                launch_config JSON
            )
            """
        )
    )
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


def test_upgrade_inserts_provenance_and_personal_credential_metadata(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        row = connection.execute(
            text(
                "SELECT transport, launch_config FROM public_mcp_apps "
                "WHERE app_id='shopify'"
            )
        ).first()

    assert row is not None
    assert row[0] == "stdio"
    assert '"credential_scope": "personal"' in row[1]
    assert '"builtin_provenance"' in row[1]


def test_upgrade_is_idempotent_for_provenance_owned_row(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='shopify'")
        ).scalar_one()
    assert count == 1


def test_upgrade_accepts_owned_row_from_an_older_provenance_version(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as connection:
        _create_tables(connection)
        launch_config = dict(migration.ROW["launch_config"])
        marker = dict(migration.BUILTIN_PROVENANCE)
        marker["version"] = 0
        launch_config["builtin_provenance"] = marker
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, launch_config) "
                "VALUES ('shopify', 'Shopify', 'stdio', :launch_config)"
            ),
            {"launch_config": json.dumps(launch_config)},
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='shopify'")
        ).scalar_one()

    assert count == 1


def test_upgrade_refuses_custom_catalog_collision(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, launch_config) "
                "VALUES ('shopify', 'Custom Shopify', 'stdio', '{\"command\": \"custom\"}')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="public_mcp_apps"):
                migration.upgrade()
        row = connection.execute(
            text("SELECT name FROM public_mcp_apps WHERE app_id='shopify'")
        ).scalar_one()
    assert row == "Custom Shopify"


def test_upgrade_refuses_custom_server_name_collision(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(text("INSERT INTO mcp_servers (name) VALUES ('shopify')"))
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="custom mcp_servers"):
                migration.upgrade()
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='shopify'")
        ).scalar_one()
    assert count == 0


@pytest.mark.parametrize(
    "app_id,name",
    [
        (" SHOPIFY ", "Unrelated"),
        ("other", " shopify "),
        ("ShOpIfY", "Unrelated"),
    ],
)
def test_upgrade_refuses_normalized_custom_catalog_collision(tmp_path, app_id, name):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, launch_config) "
                "VALUES (:app_id, :name, 'stdio', '{\"command\": \"custom\"}')"
            ),
            {"app_id": app_id, "name": name},
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="public_mcp_apps"):
                migration.upgrade()


@pytest.mark.parametrize("name", [" SHOPIFY ", "ShOpIfY"])
def test_upgrade_refuses_normalized_custom_server_collision(tmp_path, name):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text("INSERT INTO mcp_servers (name) VALUES (:name)"), {"name": name}
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="custom mcp_servers"):
                migration.upgrade()


def test_downgrade_only_deletes_provenance_owned_row(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='shopify'")
        ).scalar_one()
    assert count == 0


def test_downgrade_preserves_custom_same_id(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, launch_config) "
                "VALUES ('shopify', 'Custom Shopify', 'stdio', '{\"command\": \"custom\"}')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        name = connection.execute(
            text("SELECT name FROM public_mcp_apps WHERE app_id='shopify'")
        ).scalar_one()
    assert name == "Custom Shopify"


def test_seed_row_matches_registry():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration()
    registry = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "shopify"
    )
    assert migration.ROW == registry


def test_upgrade_connect_downgrade_reupgrade_preserves_official_connection(tmp_path):
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app
    from xagent.web.models.database import Base
    from xagent.web.models.mcp import MCPServer, UserMCPServer
    from xagent.web.models.user import User

    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'roundtrip.sqlite'}")
    Base.metadata.create_all(engine)

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

    session_factory = sessionmaker(bind=engine)
    db = session_factory()
    user = User(id=1, username="shop-owner", password_hash="x")
    db.add(user)
    db.commit()
    connect_mcp_app(
        "shopify",
        MCPAppConnectRequest(
            env={
                "SHOPIFY_STORE_DOMAIN": "acme",
                "SHOPIFY_ACCESS_TOKEN": "shpat_roundtrip_secret",
            }
        ),
        current_user=user,
        db=db,
    )
    server = db.query(MCPServer).filter(MCPServer.name == "shopify").one()
    association = (
        db.query(UserMCPServer)
        .filter(UserMCPServer.mcpserver_id == server.id, UserMCPServer.user_id == 1)
        .one()
    )
    server_id = int(server.id)
    association_id = int(association.id)
    encrypted_env = association.env
    assert server.auth == {"builtin_provenance": migration.BUILTIN_PROVENANCE}
    db.close()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
            migration.upgrade()

    db = session_factory()
    assert db.query(MCPServer).filter(MCPServer.id == server_id).one().auth == {
        "builtin_provenance": migration.BUILTIN_PROVENANCE
    }
    preserved = db.query(UserMCPServer).filter(UserMCPServer.id == association_id).one()
    assert preserved.env == encrypted_env
    assert (
        db.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='shopify'")
        ).scalar_one()
        == 1
    )
    db.close()


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
            name=" Shopify ",
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
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='shopify'")
        ).scalar_one()

    assert count == 0


def test_fresh_registry_seed_accepts_owned_server_from_older_version(tmp_path):
    from xagent.web.builtin_mcp_registry import seed_builtin_oauth_and_public_mcp_apps
    from xagent.web.models.database import Base
    from xagent.web.models.mcp import MCPServer

    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh-owned.sqlite'}")
    Base.metadata.create_all(engine)
    old_marker = dict(migration.BUILTIN_PROVENANCE)
    old_marker["version"] = 0
    session_factory = sessionmaker(bind=engine)
    db = session_factory()
    db.add(
        MCPServer(
            name="shopify",
            managed="external",
            transport="stdio",
            command="python",
            auth={"builtin_provenance": old_marker},
        )
    )
    db.commit()
    db.close()

    with engine.begin() as connection:
        seed_builtin_oauth_and_public_mcp_apps(connection)
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='shopify'")
        ).scalar_one()

    assert count == 1
