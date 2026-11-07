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
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
)

VERSIONS = Path(__file__).parent.parent.parent / "src/xagent/migrations/versions"


def _load(file_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, VERSIONS / file_name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_migration():
    return _load("20261012_migrate_zendesk_to_oauth.py", "zendesk_oauth_migration")


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _schema(metadata):
    apps = Table(
        "public_mcp_apps",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("app_id", String, unique=True),
        Column("name", String),
        Column("description", Text),
        Column("icon", String),
        Column("transport", String),
        Column("provider_name", String),
        Column("category", String),
        Column("oauth_scopes", JSON),
        Column("launch_config", JSON),
        Column("is_visible_in_connector", Boolean, default=True),
    )
    servers = Table(
        "mcp_servers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String, unique=True),
        Column("transport", String),
        Column("command", String),
        Column("args", JSON),
        Column("auth", JSON),
    )
    associations = Table(
        "user_mcpservers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("mcpserver_id", Integer),
        Column("env", JSON),
    )
    actor_connections = Table(
        "actor_mcp_server_connections",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("app_id", String),
        Column("encrypted_env", JSON),
    )
    providers = Table(
        "oauth_providers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("provider_name", String, unique=True),
        Column("name", String),
        Column("client_id", String),
        Column("client_secret", String),
        Column("auth_url", String),
        Column("token_url", String),
        Column("redirect_uri", String),
        Column("userinfo_url", String),
        Column("user_id_path", String),
        Column("email_path", String),
        Column("default_scopes", JSON),
    )
    user_oauth = Table(
        "user_oauth",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("provider", String),
        Column("access_token", String),
        Column("refresh_token", String),
        Column("expires_at", DateTime),
        Column("provider_user_id", String),
    )
    return apps, servers, associations, actor_connections, providers, user_oauth


LEGACY_ROW = {
    "app_id": "zendesk",
    "name": "Zendesk",
    "description": (
        "Connect to Zendesk with an API token to search and manage tickets, "
        "reply to customers or add internal notes, and look up users and "
        "organizations."
    ),
    "transport": "stdio",
    "provider_name": None,
    "category": "Support",
    "oauth_scopes": None,
    "is_visible_in_connector": False,
    "launch_config": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.zendesk"],
        "required_env": ["ZENDESK_SUBDOMAIN", "ZENDESK_EMAIL", "ZENDESK_API_TOKEN"],
    },
}
LEGACY_ARGS = ["-m", "xagent.web.tools.mcp.zendesk"]


@pytest.fixture()
def database(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'zendesk.db'}")
    metadata = MetaData()
    tables = _schema(metadata)
    metadata.create_all(engine)
    with engine.begin() as connection:
        yield connection, tables
    engine.dispose()


def _seed_legacy(connection, tables, **overrides):
    apps, servers, associations, actor_connections, _providers, _user_oauth = tables
    connection.execute(apps.insert().values(**{**LEGACY_ROW, **overrides}))
    connection.execute(
        servers.insert().values(
            id=1, name="zendesk", transport="stdio", command="python", args=LEGACY_ARGS
        )
    )
    connection.execute(
        associations.insert().values(
            id=1, mcpserver_id=1, env={"ZENDESK_API_TOKEN": "legacy-secret"}
        )
    )
    connection.execute(
        actor_connections.insert().values(
            id=1, app_id="zendesk", encrypted_env={"ZENDESK_API_TOKEN": "x"}
        )
    )


def _run(connection, step: str, migration=None):
    migration = migration or _load_migration()
    with patch.object(migration, "op", _operations(connection)):
        getattr(migration, step)()
    return migration


def _app(connection, tables):
    return (
        connection.execute(select(tables[0]).where(tables[0].c.app_id == "zendesk"))
        .mappings()
        .one()
    )


def _ids(connection, table):
    return [row.id for row in connection.execute(select(table.c.id))]


def test_upgrade_switches_the_catalog_row_to_account_oauth(database):
    connection, tables = database
    _seed_legacy(connection, tables)

    migration = _run(connection, "upgrade")

    app = _app(connection, tables)
    assert app["transport"] == "oauth"
    assert app["provider_name"] == "zendesk"
    assert app["oauth_scopes"] == ["read", "tickets:write"]
    assert app["launch_config"] == migration.OAUTH_LAUNCH_CONFIG
    assert app["description"] == migration.OAUTH_DESCRIPTION
    assert app["is_visible_in_connector"] is False


def test_upgrade_target_matches_the_registry_row():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app

    migration = _load_migration()
    registry_row = get_builtin_public_mcp_app("zendesk")

    assert registry_row is not None
    assert registry_row["name"] == migration.APP_NAME
    assert registry_row["description"] == migration.OAUTH_DESCRIPTION
    assert registry_row["transport"] == "oauth"
    assert registry_row["provider_name"] == migration.APP_ID
    # 20261102_zendesk_side_conv later adds side_conversations:write; its own
    # test pins the full list.
    assert set(migration.OAUTH_SCOPES).issubset(registry_row["oauth_scopes"])
    # 20261019_zendesk_marketplace later adds static_env; its own test pins it.
    assert {
        key: value
        for key, value in registry_row["launch_config"].items()
        if key != "static_env"
    } == migration.OAUTH_LAUNCH_CONFIG


def test_inserted_provider_matches_the_registry_provider_row():
    from xagent.web.builtin_mcp_registry import get_builtin_oauth_provider_rows

    migration = _load_migration()
    registry_row = next(
        row
        for row in get_builtin_oauth_provider_rows()
        if row["provider_name"] == "zendesk"
    )

    for name, expected in migration.OAUTH_PROVIDER_STATIC_FIELDS.items():
        assert registry_row[name] == expected
    # Same as the migration: credentials resolve from env at request time.
    assert registry_row["client_id"] == ""
    assert registry_row["client_secret"] == ""
    assert registry_row["redirect_uri"] == ""


def test_upgrade_keeps_an_operator_shown_row_visible(database):
    connection, tables = database
    _seed_legacy(connection, tables, is_visible_in_connector=True)

    _run(connection, "upgrade")

    app = _app(connection, tables)
    assert app["transport"] == "oauth"
    assert app["is_visible_in_connector"] is True


def test_upgrade_deletes_legacy_api_token_connections(database):
    connection, tables = database
    _apps, servers, associations, actor_connections, _providers, _uo = tables
    _seed_legacy(connection, tables)

    _run(connection, "upgrade")

    assert _ids(connection, servers) == []
    assert _ids(connection, associations) == []
    assert _ids(connection, actor_connections) == []


def test_upgrade_leaves_other_servers_and_actor_connections(database):
    connection, tables = database
    _apps, servers, associations, actor_connections, _providers, _uo = tables
    _seed_legacy(connection, tables)
    connection.execute(
        servers.insert().values(
            id=2, name="zendesk-custom", transport="stdio", command="python", args=[]
        )
    )
    connection.execute(associations.insert().values(id=2, mcpserver_id=2))
    connection.execute(actor_connections.insert().values(id=2, app_id="intercom"))

    _run(connection, "upgrade")

    assert _ids(connection, servers) == [2]
    assert _ids(connection, associations) == [2]
    assert _ids(connection, actor_connections) == [2]


def test_upgrade_does_not_adopt_a_server_running_another_command(database):
    connection, tables = database
    _apps, servers, _associations, _actor, _providers, _uo = tables
    connection.execute(_apps.insert().values(**LEGACY_ROW))
    connection.execute(
        servers.insert().values(
            id=3, name="zendesk", transport="stdio", command="node", args=["x.js"]
        )
    )

    _run(connection, "upgrade")

    assert _ids(connection, servers) == [3]


def test_upgrade_inserts_the_oauth_provider_without_pinning_env_credentials(
    database, monkeypatch
):
    """Blank credentials make the runtime read ZENDESK_CLIENT_* at request
    time, so moving to the global client later is an env change."""
    connection, tables = database
    providers = tables[4]
    monkeypatch.setenv("ZENDESK_CLIENT_ID", "zdg-local-client")
    monkeypatch.setenv("ZENDESK_CLIENT_SECRET", "local-secret")
    _seed_legacy(connection, tables)

    _run(connection, "upgrade")

    row = connection.execute(select(providers)).mappings().one()
    assert row["provider_name"] == "zendesk"
    assert row["client_id"] == ""
    assert row["client_secret"] == ""
    assert row["redirect_uri"] == ""
    assert row["auth_url"] == ""
    assert row["token_url"] == ""
    assert row["default_scopes"] == ["read"]


def test_upgrade_does_not_overwrite_an_operator_provider_row(database):
    connection, tables = database
    providers = tables[4]
    connection.execute(
        providers.insert().values(
            provider_name="zendesk",
            name="Our Zendesk",
            client_id="operator-id",
            client_secret="operator-secret",
            auth_url="https://custom.example/authorize",
            token_url="https://custom.example/token",
        )
    )
    _seed_legacy(connection, tables)

    _run(connection, "upgrade")
    _run(connection, "downgrade")

    row = connection.execute(select(providers)).mappings().one()
    assert row["name"] == "Our Zendesk"
    assert row["client_id"] == "operator-id"


@pytest.mark.parametrize(
    "overrides",
    [
        {"transport": "oauth"},
        {"provider_name": "custom"},
        {"oauth_scopes": ["read"]},
        {"name": "Zendesk (custom)"},
        {
            "launch_config": {
                "command": "python",
                "args": ["-m", "my.own.zendesk"],
                "required_env": ["ZENDESK_API_TOKEN"],
            }
        },
    ],
)
def test_upgrade_refuses_a_row_that_is_not_the_legacy_connector(database, overrides):
    connection, tables = database
    _apps, servers, associations, actor_connections, providers, _uo = tables
    _seed_legacy(connection, tables, **overrides)

    with pytest.raises(RuntimeError, match="legacy API-token Zendesk"):
        _run(connection, "upgrade")

    assert _ids(connection, servers) == [1]
    assert _ids(connection, associations) == [1]
    assert _ids(connection, actor_connections) == [1]
    assert _ids(connection, providers) == []


def test_upgrade_is_idempotent(database):
    connection, tables = database
    _seed_legacy(connection, tables)

    migration = _run(connection, "upgrade")
    _run(connection, "upgrade", migration)

    app = _app(connection, tables)
    assert app["launch_config"] == migration.OAUTH_LAUNCH_CONFIG
    assert len(_ids(connection, tables[4])) == 1


def test_upgrade_of_an_already_switched_row_still_removes_legacy_state(database):
    connection, tables = database
    _apps, servers, associations, actor_connections, _providers, _uo = tables
    migration = _load_migration()
    _seed_legacy(
        connection,
        tables,
        transport="oauth",
        provider_name="zendesk",
        oauth_scopes=migration.OAUTH_SCOPES,
        launch_config=migration.OAUTH_LAUNCH_CONFIG,
    )

    _run(connection, "upgrade", migration)

    assert _ids(connection, servers) == []
    assert _ids(connection, associations) == []
    assert _ids(connection, actor_connections) == []


def test_upgrade_without_a_catalog_row_is_a_noop(database):
    connection, tables = database

    _run(connection, "upgrade")

    assert _ids(connection, tables[4]) == []


def test_upgrade_after_the_seed_migration(database):
    """The frozen seed's row is exactly what the switch migrates from."""
    connection, tables = database
    seed = _load("20260901_seed_zendesk_mcp_app.py", "zendesk_seed_migration")
    _run(connection, "upgrade", seed)

    _run(connection, "upgrade")

    assert _app(connection, tables)["transport"] == "oauth"


def test_downgrade_restores_the_legacy_row(database):
    connection, tables = database
    _seed_legacy(connection, tables)
    migration = _run(connection, "upgrade")

    _run(connection, "downgrade", migration)

    app = _app(connection, tables)
    assert app["transport"] == "stdio"
    assert app["provider_name"] is None
    assert app["oauth_scopes"] is None
    assert app["launch_config"] == LEGACY_ROW["launch_config"]
    assert app["description"] == LEGACY_ROW["description"]
    assert _ids(connection, tables[4]) == []


@pytest.mark.parametrize(
    "credentials",
    [
        {"client_id": "admin-id", "client_secret": "admin-secret"},
        {"redirect_uri": "https://xagent.example/api/auth/zendesk/callback"},
    ],
)
def test_downgrade_keeps_a_provider_row_an_admin_configured(database, credentials):
    connection, tables = database
    providers = tables[4]
    _seed_legacy(connection, tables)
    migration = _run(connection, "upgrade")
    connection.execute(
        providers.update()
        .where(providers.c.provider_name == "zendesk")
        .values(**credentials)
    )

    _run(connection, "downgrade", migration)

    row = connection.execute(select(providers)).mappings().one()
    for name, value in credentials.items():
        assert row[name] == value


def test_downgrade_removes_oauth_connections(database):
    connection, tables = database
    _apps, servers, associations, _actor, _providers, user_oauth = tables
    _seed_legacy(connection, tables)
    migration = _run(connection, "upgrade")
    connection.execute(
        servers.insert().values(
            id=5,
            name="Zendesk",
            transport="oauth",
            auth={"app_id": "zendesk", "provider": "zendesk"},
        )
    )
    connection.execute(associations.insert().values(id=5, mcpserver_id=5))
    connection.execute(
        user_oauth.insert().values(
            id=1,
            provider="zendesk",
            access_token="oauth-access",
            refresh_token="oauth-refresh",
            provider_user_id="42",
        )
    )

    _run(connection, "downgrade", migration)

    assert _ids(connection, servers) == []
    assert _ids(connection, associations) == []
    grant = connection.execute(select(user_oauth)).mappings().one()
    assert grant["access_token"] == ""
    assert grant["refresh_token"] is None
    assert grant["provider_user_id"] == "42"


def test_downgrade_leaves_a_row_without_our_marker(database):
    connection, tables = database
    _seed_legacy(connection, tables, transport="oauth", provider_name="zendesk")

    _run(connection, "downgrade")

    app = _app(connection, tables)
    assert app["transport"] == "oauth"
    assert app["provider_name"] == "zendesk"
