"""Shopify's switch from pasted Admin API tokens to per-store OAuth (#2810).

Unlike Zendesk's legacy row, Shopify's legacy row already carries the
builtin provenance marker, so the migration has to tell the two shapes apart
by their full execution shape, not by the marker.
"""

import importlib.util
from copy import deepcopy
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
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    select,
)

from tests.shared.postgres_disposable import disposable_database_factory

VERSIONS = Path(__file__).parent.parent.parent / "src/xagent/migrations/versions"
OFFICIAL_MARKER = {"registry": "xagent", "app_id": "shopify", "version": 1}
LEGACY_ARGS = ["-m", "xagent.web.tools.mcp.shopify"]
# Tables outside this repository (xagent-saas) that also point at
# mcp_servers.id; the migration must find them through their foreign keys.
EXTERNAL_SERVER_CHILDREN = ("saas_team_mcp_servers", "saas_team_mcp_envs")


def _load(file_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, VERSIONS / file_name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_migration():
    return _load("20261119_migrate_shopify_to_oauth.py", "shopify_oauth_migration")


def _load_seed():
    return _load("20260909_seed_shopify_mcp_app.py", "shopify_seed_for_oauth_switch")


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
    server_fk = ForeignKey("mcp_servers.id", ondelete="CASCADE")
    tables = {"public_mcp_apps": apps, "mcp_servers": servers}
    tables["user_mcpservers"] = Table(
        "user_mcpservers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("mcpserver_id", Integer, server_fk),
        Column("env", JSON),
    )
    tables["mcp_oauth_clients"] = Table(
        "mcp_oauth_clients",
        metadata,
        Column("id", Integer, primary_key=True),
        Column(
            "mcp_server_id",
            Integer,
            ForeignKey("mcp_servers.id", ondelete="CASCADE"),
        ),
    )
    for name in ("mcp_oauth_grants", "mcp_oauth_flow_states"):
        tables[name] = Table(
            name,
            metadata,
            Column("id", Integer, primary_key=True),
            Column(
                "mcp_server_id",
                Integer,
                ForeignKey("mcp_servers.id", ondelete="CASCADE"),
            ),
            Column(
                "client_id",
                Integer,
                ForeignKey("mcp_oauth_clients.id", ondelete="CASCADE"),
            ),
        )
    for name in EXTERNAL_SERVER_CHILDREN:
        tables[name] = Table(
            name,
            metadata,
            Column("id", Integer, primary_key=True),
            Column(
                "mcpserver_id",
                Integer,
                ForeignKey("mcp_servers.id", ondelete="CASCADE"),
            ),
        )
    tables["actor_mcp_server_connections"] = Table(
        "actor_mcp_server_connections",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("app_id", String),
        Column("encrypted_env", JSON),
    )
    tables["oauth_providers"] = Table(
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
    tables["user_oauth"] = Table(
        "user_oauth",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("provider", String),
        Column("access_token", String),
        Column("refresh_token", String),
        Column("expires_at", DateTime),
        Column("provider_user_id", String),
    )
    return tables


SERVER_CHILDREN = (
    "user_mcpservers",
    "mcp_oauth_clients",
    "mcp_oauth_grants",
    "mcp_oauth_flow_states",
    *EXTERNAL_SERVER_CHILDREN,
)


@pytest.fixture(params=["fk-off", "fk-on"])
def database(request, tmp_path):
    """SQLite, with and without foreign-key enforcement: production may run
    either way, so the migration cleans every child row itself."""
    engine = create_engine(f"sqlite:///{tmp_path / 'shopify.db'}")
    if request.param == "fk-on":

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record):
            dbapi_connection.execute("PRAGMA foreign_keys=ON")

    metadata = MetaData()
    tables = _schema(metadata)
    metadata.create_all(engine)
    with engine.begin() as connection:
        yield connection, tables
    engine.dispose()


def _legacy_row(**overrides):
    return {**deepcopy(_load_seed().ROW), **overrides}


def _add_server(connection, tables, server_id, **values):
    """A server plus one row in every table that hangs off it."""
    connection.execute(tables["mcp_servers"].insert().values(id=server_id, **values))
    for name in SERVER_CHILDREN:
        column = "mcp_server_id" if name.startswith("mcp_oauth") else "mcpserver_id"
        row = {"id": server_id, column: server_id}
        if name in ("mcp_oauth_grants", "mcp_oauth_flow_states"):
            row["client_id"] = server_id
        connection.execute(tables[name].insert().values(**row))


def _add_legacy_server(connection, tables, server_id=1, **overrides):
    values = {
        "name": "shopify",
        "transport": "stdio",
        "command": "python",
        "args": LEGACY_ARGS,
        "auth": {"builtin_provenance": OFFICIAL_MARKER},
        **overrides,
    }
    _add_server(connection, tables, server_id, **values)


def _seed_legacy(connection, tables, **row_overrides):
    connection.execute(
        tables["public_mcp_apps"].insert().values(**_legacy_row(**row_overrides))
    )
    _add_legacy_server(connection, tables)
    connection.execute(
        tables["actor_mcp_server_connections"]
        .insert()
        .values(id=1, app_id="shopify", encrypted_env={"SHOPIFY_ACCESS_TOKEN": "x"})
    )


def _run(connection, step: str, migration=None):
    migration = migration or _load_migration()
    with patch.object(migration, "op", _operations(connection)):
        getattr(migration, step)()
    return migration


def _app(connection, tables):
    apps = tables["public_mcp_apps"]
    return (
        connection.execute(select(apps).where(apps.c.app_id == "shopify"))
        .mappings()
        .one()
    )


def _ids(connection, table):
    return sorted(row.id for row in connection.execute(select(table.c.id)))


def _child_ids(connection, tables):
    return {name: _ids(connection, tables[name]) for name in SERVER_CHILDREN}


# ---------- upgrade: catalog row ---------------------------------------------


def test_upgrade_switches_the_marked_legacy_row_to_oauth_and_hides_it(database):
    """The legacy row already carries the provenance marker; it must still be
    switched, not mistaken for an already-switched row."""
    connection, tables = database
    _seed_legacy(connection, tables)

    migration = _run(connection, "upgrade")

    app = _app(connection, tables)
    assert app["transport"] == "oauth"
    assert app["provider_name"] == "shopify"
    assert app["oauth_scopes"] == ["write_products", "write_orders", "read_customers"]
    assert app["launch_config"] == migration.OAUTH_LAUNCH_CONFIG
    assert app["description"] == migration.OAUTH_DESCRIPTION
    # Hidden until verified against a development store: an OAuth app with
    # no client configured, or a store-name prompt, must not reach users.
    assert app["is_visible_in_connector"] is False


def test_upgrade_target_matches_the_registry_row():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app

    migration = _load_migration()
    registry_row = get_builtin_public_mcp_app("shopify")

    assert registry_row is not None
    assert registry_row["name"] == migration.APP_NAME
    assert registry_row["description"] == migration.OAUTH_DESCRIPTION
    assert registry_row["transport"] == "oauth"
    assert registry_row["provider_name"] == migration.APP_ID
    assert registry_row["oauth_scopes"] == migration.OAUTH_SCOPES
    assert registry_row["launch_config"] == migration.OAUTH_LAUNCH_CONFIG
    assert registry_row["is_visible_in_connector"] is False


def test_migration_legacy_shape_is_the_frozen_seed_row():
    migration = _load_migration()
    seed_row = _load_seed().ROW

    assert migration.LEGACY_LAUNCH_CONFIG == seed_row["launch_config"]
    assert migration.LEGACY_DESCRIPTION == seed_row["description"]


def test_inserted_provider_matches_the_registry_provider_row():
    from xagent.web.builtin_mcp_registry import get_builtin_oauth_provider_rows

    migration = _load_migration()
    registry_row = next(
        row
        for row in get_builtin_oauth_provider_rows()
        if row["provider_name"] == "shopify"
    )

    for name, expected in migration.OAUTH_PROVIDER_STATIC_FIELDS.items():
        assert registry_row[name] == expected
    # Credentials resolve from SHOPIFY_CLIENT_* at request time.
    assert registry_row["client_id"] == ""
    assert registry_row["client_secret"] == ""
    assert registry_row["redirect_uri"] == ""


def test_upgrade_after_the_seed_migration(database):
    """The frozen seed's row is exactly what the switch migrates from."""
    connection, tables = database
    _run(connection, "upgrade", _load_seed())

    _run(connection, "upgrade")

    app = _app(connection, tables)
    assert app["transport"] == "oauth"
    assert app["is_visible_in_connector"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"transport": "oauth"}, id="transport"),
        pytest.param({"provider_name": "custom"}, id="provider"),
        pytest.param({"oauth_scopes": ["read_products"]}, id="scopes"),
        pytest.param({"name": "Shopify (custom)"}, id="name"),
        pytest.param(
            {
                "launch_config": {
                    **_load_seed().ROW["launch_config"],
                    "args": ["-m", "my.own.shopify"],
                }
            },
            id="module",
        ),
        pytest.param(
            {
                "launch_config": {
                    **_load_seed().ROW["launch_config"],
                    "builtin_provenance": {**OFFICIAL_MARKER, "registry": "other"},
                }
            },
            id="foreign-marker",
        ),
        pytest.param(
            {
                "launch_config": {
                    **_load_seed().ROW["launch_config"],
                    "required_env": ["SHOPIFY_ACCESS_TOKEN"],
                }
            },
            id="edited-legacy",
        ),
    ],
)
def test_upgrade_refuses_a_row_that_is_neither_shape(database, overrides):
    """An unknown shape stops the migration; nothing is deleted or added."""
    connection, tables = database
    _seed_legacy(connection, tables, **overrides)
    before = _app(connection, tables)

    with pytest.raises(RuntimeError, match="Cannot migrate Shopify connector"):
        _run(connection, "upgrade")

    assert dict(_app(connection, tables)) == dict(before)
    assert _ids(connection, tables["mcp_servers"]) == [1]
    assert all(ids == [1] for ids in _child_ids(connection, tables).values())
    assert _ids(connection, tables["actor_mcp_server_connections"]) == [1]
    assert _ids(connection, tables["oauth_providers"]) == []


def test_an_edited_description_survives_upgrade_and_downgrade(database):
    """Only the shipped description is rewritten; an operator's own text is
    theirs."""
    connection, tables = database
    _seed_legacy(connection, tables, description="Our store tools")

    migration = _run(connection, "upgrade")
    assert _app(connection, tables)["description"] == "Our store tools"

    _run(connection, "downgrade", migration)
    assert _app(connection, tables)["description"] == "Our store tools"


def test_upgrade_is_idempotent(database):
    connection, tables = database
    _seed_legacy(connection, tables)

    migration = _run(connection, "upgrade")
    _run(connection, "upgrade", migration)

    app = _app(connection, tables)
    assert app["launch_config"] == migration.OAUTH_LAUNCH_CONFIG
    assert len(_ids(connection, tables["oauth_providers"])) == 1


def test_upgrade_of_an_already_switched_row_removes_legacy_state_only(database):
    """A database seeded from the current registry (or a rerun) keeps the
    row as it is -- including an operator's choice to show it."""
    connection, tables = database
    migration = _load_migration()
    _seed_legacy(
        connection,
        tables,
        transport="oauth",
        provider_name="shopify",
        oauth_scopes=migration.OAUTH_SCOPES,
        launch_config=migration.OAUTH_LAUNCH_CONFIG,
        is_visible_in_connector=True,
    )

    _run(connection, "upgrade", migration)

    assert _app(connection, tables)["is_visible_in_connector"] is True
    assert _ids(connection, tables["mcp_servers"]) == []
    assert _ids(connection, tables["actor_mcp_server_connections"]) == []
    assert len(_ids(connection, tables["oauth_providers"])) == 1


def test_upgrade_without_a_catalog_row_is_a_noop(database):
    connection, tables = database

    _run(connection, "upgrade")

    assert _ids(connection, tables["oauth_providers"]) == []


# ---------- upgrade: legacy connections --------------------------------------


def test_upgrade_deletes_the_official_token_servers_and_everything_on_them(
    database,
):
    connection, tables = database
    _seed_legacy(connection, tables)

    _run(connection, "upgrade")

    assert _ids(connection, tables["mcp_servers"]) == []
    assert all(ids == [] for ids in _child_ids(connection, tables).values())
    assert _ids(connection, tables["actor_mcp_server_connections"]) == []


@pytest.mark.parametrize(
    "auth",
    [
        pytest.param(None, id="unstamped"),
        pytest.param({"builtin_provenance": {**OFFICIAL_MARKER, "registry": "x"}}),
        pytest.param({"app_id": "shopify"}, id="no-marker"),
    ],
)
def test_upgrade_keeps_a_same_name_same_command_server_without_our_marker(
    database, auth
):
    """Running the builtin module under the builtin name does not prove the
    server is ours; only the stamped marker does."""
    connection, tables = database
    connection.execute(tables["public_mcp_apps"].insert().values(**_legacy_row()))
    _add_legacy_server(connection, tables, auth=auth)

    _run(connection, "upgrade")

    assert _ids(connection, tables["mcp_servers"]) == [1]
    assert all(ids == [1] for ids in _child_ids(connection, tables).values())


def test_upgrade_leaves_other_servers_and_actor_connections(database):
    connection, tables = database
    _seed_legacy(connection, tables)
    _add_server(
        connection,
        tables,
        2,
        name="shopify-custom",
        transport="stdio",
        command="python",
        args=LEGACY_ARGS,
        auth={"builtin_provenance": OFFICIAL_MARKER},
    )
    _add_server(
        connection,
        tables,
        3,
        name="Shopify",
        transport="oauth",
        command=None,
        args=None,
        auth=None,
    )
    connection.execute(
        tables["actor_mcp_server_connections"].insert().values(id=2, app_id="zendesk")
    )

    _run(connection, "upgrade")

    assert _ids(connection, tables["mcp_servers"]) == [2, 3]
    assert all(ids == [2, 3] for ids in _child_ids(connection, tables).values())
    assert _ids(connection, tables["actor_mcp_server_connections"]) == [2]


# ---------- upgrade: provider row --------------------------------------------


def test_upgrade_inserts_the_provider_without_pinning_env_credentials(
    database, monkeypatch
):
    connection, tables = database
    monkeypatch.setenv("SHOPIFY_CLIENT_ID", "local-client")
    monkeypatch.setenv("SHOPIFY_CLIENT_SECRET", "local-secret")
    _seed_legacy(connection, tables)

    _run(connection, "upgrade")

    row = connection.execute(select(tables["oauth_providers"])).mappings().one()
    assert row["provider_name"] == "shopify"
    assert row["client_id"] == ""
    assert row["client_secret"] == ""
    assert row["redirect_uri"] == ""
    assert row["auth_url"] == ""
    assert row["token_url"] == ""
    assert row["default_scopes"] == []


def test_operator_provider_row_survives_upgrade_and_downgrade(database):
    connection, tables = database
    providers = tables["oauth_providers"]
    connection.execute(
        providers.insert().values(
            provider_name="shopify",
            name="Our Shopify",
            client_id="operator-id",
            client_secret="operator-secret",
        )
    )
    _seed_legacy(connection, tables)

    migration = _run(connection, "upgrade")
    _run(connection, "downgrade", migration)

    row = connection.execute(select(providers)).mappings().one()
    assert row["name"] == "Our Shopify"
    assert row["client_id"] == "operator-id"


# ---------- downgrade --------------------------------------------------------


def test_downgrade_restores_the_legacy_row_and_removes_the_inserted_provider(
    database,
):
    """The blank provider row the upgrade inserted goes, as Zendesk's does."""
    connection, tables = database
    _seed_legacy(connection, tables)
    migration = _run(connection, "upgrade")

    _run(connection, "downgrade", migration)

    app = _app(connection, tables)
    legacy = _legacy_row()
    for field in ("transport", "provider_name", "oauth_scopes", "launch_config"):
        assert app[field] == legacy[field]
    assert app["description"] == legacy["description"]
    # The legacy connector shipped visible.
    assert app["is_visible_in_connector"] is True
    assert _ids(connection, tables["oauth_providers"]) == []


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"redirect_uri": "https://example.test/cb"}, id="configured"),
        pytest.param({"auth_url": "https://example.test/auth"}, id="edited-static"),
    ],
)
def test_downgrade_keeps_a_provider_row_an_operator_changed(database, overrides):
    connection, tables = database
    _seed_legacy(connection, tables)
    migration = _run(connection, "upgrade")
    providers = tables["oauth_providers"]
    connection.execute(
        providers.update()
        .where(providers.c.provider_name == "shopify")
        .values(**overrides)
    )

    _run(connection, "downgrade", migration)

    assert len(_ids(connection, providers)) == 1


def test_downgrade_removes_oauth_servers_and_voids_grants(database):
    connection, tables = database
    _seed_legacy(connection, tables)
    migration = _run(connection, "upgrade")
    _add_server(
        connection,
        tables,
        5,
        name="Shopify",
        transport="oauth",
        command=None,
        args=None,
        auth={
            "app_id": "shopify",
            "provider": "shopify",
            "builtin_provenance": OFFICIAL_MARKER,
        },
    )
    connection.execute(
        tables["user_oauth"]
        .insert()
        .values(
            id=1,
            provider="shopify",
            access_token="oauth-access",
            refresh_token="oauth-refresh",
            provider_user_id="gid://shopify/Shop/1",
        )
    )

    _run(connection, "downgrade", migration)

    assert _ids(connection, tables["mcp_servers"]) == []
    assert all(ids == [] for ids in _child_ids(connection, tables).values())
    grant = connection.execute(select(tables["user_oauth"])).mappings().one()
    assert grant["access_token"] == ""
    assert grant["refresh_token"] is None
    # The identity stays as a reconnect tombstone.
    assert grant["provider_user_id"] == "gid://shopify/Shop/1"


def test_downgrade_leaves_a_row_that_is_not_our_oauth_shape(database):
    connection, tables = database
    _seed_legacy(connection, tables, transport="oauth", provider_name="shopify")
    before = dict(_app(connection, tables))

    _run(connection, "downgrade")

    assert dict(_app(connection, tables)) == before


def test_upgrade_downgrade_upgrade_round_trip(database):
    connection, tables = database
    _seed_legacy(connection, tables)
    migration = _run(connection, "upgrade")
    switched = dict(_app(connection, tables))

    _run(connection, "downgrade", migration)
    _run(connection, "upgrade", migration)

    assert dict(_app(connection, tables)) == switched
    assert len(_ids(connection, tables["oauth_providers"])) == 1


# ---------- downstream tables by their ON DELETE rule -----------------------


def _create(connection, ddl: str) -> None:
    connection.exec_driver_sql(ddl)


def test_upgrade_nulls_a_downstream_set_null_reference(database):
    """A downstream history table keeps its rows; only the reference goes."""
    connection, tables = database
    _create(
        connection,
        "CREATE TABLE usage_history (id INTEGER PRIMARY KEY, server_id INTEGER "
        "REFERENCES mcp_servers(id) ON DELETE SET NULL)",
    )
    _seed_legacy(connection, tables)
    connection.exec_driver_sql("INSERT INTO usage_history VALUES (7, 1)")

    _run(connection, "upgrade")

    assert _ids(connection, tables["mcp_servers"]) == []
    rows = connection.exec_driver_sql("SELECT id, server_id FROM usage_history").all()
    assert [tuple(row) for row in rows] == [(7, None)]


def test_upgrade_deletes_the_children_of_a_downstream_cascade_table(database):
    connection, tables = database
    _create(
        connection,
        "CREATE TABLE team_links (id INTEGER PRIMARY KEY, server_id INTEGER "
        "REFERENCES mcp_servers(id) ON DELETE CASCADE)",
    )
    _create(
        connection,
        "CREATE TABLE team_link_notes (id INTEGER PRIMARY KEY, link_id INTEGER "
        "REFERENCES team_links(id) ON DELETE CASCADE)",
    )
    _seed_legacy(connection, tables)
    connection.exec_driver_sql("INSERT INTO team_links VALUES (5, 1)")
    connection.exec_driver_sql("INSERT INTO team_link_notes VALUES (9, 5)")

    _run(connection, "upgrade")

    assert connection.exec_driver_sql("SELECT id FROM team_links").all() == []
    assert connection.exec_driver_sql("SELECT id FROM team_link_notes").all() == []


@pytest.mark.parametrize(
    "rule", [pytest.param("RESTRICT"), pytest.param("", id="no-rule")]
)
def test_upgrade_refuses_to_delete_through_a_protecting_reference(database, rule):
    """A reference that protects its rows stops the migration instead of
    losing them, and names the table to fix."""
    connection, tables = database
    clause = f" ON DELETE {rule}" if rule else ""
    _create(
        connection,
        "CREATE TABLE audit_trail (id INTEGER PRIMARY KEY, server_id INTEGER "
        f"REFERENCES mcp_servers(id){clause})",
    )
    _seed_legacy(connection, tables)
    connection.exec_driver_sql("INSERT INTO audit_trail VALUES (3, 1)")

    with pytest.raises(RuntimeError, match="audit_trail"):
        _run(connection, "upgrade")


def test_a_protecting_reference_without_rows_does_not_block(database):
    connection, tables = database
    _create(
        connection,
        "CREATE TABLE audit_trail (id INTEGER PRIMARY KEY, server_id INTEGER "
        "REFERENCES mcp_servers(id) ON DELETE RESTRICT)",
    )
    _seed_legacy(connection, tables)

    _run(connection, "upgrade")

    assert _ids(connection, tables["mcp_servers"]) == []


def test_upgrade_follows_references_to_the_known_child_tables(database):
    """The rule applies one level down too: a table pointing at
    user_mcpservers is handled like one pointing at mcp_servers."""
    connection, tables = database
    _create(
        connection,
        "CREATE TABLE usage_by_link (id INTEGER PRIMARY KEY, link_id INTEGER "
        "REFERENCES user_mcpservers(id) ON DELETE SET NULL)",
    )
    _seed_legacy(connection, tables)
    connection.exec_driver_sql("INSERT INTO usage_by_link VALUES (8, 1)")

    _run(connection, "upgrade")

    rows = connection.exec_driver_sql("SELECT id, link_id FROM usage_by_link").all()
    assert [tuple(row) for row in rows] == [(8, None)]


def test_upgrade_refuses_a_protecting_reference_to_a_known_child(database):
    connection, tables = database
    _create(
        connection,
        "CREATE TABLE link_audit (id INTEGER PRIMARY KEY, link_id INTEGER "
        "REFERENCES user_mcpservers(id) ON DELETE RESTRICT)",
    )
    _seed_legacy(connection, tables)
    connection.exec_driver_sql("INSERT INTO link_audit VALUES (2, 1)")

    with pytest.raises(RuntimeError, match="link_audit"):
        _run(connection, "upgrade")


def test_upgrade_survives_cascading_tables_that_reference_each_other(database):
    connection, tables = database
    _create(
        connection,
        "CREATE TABLE ring_a (id INTEGER PRIMARY KEY, server_id INTEGER "
        "REFERENCES mcp_servers(id) ON DELETE CASCADE, b_id INTEGER "
        "REFERENCES ring_b(id) ON DELETE CASCADE)",
    )
    _create(
        connection,
        "CREATE TABLE ring_b (id INTEGER PRIMARY KEY, a_id INTEGER "
        "REFERENCES ring_a(id) ON DELETE CASCADE)",
    )
    _seed_legacy(connection, tables)
    connection.exec_driver_sql("INSERT INTO ring_a VALUES (1, 1, NULL)")
    connection.exec_driver_sql("INSERT INTO ring_b VALUES (1, 1)")
    connection.exec_driver_sql("UPDATE ring_a SET b_id = 1 WHERE id = 1")

    _run(connection, "upgrade")

    assert connection.exec_driver_sql("SELECT id FROM ring_a").all() == []
    assert connection.exec_driver_sql("SELECT id FROM ring_b").all() == []


# ---------- kept servers that block the builtin OAuth connect ---------------


def test_upgrade_warns_about_a_kept_server_named_shopify(database, caplog):
    """A kept stdio server named exactly `shopify` is a second candidate for
    the builtin app, so the OAuth connector cannot resolve until it is
    renamed or removed; the operator has to hear about it."""
    connection, tables = database
    connection.execute(tables["public_mcp_apps"].insert().values(**_legacy_row()))
    _add_legacy_server(connection, tables, server_id=4, auth=None)

    with caplog.at_level("WARNING"):
        _run(connection, "upgrade")

    assert _ids(connection, tables["mcp_servers"]) == [4]
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("4" in message and "shopify" in message for message in warnings)


def test_upgrade_does_not_warn_when_nothing_named_shopify_is_kept(database, caplog):
    connection, tables = database
    _seed_legacy(connection, tables)

    with caplog.at_level("WARNING"):
        _run(connection, "upgrade")

    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


# ---------- PostgreSQL --------------------------------------------------------
# The foreign-key discovery above reads PostgreSQL's catalog through
# SQLAlchemy's inspector, not SQLite's pragma, and the full migration chain
# starts from an empty database, so it never reaches that path. Each test here
# runs in its own disposable database.


@pytest.fixture
def postgresql_engine_factory():
    with disposable_database_factory("xagent_shopify_oauth") as make:
        yield make


@pytest.mark.postgresql
def test_postgresql_upgrade_follows_each_downstream_on_delete_rule(
    postgresql_engine_factory,
):
    engine = postgresql_engine_factory("fk-rules")
    metadata = MetaData()
    tables = _schema(metadata)
    metadata.create_all(engine)
    with engine.begin() as connection:
        _create(
            connection,
            "CREATE TABLE usage_history (id INTEGER PRIMARY KEY, server_id INTEGER "
            "REFERENCES mcp_servers(id) ON DELETE SET NULL)",
        )
        _create(
            connection,
            "CREATE TABLE team_link_notes (id INTEGER PRIMARY KEY, link_id INTEGER "
            "REFERENCES saas_team_mcp_servers(id) ON DELETE CASCADE)",
        )
        _seed_legacy(connection, tables)
        connection.exec_driver_sql("INSERT INTO usage_history VALUES (7, 1)")
        connection.exec_driver_sql("INSERT INTO team_link_notes VALUES (9, 1)")

        _run(connection, "upgrade")

        assert _ids(connection, tables["mcp_servers"]) == []
        assert all(ids == [] for ids in _child_ids(connection, tables).values())
        history = connection.exec_driver_sql("SELECT id, server_id FROM usage_history")
        assert [tuple(row) for row in history] == [(7, None)]
        notes = connection.exec_driver_sql("SELECT id FROM team_link_notes").all()
        assert notes == []


@pytest.mark.postgresql
def test_postgresql_upgrade_refuses_a_protecting_reference(postgresql_engine_factory):
    engine = postgresql_engine_factory("fk-restrict")
    metadata = MetaData()
    tables = _schema(metadata)
    metadata.create_all(engine)
    with engine.connect() as connection:
        transaction = connection.begin()
        _create(
            connection,
            "CREATE TABLE audit_trail (id INTEGER PRIMARY KEY, server_id INTEGER "
            "REFERENCES mcp_servers(id) ON DELETE RESTRICT)",
        )
        _seed_legacy(connection, tables)
        connection.exec_driver_sql("INSERT INTO audit_trail VALUES (3, 1)")

        with pytest.raises(RuntimeError, match="audit_trail"):
            _run(connection, "upgrade")
        transaction.rollback()
