"""Tests for the migration that pins the remote Atlassian connector's scope."""

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
    create_engine,
    func,
    select,
)

VERSIONS = Path(__file__).parent.parent.parent / "src/xagent/migrations/versions"


def _load(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, VERSIONS / filename)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_migration():
    return _load("20261013_atlassian_mcp_scope.py", "atlassian_mcp_scope_migration")


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _run(connection, migration, *steps):
    with patch.object(migration, "op", _operations(connection)):
        for step in steps:
            getattr(migration, step)()


def _schema(metadata, *, oauth_tables=True):
    apps = Table(
        "public_mcp_apps",
        metadata,
        Column("app_id", String, primary_key=True),
        Column("name", String),
        Column("transport", String),
        Column("launch_config", JSON),
    )
    servers = Table(
        "mcp_servers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String, unique=True),
        Column("transport", String),
        Column("url", String),
        Column("auth", JSON),
    )
    links = Table(
        "user_mcpservers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("user_id", Integer),
        Column("mcpserver_id", Integer),
        Column("is_owner", Boolean, default=False),
    )
    if oauth_tables:
        Table(
            "mcp_oauth_clients",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("mcp_server_id", Integer),
        )
        Table(
            "mcp_oauth_grants",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("mcp_server_id", Integer),
            Column("mcp_oauth_client_id", Integer),
            Column("status", String),
        )
        Table(
            "mcp_oauth_flow_states",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("mcp_server_id", Integer),
            Column("mcp_oauth_client_id", Integer),
        )
    return apps, servers, links


def _insert_catalog_row(connection, apps, launch_config):
    connection.execute(
        apps.insert().values(
            app_id="atlassian",
            name="Atlassian (Jira, Confluence, Bitbucket)",
            transport="streamable_http",
            launch_config=launch_config,
        )
    )


def _setup(tmp_path, migration, *, launch_config=None):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = MetaData()
    apps, servers, links = _schema(metadata)
    metadata.create_all(engine)
    if launch_config is not False:
        with engine.begin() as connection:
            _insert_catalog_row(
                connection,
                apps,
                migration.LEGACY_LAUNCH_CONFIG
                if launch_config is None
                else launch_config,
            )
    return engine, metadata, apps, servers, links


def _add_registration(
    connection, metadata, *, server_id, grant_statuses=(), flows=0
) -> int:
    """Add a client registration for ``server_id`` with a grant per status and
    ``flows`` flow states, all pointing at it; return its id."""
    client_id = connection.execute(
        metadata.tables["mcp_oauth_clients"].insert().values(mcp_server_id=server_id)
    ).inserted_primary_key[0]
    for status in grant_statuses:
        connection.execute(
            metadata.tables["mcp_oauth_grants"]
            .insert()
            .values(
                mcp_server_id=server_id, mcp_oauth_client_id=client_id, status=status
            )
        )
    for _ in range(flows):
        connection.execute(
            metadata.tables["mcp_oauth_flow_states"]
            .insert()
            .values(mcp_server_id=server_id, mcp_oauth_client_id=client_id)
        )
    return client_id


def _add_server(
    connection,
    metadata,
    servers,
    links,
    *,
    server_id,
    auth,
    name="atlassian",
    url="https://mcp.atlassian.com/v2/mcp",
    transport="streamable_http",
    owner=False,
):
    """Add a server with a user link and a registration that an active grant
    and a flow state use; return the registration's id."""
    connection.execute(
        servers.insert().values(
            id=server_id, name=name, transport=transport, url=url, auth=auth
        )
    )
    connection.execute(
        links.insert().values(user_id=7, mcpserver_id=server_id, is_owner=owner)
    )
    return _add_registration(
        connection, metadata, server_id=server_id, grant_statuses=("active",), flows=1
    )


def _launch_config(connection, apps):
    return connection.execute(
        select(apps.c.launch_config).where(apps.c.app_id == "atlassian")
    ).scalar_one()


def _server_ids(connection, servers):
    return set(connection.execute(select(servers.c.id)).scalars())


def _server_auth(connection, servers, server_id):
    return connection.execute(
        select(servers.c.auth).where(servers.c.id == server_id)
    ).scalar_one()


def _client_ids(connection, metadata):
    table = metadata.tables["mcp_oauth_clients"]
    return set(connection.execute(select(table.c.id)).scalars())


def _flow_client_ids(connection, metadata):
    table = metadata.tables["mcp_oauth_flow_states"]
    return sorted(connection.execute(select(table.c.mcp_oauth_client_id)).scalars())


def _row_counts(connection, metadata):
    return {
        name: connection.execute(select(func.count()).select_from(table)).scalar_one()
        for name, table in metadata.tables.items()
    }


def _servers_with_policy_columns(metadata):
    return Table(
        "mcp_servers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String, unique=True),
        Column("transport", String),
        Column("url", String),
        Column("auth", JSON),
        Column("headers", JSON),
        Column("env", JSON),
        Column("timeout", Integer),
        Column("managed", String),
    )


# --- upgrade: the shared row's auth is updated in place -----------------------


def test_upgrade_pins_scope_and_updates_the_shared_server_in_place(tmp_path):
    migration = _load_migration()
    engine, metadata, apps, servers, links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.LEGACY_AUTH,
        )
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=2,
            name="unrelated",
            url="https://mcp.example.com/mcp",
            auth={"type": "mcp_oauth"},
        )
        counts_before = _row_counts(connection, metadata)

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        # Same row, same id: only its auth changed. Its user link, grant and
        # the registration that grant uses are all kept.
        assert _server_ids(connection, servers) == {1, 2}
        assert _server_auth(connection, servers, 1) == migration.SCOPED_AUTH
        assert _server_auth(connection, servers, 2) == {"type": "mcp_oauth"}
        assert _row_counts(connection, metadata) == counts_before


def test_upgrade_keeps_the_app_id_key_of_the_shared_server(tmp_path):
    migration = _load_migration()
    engine, metadata, _apps, servers, links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            transport="STREAMABLE_HTTP",
            auth={**migration.LEGACY_AUTH, "app_id": "atlassian"},
        )
        counts_before = _row_counts(connection, metadata)

        _run(connection, migration, "upgrade")

        assert _server_auth(connection, servers, 1) == {
            **migration.SCOPED_AUTH,
            "app_id": "atlassian",
        }
        assert _row_counts(connection, metadata) == counts_before


def test_upgrade_with_an_older_provenance_version_updates_the_shared_server(
    tmp_path,
):
    migration = _load_migration()
    launch_config = dict(migration.LEGACY_LAUNCH_CONFIG)
    launch_config["builtin_provenance"] = {
        **migration.BUILTIN_PROVENANCE,
        "version": 0,
    }
    engine, metadata, apps, servers, links = _setup(
        tmp_path, migration, launch_config=launch_config
    )
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.LEGACY_AUTH,
        )

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        assert _server_auth(connection, servers, 1) == migration.SCOPED_AUTH


@pytest.mark.parametrize(
    "policy",
    [
        {"headers": {"X-Debug": "1"}},
        {"timeout": 120},
        {"env": {"HTTP_PROXY": "http://proxy.internal:3128"}},
        {"managed": "internal"},
    ],
    ids=("headers", "timeout", "env", "managed"),
)
def test_upgrade_keeps_an_admin_edit_to_the_shared_server(tmp_path, policy):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = MetaData()
    apps, _servers, links = _schema(metadata)
    metadata.remove(metadata.tables["mcp_servers"])
    servers = _servers_with_policy_columns(metadata)
    metadata.create_all(engine)
    row = {
        "id": 1,
        "name": "atlassian",
        "transport": "streamable_http",
        "url": migration.SERVER_URL,
        "managed": "external",
        **policy,
    }
    with engine.begin() as connection:
        _insert_catalog_row(connection, apps, migration.LEGACY_LAUNCH_CONFIG)
        connection.execute(servers.insert().values(auth=migration.LEGACY_AUTH, **row))
        connection.execute(
            links.insert().values(user_id=7, mcpserver_id=1, is_owner=False)
        )

        _run(connection, migration, "upgrade")

        stored = connection.execute(select(servers)).mappings().one()
        assert stored["auth"] == migration.SCOPED_AUTH
        assert {key: stored[key] for key in row} == row
        assert (
            connection.execute(select(func.count()).select_from(links)).scalar_one()
            == 1
        )


def test_upgrade_updates_the_shared_server_without_the_oauth_tables(tmp_path):
    """A database without the OAuth tables has no registration to clean up;
    the row is updated the same way."""
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = MetaData()
    apps, servers, links = _schema(metadata, oauth_tables=False)
    metadata.create_all(engine)
    with engine.begin() as connection:
        _insert_catalog_row(connection, apps, migration.LEGACY_LAUNCH_CONFIG)
        connection.execute(
            servers.insert().values(
                id=1,
                name="atlassian",
                transport="streamable_http",
                url=migration.SERVER_URL,
                auth=migration.LEGACY_AUTH,
            )
        )
        connection.execute(
            links.insert().values(user_id=7, mcpserver_id=1, is_owner=False)
        )

        _run(connection, migration, "upgrade")

        assert _server_auth(connection, servers, 1) == migration.SCOPED_AUTH
        assert (
            connection.execute(select(func.count()).select_from(links)).scalar_one()
            == 1
        )


def test_upgrade_without_a_server_row_only_updates_the_catalog(tmp_path):
    migration = _load_migration()
    engine, metadata, apps, servers, _links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        _add_registration(connection, metadata, server_id=1, flows=1)

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        assert _server_ids(connection, servers) == set()
        assert _client_ids(connection, metadata) == {1}


# --- upgrade: registrations no grant references are deleted -------------------


def test_upgrade_deletes_the_registrations_no_grant_references(tmp_path):
    """The connect path reuses a stored registration and cannot register again
    if the provider rejects it, so the updated row's registrations that never
    produced a grant go, with their flow states. A registration that any grant
    references stays, whatever the grant's status, and no grant is deleted."""
    migration = _load_migration()
    engine, metadata, _apps, servers, links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        in_use = _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.LEGACY_AUTH,
        )
        revoked_only = _add_registration(
            connection, metadata, server_id=1, grant_statuses=("revoked",), flows=1
        )
        _add_registration(connection, metadata, server_id=1, flows=2)
        _add_registration(connection, metadata, server_id=1)
        other_in_use = _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=2,
            name="unrelated",
            url="https://mcp.example.com/mcp",
            auth={"type": "mcp_oauth"},
        )
        other_unused = _add_registration(connection, metadata, server_id=2, flows=1)
        counts_before = _row_counts(connection, metadata)

        _run(connection, migration, "upgrade")

        assert _server_auth(connection, servers, 1) == migration.SCOPED_AUTH
        assert _client_ids(connection, metadata) == {
            in_use,
            revoked_only,
            other_in_use,
            other_unused,
        }
        assert _flow_client_ids(connection, metadata) == sorted(
            [in_use, revoked_only, other_in_use, other_unused]
        )
        assert _row_counts(connection, metadata) == {
            **counts_before,
            "mcp_oauth_clients": counts_before["mcp_oauth_clients"] - 2,
            "mcp_oauth_flow_states": counts_before["mcp_oauth_flow_states"] - 2,
        }


@pytest.mark.parametrize("grants", ["without-client-column", "missing"])
def test_upgrade_keeps_registrations_when_grants_cannot_be_linked_to_them(
    tmp_path, grants
):
    """Fail closed: without the column that links a grant to its registration,
    a registration in use cannot be told apart, so none is deleted. The row's
    auth is still updated."""
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = MetaData()
    apps, servers, links = _schema(metadata)
    metadata.remove(metadata.tables["mcp_oauth_grants"])
    if grants == "without-client-column":
        Table(
            "mcp_oauth_grants",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("mcp_server_id", Integer),
        )
    metadata.create_all(engine)
    with engine.begin() as connection:
        _insert_catalog_row(connection, apps, migration.LEGACY_LAUNCH_CONFIG)
        connection.execute(
            servers.insert().values(
                id=1,
                name="atlassian",
                transport="streamable_http",
                url=migration.SERVER_URL,
                auth=migration.LEGACY_AUTH,
            )
        )
        connection.execute(
            links.insert().values(user_id=7, mcpserver_id=1, is_owner=False)
        )
        _add_registration(connection, metadata, server_id=1, flows=1)
        counts_before = _row_counts(connection, metadata)

        _run(connection, migration, "upgrade")

        assert _server_auth(connection, servers, 1) == migration.SCOPED_AUTH
        assert _row_counts(connection, metadata) == counts_before


def test_upgrade_deletes_an_unused_registration_without_the_flow_table(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = MetaData()
    apps, servers, links = _schema(metadata)
    metadata.remove(metadata.tables["mcp_oauth_flow_states"])
    metadata.create_all(engine)
    with engine.begin() as connection:
        _insert_catalog_row(connection, apps, migration.LEGACY_LAUNCH_CONFIG)
        connection.execute(
            servers.insert().values(
                id=1,
                name="atlassian",
                transport="streamable_http",
                url=migration.SERVER_URL,
                auth=migration.LEGACY_AUTH,
            )
        )
        in_use = _add_registration(
            connection, metadata, server_id=1, grant_statuses=("active",)
        )
        _add_registration(connection, metadata, server_id=1)

        _run(connection, migration, "upgrade")

        assert _server_auth(connection, servers, 1) == migration.SCOPED_AUTH
        assert _client_ids(connection, metadata) == {in_use}


def test_upgrade_run_twice_deletes_nothing_more(tmp_path):
    """A second run finds no row on the previous auth, so a registration added
    after the first run (one a connect has not used yet) is kept."""
    migration = _load_migration()
    engine, metadata, apps, servers, links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.LEGACY_AUTH,
        )
        _add_registration(connection, metadata, server_id=1, flows=1)
        _run(connection, migration, "upgrade")
        _add_registration(connection, metadata, server_id=1, flows=1)
        counts_before = _row_counts(connection, metadata)

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        assert _server_auth(connection, servers, 1) == migration.SCOPED_AUTH
        assert _row_counts(connection, metadata) == counts_before


# --- upgrade: rows that are not the catalog row are left alone ----------------


@pytest.mark.parametrize(
    ("server_kwargs", "auth_changes"),
    [
        ({"owner": True}, {}),
        ({}, {"client_id": "operator-client"}),
        ({}, {"scope": "read:me"}),
        ({"name": "Atlassian"}, {}),
        ({"url": "https://evil.example/mcp"}, {}),
        ({"transport": "sse"}, {}),
    ],
    ids=("owner", "client-id", "other-scope", "name", "url", "transport"),
)
def test_upgrade_leaves_servers_that_are_not_the_catalog_row(
    tmp_path, server_kwargs, auth_changes
):
    """Only the row the catalog connect path writes is changed: an owned row,
    a custom OAuth configuration, another name, URL or transport keep their
    auth and every registration, used or not."""
    migration = _load_migration()
    engine, metadata, apps, servers, links = _setup(tmp_path, migration)
    auth = {**migration.LEGACY_AUTH, **auth_changes}
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=auth,
            **server_kwargs,
        )
        _add_registration(connection, metadata, server_id=1, flows=1)
        counts_before = _row_counts(connection, metadata)

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        assert _server_auth(connection, servers, 1) == auth
        assert _row_counts(connection, metadata) == counts_before


@pytest.mark.parametrize(
    "auth",
    [None, "mcp_oauth", ["mcp_oauth"], 1],
    ids=("null", "string", "list", "number"),
)
def test_upgrade_keeps_a_server_whose_auth_is_not_an_object(tmp_path, auth):
    """Only an auth object can be the catalog definition; any other stored
    value is not a row the catalog connect path wrote."""
    migration = _load_migration()
    engine, metadata, apps, servers, links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        _add_server(connection, metadata, servers, links, server_id=1, auth=auth)
        _add_registration(connection, metadata, server_id=1)
        counts_before = _row_counts(connection, metadata)

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        assert _server_auth(connection, servers, 1) == auth
        assert _row_counts(connection, metadata) == counts_before


def test_upgrade_leaves_a_server_already_on_the_scoped_auth(tmp_path):
    """A row a connect already synced to the catalog is not the previous
    definition: neither it nor its registrations are touched."""
    migration = _load_migration()
    engine, metadata, apps, servers, links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.SCOPED_AUTH,
        )
        _add_registration(connection, metadata, server_id=1, flows=1)
        counts_before = _row_counts(connection, metadata)

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        assert _server_auth(connection, servers, 1) == migration.SCOPED_AUTH
        assert _row_counts(connection, metadata) == counts_before


def test_upgrade_leaves_an_operator_catalog_row_and_its_server_alone(tmp_path):
    migration = _load_migration()
    operator_launch = {
        "url": "https://mcp.atlassian.com/v2/mcp",
        "auth": {"type": "mcp_oauth"},
    }
    engine, metadata, apps, servers, links = _setup(
        tmp_path, migration, launch_config=operator_launch
    )
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.LEGACY_AUTH,
        )
        _add_registration(connection, metadata, server_id=1, flows=1)
        counts_before = _row_counts(connection, metadata)

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == operator_launch
        assert _server_auth(connection, servers, 1) == migration.LEGACY_AUTH
        assert _row_counts(connection, metadata) == counts_before


def test_upgrade_without_a_catalog_row_touches_no_server(tmp_path):
    migration = _load_migration()
    engine, metadata, _apps, servers, links = _setup(
        tmp_path, migration, launch_config=False
    )
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.LEGACY_AUTH,
        )
        _add_registration(connection, metadata, server_id=1, flows=1)
        counts_before = _row_counts(connection, metadata)

        _run(connection, migration, "upgrade")

        assert _server_auth(connection, servers, 1) == migration.LEGACY_AUTH
        assert _row_counts(connection, metadata) == counts_before


def test_upgrade_and_downgrade_are_noops_without_the_tables(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
        _run(connection, migration, "upgrade", "downgrade")


def test_upgrade_skips_servers_when_the_server_table_lacks_identity_columns(
    tmp_path,
):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = MetaData()
    apps = Table(
        "public_mcp_apps",
        metadata,
        Column("app_id", String, primary_key=True),
        Column("launch_config", JSON),
    )
    servers = Table(
        "mcp_servers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            apps.insert().values(
                app_id="atlassian", launch_config=migration.LEGACY_LAUNCH_CONFIG
            )
        )
        connection.execute(servers.insert().values(id=1, name="atlassian"))

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        assert _server_ids(connection, servers) == {1}


def test_upgrade_updates_no_server_when_ownership_cannot_be_checked(tmp_path):
    """Fail closed: without user_mcpservers.is_owner a user-owned server
    cannot be told apart from the catalog row, so no auth is changed and no
    registration is deleted."""
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = MetaData()
    apps, servers, _links = _schema(metadata)
    metadata.remove(metadata.tables["user_mcpservers"])
    Table(
        "user_mcpservers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("mcpserver_id", Integer),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        _insert_catalog_row(connection, apps, migration.LEGACY_LAUNCH_CONFIG)
        connection.execute(
            servers.insert().values(
                id=1,
                name="atlassian",
                transport="streamable_http",
                url=migration.SERVER_URL,
                auth=migration.LEGACY_AUTH,
            )
        )
        _add_registration(connection, metadata, server_id=1, flows=1)
        counts_before = _row_counts(connection, metadata)

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        assert _server_auth(connection, servers, 1) == migration.LEGACY_AUTH
        assert _row_counts(connection, metadata) == counts_before


# --- downgrade: auth is set back, nothing is deleted --------------------------


def test_downgrade_sets_the_shared_server_auth_back_in_place(tmp_path):
    migration = _load_migration()
    engine, metadata, apps, servers, links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        _run(connection, migration, "upgrade")
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.SCOPED_AUTH,
        )
        # Downgrade deletes no registration, used or not.
        _add_registration(connection, metadata, server_id=1, flows=1)
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=2,
            name="unrelated",
            url="https://mcp.example.com/mcp",
            auth=migration.SCOPED_AUTH,
        )
        counts_before = _row_counts(connection, metadata)

        _run(connection, migration, "downgrade")

        assert _launch_config(connection, apps) == migration.LEGACY_LAUNCH_CONFIG
        assert _server_ids(connection, servers) == {1, 2}
        assert _server_auth(connection, servers, 1) == migration.LEGACY_AUTH
        assert _server_auth(connection, servers, 2) == migration.SCOPED_AUTH
        assert _row_counts(connection, metadata) == counts_before


def test_downgrade_keeps_the_app_id_key_of_the_shared_server(tmp_path):
    migration = _load_migration()
    engine, metadata, _apps, servers, links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        _run(connection, migration, "upgrade")
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth={**migration.SCOPED_AUTH, "app_id": "atlassian"},
        )

        _run(connection, migration, "downgrade")

        assert _server_auth(connection, servers, 1) == {
            **migration.LEGACY_AUTH,
            "app_id": "atlassian",
        }


@pytest.mark.parametrize(
    ("server_kwargs", "auth_changes"),
    [
        ({"owner": True}, {}),
        ({}, {"client_id": "operator-client"}),
    ],
    ids=("owner-linked", "extra-client-id"),
)
def test_downgrade_leaves_a_scoped_server_that_is_not_the_catalog_row(
    tmp_path, server_kwargs, auth_changes
):
    """An owner-linked row and one with its own client id carry the scoped
    auth too, but they are not the row the catalog connect path writes."""
    migration = _load_migration()
    engine, metadata, apps, servers, links = _setup(tmp_path, migration)
    auth = {**migration.SCOPED_AUTH, **auth_changes}
    with engine.begin() as connection:
        _run(connection, migration, "upgrade")
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=auth,
            **server_kwargs,
        )
        _add_registration(connection, metadata, server_id=1, flows=1)
        counts_before = _row_counts(connection, metadata)

        _run(connection, migration, "downgrade")

        assert _launch_config(connection, apps) == migration.LEGACY_LAUNCH_CONFIG
        assert _server_auth(connection, servers, 1) == auth
        assert _row_counts(connection, metadata) == counts_before


def test_downgrade_without_the_provenance_marker_changes_nothing(tmp_path):
    """A catalog row without the built-in marker is an operator's own app:
    neither it nor the server under its name is touched."""
    migration = _load_migration()
    operator_launch = {
        "url": migration.SERVER_URL,
        "auth": migration.SCOPED_AUTH,
    }
    engine, metadata, apps, servers, links = _setup(
        tmp_path, migration, launch_config=operator_launch
    )
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.SCOPED_AUTH,
        )
        _add_registration(connection, metadata, server_id=1, flows=1)
        counts_before = _row_counts(connection, metadata)

        _run(connection, migration, "downgrade")

        assert _launch_config(connection, apps) == operator_launch
        assert _server_auth(connection, servers, 1) == migration.SCOPED_AUTH
        assert _row_counts(connection, metadata) == counts_before


def test_upgrade_then_downgrade_keeps_the_shared_server_and_its_grants(tmp_path):
    """A round trip returns the row's auth and keeps everything a grant uses;
    a registration no grant used is gone and is not restored."""
    migration = _load_migration()
    engine, metadata, apps, servers, links = _setup(tmp_path, migration)
    original_auth = {**migration.LEGACY_AUTH, "app_id": "atlassian"}
    with engine.begin() as connection:
        in_use = _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=original_auth,
        )
        _add_registration(connection, metadata, server_id=1, flows=1)
        counts_before = _row_counts(connection, metadata)

        _run(connection, migration, "upgrade", "downgrade")

        assert _launch_config(connection, apps) == migration.LEGACY_LAUNCH_CONFIG
        assert _server_ids(connection, servers) == {1}
        assert _server_auth(connection, servers, 1) == original_auth
        assert _client_ids(connection, metadata) == {in_use}
        assert _row_counts(connection, metadata) == {
            **counts_before,
            "mcp_oauth_clients": counts_before["mcp_oauth_clients"] - 1,
            "mcp_oauth_flow_states": counts_before["mcp_oauth_flow_states"] - 1,
        }


# --- the frozen catalog values ------------------------------------------------


def test_scoped_launch_config_matches_the_registry():
    """The migration holds a frozen copy of the catalog value; this catches
    drift between the two."""
    from xagent.web.builtin_mcp_registry import (
        ATLASSIAN_MCP_OAUTH_SCOPE,
        get_builtin_public_mcp_app,
    )

    migration = _load_migration()
    registry_row = get_builtin_public_mcp_app("atlassian")

    assert registry_row is not None
    assert migration.SCOPE == ATLASSIAN_MCP_OAUTH_SCOPE
    assert migration.SCOPED_LAUNCH_CONFIG == registry_row["launch_config"]
    assert registry_row["transport"] == migration.SERVER_TRANSPORT


def test_legacy_launch_config_is_what_the_seed_inserted():
    migration = _load_migration()
    seed = _load("20260920_seed_atlassian_mcp_app.py", "seed_atlassian_for_scope")

    assert migration.LEGACY_LAUNCH_CONFIG == seed.ROW["launch_config"]
    assert migration.SERVER_URL == seed.ROW["launch_config"]["url"]


def test_scoped_auth_requires_only_the_basic_identity_scopes():
    """The migrated catalog auth requests the subset but requires only the
    identity scopes Atlassian grants on every consent; product scopes depend
    on what the account's site has."""
    from xagent.web.builtin_mcp_registry import ATLASSIAN_MCP_OAUTH_REQUIRED_SCOPE
    from xagent.web.services.mcp_oauth import mcp_oauth_required_scope

    migration = _load_migration()

    assert migration.REQUIRED_SCOPE == "email offline_access read:account read:me"
    assert migration.REQUIRED_SCOPE == ATLASSIAN_MCP_OAUTH_REQUIRED_SCOPE
    assert migration.SCOPED_AUTH == {
        "type": "mcp_oauth",
        "scope": migration.SCOPE,
        "required_scope": migration.REQUIRED_SCOPE,
    }
    assert set(migration.REQUIRED_SCOPE.split()) < set(migration.SCOPE.split())
    assert mcp_oauth_required_scope(migration.SCOPED_AUTH) == migration.REQUIRED_SCOPE


def test_upgrade_writes_the_required_scope_into_the_catalog(tmp_path):
    migration = _load_migration()
    engine, _metadata, apps, _servers, _links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        _run(connection, migration, "upgrade")

        auth = _launch_config(connection, apps)["auth"]

    assert auth["required_scope"] == "email offline_access read:account read:me"
    assert auth["scope"] == migration.SCOPE
