"""Tests for remote_mcp_server_identity_is_claimable (seed_helpers)."""

import json

import pytest
from sqlalchemy import JSON, create_engine, text

from xagent.migrations.seed_helpers import (
    MCP_SERVERS_TABLE,
    REMOTE_MCP_SERVER_POLICY_COLUMNS,
    remote_mcp_server_identity_is_claimable,
)

APP_ID = "acme"
DISPLAY_NAME = "Acme (Boards, Docs)"
URL = "https://mcp.acme.example/mcp"
AUTH = {"type": "mcp_oauth"}

_SERVER_IDENTITY_DDL = (
    ", transport VARCHAR(50), url VARCHAR(500), auth JSON, command VARCHAR(500),"
    " args JSON, cwd VARCHAR(500), env JSON, headers JSON, timeout INTEGER,"
    " runtime_input_schema JSON, runtime_bindings JSON,"
    " concurrency_safe BOOLEAN NOT NULL DEFAULT 0, concurrent_tools JSON,"
    " allow_delegated_authorization BOOLEAN NOT NULL DEFAULT 0,"
    " managed VARCHAR(20) NOT NULL DEFAULT 'external',"
    " restart_policy VARCHAR(50) NOT NULL DEFAULT 'no'"
)
_JSON_COLUMNS = {
    "auth",
    "args",
    "env",
    "headers",
    "runtime_input_schema",
    "runtime_bindings",
    "concurrent_tools",
}


def _engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'test.db'}")


def _create_server_tables(
    connection, *, with_identity_columns=True, with_owner_column=True
):
    extra = _SERVER_IDENTITY_DDL if with_identity_columns else ""
    owner = "is_owner BOOLEAN NOT NULL DEFAULT 0," if with_owner_column else ""
    connection.execute(
        text(
            "CREATE TABLE mcp_servers ("
            "id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL UNIQUE" + extra + ")"
        )
    )
    connection.execute(
        text(
            """
            CREATE TABLE user_mcpservers (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                mcpserver_id INTEGER NOT NULL,
                """
            + owner
            + """
                is_active BOOLEAN NOT NULL DEFAULT 1
            )
            """
        )
    )


def _insert_server(
    connection,
    *,
    name=APP_ID,
    url=URL,
    auth=AUTH,
    transport="streamable_http",
    owner_user_id=None,
    server_id=1,
    **policy,
):
    values = {"id": server_id, "name": name, "transport": transport, "url": url}
    values["auth"] = json.dumps(auth) if auth is not None else None
    for column, value in policy.items():
        values[column] = (
            json.dumps(value)
            if column in _JSON_COLUMNS and value is not None
            else value
        )
    columns = ", ".join(values)
    placeholders = ", ".join(f":{c}" for c in values)
    connection.execute(
        text(f"INSERT INTO mcp_servers ({columns}) VALUES ({placeholders})"), values
    )
    if owner_user_id is not None:
        connection.execute(
            text(
                "INSERT INTO user_mcpservers (user_id, mcpserver_id, is_owner, is_active)"
                " VALUES (:user_id, :server_id, 1, 1)"
            ),
            {"user_id": owner_user_id, "server_id": server_id},
        )


def _claimable(connection):
    return remote_mcp_server_identity_is_claimable(
        connection,
        app_id=APP_ID,
        display_name=DISPLAY_NAME,
        transport="streamable_http",
        url=URL,
        auth=AUTH,
        seed_label="Acme",
    )


def _server_rows(connection):
    return list(
        connection.execute(text("SELECT id, name FROM mcp_servers ORDER BY id"))
    )


# The two refusal branches in remote_mcp_server_identity_is_claimable, by the
# phrase only that branch logs.
UNPROVABLE_SCHEMA = "schema cannot prove"
UNPROVEN_ROWS = "could not all be proven"


def _assert_refused(connection, caplog, servers_before, reason):
    """Refused without touching any server row, and the operator gets an
    actionable ERROR naming the rows, the app_id and the specific reason."""
    assert _server_rows(connection) == servers_before
    messages = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert any(
        "mcp_servers" in m and APP_ID in m and "Acme" in m and reason in m
        for m in messages
    ), messages


def test_claimable_when_mcp_servers_table_is_absent(tmp_path):
    with _engine(tmp_path).begin() as connection:
        assert _claimable(connection) is True


def test_claimable_when_no_server_collides(tmp_path):
    with _engine(tmp_path).begin() as connection:
        _create_server_tables(connection)
        _insert_server(
            connection,
            name="my-other-server",
            url="https://other.example/mcp",
            auth=None,
            owner_user_id=7,
        )
        assert _claimable(connection) is True


def test_refuses_user_owned_server_with_foreign_url(tmp_path, caplog):
    with _engine(tmp_path).begin() as connection:
        _create_server_tables(connection)
        _insert_server(connection, url="https://evil.example/mcp", owner_user_id=7)
        before = _server_rows(connection)
        assert _claimable(connection) is False
        _assert_refused(connection, caplog, before, UNPROVEN_ROWS)


def test_refuses_user_owned_server_even_with_matching_url(tmp_path, caplog):
    """The owner keeps edit rights and could later swap in a foreign URL that
    every user of the catalog card would then hit (same reasoning as
    _reject_user_owned_catalog_squat on the connect path)."""
    with _engine(tmp_path).begin() as connection:
        _create_server_tables(connection)
        _insert_server(connection, owner_user_id=7)
        before = _server_rows(connection)
        assert _claimable(connection) is False
        _assert_refused(connection, caplog, before, UNPROVEN_ROWS)


@pytest.mark.parametrize(
    "name",
    [DISPLAY_NAME, f" {DISPLAY_NAME.lower()} ", APP_ID.upper(), f" {APP_ID} "],
)
def test_refuses_servers_whose_name_normalizes_to_the_identity(tmp_path, caplog, name):
    """The listing claims rows named after the display name too, and it
    normalizes case and whitespace (see _catalog_app_keys)."""
    with _engine(tmp_path).begin() as connection:
        _create_server_tables(connection)
        _insert_server(
            connection,
            name=name,
            url="https://other.example/mcp",
            auth={"type": "bearer", "token": "x"},
        )
        before = _server_rows(connection)
        assert _claimable(connection) is False
        _assert_refused(connection, caplog, before, UNPROVEN_ROWS)


def test_refuses_unowned_row_on_a_foreign_transport(tmp_path, caplog):
    with _engine(tmp_path).begin() as connection:
        _create_server_tables(connection)
        _insert_server(connection, transport="sse")
        before = _server_rows(connection)
        assert _claimable(connection) is False
        _assert_refused(connection, caplog, before, UNPROVEN_ROWS)


@pytest.mark.parametrize(
    "auth",
    [
        {"type": "mcp_oauth", "client_id": "custom-client"},
        {"type": "mcp_oauth", "client_secret": "enc:..."},
        {"type": "mcp_oauth", "scope": "boards:read"},
        {"type": "bearer", "token": "x"},
        None,
    ],
    ids=["client_id", "client_secret", "scope", "other-type", "null"],
)
def test_refuses_unowned_row_whose_auth_differs_from_the_catalog_auth(
    tmp_path, caplog, auth
):
    """Only exact auth equality is ours. Extra OAuth keys are a custom
    configuration the connect path's auth healing would overwrite with the
    type-only catalog value, erasing the operator's client_id/secret/scope."""
    with _engine(tmp_path).begin() as connection:
        _create_server_tables(connection)
        _insert_server(connection, auth=auth)
        before = _server_rows(connection)
        assert _claimable(connection) is False
        _assert_refused(connection, caplog, before, UNPROVEN_ROWS)


@pytest.mark.parametrize(
    "policy",
    [
        {"headers": {"X-Proxy": "attacker"}},
        {"env": {"TOKEN": "x"}},
        {"command": "python"},
        {"args": ["--flag"]},
        {"cwd": "/srv/custom"},
        {"runtime_bindings": {"a": "b"}},
        {"runtime_input_schema": {"type": "object"}},
        {"concurrency_safe": True},
        {"concurrent_tools": ["create_issue"]},
        {"timeout": 30},
        {"timeout": 0},
        {"allow_delegated_authorization": True},
        {"managed": "internal"},
        {"restart_policy": "always"},
    ],
    ids=lambda p: f"{next(iter(p))}={next(iter(p.values()))!r}",
)
def test_refuses_unowned_row_carrying_its_own_policy(tmp_path, caplog, policy):
    """An ownerless row (owner account deleted, or admin-edited after a
    downgrade) matching name/transport/URL/auth but carrying any policy field
    is a custom server's configuration. timeout=0 is a configured value (an
    immediate HTTP timeout), not an absent one; concurrency_safe and
    concurrent_tools would let ReAct run non-idempotent remote calls
    concurrently under the custom row's declaration."""
    with _engine(tmp_path).begin() as connection:
        _create_server_tables(connection)
        _insert_server(connection, **policy)
        before = _server_rows(connection)
        assert _claimable(connection) is False
        _assert_refused(connection, caplog, before, UNPROVEN_ROWS)


def test_accepts_the_unowned_connect_created_row(tmp_path):
    """The row _ensure_catalog_mcp_oauth_server writes: name == app_id, catalog
    transport/URL, exactly the catalog auth, every policy column at its
    default, and only non-owner user links (the downgrade -> upgrade round
    trip). Stored defaults mirror MCPServer.from_config for an HTTP row:
    concurrent_tools=[] and env NULL."""
    with _engine(tmp_path).begin() as connection:
        _create_server_tables(connection)
        _insert_server(connection, concurrent_tools=[], env=None)
        connection.execute(
            text(
                "INSERT INTO user_mcpservers (user_id, mcpserver_id, is_owner, is_active)"
                " VALUES (7, 1, 0, 1), (8, 1, 0, 1)"
            )
        )
        assert _claimable(connection) is True


def test_refuses_when_two_rows_collide_even_if_one_is_ours(tmp_path, caplog):
    with _engine(tmp_path).begin() as connection:
        _create_server_tables(connection)
        _insert_server(connection, server_id=1)
        _insert_server(
            connection,
            server_id=2,
            name=DISPLAY_NAME,
            url="https://other.example/mcp",
            auth={"type": "bearer", "token": "x"},
        )
        before = _server_rows(connection)
        assert _claimable(connection) is False
        _assert_refused(connection, caplog, before, UNPROVEN_ROWS)


def test_refuses_when_schema_cannot_prove_ownership(tmp_path, caplog):
    """Without transport/url/auth (or without user_mcpservers) a colliding
    row's provenance cannot be established, so the identity is not claimed."""
    with _engine(tmp_path).begin() as connection:
        _create_server_tables(connection, with_identity_columns=False)
        connection.execute(
            text("INSERT INTO mcp_servers (id, name) VALUES (1, :name)"),
            {"name": APP_ID},
        )
        before = _server_rows(connection)
        assert _claimable(connection) is False
        _assert_refused(connection, caplog, before, UNPROVABLE_SCHEMA)


def test_refuses_when_user_mcpservers_table_is_missing(tmp_path, caplog):
    with _engine(tmp_path).begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE mcp_servers (id INTEGER PRIMARY KEY, "
                "name VARCHAR(100) NOT NULL UNIQUE" + _SERVER_IDENTITY_DDL + ")"
            )
        )
        _insert_server(connection)
        before = _server_rows(connection)
        assert _claimable(connection) is False
        _assert_refused(connection, caplog, before, UNPROVABLE_SCHEMA)


def test_refuses_when_user_mcpservers_lacks_the_ownership_column(tmp_path, caplog):
    """user_mcpservers exists but predates is_owner: ownership cannot be
    proven, so refuse instead of letting the ownership query raise out of
    upgrade()."""
    with _engine(tmp_path).begin() as connection:
        _create_server_tables(connection, with_owner_column=False)
        _insert_server(connection)
        before = _server_rows(connection)
        assert _claimable(connection) is False
        _assert_refused(connection, caplog, before, UNPROVABLE_SCHEMA)


def test_policy_columns_cover_every_persisted_server_policy_field():
    """Pin the reject set against the ORM: every configurable MCPServer column
    beyond identity must be in REMOTE_MCP_SERVER_POLICY_COLUMNS or be a
    lifecycle field compared against its default, so a future column cannot
    slip past the seeds the way a hand-picked subset once did on the connect
    path. Docker-only columns are excluded on purpose. Then pin the
    lightweight MCP_SERVERS_TABLE the helper selects through: it must declare
    every column the helper reads, with the ORM's JSON typing."""
    from xagent.web.models.mcp import MCPServer

    persisted = {c.name for c in MCPServer.__table__.columns}
    identity = {"id", "name", "transport", "url", "auth"}
    # Not compared at all: the catalog card renders from the static registry,
    # and timestamps carry no operator configuration.
    not_compared = {"description", "created_at", "updated_at"}
    checked_against_default = {"managed", "restart_policy"}
    docker_only = {c for c in persisted if c.startswith(("docker_", "container_"))} | {
        "volumes",
        "bind_ports",
        "auto_start",
    }
    expected = (
        persisted - identity - not_compared - docker_only - checked_against_default
    )
    assert expected == set(REMOTE_MCP_SERVER_POLICY_COLUMNS), sorted(
        expected ^ set(REMOTE_MCP_SERVER_POLICY_COLUMNS)
    )

    # The helper reads each policy and lifecycle column through
    # MCP_SERVERS_TABLE.c[...], so the lightweight sa.table() must declare every
    # one, and declare the JSON ones as sa.JSON like the ORM does: an untyped
    # column comes back as the driver's raw string and "[]" would then read as
    # configured.
    declared = {c.name: c.type for c in MCP_SERVERS_TABLE.columns}
    orm_types = {c.name: c.type for c in MCPServer.__table__.columns}
    read_by_helper = expected | checked_against_default
    assert read_by_helper <= set(declared), sorted(read_by_helper - set(declared))
    assert {n for n, t in declared.items() if isinstance(t, JSON)} == {
        n for n in declared if isinstance(orm_types.get(n), JSON)
    }
