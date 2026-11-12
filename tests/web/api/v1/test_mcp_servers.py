"""Integration tests for the /v1 MCP connector endpoints."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi import HTTPException
from sqlalchemy.exc import OperationalError

from xagent.core.tools.adapters.vibe.signed_upload import signed_upload_settings_error
from xagent.core.tools.core.mcp.model import MASKED_SECRET_VALUE, SENSITIVE_AUTH_FIELDS
from xagent.web.api.mcp import TransportFieldValidator
from xagent.web.api.v1 import mcp_servers as mcp_servers_routes
from xagent.web.api.v1.errors import _DEFAULT_MESSAGES, V1ErrorCode
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.database import get_engine
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_api_key import UserApiKey
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services import connector_team_scope, mcp_server_management
from xagent.web.services.connector_name_policy import (
    catalog_app_name_detail,
    folded_name_conflict_detail,
)
from xagent.web.services.connector_team_scope import (
    ConnectorAccess,
    ConnectorHookSessionBoundaryError,
)
from xagent.web.services.mcp_oauth import _canonical_resource

from ..conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    app_for_tests,
    client,
)
from .test_auth import _create_agent_and_key
from .test_management import _bearer

pytestmark = pytest.mark.usefixtures("_test_db")

CONNECTOR_URL = "https://mcp.example.test/mcp"


@pytest.fixture(autouse=True)
def _no_team_hooks():
    connector_team_scope.set_connector_team_hooks()
    yield
    connector_team_scope.set_connector_team_hooks()


# ===== helpers =====


@dataclass(frozen=True)
class Holder:
    username: str
    user_id: int
    key: str

    @property
    def headers(self) -> dict[str, str]:
        return _bearer(self.key)


def _holder(username: str) -> Holder:
    """A registered user with a personal key."""
    _admin_headers()
    web = _register_second_user(username)
    resp = client.post("/api/me/personal-keys", headers=web)
    assert resp.status_code == 200, resp.text
    return _holder_row(username, resp.json()["full_key"])


def _holder_row(username: str, key: str) -> Holder:
    db = _direct_db_session()
    try:
        user_id = db.query(User).filter(User.username == username).one().id
    finally:
        db.close()
    return Holder(username=username, user_id=int(user_id), key=key)


@pytest.fixture
def bob() -> Holder:
    return _holder("bob")


@pytest.fixture
def carol() -> Holder:
    return _holder("carol")


def _seed_link(server_id: int, holder: Holder, *, is_owner: bool) -> None:
    db = _direct_db_session()
    try:
        db.add(
            UserMCPServer(
                user_id=holder.user_id,
                mcpserver_id=server_id,
                is_owner=is_owner,
                can_delete=False,
                is_active=True,
            )
        )
        db.commit()
    finally:
        db.close()


def _seed(
    name: str,
    *,
    owner: Holder | None = None,
    transport: str = "streamable_http",
    url: str | None = CONNECTOR_URL,
    is_owner: bool = True,
    can_delete: bool = False,
    is_active: bool = True,
    user_env: dict[str, Any] | None = None,
    **columns: Any,
) -> int:
    """Insert a connector definition, and the owner's link when there is one."""
    db = _direct_db_session()
    try:
        server = MCPServer(
            name=name, managed="external", transport=transport, url=url, **columns
        )
        db.add(server)
        db.flush()
        if owner is not None:
            db.add(
                UserMCPServer(
                    user_id=owner.user_id,
                    mcpserver_id=server.id,
                    is_owner=is_owner,
                    can_delete=can_delete,
                    is_active=is_active,
                    env=user_env,
                )
            )
        db.commit()
        return int(server.id)
    finally:
        db.close()


def _link(owner: Holder, server_id: int, **fields: Any) -> None:
    db = _direct_db_session()
    try:
        db.add(UserMCPServer(user_id=owner.user_id, mcpserver_id=server_id, **fields))
        db.commit()
    finally:
        db.close()


def _seed_app(app_id: str, name: str) -> None:
    db = _direct_db_session()
    try:
        db.add(PublicMCPApp(app_id=app_id, name=name))
        db.commit()
    finally:
        db.close()


def _share_with_team(*server_ids: int) -> None:
    """Install a list hook that shows these connectors to every holder."""
    connector_team_scope.set_connector_team_hooks(
        visibility=lambda _db, _user_id: {
            "mcp": set(server_ids),
            "custom_api": set(),
        }
    )


def _error(resp, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}, body
    assert body["error"]["code"] == code, body
    return body["error"]


def _get_one(holder: Holder, server_id: int):
    return client.get(f"/v1/mcp-servers/{server_id}", headers=holder.headers)


def _list(holder: Holder) -> list[dict]:
    resp = client.get("/v1/mcp-servers", headers=holder.headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ===== only a live personal key opens these routes =====

_KEY_KINDS = ["no-key", "agent-key", "revoked-personal-key", "workspace-key"]


def _headers_for(kind: str) -> dict:
    if kind == "no-key":
        return {}
    if kind == "agent-key":
        return _bearer(_create_agent_and_key()[1])
    if kind == "revoked-personal-key":
        _admin_headers()
        web = _register_second_user("revoked-holder")
        issued = client.post("/api/me/personal-keys", headers=web).json()
        resp = client.delete(f"/api/me/personal-keys/{issued['id']}", headers=web)
        assert resp.status_code == 200, resp.text
        return _bearer(issued["full_key"])
    # A well-formed key of a kind this deployment does not issue.
    return _bearer("xag_workspace_" + "a" * 6 + "_" + "b" * 32)


def _own_routes() -> list[tuple[str, str]]:
    return sorted(
        (method, f"/v1{route.path}")
        for route in mcp_servers_routes.router.routes
        for method in route.methods  # type: ignore[attr-defined]
    )


@pytest.mark.parametrize("kind", _KEY_KINDS)
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/v1/mcp-servers"),
        ("GET", "/v1/mcp-servers/{server_id}"),
        ("POST", "/v1/mcp-servers"),
        ("GET", "/v1/mcp-servers/{server_id}/tools"),
        ("DELETE", "/v1/mcp-servers/{server_id}"),
        ("PUT", "/v1/mcp-servers/{server_id}/enabled"),
    ],
)
def test_mcp_routes_accept_only_personal_keys(method, path, kind):
    assert _own_routes() == [
        ("DELETE", "/v1/mcp-servers/{server_id}"),
        ("GET", "/v1/mcp-servers"),
        ("GET", "/v1/mcp-servers/{server_id}"),
        ("GET", "/v1/mcp-servers/{server_id}/tools"),
        ("POST", "/v1/mcp-servers"),
        ("PUT", "/v1/mcp-servers/{server_id}/enabled"),
    ]
    bodies = {"POST": _create_body(), "PUT": {"enabled": True}}
    resp = client.request(
        method,
        path.replace("{server_id}", "1"),
        headers=_headers_for(kind),
        json=bodies.get(method),
    )
    _error(resp, 401, "invalid_api_key")


@pytest.mark.parametrize("server_id", [0, 2**31])
def test_server_id_outside_the_column_range_is_rejected(bob, server_id):
    _error(_get_one(bob, server_id), 422, "invalid_input")


# ===== visibility =====

# cell -> (listed, detail status, access). The list hook shares "team_only" and
# "team_and_own"; every other connector is invisible to the holder through it.
_VISIBILITY_CELLS = [
    ("own_enabled", True, 200, "own"),
    ("own_disabled", True, 200, "own"),
    ("team_only", True, 200, "team"),
    ("team_and_own", True, 200, "own"),
    ("someone_elses", False, 404, None),
]


@pytest.fixture
def world(bob, carol) -> dict[str, int]:
    ids = {
        "own_enabled": _seed("own-enabled", owner=bob),
        "own_disabled": _seed("own-disabled", owner=bob, is_active=False),
        "team_only": _seed("team-only", owner=carol),
        "team_and_own": _seed("team-and-own", owner=carol),
        "someone_elses": _seed("someone-elses", owner=carol),
    }
    _link(bob, ids["team_and_own"], is_owner=False)
    _share_with_team(ids["team_only"], ids["team_and_own"])
    return ids


@pytest.mark.parametrize(
    ("cell", "listed", "status", "access"),
    _VISIBILITY_CELLS,
    ids=[cell[0] for cell in _VISIBILITY_CELLS],
)
def test_visibility_matrix(bob, world, cell, listed, status, access):
    assert len(_VISIBILITY_CELLS) == 5
    server_id = world[cell]
    in_list = [row for row in _list(bob) if row["id"] == server_id]
    assert bool(in_list) is listed

    resp = _get_one(bob, server_id)
    if status == 404:
        _error(resp, 404, "connector_not_found")
        return
    assert resp.status_code == 200, resp.text
    assert resp.json()["access"] == access
    assert in_list == [resp.json()]
    expected_enabled = {"own_disabled": False, "team_only": None}.get(cell, True)
    assert resp.json()["enabled"] is expected_enabled
    if cell == "team_only":
        body = resp.json()
        assert body["user_env"] is None
        assert (body["can_edit"], body["can_delete"]) == (False, False)


def test_missing_connector_and_custom_apis(bob, carol):
    # Nothing visible yet.
    assert _list(bob) == []
    _error(_get_one(bob, 4242), 404, "connector_not_found")

    # Custom APIs, personal or shared, are not MCP connectors.
    db = _direct_db_session()
    try:
        own, shared = CustomApi(name="own-api"), CustomApi(name="shared-api")
        db.add_all([own, shared])
        db.flush()
        db.add(UserCustomApi(user_id=bob.user_id, custom_api_id=own.id))
        db.commit()
        shared_id = int(shared.id)
    finally:
        db.close()
    connector_team_scope.set_connector_team_hooks(
        visibility=lambda _db, _uid: {"mcp": set(), "custom_api": {shared_id}}
    )
    assert _list(bob) == []


def test_list_order_puts_own_first_then_team_by_name(bob, carol):
    shared = _seed("a-shared", owner=carol)
    _seed("b-own", owner=bob)
    _seed("A-own", owner=bob)
    _share_with_team(shared)

    assert [row["name"] for row in _list(bob)] == ["A-own", "b-own", "a-shared"]


def test_visibility_hook_failure_is_500(bob, carol):
    carols = _seed("carols", owner=carol)
    _seed("bobs", owner=bob)

    def failing(_db, _user_id):
        raise RuntimeError("team backend exploded")

    connector_team_scope.set_connector_team_hooks(visibility=failing)

    # No part of the list comes back without the team half.
    _error(
        client.get("/v1/mcp-servers", headers=bob.headers),
        500,
        "internal_error",
    )
    _error(_get_one(bob, carols), 500, "internal_error")


def _admin_holder():
    key = client.post("/api/me/personal-keys", headers=_admin_headers()).json()[
        "full_key"
    ]
    return _holder_row("admin", key)


def test_admin_key_lists_only_its_own_connectors(bob):
    admin = _admin_holder()
    mine = _seed("admins", owner=admin, is_owner=False)
    _seed("bobs", owner=bob)

    resp = client.get(f"/v1/mcp-servers?user_id={bob.user_id}", headers=admin.headers)

    assert [row["id"] for row in resp.json()] == [mine]


# ===== access by id =====


def _access_hook(ids: set[int]):
    return lambda _db, _uid, refs: {
        ref: ConnectorAccess(team_owned=True, can_edit=False)
        for ref in refs
        if ref[1] in ids
    }


def _raising_access_hook(_db, _uid, _refs):
    raise RuntimeError("access backend exploded")


def _boundary_error(*_args, **_kwargs):
    raise ConnectorHookSessionBoundaryError("hook ended the transaction")


def _list_hook(ids: set[int]):
    return lambda _db, _uid: {"mcp": set(ids), "custom_api": set()}


@pytest.fixture
def shared_connector(carol) -> int:
    return _seed("shared", owner=carol)


# cell -> (hooks to install, detail status, detail error code)
_ACCESS_CELLS = {
    "access_hook_says_linked": (
        lambda sid: {"access": _access_hook({sid})},
        200,
        None,
    ),
    "access_hook_silent_list_hook_shares": (
        lambda sid: {"access": _access_hook(set()), "visibility": _list_hook({sid})},
        404,
        "connector_not_found",
    ),
    "access_hook_fails": (
        lambda sid: {"access": _raising_access_hook, "visibility": _list_hook({sid})},
        503,
        "connector_runtime_unavailable",
    ),
    "only_list_hook": (lambda sid: {"visibility": _list_hook({sid})}, 200, None),
    "no_hooks": (lambda sid: {}, 404, "connector_not_found"),
}


@pytest.mark.parametrize("cell", list(_ACCESS_CELLS))
def test_access_by_id_follows_the_installed_hooks(bob, shared_connector, cell):
    assert len(_ACCESS_CELLS) == 5
    hooks, status, code = _ACCESS_CELLS[cell]
    connector_team_scope.set_connector_team_hooks(**hooks(shared_connector))

    resp = _get_one(bob, shared_connector)

    if code is None:
        assert resp.status_code == status, resp.text
        assert resp.json()["access"] == "team"
    else:
        _error(resp, status, code)
        assert "exploded" not in resp.text


def test_access_hook_ending_the_transaction_is_an_sdk_500(
    bob, shared_connector, monkeypatch
):
    connector_team_scope.set_connector_team_hooks(
        access=_access_hook({shared_connector})
    )
    monkeypatch.setattr(
        mcp_server_management, "resolve_one_connector_access_or_raise", _boundary_error
    )
    # The application maps this exception to a body of its own; the route must
    # answer in the SDK envelope instead.
    from xagent.web.services.connector_team_scope import (
        connector_hook_session_boundary_error_handler,
    )

    handlers = app_for_tests.exception_handlers
    monkeypatch.setitem(
        handlers,
        ConnectorHookSessionBoundaryError,
        connector_hook_session_boundary_error_handler,
    )
    monkeypatch.setattr(app_for_tests, "middleware_stack", None)

    resp = _get_one(bob, shared_connector)

    assert resp.status_code == 500, resp.text
    assert resp.json() == {
        "error": {
            "code": "internal_error",
            "message": _DEFAULT_MESSAGES[V1ErrorCode.INTERNAL_ERROR],
        }
    }


# ===== secrets =====

_SECRET_AUTH = {
    "type": "oauth2",
    "client_id": "public-client-id",
    **{field: f"SECRET-auth-{field}" for field in SENSITIVE_AUTH_FIELDS},
}
_SECRETS = [
    "SECRET-header-value",
    "SECRET-env-value",
    "SECRET-user-env-value",
    *(f"SECRET-auth-{field}" for field in SENSITIVE_AUTH_FIELDS),
]


@pytest.fixture
def secret_connector(bob) -> int:
    return _seed(
        "secret-holder",
        owner=bob,
        headers={"X-Api-Key": "SECRET-header-value", "X-Empty": ""},
        auth=_SECRET_AUTH,
        env={"TOKEN": "SECRET-env-value"},
        user_env={"MINE": "SECRET-user-env-value"},
    )


def test_no_secret_on_any_path(bob, secret_connector):
    assert len(SENSITIVE_AUTH_FIELDS) == 4
    responses = [
        client.get("/v1/mcp-servers", headers=bob.headers),
        _get_one(bob, secret_connector),
    ]
    for resp in responses:
        assert resp.status_code == 200, resp.text
        for secret in _SECRETS:
            assert secret not in resp.text

    body = responses[1].json()
    assert responses[0].json() == [body]
    assert body["headers"] == {"X-Api-Key": MASKED_SECRET_VALUE, "X-Empty": ""}
    assert body["env"] == {"TOKEN": MASKED_SECRET_VALUE}
    assert body["user_env"] == {"MINE": MASKED_SECRET_VALUE}
    assert body["auth"] == {
        "type": "oauth2",
        "client_id": "public-client-id",
        **{field: MASKED_SECRET_VALUE for field in SENSITIVE_AUTH_FIELDS},
    }


def test_projection_never_decrypts(bob, secret_connector, monkeypatch):
    from xagent.core.utils import encryption

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the projection must not decrypt")

    for name in ("decrypt_value", "decrypt_env_dict"):
        monkeypatch.setattr(encryption, name, forbidden)
    monkeypatch.setattr("xagent.web.api.mcp.decrypt_value", forbidden)

    resp = _get_one(bob, secret_connector)

    assert resp.status_code == 200, resp.text
    assert resp.json()["auth"]["bearer_token"] == MASKED_SECRET_VALUE
    assert client.get("/v1/mcp-servers", headers=bob.headers).status_code == 200


# ===== transports, kinds and permissions =====


@pytest.mark.parametrize("transport", ["stdio", "websocket"])
def test_transport_matrix(bob, transport):
    server_id = _seed(
        "local-launcher",
        owner=bob,
        transport=transport,
        url=None if transport == "stdio" else "wss://mcp.example.test/ws",
        command="SECRET-command",
        args=["--token", "SECRET-arg"],
        cwd="/SECRET/cwd",
    )

    detail = _get_one(bob, server_id)
    for resp in (detail, client.get("/v1/mcp-servers", headers=bob.headers)):
        assert resp.status_code == 200, resp.text
        for leaked in ("SECRET-command", "SECRET-arg", "/SECRET/cwd"):
            assert leaked not in resp.text
    body = detail.json()
    assert {"command", "args", "cwd"}.isdisjoint(body)
    assert body["transport"] == transport
    assert body["kind"] == "custom"
    assert (body["can_edit"], body["can_delete"]) == (False, True)


def _seed_catalog_apps() -> None:
    _seed_app("real-app", "Real App")
    # Two different apps match this name, one by app id and one by name, which
    # the web code treats as "cannot tell".
    _seed_app("ambiguous", "First")
    _seed_app("second", "ambiguous")
    db = _direct_db_session()
    try:
        db.add(
            PublicMCPApp(
                app_id="hidden-app", name="Hidden App", is_visible_in_connector=False
            )
        )
        db.commit()
    finally:
        db.close()


# cell -> (connector name, columns, expected kind)
_CATALOG_CELLS: dict[str, tuple[str, dict[str, Any], str]] = {
    "oauth_transport": ("conn-oauth", {"transport": "oauth", "url": None}, "catalog"),
    "auth_app_id_of_existing_app": (
        "conn-auth-app-id",
        {"auth": {"app_id": "real-app"}},
        "catalog",
    ),
    "auth_app_id_of_deleted_app": (
        "conn-auth-deleted-app-id",
        {"auth": {"app_id": "deleted-app"}},
        "catalog",
    ),
    "auth_provider_only": (
        "conn-auth-provider",
        {"auth": {"provider": "acme"}},
        "catalog",
    ),
    "name_matches_two_apps": ("ambiguous", {}, "catalog"),
    "name_equals_one_app_id": ("real-app", {}, "catalog"),
    "name_equals_one_app_name": ("Real App", {}, "catalog"),
    "name_equals_hidden_app": ("hidden-app", {}, "catalog"),
    "custom_case_variant": ("Real-App", {}, "custom"),
    "custom_no_match": (
        "mine",
        {"auth": {"type": "bearer", "bearer_token": "t"}},
        "custom",
    ),
}


def _assert_kind(body: dict, kind: str) -> None:
    assert body["kind"] == kind
    expected = (kind == "custom", kind == "custom")
    assert (body["can_edit"], body["can_delete"]) == expected


@pytest.mark.parametrize("cell", list(_CATALOG_CELLS))
def test_catalog_classification_is_identical_in_list_and_detail(bob, cell):
    assert len(_CATALOG_CELLS) == 10
    name, columns, kind = _CATALOG_CELLS[cell]
    _seed_catalog_apps()
    server_id = _seed(name, owner=bob, **columns)

    _assert_kind(_get_one(bob, server_id).json(), kind)
    listed = [row for row in _list(bob) if row["id"] == server_id]
    assert len(listed) == 1
    _assert_kind(listed[0], kind)

    db = _direct_db_session()
    try:
        server = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert mcp_server_management.connector_kind(db, server) == kind
    finally:
        db.close()


def test_catalog_classification_of_a_mixed_list(bob):
    _seed_catalog_apps()
    ids = {
        cell: _seed(name, owner=bob, **columns)
        for cell, (name, columns, _kind) in _CATALOG_CELLS.items()
    }

    rows = {row["id"]: row for row in _list(bob)}

    assert len(rows) == len(_CATALOG_CELLS)
    for cell, (_name, _columns, kind) in _CATALOG_CELLS.items():
        _assert_kind(rows[ids[cell]], kind)


# ===== the catalog is read once per request =====


def _catalog_statements(call) -> int:
    engine = get_engine()
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if "public_mcp_apps" in statement:
            statements.append(statement)

    sa.event.listen(engine, "before_cursor_execute", record)
    try:
        call()
    finally:
        sa.event.remove(engine, "before_cursor_execute", record)
    return len(statements)


def _seed_plain(bob, count: int) -> list[int]:
    return [_seed(f"plain-{i}", owner=bob) for i in range(count)]


def _seed_oauth(bob, count: int) -> None:
    for i in range(count):
        _seed(
            f"oauth-{i}",
            owner=bob,
            transport="oauth",
            url=None,
            auth={"app_id": f"app-{i}"},
        )


# cell -> (plain connectors, auth-marker-only connectors, oauth connectors, detail?, count)
_QUERY_CELLS = {
    "c0": (0, 0, 0, False, 0),
    "c1": (1, 0, 0, False, 1),
    "c5": (5, 0, 0, False, 1),
    "c20": (20, 0, 0, False, 1),
    "auth_marker_only": (0, 5, 0, False, 0),
    "detail": (1, 0, 0, True, 1),
    # One read, plus one app lookup per oauth row that is not batched.
    "oauth_mix": (3, 0, 3, False, 4),
}


@pytest.mark.parametrize("cell", list(_QUERY_CELLS))
def test_list_catalog_lookups_do_not_grow_with_rows(bob, cell):
    assert len(_QUERY_CELLS) == 7
    plain, marker, oauth, detail, expected = _QUERY_CELLS[cell]
    plain_ids = _seed_plain(bob, plain)
    for i in range(marker):
        _seed(f"auth-marker-{i}", owner=bob, auth={"provider": "acme"})
    _seed_oauth(bob, oauth)

    if detail:
        call = lambda: _get_one(bob, plain_ids[0])  # noqa: E731
    else:
        call = lambda: _list(bob)  # noqa: E731

    assert _catalog_statements(call) == expected


@pytest.mark.parametrize("route", ["list", "detail"])
def test_a_failing_catalog_query_fails_the_whole_response(bob, route):
    server_id = _seed("plain", owner=bob)
    engine = get_engine()

    def fail_catalog_read(conn, cursor, statement, parameters, context, executemany):
        if "public_mcp_apps" in statement:
            raise OperationalError(statement, parameters, Exception("boom"))

    sa.event.listen(engine, "before_cursor_execute", fail_catalog_read)
    try:
        if route == "list":
            resp = client.get("/v1/mcp-servers", headers=bob.headers)
        else:
            resp = _get_one(bob, server_id)
    finally:
        sa.event.remove(engine, "before_cursor_execute", fail_catalog_read)

    assert resp.status_code == 500, resp.text
    assert resp.json() == {
        "error": {
            "code": "internal_error",
            "message": _DEFAULT_MESSAGES[V1ErrorCode.INTERNAL_ERROR],
        }
    }
    assert "boom" not in resp.text


# cell -> (holder is admin, link fields, expected (can_edit, can_delete))
_PERMISSION_CELLS = {
    "owner": (False, {"is_owner": True}, (True, True)),
    "non_owner": (False, {"is_owner": False}, (False, False)),
    "non_owner_with_delete_grant": (
        False,
        {"is_owner": False, "can_delete": True},
        (False, True),
    ),
    "admin_non_owner": (True, {"is_owner": False}, (True, True)),
}


@pytest.mark.parametrize("cell", list(_PERMISSION_CELLS))
def test_edit_and_delete_flags_follow_the_web_permission_rules(bob, cell):
    assert len(_PERMISSION_CELLS) == 4
    is_admin, link_fields, expected = _PERMISSION_CELLS[cell]
    if is_admin:
        holder = _admin_holder()
    else:
        holder = bob
    server_id = _seed("permission-probe", owner=holder, **link_fields)

    body = _get_one(holder, server_id).json()

    assert (body["can_edit"], body["can_delete"]) == expected


# ===== computed fields =====

_VALID_SIGNED_UPLOAD = {
    "tool_name": "attach_document",
    "file_argument": "document",
    "issuing_tool_name": "request_upload",
    "allowed_upload_hosts": ["uploads.example.test"],
    "upload_path_prefix": "/staged/",
    "content_types": {".pdf": "application/pdf"},
}


@pytest.mark.parametrize(
    ("stored", "usable"),
    [([_VALID_SIGNED_UPLOAD], True), ([{"tool_name": "attach_document"}], False)],
    ids=["valid", "invalid"],
)
def test_signed_uploads_error_is_computed_from_the_stored_value(bob, stored, usable):
    server_id = _seed("uploader", owner=bob, signed_uploads=stored)

    body = _get_one(bob, server_id).json()

    expected = signed_upload_settings_error(
        stored, transport="streamable_http", url=CONNECTOR_URL
    )
    assert (expected is None) is usable
    assert body["signed_uploads_error"] == expected


@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "null"])
def test_timestamps_are_iso_strings_or_null(bob, stamped):
    server_id = _seed("timed", owner=bob)
    db = _direct_db_session()
    try:
        server = db.get(MCPServer, server_id)
        if stamped:
            expected = (server.created_at.isoformat(), server.updated_at.isoformat())
        else:
            db.execute(
                MCPServer.__table__.update()
                .where(MCPServer.id == server_id)
                .values(created_at=None, updated_at=None)
            )
            db.commit()
            expected = (None, None)
    finally:
        db.close()

    body = _get_one(bob, server_id).json()

    assert (body["created_at"], body["updated_at"]) == expected


# ===== the holder's own OAuth authorization =====

AUTHORIZATION_MESSAGE = (
    "Authorization can only be completed in the xagent web app, not with an "
    "API key. Open Connectors and select this connector; if it offers Connect, "
    "choose it, otherwise ask the connector's owner."
)
CONFIGURATION_ERROR_MESSAGE = (
    "This connector's OAuth settings are unusable. Its owner must correct them "
    "in the xagent web app."
)
OTHER_URL = "https://old.example.test/mcp"
OAUTH_AUTH = {"type": "mcp_oauth", "scope": "read write"}


def _seed_grant(
    holder: Holder,
    server_id: int,
    *,
    resource: str | None = None,
    owner_key: str | None = None,
    status: str = "active",
    scope: str = "read write",
    client_id: str | None = None,
    expires_at: datetime | None = None,
    refresh_token: str | None = None,
) -> None:
    """One OAuth grant of ``holder`` on a connector, with its client row."""
    db = _direct_db_session()
    try:
        client_row = MCPOAuthClient(
            mcp_server_id=server_id,
            issuer="https://auth.example.test",
            authorization_endpoint="https://auth.example.test/authorize",
            token_endpoint="https://auth.example.test/token",
            client_id=client_id or f"client-{holder.user_id}-{server_id}",
            redirect_uri="https://xagent.example.test/callback",
            lookup_hash=f"client-{holder.user_id}-{server_id}",
        )
        db.add(client_row)
        db.flush()
        db.add(
            MCPOAuthGrant(
                mcp_server_id=server_id,
                user_id=holder.user_id,
                mcp_oauth_client_id=client_row.id,
                lookup_hash=f"grant-{holder.user_id}-{server_id}-{owner_key}-{status}",
                resource_owner_key=owner_key or f"xagent:user:{holder.user_id}",
                issuer="https://auth.example.test",
                resource=resource or _canonical_resource(CONNECTOR_URL),
                scope=scope,
                access_token="encrypted",
                refresh_token=refresh_token,
                expires_at=expires_at,
                status=status,
            )
        )
        db.commit()
    finally:
        db.close()


def _seed_user_oauth(holder: Holder, *, connected: bool) -> None:
    db = _direct_db_session()
    try:
        db.add(
            UserOAuth(
                user_id=holder.user_id,
                provider="acme",
                provider_user_id="acme-1",
                access_token="token" if connected else "",
            )
        )
        db.commit()
    finally:
        db.close()


def _authorization(holder: Holder, server_id: int) -> dict | None:
    """The field as the detail route reports it; the list must agree."""
    detail = _get_one(holder, server_id)
    assert detail.status_code == 200, detail.text
    listed = [row for row in _list(holder) if row["id"] == server_id]
    assert len(listed) == 1
    assert listed[0]["authorization"] == detail.json()["authorization"]
    return detail.json()["authorization"]


def _assert_authorization(body: dict | None, kind: str, status: str) -> None:
    message = {
        "authorized": None,
        "configuration_error": CONFIGURATION_ERROR_MESSAGE,
    }.get(status, AUTHORIZATION_MESSAGE)
    assert body == {"kind": kind, "status": status, "message": message}


def _own_oauth(holder: Holder, **columns: Any) -> int:
    columns.setdefault("auth", OAUTH_AUTH)
    return _seed("own-oauth", owner=holder, **columns)


def _team_oauth(owner: Holder, bob: Holder) -> int:
    """A connector owned by ``owner`` that ``bob`` sees only through the team."""
    server_id = _seed("team-oauth", owner=owner, auth=OAUTH_AUTH)
    _share_with_team(server_id)
    return server_id


def _cell_own_grant(bob, carol):
    server_id = _own_oauth(bob)
    _seed_grant(bob, server_id)
    return bob, server_id, "mcp_oauth", "authorized"


def _cell_own_no_grant(bob, carol):
    return bob, _own_oauth(bob), "mcp_oauth", "not_authorized"


def _cell_only_others_grant(bob, carol):
    server_id = _own_oauth(bob)
    _seed_grant(carol, server_id)
    return bob, server_id, "mcp_oauth", "not_authorized"


def _cell_own_grant_revoked(bob, carol):
    server_id = _own_oauth(bob)
    _seed_grant(bob, server_id, status="revoked")
    return bob, server_id, "mcp_oauth", "not_authorized"


def _cell_invalid_required_scope(bob, carol):
    server_id = _own_oauth(
        bob, auth={"type": "mcp_oauth", "scope": "read", "required_scope": "write"}
    )
    _seed_grant(bob, server_id)
    return bob, server_id, "mcp_oauth", "configuration_error"


def _cell_catalog_connected(bob, carol):
    _seed_app("acme-app", "Acme App")
    db = _direct_db_session()
    try:
        db.query(PublicMCPApp).filter_by(app_id="acme-app").one().provider_name = "acme"
        db.commit()
    finally:
        db.close()
    _seed_user_oauth(bob, connected=True)
    server_id = _seed(
        "acme-app", owner=bob, transport="oauth", url=None, auth={"app_id": "acme-app"}
    )
    return bob, server_id, "builtin_oauth", "authorized"


def _cell_catalog_needs_reconnect(bob, carol):
    _holder, server_id, _kind, _status = _cell_catalog_connected(bob, carol)
    db = _direct_db_session()
    try:
        db.query(UserOAuth).update({"access_token": ""})
        db.commit()
    finally:
        db.close()
    return bob, server_id, "builtin_oauth", "needs_reconnect"


def _cell_catalog_no_account(bob, carol):
    _holder, server_id, _kind, _status = _cell_catalog_connected(bob, carol)
    db = _direct_db_session()
    try:
        db.query(UserOAuth).delete()
        db.commit()
    finally:
        db.close()
    return bob, server_id, "builtin_oauth", "not_authorized"


def _cell_bearer_connector(bob, carol):
    server_id = _seed(
        "bearer", owner=bob, auth={"type": "bearer", "bearer_token": "secret"}
    )
    return bob, server_id, None, None


def _cell_team_row_member_grant(bob, carol):
    server_id = _team_oauth(carol, bob)
    _seed_grant(bob, server_id)
    return bob, server_id, "mcp_oauth", "authorized"


def _cell_team_row_grant_other_url(bob, carol):
    server_id = _team_oauth(carol, bob)
    _seed_grant(bob, server_id, resource=_canonical_resource(OTHER_URL))
    return bob, server_id, "mcp_oauth", "not_authorized"


def _cell_team_row_grant_non_default_identity(bob, carol):
    server_id = _team_oauth(carol, bob)
    _seed_grant(bob, server_id, owner_key="acme:actor:1")
    return bob, server_id, "mcp_oauth", "not_authorized"


def _cell_own_grant_other_url(bob, carol):
    # A holder with a link of their own is matched without a URL comparison.
    server_id = _own_oauth(bob)
    _seed_grant(bob, server_id, resource=_canonical_resource(OTHER_URL))
    return bob, server_id, "mcp_oauth", "authorized"


def _cell_own_grant_non_default_identity(bob, carol):
    # Only another owner key holds a grant; the holder's default owner key has none.
    server_id = _own_oauth(bob)
    _seed_grant(bob, server_id, owner_key="acme:actor:1")
    return bob, server_id, "mcp_oauth", "not_authorized"


def _cell_mcp_oauth_on_stdio(bob, carol):
    server_id = _own_oauth(bob, transport="stdio", url=None, command="acme-mcp")
    _seed_grant(bob, server_id)
    return bob, server_id, None, None


def _cell_own_unparseable_resource(bob, carol):
    # The grant selector raises ValueError for a resource that does not parse.
    server_id = _own_oauth(bob, auth={**OAUTH_AUTH, "resource": "http://[::1"})
    _seed_grant(bob, server_id)
    return bob, server_id, "mcp_oauth", "configuration_error"


def _cell_team_row_unparseable_issuer(bob, carol):
    server_id = _seed(
        "team-oauth", owner=carol, auth={**OAUTH_AUTH, "issuer": "http://[::1"}
    )
    _share_with_team(server_id)
    return bob, server_id, "mcp_oauth", "configuration_error"


def _cell_own_grant_disabled_link(bob, carol):
    # The grant counts through the holder's own link even when it is disabled.
    server_id = _own_oauth(bob, is_active=False)
    _seed_grant(bob, server_id)
    return bob, server_id, "mcp_oauth", "authorized"


def _cell_own_grant_scope_short(bob, carol):
    server_id = _own_oauth(bob)
    _seed_grant(bob, server_id, scope="read")
    return bob, server_id, "mcp_oauth", "not_authorized"


def _cell_own_grant_other_client_id(bob, carol):
    server_id = _own_oauth(bob, auth={**OAUTH_AUTH, "client_id": "configured"})
    _seed_grant(bob, server_id, client_id="other")
    return bob, server_id, "mcp_oauth", "not_authorized"


def _cell_own_grant_sse(bob, carol):
    server_id = _own_oauth(bob, transport="sse")
    _seed_grant(bob, server_id)
    return bob, server_id, "mcp_oauth", "authorized"


def _cell_own_grant_websocket(bob, carol):
    server_id = _own_oauth(bob, transport="websocket")
    _seed_grant(bob, server_id)
    return bob, server_id, "mcp_oauth", "authorized"


def _cell_own_grant_expired_no_refresh(bob, carol):
    server_id = _own_oauth(bob)
    _seed_grant(bob, server_id, expires_at=_AN_HOUR_AGO)
    return bob, server_id, "mcp_oauth", "needs_reconnect"


def _cell_own_grant_expired_with_refresh(bob, carol):
    server_id = _own_oauth(bob)
    _seed_grant(bob, server_id, expires_at=_AN_HOUR_AGO, refresh_token="encrypted")
    return bob, server_id, "mcp_oauth", "authorized"


def _cell_catalog_app_unresolved(bob, carol):
    server_id = _seed(
        "missing-app",
        owner=bob,
        transport="oauth",
        url=None,
        auth={"app_id": "missing-app"},
    )
    return bob, server_id, "builtin_oauth", "configuration_error"


_AN_HOUR_AGO = datetime.now(timezone.utc) - timedelta(hours=1)

_AUTHORIZATION_CELLS = {
    "own_grant": _cell_own_grant,
    "own_no_grant": _cell_own_no_grant,
    "only_others_grant": _cell_only_others_grant,
    "own_grant_revoked": _cell_own_grant_revoked,
    "invalid_required_scope": _cell_invalid_required_scope,
    "catalog_connected": _cell_catalog_connected,
    "catalog_needs_reconnect": _cell_catalog_needs_reconnect,
    "catalog_no_account": _cell_catalog_no_account,
    "bearer_connector": _cell_bearer_connector,
    "team_row_member_grant": _cell_team_row_member_grant,
    "team_row_grant_other_url": _cell_team_row_grant_other_url,
    "team_row_grant_non_default_identity": _cell_team_row_grant_non_default_identity,
    "own_grant_other_url": _cell_own_grant_other_url,
    "own_grant_non_default_identity": _cell_own_grant_non_default_identity,
    "mcp_oauth_on_stdio": _cell_mcp_oauth_on_stdio,
    "own_unparseable_resource": _cell_own_unparseable_resource,
    "team_row_unparseable_issuer": _cell_team_row_unparseable_issuer,
    "own_grant_disabled_link": _cell_own_grant_disabled_link,
    "own_grant_scope_short": _cell_own_grant_scope_short,
    "own_grant_other_client_id": _cell_own_grant_other_client_id,
    "own_grant_sse": _cell_own_grant_sse,
    "own_grant_websocket": _cell_own_grant_websocket,
    "own_grant_expired_no_refresh": _cell_own_grant_expired_no_refresh,
    "own_grant_expired_with_refresh": _cell_own_grant_expired_with_refresh,
    "catalog_app_unresolved": _cell_catalog_app_unresolved,
}


@pytest.mark.parametrize("cell", list(_AUTHORIZATION_CELLS))
def test_authorization_projection(bob, carol, cell):
    holder, server_id, kind, status = _AUTHORIZATION_CELLS[cell](bob, carol)

    body = _authorization(holder, server_id)

    if kind is None:
        assert body is None
    else:
        _assert_authorization(body, kind, status)


def _is_selector_read(statement: str) -> bool:
    """The grant selector's query, not the read of the holder's grant server ids."""
    return "mcp_oauth_grants" in statement and "mcp_oauth_clients" in statement


def _is_grant_id_set_read(statement: str) -> bool:
    """The read of the server ids that hold an active grant for the holder."""
    return "mcp_oauth_grants" in statement and "mcp_oauth_clients" not in statement


@pytest.mark.parametrize(
    "is_failing_read",
    [_is_selector_read, _is_grant_id_set_read],
    ids=["selector", "id_set"],
)
@pytest.mark.parametrize("access", ["own", "team"])
@pytest.mark.parametrize("route", ["list", "detail"])
def test_authorization_read_failure_fails_the_whole_response(
    bob, carol, route, access, is_failing_read
):
    server_id = _own_oauth(bob) if access == "own" else _team_oauth(carol, bob)
    _seed_grant(bob, server_id)
    engine = get_engine()

    def fail_grant_read(conn, cursor, statement, parameters, context, executemany):
        if is_failing_read(statement):
            raise OperationalError(statement, parameters, Exception("boom"))

    sa.event.listen(engine, "before_cursor_execute", fail_grant_read)
    try:
        if route == "list":
            resp = client.get("/v1/mcp-servers", headers=bob.headers)
        else:
            resp = _get_one(bob, server_id)
    finally:
        sa.event.remove(engine, "before_cursor_execute", fail_grant_read)

    assert resp.status_code == 500, resp.text
    assert resp.json() == {
        "error": {
            "code": "internal_error",
            "message": _DEFAULT_MESSAGES[V1ErrorCode.INTERNAL_ERROR],
        }
    }
    assert "boom" not in resp.text


def test_an_unparseable_oauth_address_does_not_fail_the_list(bob, carol, caplog):
    good_id = _own_oauth(bob)
    _seed_grant(bob, good_id)
    bad_id = _seed(
        "team-oauth", owner=carol, auth={**OAUTH_AUTH, "resource": "http://[::1"}
    )
    _share_with_team(bad_id)

    with caplog.at_level(logging.WARNING, logger=mcp_server_management.__name__):
        rows = {row["id"]: row for row in _list(bob)}

    assert rows[good_id]["authorization"]["status"] == "authorized"
    assert rows[bad_id]["authorization"]["status"] == "configuration_error"
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.name == mcp_server_management.__name__
    ]
    assert warnings == [
        "Reporting MCP OAuth connector authorization as configuration_error, "
        f"its OAuth resource or issuer does not parse: server_id={bad_id} "
        "error=ValueError"
    ]
    assert not any("[::1" in message for message in warnings)


def test_authorization_value_error_from_the_grant_read_propagates(bob):
    server_id = _own_oauth(bob)
    _seed_grant(bob, server_id)
    engine = get_engine()

    def fail_grant_read(conn, cursor, statement, parameters, context, executemany):
        if _is_selector_read(statement):
            raise ValueError("bad stored value")

    sa.event.listen(engine, "before_cursor_execute", fail_grant_read)
    try:
        resp = _get_one(bob, server_id)
    finally:
        sa.event.remove(engine, "before_cursor_execute", fail_grant_read)

    assert resp.status_code == 500, resp.text
    assert resp.json()["error"]["code"] == "internal_error"
    assert "bad stored value" not in resp.text


def test_authorization_reads_grants_once_and_skips_rows_without_one(bob, carol):
    for index in range(2):
        _seed(f"no-grant-{index}", owner=bob, auth=OAUTH_AUTH)
    team_id = _team_oauth(carol, bob)
    _seed_grant(bob, team_id)
    engine = get_engine()
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    sa.event.listen(engine, "before_cursor_execute", record)
    try:
        rows = _list(bob)
    finally:
        sa.event.remove(engine, "before_cursor_execute", record)

    assert {row["authorization"]["status"] for row in rows} == {
        "authorized",
        "not_authorized",
    }
    # One read of the holder's grant server ids, and one selector read for the
    # team row that has a grant.
    assert sum("mcp_oauth_grants" in s for s in statements) == 2
    # The selector gets the row's URL, so it never reads it again.
    assert not any(s.lstrip().startswith("SELECT mcp_servers.url") for s in statements)


# ===== creating a connector =====


def _create_body(**fields: Any) -> dict[str, Any]:
    return {
        "name": "new-connector",
        "transport": "streamable_http",
        "url": CONNECTOR_URL,
        **fields,
    }


def _create(holder: Holder, **fields: Any):
    return client.post(
        "/v1/mcp-servers", headers=holder.headers, json=_create_body(**fields)
    )


def _store() -> tuple[list, list]:
    """Every connector definition and link, for before/after comparison."""
    db = _direct_db_session()
    try:
        return (
            sorted(db.query(MCPServer.id, MCPServer.name, MCPServer.url).all()),
            sorted(
                db.query(
                    UserMCPServer.user_id,
                    UserMCPServer.mcpserver_id,
                    UserMCPServer.is_active,
                    UserMCPServer.is_owner,
                ).all()
            ),
        )
    finally:
        db.close()


def _assert_nothing_written(before: tuple[list, list]) -> None:
    assert _store() == before


def _key_prefix(holder: Holder) -> str:
    db = _direct_db_session()
    try:
        return str(
            db.query(UserApiKey.key_prefix)
            .filter(UserApiKey.user_id == holder.user_id)
            .one()[0]
        )
    finally:
        db.close()


def _fail_web_create(monkeypatch, exc: BaseException) -> None:
    def fail(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(mcp_server_management, "create_mcp_server", fail)


@pytest.mark.parametrize("enabled", [True, False])
def test_create_returns_the_new_connector(bob, enabled):
    resp = _create(
        bob,
        description="A search tool",
        headers={"X-Api-Key": "SECRET-header-value"},
        auth={"type": "bearer", "bearer_token": "SECRET-bearer"},
        enabled=enabled,
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "new-connector"
    assert body["description"] == "A search tool"
    assert (body["transport"], body["url"]) == ("streamable_http", CONNECTOR_URL)
    assert (body["access"], body["kind"], body["enabled"]) == (
        "own",
        "custom",
        enabled,
    )
    assert (body["can_edit"], body["can_delete"]) == (True, True)
    assert body["headers"] == {"X-Api-Key": MASKED_SECRET_VALUE}
    assert body["auth"] == {"type": "bearer", "bearer_token": MASKED_SECRET_VALUE}
    assert _get_one(bob, body["id"]).json() == body
    db = _direct_db_session()
    try:
        link = db.query(UserMCPServer).filter_by(mcpserver_id=body["id"]).one()
        assert (link.user_id, link.is_owner, link.is_active) == (
            bob.user_id,
            True,
            enabled,
        )
    finally:
        db.close()


def test_create_accepts_runtime_declarations(bob):
    schema = {"context": {"account_id": {"type": "string", "required": True}}}
    bindings = [
        {
            "source": {"input_type": "context", "key": "account_id"},
            "target": {"target_type": "mcp_meta", "key": "account_id"},
        }
    ]

    resp = _create(
        bob,
        runtime_input_schema=schema,
        runtime_bindings=bindings,
        allow_delegated_authorization=True,
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["runtime_input_schema"] == schema
    assert body["runtime_bindings"] == bindings
    assert body["allow_delegated_authorization"] is True


# ===== the request rules checked before the web code runs =====

# Transports the API refuses: everything the web code knows how to store except
# the two remote ones, plus spellings that are not the exact names.
_REFUSED_TRANSPORTS = sorted(
    (set(TransportFieldValidator.TRANSPORT_REQUIRED_FIELDS) | {"oauth"})
    - {"streamable_http", "sse"}
) + ["SSE", "http"]


@pytest.mark.parametrize("transport", _REFUSED_TRANSPORTS)
def test_create_refuses_other_transports(bob, transport):
    assert set(_REFUSED_TRANSPORTS) - {"SSE", "http"} == {
        "oauth",
        "stdio",
        "websocket",
    }
    before = _store()

    resp = _create(bob, transport=transport)

    error = _error(resp, 400, "transport_not_supported")
    assert error["message"] == _DEFAULT_MESSAGES[V1ErrorCode.TRANSPORT_NOT_SUPPORTED]
    _assert_nothing_written(before)


def test_create_accepts_the_sse_transport(bob):
    resp = _create(bob, transport="sse")

    assert resp.status_code == 200, resp.text
    assert resp.json()["transport"] == "sse"


# cell -> (request fields, error message)
_INPUT_RULE_CELLS: dict[str, tuple[dict[str, Any], str]] = {
    "blank_name": ({"name": "   "}, "Connector name cannot be blank."),
    "name_ending_in_newline": (
        {"name": "abc\n"},
        "Connector names cannot start or end with tabs, line breaks or other "
        "whitespace besides spaces.",
    ),
    "url_without_scheme": (
        {"url": "mcp.example.test/mcp"},
        "url must be an absolute http(s) URL without user info.",
    ),
    "url_with_user_info": (
        {"url": "https://user:pass@mcp.example.test/mcp"},
        "url must be an absolute http(s) URL without user info.",
    ),
    "url_with_other_scheme": (
        {"url": "ftp://mcp.example.test/mcp"},
        "url must be an absolute http(s) URL without user info.",
    ),
    "url_with_port_out_of_range": (
        {"url": "https://mcp.example.test:99999/mcp"},
        "url must be an absolute http(s) URL without user info.",
    ),
    "url_with_null_character": (
        {"url": "https://mcp.example.test/m\x00cp"},
        "url must be an absolute http(s) URL without user info.",
    ),
    "auth_claims_a_catalog_app": (
        {"auth": {"type": "bearer", "app_id": "gmail"}},
        "auth cannot set app_id, provider or builtin_provenance; these mark "
        "catalog connectors.",
    ),
    "auth_names_a_provider": (
        {"auth": {"provider": "acme"}},
        "auth cannot set app_id, provider or builtin_provenance; these mark "
        "catalog connectors.",
    ),
    "auth_sets_builtin_provenance": (
        {"auth": {"builtin_provenance": "x"}},
        "auth cannot set app_id, provider or builtin_provenance; these mark "
        "catalog connectors.",
    ),
    "masked_header_value": (
        {"headers": {"X-Key": MASKED_SECRET_VALUE}},
        "A new connector has no stored value for '********'; send the real value.",
    ),
    "masked_auth_value": (
        {"auth": {"type": "bearer", "bearer_token": MASKED_SECRET_VALUE}},
        "A new connector has no stored value for '********'; send the real value.",
    ),
    "auth_secret_is_a_number": (
        {"auth": {"type": "bearer", "bearer_token": 123}},
        "auth.bearer_token must be a string or null.",
    ),
    "auth_secret_is_false": (
        {"auth": {"type": "api_key", "api_key_name": "X-Key", "api_key_value": False}},
        "auth.api_key_value must be a string or null.",
    ),
    "auth_secret_is_a_list": (
        {"auth": {"type": "oauth2", "client_secret": ["s"]}},
        "auth.client_secret must be a string or null.",
    ),
    "auth_secret_is_an_object": (
        {"auth": {"type": "oauth2", "access_token": {"v": "s"}}},
        "auth.access_token must be a string or null.",
    ),
}


@pytest.mark.parametrize(
    ("fields", "message"),
    _INPUT_RULE_CELLS.values(),
    ids=list(_INPUT_RULE_CELLS),
)
def test_create_input_rules(bob, fields, message):
    assert len(_INPUT_RULE_CELLS) == 16
    before = _store()

    error = _error(_create(bob, **fields), 400, "invalid_input")

    assert error["message"] == message
    _assert_nothing_written(before)


def test_create_keeps_null_secrets_and_other_auth_values(bob):
    auth = {
        "type": "bearer",
        "bearer_token": None,
        "token_type": "Bearer",
        "extra": 123,
        "nested": {"k": [1, 2]},
    }

    resp = _create(bob, auth=auth)

    assert resp.status_code == 200, resp.text
    assert resp.json()["auth"] == auth


_SHAPE_CELLS: dict[str, dict[str, Any]] = {
    "name_over_100_characters": {"name": "n" * 101},
    "empty_transport": {"transport": ""},
    "transport_over_32_characters": {"transport": "t" * 33},
    "header_name_with_a_space": {"headers": {"X Key": "v"}},
    "header_value_with_a_line_break": {"headers": {"X-Key": "a\r\nInjected: 1"}},
    "header_value_not_ascii": {"headers": {"X-Key": "café"}},
    "more_than_20_headers": {"headers": {f"X-{i}": "v" for i in range(21)}},
    "unknown_field": {"user_env": {"A": "b"}},
    "url_over_500_characters": {"url": "https://example.test/" + "a" * 500},
}


@pytest.mark.parametrize("fields", _SHAPE_CELLS.values(), ids=list(_SHAPE_CELLS))
def test_create_rejects_a_malformed_body(bob, fields):
    assert len(_SHAPE_CELLS) == 9
    before = _store()

    resp = _create(bob, **fields)

    _error(resp, 422, "invalid_input")
    for leaked in ("Injected", "café"):
        assert leaked not in resp.text
    _assert_nothing_written(before)


# ===== secrets =====

_REQUEST_SECRETS = {
    "headers": {"X-Api-Key": "SECRET-request-header"},
    "auth": {"type": "bearer", "bearer_token": "SECRET-request-bearer"},
}
_REQUEST_SECRET_VALUES = ["SECRET-request-header", "SECRET-request-bearer"]

_BAD_CONFIGURATION = "Invalid configuration: Configuration error in field 'url': boom"
_DEFAULT_INVALID_INPUT = _DEFAULT_MESSAGES[V1ErrorCode.INVALID_INPUT]
_DEFAULT_INTERNAL = _DEFAULT_MESSAGES[V1ErrorCode.INTERNAL_ERROR]
_NAME_CONFLICT = _DEFAULT_MESSAGES[V1ErrorCode.CONNECTOR_NAME_CONFLICT]

# (row id, web error to inject, status, code and message the SDK answers). A row
# may have several cases.
_ROW_CASES: list[tuple[str, HTTPException, int, str, str]] = [
    (
        "exact_name_conflict",
        HTTPException(400, "MCP server 'new-connector' already exists"),
        409,
        "connector_name_conflict",
        _NAME_CONFLICT,
    ),
    (
        "folded_name_conflict",
        HTTPException(400, folded_name_conflict_detail("new-connector")),
        409,
        "connector_name_conflict",
        _NAME_CONFLICT,
    ),
    (
        "catalog_name_conflict",
        HTTPException(400, catalog_app_name_detail("new-connector")),
        409,
        "connector_name_conflict",
        catalog_app_name_detail("new-connector"),
    ),
    (
        "passed_through_message",
        HTTPException(400, _BAD_CONFIGURATION),
        400,
        "invalid_input",
        _BAD_CONFIGURATION,
    ),
    (
        "passed_through_message",
        HTTPException(400, f"{_BAD_CONFIGURATION} {_REQUEST_SECRET_VALUES[0]}"),
        400,
        "invalid_input",
        _DEFAULT_INVALID_INPUT,
    ),
    (
        "passed_through_message",
        HTTPException(400, "x" * 1001),
        400,
        "invalid_input",
        _DEFAULT_INVALID_INPUT,
    ),
    (
        "unclassified_bad_request",
        HTTPException(400, {"detail": _REQUEST_SECRET_VALUES[1]}),
        400,
        "invalid_input",
        _DEFAULT_INVALID_INPUT,
    ),
    (
        "unclassified_bad_request",
        HTTPException(400, ""),
        400,
        "invalid_input",
        _DEFAULT_INVALID_INPUT,
    ),
    (
        "forbidden",
        HTTPException(403, f"Not allowed: {_REQUEST_SECRET_VALUES[0]}"),
        403,
        "connector_forbidden",
        _DEFAULT_MESSAGES[V1ErrorCode.CONNECTOR_FORBIDDEN],
    ),
    (
        "not_found",
        HTTPException(404, f"MCP server not found {_REQUEST_SECRET_VALUES[1]}"),
        404,
        "connector_not_found",
        _DEFAULT_MESSAGES[V1ErrorCode.CONNECTOR_NOT_FOUND],
    ),
    (
        "managed_elsewhere",
        HTTPException(409, f"Has connections: {_REQUEST_SECRET_VALUES[0]}"),
        409,
        "connector_not_editable",
        "This connector has connections managed outside your account; remove it "
        "in the web app.",
    ),
    (
        "internal_error",
        HTTPException(500, f"Failed to create MCP server: {_REQUEST_SECRET_VALUES[0]}"),
        500,
        "internal_error",
        _DEFAULT_INTERNAL,
    ),
    (
        "internal_error",
        HTTPException(502, "anything else"),
        500,
        "internal_error",
        _DEFAULT_INTERNAL,
    ),
]


def test_create_success_carries_no_secret(bob):
    rows = mcp_servers_routes._MCP_WRITE_ERROR_ROWS
    assert len(rows) == 9
    assert {row.id for row in rows} == {case[0] for case in _ROW_CASES}

    ok = _create(bob, name="made-connector", **_REQUEST_SECRETS)
    assert ok.status_code == 200, ok.text
    for secret in _REQUEST_SECRET_VALUES:
        assert secret not in ok.text
    body = ok.json()
    assert body["headers"] == {"X-Api-Key": MASKED_SECRET_VALUE}
    assert body["auth"]["bearer_token"] == MASKED_SECRET_VALUE


# ===== error translation =====


@pytest.mark.parametrize(
    ("row_id", "exc", "status", "code", "message"),
    _ROW_CASES,
    ids=[f"{case[0]}-{i}" for i, case in enumerate(_ROW_CASES)],
)
def test_write_error_translation(bob, monkeypatch, row_id, exc, status, code, message):
    assert len(_ROW_CASES) == 13
    before = _store()
    _fail_web_create(monkeypatch, exc)

    resp = _create(bob, **_REQUEST_SECRETS)

    error = _error(resp, status, code)
    assert error["message"] == message
    assert "details" not in error
    assert "detail" not in resp.json()
    for secret in _REQUEST_SECRET_VALUES:
        assert secret not in resp.text
    _assert_nothing_written(before)


def _own_taken(bob, carol):
    return _seed("taken", owner=bob)


def _someone_elses_taken(bob, carol):
    return _seed("taken", owner=carol)


def _taken_with_a_non_owner_link(bob, carol):
    server_id = _seed("taken", owner=carol)
    _seed_link(server_id, bob, is_owner=False)
    return server_id


def _folded_taken(bob, carol):
    return _seed("my_server", owner=carol)


def _reserved_by_catalog(bob, carol):
    _seed_app("real-app", "Real App")


# cell -> (name in the request, seeder, whether the error names the holder's own
# connector)
_NAME_CONFLICT_CELLS = {
    "own_connector_with_the_same_name": ("taken", _own_taken, True),
    "someone_elses_connector_with_the_same_name": (
        "taken",
        _someone_elses_taken,
        False,
    ),
    "someone_elses_connector_the_holder_is_linked_to": (
        "taken",
        _taken_with_a_non_owner_link,
        False,
    ),
    "name_that_folds_to_an_existing_one": ("My-Server", _folded_taken, False),
    "name_reserved_for_a_catalog_app": ("real-app", _reserved_by_catalog, False),
}


@pytest.mark.parametrize(
    ("name", "seeder", "names_the_connector"),
    _NAME_CONFLICT_CELLS.values(),
    ids=list(_NAME_CONFLICT_CELLS),
)
def test_name_conflicts_from_the_web_code(
    bob, carol, name, seeder, names_the_connector
):
    assert len(_NAME_CONFLICT_CELLS) == 5
    existing = seeder(bob, carol)
    before = _store()

    resp = _create(bob, name=name)

    error = _error(resp, 409, "connector_name_conflict")
    if names_the_connector:
        assert error["details"] == {"connector_id": existing}
    else:
        assert "details" not in error
    if name == "real-app":
        assert error["message"] == catalog_app_name_detail("real-app")
    else:
        assert error["message"] == _NAME_CONFLICT
    _assert_nothing_written(before)


_PYDANTIC_TAIL = (
    "    For further information visit https://errors.pydantic.dev/2.12/v/value_error"
)
_NAME_REASON = (
    "Value error, Name can only contain letters, numbers, hyphens and underscores"
)
_HEAD = "Invalid configuration: 1 validation error for MCPServerConfig"
_FIELD_SUMMARY = f"{_HEAD}: name: {_NAME_REASON}"

# cell -> (web detail, expected message)
_SUMMARY_CELLS: dict[str, tuple[str, str]] = {
    "field_error": (
        f"{_HEAD}\nname\n  {_NAME_REASON} "
        f"[type=value_error, input_value='ab c-INPUT-SENTINEL', input_type=str]\n"
        f"{_PYDANTIC_TAIL}",
        _FIELD_SUMMARY,
    ),
    "model_error_whose_input_is_the_whole_request": (
        f"{_HEAD}\n  Value error, external managed servers should not have "
        "auto_start parameter [type=value_error, input_value={'name': 'ok', "
        f"'url': 'INPUT-SENTINEL'}}, input_type=dict]\n{_PYDANTIC_TAIL}",
        f"{_HEAD}: Value error, external managed servers should not have "
        "auto_start parameter",
    ),
    "two_errors": (
        "Invalid configuration: 2 validation errors for MCPServerConfig\n"
        f"name\n  {_NAME_REASON} [type=value_error, input_value='a b', "
        f"input_type=str]\n{_PYDANTIC_TAIL}\n"
        "url\n  Input should be a valid string [type=string_type, "
        f"input_value=7, input_type=int]\n{_PYDANTIC_TAIL}",
        "Invalid configuration: 2 validation errors for MCPServerConfig: "
        f"name: {_NAME_REASON}",
    ),
    "reason_line_of_unknown_shape": (
        f"{_HEAD}\nname\n  something odd INPUT-SENTINEL",
        _HEAD,
    ),
    "reason_contains_a_request_secret": (
        f"{_HEAD}\nname\n  Value error, rejected {_REQUEST_SECRET_VALUES[0]} "
        "[type=value_error, input_value='x', input_type=str]\n"
        f"{_PYDANTIC_TAIL}",
        _DEFAULT_INVALID_INPUT,
    ),
    "one_line_configuration_error": (
        "Invalid configuration: Masked secret 'X-Key' has no stored value; "
        "provide a new value",
        "Invalid configuration: Masked secret 'X-Key' has no stored value; "
        "provide a new value",
    ),
    "multi_line_text_of_another_shape": (
        "First line\nsecond line INPUT-SENTINEL",
        "First line",
    ),
}


@pytest.mark.parametrize(
    ("detail", "message"), _SUMMARY_CELLS.values(), ids=list(_SUMMARY_CELLS)
)
def test_invalid_configuration_is_summarized(bob, monkeypatch, detail, message):
    assert len(_SUMMARY_CELLS) == 7
    _fail_web_create(monkeypatch, HTTPException(400, detail))

    resp = _create(bob, **_REQUEST_SECRETS)

    error = _error(resp, 400, "invalid_input")
    assert error["message"] == message
    assert "INPUT-SENTINEL" not in resp.text


def test_a_name_the_web_code_rejects_is_summarized_without_the_input(bob):
    before = _store()

    resp = _create(bob, name="ab c", **_REQUEST_SECRETS)

    error = _error(resp, 400, "invalid_input")
    assert error["message"] == _FIELD_SUMMARY
    assert "input_value" not in resp.text
    _assert_nothing_written(before)


# ===== internal errors, retries and the write log =====

_SENTINEL = "SENTINEL-internal-detail"


@pytest.mark.parametrize("failure", ["http_500", "runtime_error"])
def test_internal_errors_are_sanitized(bob, monkeypatch, caplog, failure):
    exc = {
        "http_500": HTTPException(500, f"Failed to create MCP server: {_SENTINEL}"),
        "runtime_error": RuntimeError(_SENTINEL),
    }[failure]
    _fail_web_create(monkeypatch, exc)

    with caplog.at_level(logging.DEBUG):
        resp = _create(bob)

    error = _error(resp, 500, "internal_error")
    assert error["message"] == _DEFAULT_INTERNAL
    assert _SENTINEL not in resp.text
    if failure == "http_500":
        ours = [
            r
            for r in caplog.records
            if r.name.startswith(("xagent.web.api.v1", mcp_server_management.__name__))
        ]
        assert any("action=create status=500" in r.getMessage() for r in ours)
        for record in ours:
            assert _SENTINEL not in record.getMessage()
            assert _SENTINEL not in (record.exc_text or "")
            assert record.exc_info is None


def test_retrying_a_create_names_the_connector_it_made(bob):
    first = _create(bob, headers={"X-Key": "v"})
    assert first.status_code == 200, first.text
    before = _store()

    again = _create(bob, headers={"X-Key": "v"})

    error = _error(again, 409, "connector_name_conflict")
    assert error["details"] == {"connector_id": first.json()["id"]}
    _assert_nothing_written(before)


def test_a_failing_owned_connector_lookup_answers_500(bob):
    _seed("taken", owner=bob)
    engine = get_engine()

    def fail_lookup(conn, cursor, statement, parameters, context, executemany):
        # The lookup of the holder's own connector id: it selects only the id.
        flat = " ".join(statement.split())
        if flat.startswith("SELECT mcp_servers.id AS mcp_servers_id FROM mcp_servers"):
            raise OperationalError(statement, parameters, Exception(_SENTINEL))

    sa.event.listen(engine, "before_cursor_execute", fail_lookup)
    try:
        resp = _create(bob, name="taken")
    finally:
        sa.event.remove(engine, "before_cursor_execute", fail_lookup)

    error = _error(resp, 500, "internal_error")
    assert error["message"] == _DEFAULT_INTERNAL
    assert _SENTINEL not in resp.text


def test_a_created_connector_that_cannot_be_read_back_is_logged(
    bob, monkeypatch, caplog
):
    monkeypatch.setattr(
        mcp_server_management, "resolve_holder_connector", lambda *_a, **_k: None
    )

    with caplog.at_level(logging.DEBUG):
        resp = _create(bob, name="unreadable-subject", **_REQUEST_SECRETS)

    error = _error(resp, 500, "internal_error")
    assert error["message"] == _DEFAULT_INTERNAL
    errors = [
        r
        for r in caplog.records
        if r.levelno >= logging.ERROR
        and r.name.startswith(("xagent.web.api.v1", mcp_server_management.__name__))
    ]
    assert len(errors) == 1
    db = _direct_db_session()
    try:
        server_id = db.query(MCPServer.id).filter_by(name="unreadable-subject").one()[0]
    finally:
        db.close()
    assert errors[0].getMessage() == (
        "personal key mcp server write failed: action=create "
        f"user_id={bob.user_id} server_id={server_id}"
    )
    assert errors[0].exc_info is None
    assert "unreadable-subject" not in errors[0].getMessage()
    assert not any(s in errors[0].getMessage() for s in _REQUEST_SECRET_VALUES)
    # The connector itself was created; only the read-back failed.
    db = _direct_db_session()
    try:
        assert db.query(UserMCPServer).filter_by(mcpserver_id=server_id).count() == 1
    finally:
        db.close()
    # No success line, and no record at any level carries the name, a secret or
    # the url. The shared create path (web ``create_mcp_server`` and the
    # connector manager) logs its own info lines that name the connector; they
    # predate this endpoint and are left out.
    shared_loggers = ("xagent.web.api.mcp", "xagent.core.tools.core.mcp.manager.db")
    records = [r for r in caplog.records if r.name not in shared_loggers]
    messages = [r.getMessage() for r in records]
    assert not any("action=create key_prefix=" in m for m in messages)
    forbidden = ["unreadable-subject", CONNECTOR_URL, *_REQUEST_SECRET_VALUES]
    assert [(m, f) for m in messages for f in forbidden if f in m] == []


def test_a_create_logs_one_line_without_the_name_or_secrets(bob, caplog):
    with caplog.at_level(logging.INFO):
        resp = _create(bob, name="log-subject", **_REQUEST_SECRETS)

    assert resp.status_code == 200, resp.text
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("personal key mcp server write:")
    ]
    assert lines == [
        f"personal key mcp server write: action=create "
        f"key_prefix={_key_prefix(bob)} user_id={bob.user_id} "
        f"server_id={resp.json()['id']}"
    ]
    assert "log-subject" not in lines[0]
    assert not any(secret in lines[0] for secret in _REQUEST_SECRET_VALUES)
