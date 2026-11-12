"""Integration tests for deleting and switching connectors through /v1."""

import asyncio
import logging
import threading
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.exc import OperationalError

from xagent.web.api.v1.deps import _bearer, get_user_from_personal_key
from xagent.web.api.v1.errors import _DEFAULT_MESSAGES, V1ErrorCode
from xagent.web.models.database import get_engine
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import MCPOAuthGrant
from xagent.web.services import connector_team_scope, mcp_server_management
from xagent.web.services.connector_team_scope import ConnectorDeleteDecision

from ..conftest import _direct_db_session, app_for_tests, client
from .test_mcp_servers import secret_connector  # noqa: F401  (fixture)
from .test_mcp_servers import (
    _CATALOG_CELLS,
    _SECRETS,
    CONNECTOR_URL,
    Holder,
    _assert_nothing_written,
    _error,
    _get_one,
    _holder,
    _key_prefix,
    _link,
    _list,
    _seed,
    _seed_catalog_apps,
    _seed_grant,
    _share_with_team,
    _store,
)

pytestmark = pytest.mark.usefixtures("_test_db")

_TEAM_READ_ONLY = (
    "This connector is shared with your team and is read-only through the API."
)
_NOT_EDITABLE_CATALOG = (
    "This connector is a catalog app connection, or cannot be shown not to be "
    "one, so it can only be changed in the web app."
)
_MANAGED_ELSEWHERE = (
    "This connector has connections managed outside your account; remove it in "
    "the web app."
)
_FORBIDDEN = _DEFAULT_MESSAGES[V1ErrorCode.CONNECTOR_FORBIDDEN]
_NOT_FOUND = _DEFAULT_MESSAGES[V1ErrorCode.CONNECTOR_NOT_FOUND]
_INTERNAL = _DEFAULT_MESSAGES[V1ErrorCode.INTERNAL_ERROR]

# The three write actions this module adds, as the log line names them.
_WRITE_ACTIONS = ["delete", "enable", "disable"]


@pytest.fixture(autouse=True)
def _no_team_hooks():
    connector_team_scope.set_connector_team_hooks()
    yield
    connector_team_scope.set_connector_team_hooks()


@pytest.fixture
def bob() -> Holder:
    return _holder("bob")


@pytest.fixture
def carol() -> Holder:
    return _holder("carol")


# ===== helpers =====


def _delete(holder: Holder, server_id: int):
    return client.delete(f"/v1/mcp-servers/{server_id}", headers=holder.headers)


def _set_enabled(holder: Holder, server_id: int, enabled: Any):
    return client.put(
        f"/v1/mcp-servers/{server_id}/enabled",
        headers=holder.headers,
        json={"enabled": enabled},
    )


def _act(action: str, holder: Holder, server_id: int):
    if action == "delete":
        return _delete(holder, server_id)
    return _set_enabled(holder, server_id, action == "enable")


def _link_of(holder: Holder, server_id: int) -> UserMCPServer | None:
    db = _direct_db_session()
    try:
        return (
            db.query(UserMCPServer)
            .filter_by(user_id=holder.user_id, mcpserver_id=server_id)
            .one_or_none()
        )
    finally:
        db.close()


def _definition_exists(server_id: int) -> bool:
    db = _direct_db_session()
    try:
        return db.query(MCPServer.id).filter_by(id=server_id).first() is not None
    finally:
        db.close()


def _grant_count(server_id: int) -> int:
    db = _direct_db_session()
    try:
        return db.query(MCPOAuthGrant).filter_by(mcp_server_id=server_id).count()
    finally:
        db.close()


class _Calls:
    """Counts calls to the web functions, which must not run on a refusal."""

    def __init__(self, monkeypatch) -> None:
        self.calls: list[str] = []

        async def delete(*_a, **_k):
            self.calls.append("delete")
            raise AssertionError("web delete must not run")

        def update(*_a, **_k):
            self.calls.append("update")
            raise AssertionError("web update must not run")

        monkeypatch.setattr(mcp_server_management, "delete_mcp_server", delete)
        monkeypatch.setattr(mcp_server_management, "update_mcp_server", update)


def _lose_the_link():
    """A ``resolve_holder_connector`` that finds the connector on its first call
    and nothing on its second, as if another request deleted it in between."""
    real = mcp_server_management.resolve_holder_connector
    calls = {"n": 0}

    def resolve(db, user_id, server_id):
        calls["n"] += 1
        return real(db, user_id, server_id) if calls["n"] == 1 else None

    return resolve


def _fail_statements(match) -> Any:
    """Make every SQL statement accepted by ``match`` fail in the real database."""
    engine = get_engine()

    def fail(conn, cursor, statement, parameters, context, executemany):
        if match(statement):
            raise OperationalError(statement, parameters, Exception("boom"))

    sa.event.listen(engine, "before_cursor_execute", fail)
    return lambda: sa.event.remove(engine, "before_cursor_execute", fail)


# ===== write isolation =====

_REFUSED_CELLS = {
    "someone_elses": (404, "connector_not_found", _NOT_FOUND),
    "shared_with_the_team": (403, "connector_forbidden", _TEAM_READ_ONLY),
    "missing": (404, "connector_not_found", _NOT_FOUND),
}


@pytest.mark.parametrize("cell", list(_REFUSED_CELLS))
@pytest.mark.parametrize("action", _WRITE_ACTIONS)
def test_a_connector_the_holder_may_not_change_is_refused_without_a_write(
    bob, carol, monkeypatch, caplog, action, cell
):
    assert len(_REFUSED_CELLS) == 3
    status, code, message = _REFUSED_CELLS[cell]
    server_id = _seed("carols", owner=carol)
    if cell == "shared_with_the_team":
        _share_with_team(server_id)
    if cell == "missing":
        server_id += 1000
    before = _store()
    spy = _Calls(monkeypatch)

    with caplog.at_level(logging.INFO):
        resp = _act(action, bob, server_id)

    assert _error(resp, status, code)["message"] == message
    assert spy.calls == []
    _assert_nothing_written(before)
    assert not [r for r in caplog.records if "mcp server write:" in r.getMessage()]


@pytest.mark.parametrize("action", _WRITE_ACTIONS)
def test_a_failing_access_hook_is_a_503_and_writes_nothing(
    bob, carol, monkeypatch, action
):
    server_id = _seed("carols", owner=carol)
    connector_team_scope.set_connector_team_hooks(
        access=lambda _db, _uid, _refs: (_ for _ in ()).throw(RuntimeError("exploded")),
        visibility=lambda _db, _uid: {"mcp": {server_id}, "custom_api": set()},
    )
    before = _store()
    spy = _Calls(monkeypatch)

    resp = _act(action, bob, server_id)

    # The hook failure is not folded into "not found" or "forbidden".
    _error(resp, 503, "connector_runtime_unavailable")
    assert "exploded" not in resp.text
    assert spy.calls == []
    _assert_nothing_written(before)


# ===== deleting =====


@pytest.mark.parametrize("transport", ["streamable_http", "stdio", "websocket"])
def test_delete_removes_the_holders_connector(bob, carol, transport):
    server_id = _seed(
        "doomed",
        owner=bob,
        transport=transport,
        url=CONNECTOR_URL if transport == "streamable_http" else None,
        command="launch" if transport == "stdio" else None,
    )
    other = _seed("kept", owner=bob)
    carols = _seed("carols", owner=carol)

    resp = _delete(bob, server_id)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"id": server_id}
    assert not _definition_exists(server_id)
    assert _link_of(bob, server_id) is None
    assert [row["id"] for row in _list(bob)] == [other]
    assert _definition_exists(carols) and _link_of(carol, carols) is not None
    _error(_get_one(bob, server_id), 404, "connector_not_found")


def test_delete_leaves_the_definition_while_another_user_still_links_it(bob, carol):
    server_id = _seed("shared-by-link", owner=bob)
    _link(carol, server_id, is_owner=False)

    resp = _delete(bob, server_id)

    assert resp.status_code == 200, resp.text
    assert _link_of(bob, server_id) is None
    assert _definition_exists(server_id)
    assert _link_of(carol, server_id) is not None


def test_a_non_owner_with_delete_rights_deletes_only_their_own_link(bob, carol):
    server_id = _seed("owned-by-bob", owner=bob)
    _link(carol, server_id, is_owner=False, can_delete=True)

    resp = _delete(carol, server_id)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"id": server_id}
    assert _link_of(carol, server_id) is None
    assert _definition_exists(server_id)
    assert _link_of(bob, server_id) is not None


def test_a_second_delete_is_a_404(bob):
    server_id = _seed("once", owner=bob)
    assert _delete(bob, server_id).status_code == 200

    _error(_delete(bob, server_id), 404, "connector_not_found")


def test_deleting_a_team_owned_definition_leaves_it_to_the_team(bob):
    server_id = _seed("team-owned", owner=bob)
    connector_team_scope.set_connector_team_hooks(
        deleted=lambda _db, _uid, _type, _id: ConnectorDeleteDecision(
            team_owned=True, authorized=True, delete_definition=False
        ),
        visibility=lambda _db, _uid: {"mcp": {server_id}, "custom_api": set()},
    )

    resp = _delete(bob, server_id)

    assert resp.status_code == 200, resp.text
    assert _definition_exists(server_id)
    assert _link_of(bob, server_id) is None
    assert _get_one(bob, server_id).json()["access"] == "team"
    # The team still shares it, so a repeat is refused rather than not found.
    _error(_delete(bob, server_id), 403, "connector_forbidden")


@pytest.mark.parametrize(
    "reason",
    ["blocked_by_team_agent", "not_a_team_admin"],
)
def test_a_web_refusal_to_delete_is_a_403_that_changes_nothing(bob, reason):
    server_id = _seed("held", owner=bob)
    if reason == "blocked_by_team_agent":
        connector_team_scope.set_connector_team_hooks(
            deleted=lambda *_a: ConnectorDeleteDecision(
                team_owned=True, blocked_reason="Still used by a team agent: SECRET"
            )
        )
    else:
        connector_team_scope.set_connector_team_hooks(
            deleted=lambda *_a: ConnectorDeleteDecision(
                team_owned=True, authorized=False
            )
        )
    before = _store()

    resp = _delete(bob, server_id)

    error = _error(resp, 403, "connector_forbidden")
    assert error["message"] == _FORBIDDEN
    assert "SECRET" not in resp.text
    _assert_nothing_written(before)


def test_a_link_without_delete_rights_is_a_403_from_the_web_rules(bob):
    server_id = _seed("not-mine", owner=bob, is_owner=False, can_delete=False)
    before = _store()

    _error(_delete(bob, server_id), 403, "connector_forbidden")

    _assert_nothing_written(before)


def test_other_accounts_oauth_state_makes_a_delete_a_409(bob):
    server_id = _seed("actor-bound", owner=bob)
    _seed_grant(bob, server_id, owner_key="xagent:actor:machine-account")
    before = _store()

    resp = _delete(bob, server_id)

    error = _error(resp, 409, "connector_not_editable")
    assert error["message"] == _MANAGED_ELSEWHERE
    _assert_nothing_written(before)
    assert _grant_count(server_id) == 1


@pytest.mark.parametrize("cell", list(_CATALOG_CELLS))
def test_delete_refuses_catalog_connectors_and_deletes_custom_ones(
    bob, monkeypatch, cell
):
    name, columns, kind = _CATALOG_CELLS[cell]
    _seed_catalog_apps()
    server_id = _seed(name, owner=bob, **columns)
    before = _store()

    if kind == "catalog":
        spy = _Calls(monkeypatch)
        resp = _delete(bob, server_id)
        error = _error(resp, 409, "connector_not_editable")
        assert error["message"] == _NOT_EDITABLE_CATALOG
        assert spy.calls == []
        _assert_nothing_written(before)
    else:
        resp = _delete(bob, server_id)
        assert resp.status_code == 200, resp.text
        assert not _definition_exists(server_id)


def test_a_database_failure_during_delete_is_a_500_that_changes_nothing(bob, caplog):
    server_id = _seed("fragile", owner=bob)
    before = _store()
    stop = _fail_statements(lambda s: "FROM mcp_oauth_grants" in s)
    try:
        with caplog.at_level(logging.INFO):
            resp = _delete(bob, server_id)
    finally:
        stop()

    error = _error(resp, 500, "internal_error")
    assert error["message"] == _INTERNAL
    assert "boom" not in resp.text
    _assert_nothing_written(before)
    assert any("action=delete status=500" in r.getMessage() for r in caplog.records)
    assert not [r for r in caplog.records if "mcp server write:" in r.getMessage()]


# ===== the delete runs outside the request's event loop =====


def test_delete_runs_on_a_different_loop_than_the_request(bob, monkeypatch):
    server_id = _seed("elsewhere", owner=bob)
    seen: dict[str, int] = {}

    async def recording_delete(*_args, **_kwargs):
        seen["delete_loop"] = id(asyncio.get_running_loop())
        seen["delete_thread"] = threading.get_ident()

    monkeypatch.setattr(mcp_server_management, "delete_mcp_server", recording_delete)

    # Same signature as the real dependency, so FastAPI reads the bearer token
    # from the request as before.
    async def recording_personal_key(
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ):
        seen["request_loop"] = id(asyncio.get_running_loop())
        seen["request_thread"] = threading.get_ident()
        return await get_user_from_personal_key(credentials)

    app_for_tests.dependency_overrides[get_user_from_personal_key] = (
        recording_personal_key
    )
    try:
        resp = _delete(bob, server_id)
    finally:
        del app_for_tests.dependency_overrides[get_user_from_personal_key]

    assert resp.status_code == 200, resp.text
    assert set(seen) == {
        "request_loop",
        "request_thread",
        "delete_loop",
        "delete_thread",
    }
    assert seen["request_loop"] != seen["delete_loop"]
    assert seen["request_thread"] != seen["delete_thread"]


# ===== switching on and off =====


@pytest.mark.parametrize("enabled", [True, False])
def test_enable_and_disable_switch_only_the_holders_link(bob, carol, enabled):
    server_id = _seed("switch", owner=bob, is_active=not enabled)
    _link(carol, server_id, is_owner=False, is_active=not enabled)
    definition_before = _store()[0]

    resp = _set_enabled(bob, server_id, enabled)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["id"], body["enabled"], body["access"]) == (server_id, enabled, "own")
    link, carols = _link_of(bob, server_id), _link_of(carol, server_id)
    assert link is not None and carols is not None
    assert (link.is_active, carols.is_active) == (enabled, not enabled)
    assert _store()[0] == definition_before
    assert _get_one(bob, server_id).json() == body
    assert [row["enabled"] for row in _list(bob)] == [enabled]


def test_a_non_owner_switches_only_their_own_link(bob, carol):
    server_id = _seed("owned-by-bob", owner=bob)
    _link(carol, server_id, is_owner=False)
    definition_before = _store()[0]

    resp = _set_enabled(carol, server_id, False)

    assert resp.status_code == 200, resp.text
    assert resp.json()["enabled"] is False
    carols, bobs = _link_of(carol, server_id), _link_of(bob, server_id)
    assert carols is not None and bobs is not None
    assert (carols.is_active, bobs.is_active) == (False, True)
    assert _store()[0] == definition_before


@pytest.mark.parametrize("enabled", [True, False])
def test_a_switch_sends_only_is_active_to_the_web_function(bob, monkeypatch, enabled):
    server_id = _seed("switch-fields", owner=bob, is_active=not enabled)
    seen: list[set[str]] = []
    real_update = mcp_server_management.update_mcp_server

    def spy(server_id_arg, server_data, **kwargs):
        seen.append(set(server_data.model_fields_set))
        return real_update(server_id_arg, server_data, **kwargs)

    monkeypatch.setattr(mcp_server_management, "update_mcp_server", spy)

    resp = _set_enabled(bob, server_id, enabled)

    assert resp.status_code == 200, resp.text
    assert seen == [{"is_active"}]


def test_a_connector_deleted_during_a_switch_is_a_404(bob, monkeypatch, caplog):
    server_id = _seed("vanishing", owner=bob)
    monkeypatch.setattr(
        mcp_server_management, "resolve_holder_connector", _lose_the_link()
    )

    with caplog.at_level(logging.INFO):
        resp = _set_enabled(bob, server_id, False)

    error = _error(resp, 404, "connector_not_found")
    assert error["message"] == _NOT_FOUND
    assert not [
        r for r in caplog.records if r.getMessage().startswith("personal key mcp")
    ]


def test_setting_the_same_value_twice_gives_the_same_answer(bob):
    server_id = _seed("twice", owner=bob, is_active=False)

    first = _set_enabled(bob, server_id, True)
    second = _set_enabled(bob, server_id, True)

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["enabled"] is True
    off = _set_enabled(bob, server_id, False)
    again = _set_enabled(bob, server_id, False)
    assert off.json() == again.json()
    assert off.json()["enabled"] is False


@pytest.mark.parametrize("transport", ["stdio", "websocket"])
def test_launcher_connectors_can_be_switched_and_deleted(bob, transport):
    server_id = _seed(
        "launcher",
        owner=bob,
        transport=transport,
        url=None if transport == "stdio" else "wss://mcp.example.test/ws",
        command="SECRET-command",
    )

    off = _set_enabled(bob, server_id, False)
    assert off.status_code == 200, off.text
    assert off.json()["enabled"] is False
    assert "SECRET-command" not in off.text
    assert _delete(bob, server_id).status_code == 200
    assert not _definition_exists(server_id)


@pytest.mark.parametrize("cell", list(_CATALOG_CELLS))
def test_catalog_connectors_can_be_switched(bob, cell):
    name, columns, kind = _CATALOG_CELLS[cell]
    _seed_catalog_apps()
    server_id = _seed(name, owner=bob, **columns)

    resp = _set_enabled(bob, server_id, False)

    assert resp.status_code == 200, resp.text
    assert (resp.json()["kind"], resp.json()["enabled"]) == (kind, False)
    link = _link_of(bob, server_id)
    assert link is not None and link.is_active is False


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"enabled": None},
        {"enabled": "yes"},
        {"enabled": 1},
        {"enabled": True, "x": 1},
    ],
    ids=["missing", "null", "string", "integer", "extra_key"],
)
def test_enabled_body_must_be_exactly_a_boolean(bob, body):
    server_id = _seed("strict", owner=bob)
    before = _store()

    resp = client.put(
        f"/v1/mcp-servers/{server_id}/enabled", headers=bob.headers, json=body
    )

    _error(resp, 422, "invalid_input")
    _assert_nothing_written(before)


def test_a_database_failure_while_switching_is_a_500_that_changes_nothing(bob, caplog):
    server_id = _seed("fragile", owner=bob)
    before = _store()
    stop = _fail_statements(lambda s: s.startswith("UPDATE user_mcpservers"))
    try:
        with caplog.at_level(logging.INFO):
            resp = _set_enabled(bob, server_id, False)
    finally:
        stop()

    error = _error(resp, 500, "internal_error")
    assert error["message"] == _INTERNAL
    assert "boom" not in resp.text
    _assert_nothing_written(before)
    assert any("action=disable status=500" in r.getMessage() for r in caplog.records)


# ===== secrets =====


def test_a_switch_response_carries_no_secret(bob, secret_connector):  # noqa: F811
    resp = _set_enabled(bob, secret_connector, False)

    assert resp.status_code == 200, resp.text
    for secret in _SECRETS:
        assert secret not in resp.text
    assert resp.json() == _get_one(bob, secret_connector).json()


_SECRET_DETAIL = "SECRET-web-detail"
_REFUSAL_ROWS = ["forbidden", "not_found", "not_editable"]
_REFUSALS = {
    "forbidden": (HTTPException(403, _SECRET_DETAIL), 403, "connector_forbidden"),
    "not_found": (HTTPException(404, _SECRET_DETAIL), 404, "connector_not_found"),
    "not_editable": (
        HTTPException(409, _SECRET_DETAIL),
        409,
        "connector_not_editable",
    ),
}


@pytest.mark.parametrize("row", _REFUSAL_ROWS)
@pytest.mark.parametrize("action", _WRITE_ACTIONS)
def test_web_refusals_become_sdk_errors_without_the_web_text(
    bob, monkeypatch, action, row
):
    exc, status, code = _REFUSALS[row]
    server_id = _seed("refused", owner=bob)

    async def delete(*_a, **_k):
        raise exc

    def update(*_a, **_k):
        raise exc

    monkeypatch.setattr(mcp_server_management, "delete_mcp_server", delete)
    monkeypatch.setattr(mcp_server_management, "update_mcp_server", update)
    before = _store()

    resp = _act(action, bob, server_id)

    error = _error(resp, status, code)
    assert _SECRET_DETAIL not in resp.text
    assert "details" not in error
    assert "detail" not in resp.json()
    _assert_nothing_written(before)


# ===== the write log =====


@pytest.mark.parametrize("action", _WRITE_ACTIONS)
def test_each_successful_write_logs_one_line(bob, caplog, action):
    assert len(_WRITE_ACTIONS) == 3
    server_id = _seed("log-subject", owner=bob, is_active=action != "enable")

    with caplog.at_level(logging.INFO):
        resp = _act(action, bob, server_id)

    assert resp.status_code == 200, resp.text
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("personal key mcp server write:")
    ]
    assert lines == [
        f"personal key mcp server write: action={action} "
        f"key_prefix={_key_prefix(bob)} user_id={bob.user_id} "
        f"server_id={server_id}"
    ]
    assert "log-subject" not in lines[0]
