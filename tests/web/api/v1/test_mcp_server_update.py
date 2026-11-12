"""Integration tests for changing connectors through PATCH /v1/mcp-servers/{id}."""

import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import sqlalchemy as sa
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Query, object_session

from xagent.core.tools.core.mcp.model import MASKED_SECRET_VALUE, SENSITIVE_AUTH_FIELDS
from xagent.web.api import mcp as mcp_api
from xagent.web.api.mcp import _delete_team_path_mcp_oauth_state
from xagent.web.api.v1.errors import _DEFAULT_MESSAGES, V1ErrorCode
from xagent.web.models.database import get_engine
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import (
    MCPOAuthClient,
    MCPOAuthFlowState,
    MCPOAuthGrant,
)
from xagent.web.schemas.v1 import V1McpServerUpdateRequest
from xagent.web.services import connector_team_scope, mcp_server_management
from xagent.web.services.connector_name_policy import catalog_app_name_detail

from ..conftest import _direct_db_session, client
from .test_mcp_servers import (
    _ACCESS_CELLS,
    _CATALOG_CELLS,
    _INPUT_RULE_CELLS,
    _NAME_CONFLICT,
    _NAME_CONFLICT_CELLS,
    _REFUSED_TRANSPORTS,
    CONNECTOR_URL,
    Holder,
    _admin_holder,
    _error,
    _holder,
    _key_prefix,
    _link,
    _reserved_by_catalog,
    _seed,
    _seed_catalog_apps,
    _share_with_team,
)

pytestmark = pytest.mark.usefixtures("_test_db")

ISSUER = "https://auth.example.test"
METADATA_URL = "https://mcp.example.test/.well-known/oauth-protected-resource"
MOVED_URL = "https://moved.example.test/mcp"
OAUTH_AUTH = {
    "type": "mcp_oauth",
    "client_id": "web-client",
    "scope": "read",
    "issuer": ISSUER,
    "resource_metadata_url": METADATA_URL,
}
REFUSED_REBIND = (
    "Other accounts have authorized this connector with their own sign-in; "
    "change its URL, issuer or resource metadata URL in the web app."
)
_FORBIDDEN = _DEFAULT_MESSAGES[V1ErrorCode.CONNECTOR_FORBIDDEN]
_NOT_FOUND = _DEFAULT_MESSAGES[V1ErrorCode.CONNECTOR_NOT_FOUND]
_INVALID_KEY = _DEFAULT_MESSAGES[V1ErrorCode.INVALID_API_KEY]


@pytest.fixture(autouse=True)
def _no_team_hooks():
    connector_team_scope.set_connector_team_hooks()
    yield
    connector_team_scope.set_connector_team_hooks()


# ===== helpers =====


def _patch(holder: Holder, server_id: int, body: dict[str, Any]):
    return client.patch(
        f"/v1/mcp-servers/{server_id}", headers=holder.headers, json=body
    )


_TABLES = (MCPServer, UserMCPServer, MCPOAuthGrant, MCPOAuthFlowState)


def _snapshot() -> dict[str, list]:
    """Every row of the tables an update can touch, for before/after checks."""
    db = _direct_db_session()
    try:
        return {
            model.__tablename__: sorted(
                (
                    tuple(getattr(row, col.key) for col in sa.inspect(model).columns)
                    for row in db.query(model)
                ),
                key=repr,
            )
            for model in _TABLES
        }
    finally:
        db.close()


def _fail_statements(prefix: str) -> Callable[[], None]:
    """Make every SQL statement starting with ``prefix`` fail in the real database."""
    engine = get_engine()

    def fail(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().startswith(prefix):
            raise OperationalError(statement, parameters, Exception("boom"))

    sa.event.listen(engine, "before_cursor_execute", fail)
    return lambda: sa.event.remove(engine, "before_cursor_execute", fail)


def _stored(server_id: int) -> MCPServer:
    db = _direct_db_session()
    try:
        server = db.query(MCPServer).filter_by(id=server_id).one()
        db.expunge(server)
        return server
    finally:
        db.close()


def _client_row(db, server_id: int, client_id: str = "web-client") -> MCPOAuthClient:
    existing = (
        db.query(MCPOAuthClient)
        .filter_by(mcp_server_id=server_id, client_id=client_id)
        .first()
    )
    if existing is not None:
        return existing
    row = MCPOAuthClient(
        mcp_server_id=server_id,
        issuer=ISSUER,
        authorization_endpoint=f"{ISSUER}/authorize",
        token_endpoint=f"{ISSUER}/token",
        client_id=client_id,
        redirect_uri="https://xagent.example.test/callback",
    )
    db.add(row)
    db.flush()
    return row


def _grant(
    holder: Holder,
    server_id: int,
    *,
    owner_key: str | None = None,
    status: str = "active",
    expires_at: datetime | None = None,
    client_id: str = "web-client",
) -> int:
    """One OAuth grant of ``holder`` on a connector."""
    db = _direct_db_session()
    try:
        grant = MCPOAuthGrant(
            mcp_server_id=server_id,
            user_id=holder.user_id,
            mcp_oauth_client_id=_client_row(db, server_id, client_id).id,
            lookup_hash=uuid.uuid4().hex,
            resource_owner_key=owner_key or f"xagent:user:{holder.user_id}",
            issuer=ISSUER,
            resource=CONNECTOR_URL,
            scope=uuid.uuid4().hex,
            access_token="encrypted",
            status=status,
            expires_at=expires_at,
        )
        db.add(grant)
        db.commit()
        return int(grant.id)
    finally:
        db.close()


def _flow(holder: Holder, server_id: int) -> None:
    """An unexpired team-path sign-in in progress for ``holder``."""
    db = _direct_db_session()
    try:
        db.add(
            MCPOAuthFlowState(
                state=uuid.uuid4().hex,
                mcp_server_id=server_id,
                user_id=holder.user_id,
                mcp_oauth_client_id=_client_row(db, server_id).id,
                resource_owner_key=f"xagent:user:{holder.user_id}",
                access_path="team_member",
                issuer=ISSUER,
                resource=CONNECTOR_URL,
                code_verifier="verifier",
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
        )
        db.commit()
    finally:
        db.close()


class _Calls:
    """Counts calls to the web update and to the rebind checks, which run the
    real code."""

    def __init__(self, monkeypatch) -> None:
        self.counts = {"update": 0, "editor": 0, "grants": 0}
        for key, owner, name in (
            ("update", mcp_server_management, "update_mcp_server"),
            ("editor", mcp_server_management, "_require_rebind_editor"),
            ("grants", mcp_server_management, "_has_active_grant_surviving_rebind"),
        ):
            monkeypatch.setattr(owner, name, self._counting(key, getattr(owner, name)))

    def _counting(self, key: str, real):
        def counted(*args, **kwargs):
            self.counts[key] += 1
            return real(*args, **kwargs)

        return counted


def _inject_after_lock(monkeypatch, *statements: tuple[str, dict]) -> list[str]:
    """Run SQL in the update's own transaction once it holds the definition row.

    The first thing the update does with the locked row is read its binding,
    so that read is the hook; the SQL then runs in the update's session, and
    is rolled back with it when the update refuses.
    """
    injected: list[str] = []
    real = mcp_server_management._member_grant_binding

    def binding_then_inject(server):
        if isinstance(server, MCPServer) and not injected:
            session = object_session(server)
            for sql, params in statements:
                session.execute(text(sql), params)
            injected.append("done")
        return real(server)

    monkeypatch.setattr(
        mcp_server_management, "_member_grant_binding", binding_then_inject
    )
    return injected


@pytest.fixture
def world() -> SimpleNamespace:
    """bob owns an OAuth connector; erin links it without owning it."""
    bob, carol, dave, erin = (_holder(n) for n in ("bob", "carol", "dave", "erin"))
    server_id = _seed("oauth-connector", owner=bob, auth=OAUTH_AUTH)
    return SimpleNamespace(
        bob=bob, carol=carol, dave=dave, erin=erin, server_id=server_id
    )


def _link_member(who: str) -> Callable[[SimpleNamespace], None]:
    """Seed: ``who`` links the connector without owning it."""

    def seed(w: SimpleNamespace) -> None:
        _link(getattr(w, who), w.server_id, is_owner=False, is_active=True)

    return seed


def _carol_has_a_grant(w: SimpleNamespace) -> None:
    _link_member("carol")(w)
    _grant(w.carol, w.server_id)


def _sql(w: SimpleNamespace, key: str) -> list[tuple[str, dict]]:
    """Statements run in the update's transaction once it holds the lock."""
    params = {"bob": w.bob.user_id, "s": w.server_id}
    return {
        "demote_owner": [
            (
                "UPDATE user_mcpservers SET is_owner = 0 "
                "WHERE user_id = :bob AND mcpserver_id = :s",
                params,
            )
        ],
        "delete_link": [
            (
                "DELETE FROM user_mcpservers "
                "WHERE user_id = :bob AND mcpserver_id = :s",
                params,
            )
        ],
        # The foreign keys are checked at commit, which the refusal never
        # reaches, so the account can be made to vanish without losing its link.
        "drop_account": [
            ("PRAGMA defer_foreign_keys = ON", {}),
            ("UPDATE users SET id = id + 100000 WHERE id = :bob", params),
        ],
    }[key]


def _grants_of(who: str, **grant: Any) -> Callable[[SimpleNamespace], None]:
    """Seed: ``who`` holds one grant on the connector, and links it unless ``who``
    is bob (who owns it) or dave (who reaches it only through the team)."""

    def seed(w: SimpleNamespace) -> None:
        holder = getattr(w, who)
        if who not in ("bob", "dave"):
            _link(holder, w.server_id, is_owner=False, is_active=grant.pop("on", True))
        _grant(holder, w.server_id, **grant)

    return seed


def _auth_with(**changes: Any) -> Callable[[SimpleNamespace], dict]:
    return lambda w: {"auth": {**OAUTH_AUTH, **changes}}


def _moved(w: SimpleNamespace) -> None:
    assert _stored(w.server_id).url == MOVED_URL


def _team_path_state_deleted(w: SimpleNamespace) -> None:
    db = _direct_db_session()
    try:
        owners = {
            g.user_id
            for g in db.query(MCPOAuthGrant).filter_by(mcp_server_id=w.server_id)
        }
        assert owners == {w.bob.user_id}
        assert (
            db.query(MCPOAuthFlowState).filter_by(mcp_server_id=w.server_id).count()
            == 0
        )
    finally:
        db.close()


def _bearer_with_issuer(w: SimpleNamespace) -> None:
    db = _direct_db_session()
    try:
        db.query(MCPServer).filter_by(id=w.server_id).update(
            {"auth": {**OAUTH_AUTH, "type": "bearer"}}
        )
        db.commit()
    finally:
        db.close()
    _grants_of("carol")(w)


def _grant_on_another_connector(w: SimpleNamespace) -> None:
    _grant(w.carol, _seed("another", owner=w.carol, auth=OAUTH_AUTH))


def _dave_holds_state_bob_holds_a_grant(w: SimpleNamespace) -> None:
    _grants_of("dave")(w)
    _flow(w.dave, w.server_id)
    _grant(w.bob, w.server_id)


@dataclass(frozen=True)
class _Cell:
    seed: Callable[[SimpleNamespace], None] = _carol_has_a_grant
    holder: str = "bob"
    body: Callable[[SimpleNamespace], dict] = lambda w: {"url": MOVED_URL}
    status: int = 409
    code: str = "connector_not_editable"
    message: str = REFUSED_REBIND
    # Calls to the web update, the editor check and the grant check.
    calls: tuple[int, int, int] = (0, 1, 1)
    inject: str | None = None
    # A statement prefix that fails in the real database.
    fail_sql: str | None = None
    after: Callable[[SimpleNamespace], None] = field(default=lambda w: None)


# The reason every refusal of a given code carries in this table.
_CELL_REASONS = {
    "connector_not_editable": "authorized_by_others",
    "connector_forbidden": "not_permitted",
}

_CELLS: dict[str, _Cell] = {
    "another_account_with_a_link": _Cell(),
    "holders_machine_account_grant": _Cell(
        seed=_grants_of("bob", owner_key="acme:actor:1")
    ),
    "grant_of_another_client": _Cell(
        seed=_grants_of("carol", client_id="other-client")
    ),
    "grant_expired_but_active": _Cell(
        seed=_grants_of(
            "carol", expires_at=datetime.now(timezone.utc) - timedelta(days=1)
        )
    ),
    "same_url_resent": _Cell(
        body=lambda w: {"url": CONNECTOR_URL}, status=200, calls=(1, 0, 0)
    ),
    "description_only": _Cell(
        body=lambda w: {"description": "new text"}, status=200, calls=(1, 0, 0)
    ),
    "stored_auth_is_not_oauth": _Cell(seed=_bearer_with_issuer),
    "grant_query_fails": _Cell(
        status=500,
        code="internal_error",
        message=_DEFAULT_MESSAGES[V1ErrorCode.INTERNAL_ERROR],
        fail_sql="SELECT mcp_oauth_grants.id",
    ),
    "write_fails_after_flows_are_deleted": _Cell(
        seed=_dave_holds_state_bob_holds_a_grant,
        status=500,
        code="internal_error",
        message=_DEFAULT_MESSAGES[V1ErrorCode.INTERNAL_ERROR],
        calls=(1, 1, 1),
        # The web update has deleted dave's flow state by the time this fails.
        fail_sql="DELETE FROM mcp_oauth_grants",
    ),
    "grant_on_another_connector": _Cell(
        seed=_grant_on_another_connector, status=200, calls=(1, 1, 1), after=_moved
    ),
    "team_path_state_is_deleted": _Cell(
        seed=_dave_holds_state_bob_holds_a_grant,
        status=200,
        calls=(1, 1, 1),
        after=_team_path_state_deleted,
    ),
    "issuer_changed": _Cell(body=_auth_with(issuer="https://other.example.test")),
    "metadata_url_changed": _Cell(
        body=_auth_with(resource_metadata_url="https://other.example.test/meta")
    ),
    "other_auth_key_changed": _Cell(
        body=_auth_with(client_id="renamed"), status=200, calls=(1, 0, 0)
    ),
    "trailing_slash_added": _Cell(body=lambda w: {"url": CONNECTOR_URL + "/"}),
    "link_switched_off": _Cell(seed=_grants_of("carol", on=False)),
    "auth_cleared": _Cell(body=lambda w: {"auth": {}}),
    "non_editor_with_kept_grants": _Cell(
        seed=lambda w: (_link_member("erin")(w), _carol_has_a_grant(w)),
        holder="erin",
        status=403,
        code="connector_forbidden",
        message=_FORBIDDEN,
        calls=(0, 0, 0),
    ),
    "owner_demoted_while_waiting_for_the_lock": _Cell(
        status=403,
        code="connector_forbidden",
        message=_FORBIDDEN,
        calls=(0, 1, 0),
        inject="demote_owner",
    ),
    "link_deleted_while_waiting_for_the_lock": _Cell(
        status=404,
        code="connector_not_found",
        message=_NOT_FOUND,
        calls=(0, 1, 0),
        inject="delete_link",
    ),
    "account_gone_while_waiting_for_the_lock": _Cell(
        status=401,
        code="invalid_api_key",
        message=_INVALID_KEY,
        calls=(0, 1, 0),
        inject="drop_account",
    ),
}


@pytest.mark.parametrize("cell", list(_CELLS))
def test_patch_rebind_refused_while_kept_grants_exist(world, monkeypatch, cell):
    spec = _CELLS[cell]
    spec.seed(world)
    before = _snapshot()
    calls = _Calls(monkeypatch)
    injected = (
        _inject_after_lock(monkeypatch, *_sql(world, spec.inject))
        if spec.inject
        else None
    )
    stop_failing = _fail_statements(spec.fail_sql) if spec.fail_sql else None

    try:
        resp = _patch(getattr(world, spec.holder), world.server_id, spec.body(world))
    finally:
        if stop_failing is not None:
            stop_failing()

    assert len(_CELLS) == 21
    assert tuple(calls.counts.values()) == spec.calls
    if spec.status == 200:
        assert resp.status_code == 200, resp.text
    else:
        # A refusal answers exactly this body and writes nothing.
        expected: dict[str, Any] = {"code": spec.code, "message": spec.message}
        if spec.code in _CELL_REASONS:
            expected["details"] = {"reason": _CELL_REASONS[spec.code]}
        assert resp.json() == {"error": expected}
        assert resp.status_code == spec.status
        assert _snapshot() == before
    if injected is not None:
        assert injected == ["done"]
    spec.after(world)


# ===== the grants the SDK counts are the ones the web update keeps =====


def test_kept_grant_query_complements_web_team_path_deletion(world):
    w = world
    _link(w.carol, w.server_id, is_owner=False, is_active=True)
    p = {
        "team_path": _grant(w.dave, w.server_id),
        "has_a_link": _grant(w.carol, w.server_id),
        "other_identity": _grant(w.dave, w.server_id, owner_key="acme:actor:7"),
        "holders_own": _grant(w.bob, w.server_id),
        "revoked": _grant(w.carol, w.server_id, status="revoked"),
    }
    db = _direct_db_session()
    try:
        everything = db.query(MCPOAuthGrant).filter_by(mcp_server_id=w.server_id).all()
        in_scope = {
            g.id
            for g in everything
            if g.status == "active"
            and not (
                g.user_id == w.bob.user_id
                and g.resource_owner_key == f"xagent:user:{w.bob.user_id}"
            )
        }
        counted = {
            row_id
            for (row_id,) in mcp_server_management._grants_surviving_rebind(
                db, server_id=w.server_id, user_id=w.bob.user_id
            ).all()
        }
        # Rolling the savepoint back, not committing it, undoes the deletion.
        savepoint = db.begin_nested()
        _delete_team_path_mcp_oauth_state(db, server_id=w.server_id)
        remaining = {
            row_id
            for (row_id,) in db.query(MCPOAuthGrant.id).filter_by(
                mcp_server_id=w.server_id
            )
        }
        savepoint.rollback()
        after = {
            g.id for g in db.query(MCPOAuthGrant).filter_by(mcp_server_id=w.server_id)
        }
    finally:
        db.close()
    deleted = set(p.values()) - remaining

    assert in_scope == {p["team_path"], p["has_a_link"], p["other_identity"]}
    assert in_scope <= deleted | counted
    assert not deleted & counted
    assert p["team_path"] in deleted
    assert {p["has_a_link"], p["other_identity"]} <= counted
    assert not {p["holders_own"], p["revoked"]} & counted
    assert after == set(p.values())


# ===== when the SDK takes the definition row lock =====

_LOCK_CELLS = {
    "headers_only": (
        {"headers": {"X-Key": "stored-value"}},
        {"headers": {"X-Key": MASKED_SECRET_VALUE}},
        ["web"],
    ),
    "auth_only": (
        {"auth": {**OAUTH_AUTH, "client_secret": "stored-secret"}},
        {"auth": {**OAUTH_AUTH, "client_secret": MASKED_SECRET_VALUE}},
        ["sdk", "web"],
    ),
    "url_changed": ({}, {"url": MOVED_URL}, ["sdk", "grants", "web"]),
    "name_only": ({}, {"name": "renamed"}, ["web"]),
}


@pytest.mark.parametrize("cell", list(_LOCK_CELLS))
def test_patch_takes_the_definition_lock_only_for_url_or_auth(world, monkeypatch, cell):
    assert len(_LOCK_CELLS) == 4
    columns, body, order = _LOCK_CELLS[cell]
    server_id = _seed("lock-subject", owner=world.bob, **columns)
    events: list[str] = []
    merges: list[dict] = []
    real_lock = Query.with_for_update
    real_merge = mcp_api._merge_masked_env
    real_check = mcp_server_management._has_active_grant_surviving_rebind

    def lock(self, *args, **kwargs):
        if self.column_descriptions[0]["entity"] is MCPServer:
            assert kwargs == {"key_share": True}
            module = sys._getframe(1).f_globals["__name__"]
            events.append("sdk" if module.endswith("mcp_server_management") else "web")
        return real_lock(self, *args, **kwargs)

    def merge(new, old):
        merges.append(old)
        return real_merge(new, old)

    def check(*args, **kwargs):
        events.append("grants")
        return real_check(*args, **kwargs)

    monkeypatch.setattr(Query, "with_for_update", lock)
    monkeypatch.setattr(mcp_api, "_merge_masked_env", merge)
    monkeypatch.setattr(
        mcp_server_management, "_has_active_grant_surviving_rebind", check
    )

    resp = _patch(world.bob, server_id, body)

    assert resp.status_code == 200, resp.text
    assert events == order
    if cell == "headers_only":
        assert merges == [{"X-Key": "stored-value"}]
        assert _stored(server_id).headers == {"X-Key": "stored-value"}


# ===== the web code is not asked to judge team sharing =====


def test_patch_does_not_consult_team_sharing(world):
    seen: list[int] = []

    def hook(_db, _user_id):
        seen.append(1)
        raise RuntimeError("team backend exploded")

    connector_team_scope.set_connector_team_hooks(visibility=hook)

    resp = _patch(world.bob, world.server_id, {"url": MOVED_URL})

    assert resp.status_code == 200, resp.text
    assert seen == []
    stored = _stored(world.server_id)
    assert stored.url == MOVED_URL
    assert stored.auth["type"] == "mcp_oauth"


# ===== which connectors the holder may change =====

_TRANSPORT_REFUSED = (
    "This connector uses the 'stdio' transport; change it in the web app."
)
_CATALOG_REFUSED = (
    "This connector is a catalog app connection, or cannot be shown not to be "
    "one, so it can only be changed in the web app."
)


def _assert_refused(resp, status: int, code: str, message: str | None = None) -> None:
    error = _error(resp, status, code)
    if message is not None:
        assert error["message"] == message


@pytest.mark.parametrize("cell", list(_ACCESS_CELLS))
def test_access_by_id_decides_what_a_change_answers(world, monkeypatch, cell):
    assert len(_ACCESS_CELLS) == 5
    server_id = _seed("shared", owner=world.carol)
    hooks, _status, _code = _ACCESS_CELLS[cell]
    connector_team_scope.set_connector_team_hooks(**hooks(server_id))
    calls = _Calls(monkeypatch)

    resp = _patch(world.dave, server_id, {"description": "x"})

    expected = {
        "access_hook_says_linked": (403, "connector_forbidden"),
        "access_hook_fails": (503, "connector_runtime_unavailable"),
        "only_list_hook": (403, "connector_forbidden"),
    }.get(cell, (404, "connector_not_found"))
    _assert_refused(resp, *expected)
    assert "exploded" not in resp.text
    assert calls.counts["update"] == 0


@pytest.mark.parametrize(
    "body",
    [
        {"headers": {"X-Api-Key": "rotated"}},
        {"auth": {**OAUTH_AUTH, "client_secret": "rotated"}},
        {"headers": {}},
        {"description": "changed"},
        {"name": "oauth-connector"},
    ],
    ids=["headers", "secret_only_auth", "headers_cleared", "description", "same_name"],
)
def test_a_holder_who_may_not_edit_changes_nothing(world, monkeypatch, caplog, body):
    _link(world.erin, world.server_id, is_owner=False, is_active=True)
    calls = _Calls(monkeypatch)
    before = _snapshot()

    with caplog.at_level(logging.INFO):
        resp = _patch(world.erin, world.server_id, body)

    _assert_refused(resp, 403, "connector_forbidden", _FORBIDDEN)
    assert resp.json()["error"]["details"] == {"reason": "not_permitted"}
    assert calls.counts["update"] == 0
    assert _snapshot() == before
    assert not [r for r in caplog.records if "mcp server write:" in r.getMessage()]


def test_an_empty_body_from_a_holder_who_may_not_edit_answers_200(world):
    _link(world.erin, world.server_id, is_owner=False, is_active=True)

    resp = _patch(world.erin, world.server_id, {})

    assert resp.status_code == 200, resp.text
    assert resp.json()["can_edit"] is False


@pytest.mark.parametrize("transport", _REFUSED_TRANSPORTS)
def test_patch_refuses_to_move_a_connector_to_another_transport(world, transport):
    before = _snapshot()

    resp = _patch(world.bob, world.server_id, {"transport": transport})

    _assert_refused(resp, 400, "transport_not_supported")
    assert _snapshot() == before


@pytest.mark.parametrize("transport", ["stdio", "websocket"])
def test_patch_refuses_a_local_connector(world, transport):
    server_id = _seed(
        "local", owner=world.bob, transport=transport, url=None, command="launch"
    )
    before = _snapshot()

    resp = _patch(world.bob, server_id, {"description": "x"})

    _assert_refused(
        resp,
        400,
        "transport_not_supported",
        _TRANSPORT_REFUSED.replace("stdio", transport),
    )
    assert _snapshot() == before


@pytest.mark.parametrize("cell", list(_CATALOG_CELLS))
def test_patch_refuses_catalog_connectors(world, cell):
    assert len(_CATALOG_CELLS) == 10
    name, columns, kind = _CATALOG_CELLS[cell]
    _seed_catalog_apps()
    server_id = _seed(name, owner=world.bob, **columns)

    resp = _patch(world.bob, server_id, {"description": "x"})

    if kind == "catalog":
        _assert_refused(resp, 409, "connector_not_editable", _CATALOG_REFUSED)
        assert resp.json()["error"]["details"] == {"reason": "catalog"}
    else:
        assert resp.status_code == 200, resp.text


@pytest.mark.parametrize(
    "body",
    [{"transport": ""}, {"transport": "t" * 33}, {"colour": "red"}],
    ids=["empty_transport", "long_transport", "unknown_field"],
)
def test_patch_rejects_a_malformed_body(world, body):
    before = _snapshot()

    _error(_patch(world.bob, world.server_id, body), 422, "invalid_input")
    assert _snapshot() == before


# ===== null and empty bodies =====

_NULLABLE = {"runtime_input_schema", "runtime_bindings", "signed_uploads"}
_NOT_NULLABLE = [f for f in V1McpServerUpdateRequest.model_fields if f not in _NULLABLE]


@pytest.mark.parametrize("field", _NOT_NULLABLE)
def test_patch_null_is_refused_for_fields_that_cannot_be_cleared(world, field):
    assert len(V1McpServerUpdateRequest.model_fields) == 10
    assert len(_NOT_NULLABLE) == 7
    before = _snapshot()

    resp = _patch(world.bob, world.server_id, {field: None})

    _assert_refused(
        resp,
        400,
        "invalid_input",
        f"{field} cannot be null; omit it to keep the stored value.",
    )
    assert _snapshot() == before


def test_patch_null_and_empty_dicts_clear_and_an_empty_body_changes_nothing(
    world, monkeypatch
):
    schema = {"context": {"account_id": {"type": "string", "required": True}}}
    bindings = [
        {
            "source": {"input_type": "context", "key": "account_id"},
            "target": {"target_type": "mcp_meta", "key": "account_id"},
        }
    ]
    server_id = _seed(
        "declared",
        owner=world.bob,
        runtime_input_schema=schema,
        runtime_bindings=bindings,
        signed_uploads=[{"tool": "attach_document"}],
        headers={"X-Key": "stored-value"},
        auth={"type": "bearer", "bearer_token": "stored-token"},
    )
    calls = _Calls(monkeypatch)
    before = _snapshot()

    unchanged = _patch(world.bob, server_id, {})

    assert unchanged.status_code == 200, unchanged.text
    assert unchanged.json()["runtime_bindings"] == bindings
    assert calls.counts["update"] == 0
    assert _snapshot() == before
    cleared = _patch(
        world.bob,
        server_id,
        {
            "runtime_input_schema": None,
            "runtime_bindings": None,
            "signed_uploads": None,
            "headers": {},
            "auth": {},
        },
    )
    assert cleared.status_code == 200, cleared.text
    assert (cleared.json()["headers"], cleared.json()["auth"]) == ({}, {})
    stored = _stored(server_id)
    assert (
        stored.runtime_input_schema,
        stored.runtime_bindings,
        stored.signed_uploads,
        stored.headers,
        stored.auth,
    ) == (None, None, None, {}, {})


# ===== masked values =====

_NO_VALUE = "Masked secret '{}' has no stored value; provide a new value"


def test_patch_header_placeholders(world):
    kept = _seed("with-header", owner=world.bob, headers={"X-Key": "stored-value"})
    bare = _seed("without-header", owner=world.bob)
    mask = {"X-Key": MASKED_SECRET_VALUE}

    for server_id, headers, key in (
        (bare, mask, "X-Key"),
        (kept, {"X-Key-2": MASKED_SECRET_VALUE}, "X-Key-2"),
    ):
        before = _snapshot()
        _assert_refused(
            _patch(world.bob, server_id, {"headers": headers}),
            400,
            "invalid_input",
            "Invalid configuration: " + _NO_VALUE.format(key),
        )
        assert _snapshot() == before


@pytest.mark.parametrize("key", SENSITIVE_AUTH_FIELDS)
def test_patch_secret_auth_placeholders(world, key):
    kept = _seed("with-secret", owner=world.bob, auth={"type": "bearer", key: "stored"})
    bare = _seed("without-secret", owner=world.bob, auth={"type": "bearer"})
    body = {"auth": {"type": "bearer", key: MASKED_SECRET_VALUE}}

    resp = _patch(world.bob, kept, body)
    assert resp.status_code == 200, resp.text
    assert _stored(kept).auth[key] == "stored"
    before = _snapshot()

    _assert_refused(
        _patch(world.bob, bare, body), 400, "invalid_input", _NO_VALUE.format(key)
    )
    assert _snapshot() == before


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"name": "  "}, "Connector name cannot be blank."),
        (
            {"url": "ftp://mcp.example.test"},
            "url must be an absolute http(s) URL without user info.",
        ),
        (
            {"auth": {"app_id": "real-app"}},
            "auth cannot set app_id, provider or builtin_provenance; "
            "these mark catalog connectors.",
        ),
        (
            {"auth": {"type": "bearer", "scope": MASKED_SECRET_VALUE}},
            "Only secret auth fields can keep their stored value with '********'.",
        ),
    ],
    ids=[
        "blank_name",
        "url_scheme",
        "catalog_marker",
        "masked_scope",
    ],
)
def test_patch_input_rules(world, body, message):
    before = _snapshot()

    _assert_refused(
        _patch(world.bob, world.server_id, body), 400, "invalid_input", message
    )
    assert _snapshot() == before


_SECRET_TYPE_CELLS = {
    key: cell
    for key, cell in _INPUT_RULE_CELLS.items()
    if key.startswith("auth_secret_")
}


@pytest.mark.parametrize(
    ("body", "message"), _SECRET_TYPE_CELLS.values(), ids=list(_SECRET_TYPE_CELLS)
)
def test_patch_refuses_a_secret_auth_field_that_is_not_a_string(world, body, message):
    assert len(_SECRET_TYPE_CELLS) == 4
    before = _snapshot()

    _assert_refused(
        _patch(world.bob, world.server_id, body), 400, "invalid_input", message
    )
    assert _snapshot() == before


# ===== answers and translation =====


def test_patch_changes_what_was_sent_and_returns_no_secret(world):
    secrets = ["SECRET-new-header", "SECRET-new-bearer"]
    body = {
        "name": "renamed-connector",
        "transport": "sse",
        "description": "changed",
        "url": MOVED_URL,
        "headers": {"X-Api-Key": secrets[0]},
        "auth": {"type": "bearer", "bearer_token": secrets[1]},
    }

    first = _patch(world.bob, world.server_id, body)
    second = _patch(world.bob, world.server_id, body)

    assert first.status_code == 200, first.text
    assert all(secret not in first.text for secret in secrets)
    shown = first.json()
    assert (shown["name"], shown["transport"], shown["url"]) == (
        "renamed-connector",
        "sse",
        MOVED_URL,
    )
    assert shown["headers"] == {"X-Api-Key": MASKED_SECRET_VALUE}
    assert shown["auth"]["bearer_token"] == MASKED_SECRET_VALUE
    assert shown["access"] == "own" and shown["can_edit"] is True
    # Sending the same body again answers the same.
    assert second.status_code == 200, second.text
    assert {k: v for k, v in second.json().items() if k != "updated_at"} == {
        k: v for k, v in shown.items() if k != "updated_at"
    }


@pytest.mark.parametrize(
    "body",
    [
        {"url": MOVED_URL},
        {"url": CONNECTOR_URL + "/"},
        {"auth": {**OAUTH_AUTH, "issuer": "https://other.example.test"}},
        {"auth": {}},
        {"auth": {**OAUTH_AUTH, "client_secret": MASKED_SECRET_VALUE}},
    ],
    ids=["url", "trailing_slash", "issuer", "auth_cleared", "masked_secret"],
)
def test_the_predicted_binding_is_the_one_the_web_function_stores(
    world, monkeypatch, body
):
    seen: list[tuple] = []
    real = mcp_server_management._member_grant_binding

    def record(server):
        seen.append(real(server))
        return seen[-1]

    monkeypatch.setattr(mcp_server_management, "_member_grant_binding", record)
    server_id = _seed(
        "with-secret",
        owner=world.bob,
        auth={**OAUTH_AUTH, "client_secret": "stored-secret"},
    )

    resp = _patch(world.bob, server_id, body)

    assert resp.status_code == 200, resp.text
    # The binding before the change, then the predicted one.
    assert len(seen) == 2
    assert seen[1] == mcp_api._member_grant_binding(_stored(server_id))


def test_the_web_account_gone_answer_is_a_401(world, monkeypatch):
    """The real web function, not a stand-in, raises the account-gone text."""
    real = mcp_server_management.update_mcp_server

    def drop_the_account_then_update(server_id, data, *, current_user, db):
        # The foreign keys are checked at commit, which the refusal never
        # reaches, so the account vanishes without losing its link.
        db.execute(text("PRAGMA defer_foreign_keys = ON"))
        db.execute(
            text("UPDATE users SET id = id + 100000 WHERE id = :bob"),
            {"bob": world.bob.user_id},
        )
        return real(server_id, data, current_user=current_user, db=db)

    monkeypatch.setattr(
        mcp_server_management, "update_mcp_server", drop_the_account_then_update
    )
    before = _snapshot()

    resp = _patch(world.bob, world.server_id, {"url": MOVED_URL})

    _assert_refused(resp, 401, "invalid_api_key", _INVALID_KEY)
    assert _snapshot() == before


@pytest.mark.parametrize(
    ("body", "dumped"),
    [
        ({"name": "renamed"}, {"name": "renamed"}),
        (
            {
                "description": "d",
                "allow_delegated_authorization": False,
                "runtime_input_schema": None,
                "headers": {},
            },
            {
                "description": "d",
                "allow_delegated_authorization": False,
                "runtime_input_schema": None,
                "config": {"headers": {}},
            },
        ),
    ],
    ids=["name_only", "mixed_with_empty_headers"],
)
def test_the_web_update_request_carries_exactly_the_fields_sent(body, dumped):
    web = mcp_server_management._web_update_request(V1McpServerUpdateRequest(**body))

    assert web.model_dump(exclude_unset=True) == dumped
    assert web.model_fields_set == set(dumped)


@pytest.mark.parametrize(
    ("name", "seeder", "names_the_connector"),
    [*_NAME_CONFLICT_CELLS.values(), ("Real_App", _reserved_by_catalog, False)],
    ids=[*_NAME_CONFLICT_CELLS, "name_that_folds_to_a_catalog_app"],
)
def test_patch_name_conflicts_from_the_web_code(
    world, name, seeder, names_the_connector
):
    existing = seeder(world.bob, world.carol)
    before = _snapshot()

    resp = _patch(world.bob, world.server_id, {"name": name})

    error = _error(resp, 409, "connector_name_conflict")
    if names_the_connector:
        assert error["details"] == {"connector_id": existing}
    else:
        assert "details" not in error
    assert error["message"] == (
        catalog_app_name_detail(name)
        if seeder is _reserved_by_catalog
        else _NAME_CONFLICT
    )
    assert _snapshot() == before


def test_an_error_from_the_update_does_not_echo_the_request_secrets(world, monkeypatch):
    secret = "SECRET-probe-value"

    def refuse(*_args, **_kwargs):
        raise HTTPException(400, f"Invalid configuration: bad {secret}")

    monkeypatch.setattr(mcp_server_management, "update_mcp_server", refuse)

    resp = _patch(world.bob, world.server_id, {"headers": {"X-Key": secret}})

    _assert_refused(
        resp, 400, "invalid_input", _DEFAULT_MESSAGES[V1ErrorCode.INVALID_INPUT]
    )
    assert secret not in resp.text


def test_an_admin_key_changes_a_connector_it_links(world):
    admin = _admin_holder()
    server_id = _seed("admins", owner=admin, is_owner=False)

    resp = _patch(admin, server_id, {"url": MOVED_URL})

    assert resp.status_code == 200, resp.text
    assert _stored(server_id).url == MOVED_URL


def test_an_update_logs_one_line_and_an_empty_body_none(world, caplog):
    prefix = "personal key mcp server write:"
    with caplog.at_level(logging.INFO):
        _patch(world.bob, world.server_id, {})
        assert not [r for r in caplog.records if r.getMessage().startswith(prefix)]
        resp = _patch(world.bob, world.server_id, {"description": "SECRET-text"})

    assert resp.status_code == 200, resp.text
    lines = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)
    ]
    assert lines == [
        f"{prefix} action=update key_prefix={_key_prefix(world.bob)} "
        f"user_id={world.bob.user_id} server_id={world.server_id}"
    ]


@pytest.mark.parametrize("shared", [False, True], ids=["unshared", "team_shares_it"])
def test_a_link_removed_during_an_update_is_a_404(world, monkeypatch, caplog, shared):
    if shared:
        _share_with_team(world.server_id)
    real = mcp_server_management.update_mcp_server

    def update_then_remove_the_link(*args, **kwargs):
        answer = real(*args, **kwargs)
        db = _direct_db_session()
        try:
            db.query(UserMCPServer).filter_by(
                user_id=world.bob.user_id, mcpserver_id=world.server_id
            ).delete()
            db.commit()
        finally:
            db.close()
        return answer

    monkeypatch.setattr(
        mcp_server_management, "update_mcp_server", update_then_remove_the_link
    )
    with caplog.at_level(logging.INFO):
        resp = _patch(world.bob, world.server_id, {"description": "x"})

    _assert_refused(resp, 404, "connector_not_found", _NOT_FOUND)
    assert not [r for r in caplog.records if "mcp server write:" in r.getMessage()]
