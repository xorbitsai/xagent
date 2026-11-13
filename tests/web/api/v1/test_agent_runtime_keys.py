"""Integration tests for listing and revoking an agent's runtime keys over /v1."""

import logging
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from xagent.core.utils.api_key import ApiKeyKind, generate_api_key
from xagent.web.models.agent import AgentOrigin
from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.database import get_engine
from xagent.web.models.workforce import Workforce
from xagent.web.services import agent_management
from xagent.web.services.agent_team_scope import (
    AgentTeamScope,
    set_agent_team_scope_hook,
)
from xagent.web.services.api_keys import (
    AgentApiKeyService,
    AgentRuntimeKeyNotFoundError,
)

from ..conftest import _admin_headers, _direct_db_session, _register_second_user, client
from .test_agent_skills import _seed_agent, _set_visibility, _user_id
from .test_management import _bearer, _personal_key
from .test_skills import _error

pytestmark = pytest.mark.usefixtures("_test_db")

_ITEM_FIELDS = {
    "id",
    "label",
    "key_prefix",
    "masked_key",
    "status",
    "created_at",
    "last_used_at",
    "revoked_at",
}
_REVOCATION_FIELDS = {"id", "key_prefix", "revoked", "revoked_at"}
_MISSING_ID = 999_999


@pytest.fixture
def team_scopes():
    """Team scopes by user id, as an application that groups users installs."""
    scopes: dict[int, AgentTeamScope] = {}
    set_agent_team_scope_hook(lambda _db, user_id: scopes.get(int(user_id)))
    yield scopes
    set_agent_team_scope_hook(None)


# ===== helpers =====


class _Key:
    def __init__(self, key_id: int, full_key: str, key_prefix: str, key_hash: str):
        self.id = key_id
        self.full_key = full_key
        self.key_prefix = key_prefix
        self.key_hash = key_hash


def _seed_key(
    agent_id: int | None = None,
    *,
    workforce_id: int | None = None,
    label: str | None = None,
    paused: bool = False,
    revoked_at: datetime | None = None,
    created_at: datetime | None = None,
) -> _Key:
    db = _direct_db_session()
    try:
        full_key, key_prefix, key_hash = generate_api_key(db, kind=ApiKeyKind.AGENT)
        row = AgentApiKey(
            agent_id=agent_id,
            workforce_id=workforce_id,
            label=label,
            key_prefix=key_prefix,
            key_hash=key_hash,
            paused_at=datetime.now(timezone.utc) if paused else None,
            revoked_at=revoked_at,
        )
        if created_at is not None:
            row.created_at = created_at
        db.add(row)
        db.commit()
        return _Key(int(row.id), full_key, key_prefix, key_hash)
    finally:
        db.close()


def _seed_workforce_key(owner: str = "admin") -> _Key:
    manager_id = _seed_agent(
        owner, origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value
    )
    db = _direct_db_session()
    try:
        workforce = Workforce(
            owner_user_id=_user_id(owner),
            scope_id=str(_user_id(owner)),
            name="crew",
            manager_agent_id=manager_id,
        )
        db.add(workforce)
        db.commit()
        workforce_id = int(workforce.id)
    finally:
        db.close()
    return _seed_key(workforce_id=workforce_id)


def _key_table() -> list[tuple]:
    db = _direct_db_session()
    try:
        return [
            (
                row.id,
                row.agent_id,
                row.workforce_id,
                row.label,
                row.key_prefix,
                row.key_hash,
                row.paused_at,
                row.revoked_at,
                row.updated_at,
            )
            for row in db.query(AgentApiKey).order_by(AgentApiKey.id).all()
        ]
    finally:
        db.close()


def _stored(key_id: int) -> AgentApiKey:
    db = _direct_db_session()
    try:
        row = db.get(AgentApiKey, key_id)
        db.expunge(row)
        return row
    finally:
        db.close()


def _list(key: str | None, agent_id: int, **params):
    return client.get(
        f"/v1/agents/{agent_id}/api-keys",
        headers=_bearer(key) if key is not None else {},
        params=params,
    )


def _revoke(key: str | None, agent_id: int, key_id: int):
    return client.delete(
        f"/v1/agents/{agent_id}/api-keys/{key_id}",
        headers=_bearer(key) if key is not None else {},
    )


def _authenticates(runtime_key: str) -> bool:
    """Whether a runtime key is still accepted by a runtime-key route."""
    resp = client.get(f"/v1/chat/tasks/{_MISSING_ID}", headers=_bearer(runtime_key))
    assert resp.status_code in (401, 404), resp.text
    return resp.status_code == 404


def _unmanageable_agent(kind: str, team_scopes: dict) -> int:
    """An agent id the admin caller must not be able to manage."""
    if kind == "missing":
        return _MISSING_ID
    if kind == "workforce-manager":
        return _seed_agent(origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value)
    _register_second_user("bob")
    if kind == "another-users-agent":
        return _seed_agent("bob")
    team_scopes[_user_id()] = AgentTeamScope(team_id=1, is_team_admin=False)
    if kind == "another-teams-agent":
        return _seed_agent("bob", team_id=2)
    assert kind == "teammates-restricted-agent"
    return _seed_agent("bob", team_id=1, visibility="admins")


_UNMANAGEABLE = [
    "another-users-agent",
    "another-teams-agent",
    "teammates-restricted-agent",
    "missing",
    "workforce-manager",
]


# ===== listing =====


def test_lists_keys_of_own_agent():
    key = _personal_key()
    agent_id = _seed_agent()
    now = datetime.now(timezone.utc)
    older = _seed_key(agent_id, label="ci", created_at=now - timedelta(hours=1))
    newer = _seed_key(agent_id, created_at=now)

    resp = _list(key, agent_id)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [item["id"] for item in body] == [newer.id, older.id]
    assert body[1]["label"] == "ci"
    assert body[1]["key_prefix"] == older.key_prefix
    assert body[1]["masked_key"] == f"xag_{older.key_prefix}_••••••••"
    assert body[1]["status"] == "active"
    assert body[1]["last_used_at"] is None
    assert body[1]["revoked_at"] is None


def test_agent_without_keys_lists_empty():
    key = _personal_key()

    resp = _list(key, _seed_agent())

    assert resp.status_code == 200, resp.text
    assert resp.json() == []


def test_list_exposes_only_the_documented_fields():
    key = _personal_key()
    agent_id = _seed_agent()
    live = _seed_key(agent_id)
    gone = _seed_key(agent_id, revoked_at=datetime.now(timezone.utc))

    listed = _list(key, agent_id, include_revoked=True)
    revoked = _revoke(key, agent_id, live.id)

    assert listed.status_code == 200, listed.text
    assert [set(item) for item in listed.json()] == [_ITEM_FIELDS, _ITEM_FIELDS]
    assert revoked.status_code == 200, revoked.text
    assert set(revoked.json()) == _REVOCATION_FIELDS
    for text in (listed.text, revoked.text):
        for seeded in (live, gone):
            assert seeded.full_key not in text
            assert seeded.key_hash not in text


def test_list_hides_revoked_keys_unless_asked():
    key = _personal_key()
    agent_id = _seed_agent()
    live = _seed_key(agent_id)
    gone = _seed_key(agent_id, revoked_at=datetime.now(timezone.utc))

    default = _list(key, agent_id).json()
    everything = _list(key, agent_id, include_revoked=True).json()

    assert [item["id"] for item in default] == [live.id]
    assert {item["id"]: item["status"] for item in everything} == {
        live.id: "active",
        gone.id: "revoked",
    }
    assert next(i for i in everything if i["id"] == gone.id)["revoked_at"] is not None


@pytest.mark.parametrize("include_revoked", [False, True])
def test_list_reports_paused_keys(include_revoked):
    key = _personal_key()
    agent_id = _seed_agent()
    paused = _seed_key(agent_id, paused=True)

    body = _list(key, agent_id, include_revoked=include_revoked).json()

    assert [(item["id"], item["status"]) for item in body] == [(paused.id, "paused")]


def test_list_never_includes_other_owners_keys():
    key = _personal_key()
    agent_id = _seed_agent()
    own = _seed_key(agent_id)
    _seed_key(_seed_agent())
    _seed_workforce_key()
    _register_second_user("bob")
    _seed_key(_seed_agent("bob"))

    body = _list(key, agent_id, include_revoked=True).json()

    assert [item["id"] for item in body] == [own.id]


# ===== revoking =====


def test_revoke_disables_only_the_named_key():
    key = _personal_key()
    agent_id = _seed_agent()
    target = _seed_key(agent_id)
    sibling = _seed_key(agent_id)
    elsewhere = _seed_key(_seed_agent())
    assert _authenticates(target.full_key)

    resp = _revoke(key, agent_id, target.id)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == target.id
    assert body["key_prefix"] == target.key_prefix
    assert body["revoked"] is True
    assert body["revoked_at"] is not None
    assert _stored(target.id).revoked_at is not None
    assert not _authenticates(target.full_key)
    assert _authenticates(sibling.full_key)
    assert _authenticates(elsewhere.full_key)
    assert [item["id"] for item in _list(key, agent_id).json()] == [sibling.id]


def test_revoke_applies_to_a_paused_key():
    key = _personal_key()
    agent_id = _seed_agent()
    paused = _seed_key(agent_id, paused=True)
    paused_at = _stored(paused.id).paused_at

    resp = _revoke(key, agent_id, paused.id)

    assert resp.status_code == 200, resp.text
    assert resp.json()["revoked"] is True
    stored = _stored(paused.id)
    assert stored.revoked_at is not None
    assert stored.paused_at == paused_at


def test_revoke_is_idempotent(caplog):
    key = _personal_key()
    agent_id = _seed_agent()
    target = _seed_key(agent_id)
    first = _revoke(key, agent_id, target.id)
    before = _key_table()
    caplog.set_level(logging.INFO, logger=agent_management.__name__)
    caplog.clear()

    second = _revoke(key, agent_id, target.id)

    assert second.status_code == 200, second.text
    assert second.json() == {**first.json(), "revoked": False}
    assert _key_table() == before
    assert "runtime key revoked" not in caplog.text


def test_revoke_logs_the_calling_credential(caplog):
    headers = _admin_headers()
    created = client.post("/api/me/personal-keys", headers=headers).json()
    agent_id = _seed_agent()
    target = _seed_key(agent_id)
    caplog.set_level(logging.INFO, logger=agent_management.__name__)

    assert _revoke(created["full_key"], agent_id, target.id).status_code == 200

    lines = [r.getMessage() for r in caplog.records if "runtime key revoked" in r.msg]
    assert lines == [
        f"runtime key revoked: credential=personal_key:{created['key_prefix']} "
        f"user_id={_user_id()} agent_id={agent_id} key_id={target.id} "
        f"key_prefix={target.key_prefix}"
    ]
    assert target.full_key not in caplog.text
    assert created["full_key"] not in caplog.text


def test_revoke_reports_a_failed_commit_as_an_error(monkeypatch, caplog):
    key = _personal_key()
    agent_id = _seed_agent()
    target = _seed_key(agent_id)
    real_stage = AgentApiKeyService.revoke_key_for_agent
    real_commit = Session.commit
    staged: list[Session] = []

    def stage(self, *args, **kwargs):
        staged.append(self.db)
        return real_stage(self, *args, **kwargs)

    def commit(self):
        if self in staged:
            raise RuntimeError("commit failed")
        return real_commit(self)

    monkeypatch.setattr(AgentApiKeyService, "revoke_key_for_agent", stage)
    monkeypatch.setattr(Session, "commit", commit)
    caplog.set_level(logging.INFO, logger=agent_management.__name__)

    resp = _revoke(key, agent_id, target.id)

    assert staged, "the revocation was staged before the commit failed"
    assert resp.status_code == 500, resp.text
    assert "runtime key revoked" not in caplog.text
    assert _stored(target.id).revoked_at is None
    assert _authenticates(target.full_key)


@pytest.mark.parametrize("bound_to", ["another-agent", "workforce"])
def test_staged_revocation_never_writes_a_key_bound_elsewhere(bound_to):
    _admin_headers()
    agent_id = _seed_agent()
    _seed_key(agent_id)
    other = (
        _seed_key(_seed_agent())
        if bound_to == "another-agent"
        else _seed_workforce_key()
    )
    before = _key_table()
    db = _direct_db_session()
    try:
        with pytest.raises(AgentRuntimeKeyNotFoundError):
            AgentApiKeyService(db).revoke_key_for_agent(agent_id, other.id)
        # Commit instead of rolling back: nothing may have been staged.
        db.commit()
    finally:
        db.close()

    assert _key_table() == before


# ===== who may manage an agent's keys =====


@pytest.mark.parametrize("kind", _UNMANAGEABLE)
def test_unmanageable_agent_is_agent_not_found(kind, team_scopes):
    key = _personal_key()
    own_key = _seed_key(_seed_agent())
    agent_id = _unmanageable_agent(kind, team_scopes)
    its_key = own_key if kind == "missing" else _seed_key(agent_id)
    before = _key_table()

    listed = _list(key, agent_id, include_revoked=True)
    with_real_key = _revoke(key, agent_id, its_key.id)
    with_own_key = _revoke(key, agent_id, own_key.id)
    with_missing_key = _revoke(key, agent_id, _MISSING_ID)

    for resp in (listed, with_real_key, with_own_key, with_missing_key):
        _error(resp, 404, "agent_not_found")
    assert with_real_key.content == with_missing_key.content == with_own_key.content
    assert _key_table() == before
    assert _authenticates(own_key.full_key)


@pytest.mark.parametrize("kind", _UNMANAGEABLE)
def test_unmanageable_agent_never_touches_the_key_table(kind, team_scopes):
    key = _personal_key()
    own_agent = _seed_agent()
    own_key = _seed_key(own_agent)
    agent_id = _unmanageable_agent(kind, team_scopes)
    statements: list[str] = []

    def record(_conn, _cursor, statement, *_rest):
        statements.append(statement)

    def recorded(send) -> list[str]:
        statements.clear()
        send()
        return list(statements)

    engine = get_engine()
    event.listen(engine, "before_cursor_execute", record)
    try:
        refused = recorded(lambda: _list(key, agent_id)) + recorded(
            lambda: _revoke(key, agent_id, own_key.id)
        )
        allowed = recorded(lambda: _list(key, own_agent))
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert any("agents" in statement for statement in refused)
    assert not [s for s in refused if "agent_api_keys" in s]
    assert [s for s in allowed if "agent_api_keys" in s]


@pytest.mark.parametrize("kind", _UNMANAGEABLE)
def test_unmanageable_agent_row_is_never_locked(kind, team_scopes, monkeypatch):
    key = _personal_key()
    own_agent = _seed_agent()
    own_key = _seed_key(own_agent)
    agent_id = _unmanageable_agent(kind, team_scopes)
    real = agent_management.acquire_runtime_key_transition_fence
    locked: list[int] = []

    def record(db, locked_agent_id):
        locked.append(locked_agent_id)
        return real(db, locked_agent_id)

    monkeypatch.setattr(
        agent_management, "acquire_runtime_key_transition_fence", record
    )

    _error(_revoke(key, agent_id, own_key.id), 404, "agent_not_found")
    assert locked == []
    assert _revoke(key, own_agent, own_key.id).status_code == 200
    assert locked == [own_agent]


@pytest.mark.parametrize(
    "outside",
    ["missing", "own-other-agent", "another-users-agent", "workforce"],
)
def test_key_outside_the_agent_is_api_key_not_found(outside):
    key = _personal_key()
    agent_id = _seed_agent()
    _seed_key(agent_id)
    reference = _revoke(key, agent_id, _MISSING_ID)
    if outside == "missing":
        other = None
    elif outside == "own-other-agent":
        other = _seed_key(_seed_agent())
    elif outside == "another-users-agent":
        _register_second_user("bob")
        other = _seed_key(_seed_agent("bob"))
    else:
        other = _seed_workforce_key()
    before = _key_table()

    resp = _revoke(key, agent_id, other.id if other is not None else _MISSING_ID)

    _error(resp, 404, "api_key_not_found")
    assert resp.content == reference.content
    assert _key_table() == before
    if other is not None:
        assert _stored(other.id).revoked_at is None


@pytest.mark.parametrize(
    ("scope", "stored_as", "reachable"),
    [
        (
            AgentTeamScope(team_id=1, is_team_admin=False),
            {"team_id": 1, "visibility": "team"},
            True,
        ),
        (
            AgentTeamScope(team_id=1, is_team_admin=False),
            {"team_id": 1, "visibility": "admins"},
            False,
        ),
        (
            AgentTeamScope(team_id=1, is_team_admin=True),
            {"team_id": 1, "visibility": "admins"},
            True,
        ),
        (None, {"owner": "admin", "team_id": 1, "visibility": "team"}, True),
    ],
    ids=[
        "member-team-visible",
        "member-restricted",
        "team-admin-restricted",
        "no-membership-own-team-agent",
    ],
)
def test_team_access_follows_the_agent_ownership_rule(
    scope, stored_as, reachable, team_scopes
):
    key = _personal_key()
    _register_second_user("bob")
    if scope is not None:
        team_scopes[_user_id()] = scope
    agent_id = _seed_agent(stored_as.pop("owner", "bob"), **stored_as)
    target = _seed_key(agent_id)

    listed = _list(key, agent_id)
    revoked = _revoke(key, agent_id, target.id)

    if reachable:
        assert [item["id"] for item in listed.json()] == [target.id]
        assert revoked.status_code == 200, revoked.text
        assert _stored(target.id).revoked_at is not None
    else:
        _error(listed, 404, "agent_not_found")
        _error(revoked, 404, "agent_not_found")
        assert _stored(target.id).revoked_at is None


@pytest.mark.parametrize("lost", ["agent-restricted", "caller-demoted"])
def test_revoke_rechecks_ownership_under_the_agent_lock(lost, team_scopes, monkeypatch):
    key = _personal_key()
    _register_second_user("bob")
    if lost == "agent-restricted":
        team_scopes[_user_id()] = AgentTeamScope(team_id=1, is_team_admin=False)
        agent_id = _seed_agent("bob", team_id=1, visibility="team")
    else:
        team_scopes[_user_id()] = AgentTeamScope(team_id=1, is_team_admin=True)
        agent_id = _seed_agent("bob", team_id=1, visibility="admins")
    target = _seed_key(agent_id)
    before = _key_table()
    real = agent_management.acquire_runtime_key_transition_fence

    def lose_access_then_lock(db, locked_agent_id):
        if lost == "agent-restricted":
            _set_visibility(agent_id, "admins")
        else:
            team_scopes[_user_id()] = AgentTeamScope(team_id=1, is_team_admin=False)
        return real(db, locked_agent_id)

    monkeypatch.setattr(
        agent_management, "acquire_runtime_key_transition_fence", lose_access_then_lock
    )

    _error(_revoke(key, agent_id, target.id), 404, "agent_not_found")
    assert _key_table() == before
    assert _authenticates(target.full_key)


def test_revoke_reports_an_earlier_revocation_unchanged(monkeypatch):
    key = _personal_key()
    agent_id = _seed_agent()
    target = _seed_key(agent_id)
    earlier = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    real = agent_management.acquire_runtime_key_transition_fence

    def revoke_elsewhere_then_lock(db, locked_agent_id):
        other = _direct_db_session()
        try:
            other.get(AgentApiKey, target.id).revoked_at = earlier
            other.commit()
        finally:
            other.close()
        return real(db, locked_agent_id)

    monkeypatch.setattr(
        agent_management,
        "acquire_runtime_key_transition_fence",
        revoke_elsewhere_then_lock,
    )

    resp = _revoke(key, agent_id, target.id)

    assert resp.status_code == 200, resp.text
    assert resp.json()["revoked"] is False
    assert datetime.fromisoformat(resp.json()["revoked_at"]) == earlier
    assert _stored(target.id).revoked_at.replace(tzinfo=timezone.utc) == earlier


# ===== credentials and path bounds =====


@pytest.mark.parametrize(
    "credential", ["none", "malformed", "runtime-key", "revoked-personal-key"]
)
def test_requires_a_usable_personal_key(credential):
    headers = _admin_headers()
    agent_id = _seed_agent()
    target = _seed_key(agent_id)
    if credential == "none":
        presented = None
    elif credential == "malformed":
        presented = "xag_personal_nope"
    elif credential == "runtime-key":
        presented = target.full_key
    else:
        created = client.post("/api/me/personal-keys", headers=headers).json()
        deleted = client.delete(
            f"/api/me/personal-keys/{created['id']}", headers=headers
        )
        assert deleted.json()["revoked"] is True
        presented = created["full_key"]
    before = _key_table()

    _error(_list(presented, agent_id), 401, "invalid_api_key")
    _error(_revoke(presented, agent_id, target.id), 401, "invalid_api_key")
    assert _key_table() == before


@pytest.mark.parametrize(
    ("agent_id", "key_id"), [(0, 1), (2**31, 1), (1, 0), (1, 2**31)]
)
def test_ids_outside_the_column_range_are_rejected(agent_id, key_id):
    key = _personal_key()

    assert _revoke(key, agent_id, key_id).status_code == 422
    if key_id == 1:
        assert _list(key, agent_id).status_code == 422
