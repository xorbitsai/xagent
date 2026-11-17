"""Owners read and write the widget voice switch of what they can edit."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from tests.shared.widget_voice_env import clear_widget_voice_env
from xagent.web.api import widget as widget_api
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.deployment import Deployment, DeploymentOwnerType
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce
from xagent.web.services import agent_team_scope, widget_voice, widget_voice_switch
from xagent.web.services.agent_team_scope import AgentTeamScope
from xagent.web.services.widget_voice import WidgetVoiceAvailability

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _install_one_slot_queue_pool,
    _login,
    _register_second_user,
    client,
)
from .test_widget_voice_admin import _agent_switch, _deployment_switch
from .test_workforce_widget import _create_workforce, _enable_widget

pytestmark = pytest.mark.usefixtures("_test_db")

AVAILABLE = WidgetVoiceAvailability(True, None)
NOT_CONFIGURED = WidgetVoiceAvailability(False, "voice_models_not_configured")
UNAVAILABLE = WidgetVoiceAvailability(False, "voice_models_unavailable")
TEAM_ID = 7


@pytest.fixture(autouse=True)
def _widget_voice_env(monkeypatch):
    clear_widget_voice_env(monkeypatch)


@pytest.fixture
def provider(monkeypatch):
    """Replace the voice provider lookup; ``result`` is what it answers."""
    state = SimpleNamespace(calls=[], result=AVAILABLE)

    async def resolve(owner_user_id, *, request_db):
        state.calls.append(owner_user_id)
        if isinstance(state.result, Exception):
            raise state.result
        return state.result

    monkeypatch.setattr(
        widget_voice_switch, "resolve_widget_voice_availability", resolve
    )
    monkeypatch.setattr(widget_api, "resolve_widget_voice_availability", resolve)
    return state


def _user_id(username: str) -> int:
    db = _direct_db_session()
    try:
        return int(db.query(User).filter(User.username == username).one().id)
    finally:
        db.close()


def _new_agent(owner: dict[str, str], name: str = "Owner voice agent") -> int:
    response = client.post("/api/agents", headers=owner, json={"name": name})
    assert response.status_code == 200, response.text
    return int(response.json()["id"])


def _set_agent(agent_id: int, **values: Any) -> None:
    db = _direct_db_session()
    try:
        agent = db.get(Agent, agent_id)
        for key, value in values.items():
            setattr(agent, key, value)
        db.commit()
    finally:
        db.close()


def _new_workforce(name: str, owner: str | None = None) -> int:
    """A workforce with a deployment row, owned by ``owner`` (default admin)."""
    workforce_id = _create_workforce(name)
    _enable_widget(workforce_id)
    if owner is not None:
        db = _direct_db_session()
        try:
            db.get(Workforce, workforce_id).owner_user_id = _user_id(owner)
            db.commit()
        finally:
            db.close()
    return workforce_id


def _set_deployment(workforce_id: int, **values: Any) -> None:
    db = _direct_db_session()
    try:
        row = (
            db.query(Deployment)
            .filter_by(
                owner_type=DeploymentOwnerType.WORKFORCE.value, owner_id=workforce_id
            )
            .one()
        )
        for key, value in values.items():
            setattr(row, key, value)
        db.commit()
    finally:
        db.close()


def _set_workforce(workforce_id: int, **values: Any) -> None:
    db = _direct_db_session()
    try:
        workforce = db.get(Workforce, workforce_id)
        for key, value in values.items():
            setattr(workforce, key, value)
        db.commit()
    finally:
        db.close()


def _team_hook(monkeypatch, scopes: dict[str, AgentTeamScope]) -> None:
    by_id = {_user_id(name): scope for name, scope in scopes.items()}
    monkeypatch.setattr(
        agent_team_scope,
        "_agent_team_scope_hook",
        lambda db, user_id: by_id.get(user_id),
    )


def _agent_target(entity: str, owner: str = "bob") -> tuple[str, Any, Any]:
    """The route and the readers/writers of the switch for a fresh entity."""
    if entity == "agent":
        agent_id = _new_agent(_login_as(owner))
        return (
            f"/api/agents/{agent_id}/widget-voice",
            lambda: _agent_switch(agent_id),
            lambda value: _set_agent(agent_id, widget_voice_enabled=value),
        )
    workforce_id = _new_workforce("Owner voice workforce", owner)
    return (
        f"/api/workforces/{workforce_id}/widget-voice",
        lambda: _deployment_switch(workforce_id),
        lambda value: _set_deployment(workforce_id, widget_voice_enabled=value),
    )


def _register(username: str) -> dict[str, str]:
    _admin_headers()
    return _register_second_user(username, f"{username}pass1")


def _login_as(username: str) -> dict[str, str]:
    return _login(username, f"{username}pass1")


# ----- access -----


def test_owner_agent_voice_access_matrix(provider, monkeypatch) -> None:
    _admin_headers()
    users = {
        name: _register_second_user(name, f"{name}pass1")
        for name in ("bob", "carol", "dave", "erin", "frank")
    }
    admin = _admin_headers()
    agent_id = _new_agent(users["bob"])
    _team_hook(
        monkeypatch,
        {
            "carol": AgentTeamScope(team_id=TEAM_ID, is_team_admin=True),
            "dave": AgentTeamScope(team_id=TEAM_ID, is_team_admin=False),
            "erin": AgentTeamScope(team_id=TEAM_ID, is_team_admin=False),
        },
    )
    path = f"/api/agents/{agent_id}/widget-voice"
    # (caller headers, agent team id, visibility, expected status)
    cases = [
        ("creator", users["bob"], None, "team", 200),
        ("team admin", users["carol"], TEAM_ID, "admins", 200),
        ("member, team visibility", users["dave"], TEAM_ID, "team", 200),
        ("member, admins visibility", users["erin"], TEAM_ID, "admins", 404),
        ("other user", users["frank"], None, "team", 404),
        ("system admin who is not the creator", admin, None, "team", 404),
        ("no login", {}, None, "team", 401),
    ]
    for label, headers, team_id, visibility, expected in cases:
        _set_agent(
            agent_id,
            team_id=team_id,
            visibility=visibility,
            widget_voice_enabled=False,
        )
        for method, body in (("get", None), ("put", True), ("put", False)):
            if method == "get":
                response = client.get(path, headers=headers)
            else:
                response = client.put(path, headers=headers, json={"enabled": body})
            assert response.status_code == expected, (label, method, response.text)
        if expected != 200:
            assert _agent_switch(agent_id) is False, label


@pytest.mark.parametrize("exists", [True, False])
def test_owner_workforce_voice_access_matrix(exists, provider) -> None:
    admin = _admin_headers()
    users = {
        name: _register_second_user(name, f"{name}pass1") for name in ("bob", "carol")
    }
    workforce_id = _new_workforce("Matrix workforce", "bob") if exists else 999999
    path = f"/api/workforces/{workforce_id}/widget-voice"
    cases = [
        ("owner", users["bob"], 200),
        ("system admin", admin, 200),
        ("other user", users["carol"], 403),
        ("no login", {}, 401),
    ]
    for label, headers, ok in cases:
        expected = ok if exists or ok == 401 else 404
        for method, body in (("get", None), ("put", True), ("put", False)):
            if method == "get":
                response = client.get(path, headers=headers)
            else:
                response = client.put(path, headers=headers, json={"enabled": body})
            assert response.status_code == expected, (label, method, response.text)
        if exists and expected != 200:
            assert _deployment_switch(workforce_id) is False


# ----- the provider is asked only when it matters -----


@pytest.mark.parametrize(
    "entity, switch",
    [
        ("agent", False),
        ("agent", True),
        ("workforce", False),
        ("workforce", True),
        ("workforce without deployment", False),
    ],
)
def test_owner_get_skips_provider_when_off(entity, switch, provider) -> None:
    _register("bob")
    provider.result = NOT_CONFIGURED
    if entity == "workforce without deployment":
        workforce_id = _create_workforce("No deployment", publish=False)
        path = f"/api/workforces/{workforce_id}/widget-voice"
        headers = _admin_headers()
    else:
        path, _, set_switch = _agent_target(entity)
        set_switch(switch)
        headers = _login_as("bob")
    response = client.get(path, headers=headers)
    assert response.status_code == 200, response.text
    if switch:
        assert response.json() == {
            "enabled": True,
            "available": False,
            "reason": "voice_models_not_configured",
        }
        assert len(provider.calls) == 1
    else:
        assert response.json() == {"enabled": False, "available": None, "reason": None}
        assert provider.calls == []


@pytest.mark.parametrize("entity", ["agent", "workforce"])
@pytest.mark.parametrize("before", [False, True])
def test_owner_put_false_skips_provider(entity, before, provider) -> None:
    _register("bob")
    path, read, set_switch = _agent_target(entity)
    set_switch(before)
    response = client.put(path, headers=_login_as("bob"), json={"enabled": False})
    assert response.status_code == 200, response.text
    assert response.json() == {"enabled": False, "available": None, "reason": None}
    assert read() is False
    assert provider.calls == []


@pytest.mark.parametrize("entity", ["agent", "workforce"])
@pytest.mark.parametrize(
    "answer, status, written, body",
    [
        (AVAILABLE, 200, True, {"enabled": True, "available": True, "reason": None}),
        (
            NOT_CONFIGURED,
            200,
            True,
            {
                "enabled": True,
                "available": False,
                "reason": "voice_models_not_configured",
            },
        ),
        (
            UNAVAILABLE,
            200,
            True,
            {"enabled": True, "available": False, "reason": "voice_models_unavailable"},
        ),
        (OperationalError("SELECT", {}, Exception("down")), 500, False, None),
    ],
)
def test_owner_put_true_reports_availability(
    entity, answer, status, written, body, provider
) -> None:
    _register("bob")
    path, read, _ = _agent_target(entity)
    provider.result = answer
    response = client.put(path, headers=_login_as("bob"), json={"enabled": True})
    assert response.status_code == status, response.text
    if body is not None:
        assert response.json() == body
    assert read() is written


def _row(model: type, **filters: Any) -> dict[str, Any]:
    db = _direct_db_session()
    try:
        row = db.query(model).filter_by(**filters).one()
        return {c.name: getattr(row, c.name) for c in model.__table__.columns}
    finally:
        db.close()


@pytest.mark.parametrize("entity", ["agent", "workforce"])
@pytest.mark.parametrize("value", [True, False])
def test_owner_put_changes_only_the_switch(entity, value, provider) -> None:
    _register("bob")
    path, _, set_switch = _agent_target(entity)
    set_switch(not value)
    if entity == "agent":
        model, filters = Agent, {"id": int(path.split("/")[3])}
    else:
        model, filters = (
            Deployment,
            {
                "owner_type": DeploymentOwnerType.WORKFORCE.value,
                "owner_id": int(path.split("/")[3]),
            },
        )
    before = _row(model, **filters)
    response = client.put(path, headers=_login_as("bob"), json={"enabled": value})
    assert response.status_code == 200, response.text
    after = _row(model, **filters)
    changed = {k for k in before if before[k] != after[k]}
    assert changed - {"updated_at"} == {"widget_voice_enabled"}
    assert after["widget_voice_enabled"] is value


@pytest.mark.parametrize("entity", ["agent", "workforce"])
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"enabled": None},
        {"enabled": "true"},
        {"enabled": 1},
        {"enabled": True, "x": 1},
    ],
)
def test_owner_put_rejects_bad_bodies(entity, body, provider) -> None:
    _register("bob")
    path, read, _ = _agent_target(entity)
    response = client.put(path, headers=_login_as("bob"), json=body)
    assert response.status_code == 422, response.text
    assert read() is False
    assert provider.calls == []


@pytest.mark.parametrize("row", ["widget on", "widget off", "none"])
@pytest.mark.parametrize("status", ["active", "archived"])
@pytest.mark.parametrize("value", [True, False])
def test_workforce_voice_deployment_and_status_guards(
    row, status, value, provider
) -> None:
    """A write needs a deployment row (never created here); turning voice on
    needs an active workforce. A row whose widget is turned off still takes the
    value."""
    admin = _admin_headers()
    if row == "none":
        workforce_id = _create_workforce("Guarded no row", publish=False)
    else:
        workforce_id = _new_workforce("Guarded workforce")
        if row == "widget off":
            _set_deployment(workforce_id, widget_enabled=False)
    _set_workforce(workforce_id, status=status)
    path = f"/api/workforces/{workforce_id}/widget-voice"
    response = client.put(path, headers=admin, json={"enabled": value})
    if row == "none":
        assert response.status_code == 404, response.text
        db = _direct_db_session()
        try:
            assert db.query(Deployment).filter_by(owner_id=workforce_id).count() == 0
        finally:
            db.close()
        assert provider.calls == []
    elif status == "archived" and value:
        assert response.status_code == 400, response.text
        assert provider.calls == []
        assert _deployment_switch(workforce_id) is False
    else:
        assert response.status_code == 200, response.text
        assert response.json()["enabled"] is value
        assert _deployment_switch(workforce_id) is value


def _lose_agent_ownership(agent_id: int) -> None:
    _set_agent(agent_id, user_id=_user_id("carol"))


def _delete_agent(agent_id: int) -> None:
    db = _direct_db_session()
    try:
        db.delete(db.get(Agent, agent_id))
        db.commit()
    finally:
        db.close()


def _lose_workforce_edit_right(workforce_id: int) -> None:
    _set_workforce(workforce_id, owner_user_id=_user_id("carol"))


def _delete_deployment(workforce_id: int) -> None:
    db = _direct_db_session()
    try:
        db.query(Deployment).filter_by(
            owner_type=DeploymentOwnerType.WORKFORCE.value, owner_id=workforce_id
        ).delete()
        db.commit()
    finally:
        db.close()


def _workforce_deployment_count(workforce_id: int) -> int:
    db = _direct_db_session()
    try:
        return (
            db.query(Deployment)
            .filter_by(
                owner_type=DeploymentOwnerType.WORKFORCE.value, owner_id=workforce_id
            )
            .count()
        )
    finally:
        db.close()


@pytest.mark.parametrize(
    "entity, change, status",
    [
        ("agent", _lose_agent_ownership, 404),
        ("agent", _delete_agent, 404),
        ("workforce", _lose_workforce_edit_right, 403),
        ("workforce", _delete_deployment, 404),
    ],
)
def test_owner_put_checks_access_again_before_writing(
    entity, change, status, monkeypatch
) -> None:
    """Access lost while the provider is being asked stops the write."""
    _register("bob")
    _register("carol")
    path, read, _ = _agent_target(entity)
    entity_id = int(path.split("/")[3])

    async def resolve(owner_user_id, *, request_db):
        change(entity_id)
        return AVAILABLE

    monkeypatch.setattr(
        widget_voice_switch, "resolve_widget_voice_availability", resolve
    )
    response = client.put(path, headers=_login_as("bob"), json={"enabled": True})
    assert response.status_code == status, response.text
    if change is _delete_agent:
        db = _direct_db_session()
        try:
            assert db.get(Agent, entity_id) is None
        finally:
            db.close()
    elif change is _delete_deployment:
        assert _workforce_deployment_count(entity_id) == 0
    else:
        assert read() is False


@pytest.mark.parametrize(
    "entity, detail", [("agent", "Agent not found"), ("workforce", "User not found")]
)
@pytest.mark.parametrize(
    "method, value", [("get", None), ("put", True), ("put", False)]
)
def test_owner_voice_answers_404_when_the_caller_is_gone(
    entity, detail, method, value, provider, monkeypatch
) -> None:
    """The caller's row deleted after the login check: 404 on both routes, the
    provider is not asked, and nothing is written."""
    _register("bob")
    path, read, _ = _agent_target(entity)
    headers = _login_as("bob")

    def gone(cls, user):
        return cls(user_id=999_999, is_admin=bool(user.is_admin))

    monkeypatch.setattr(widget_voice_switch.OwnerCaller, "of", classmethod(gone))
    if method == "get":
        response = client.get(path, headers=headers)
    else:
        response = client.put(path, headers=headers, json={"enabled": value})
    assert response.status_code == 404, response.text
    assert response.json()["detail"] == detail
    assert provider.calls == []
    assert read() is False


# ----- the old owner entry points still do not take the switch -----
# (see test_widget_voice_admin.py::test_owner_endpoints_leave_the_switch_off)


# ----- records, guests, pool, caller values, concurrency -----


@pytest.mark.parametrize("entity", ["agent", "workforce"])
@pytest.mark.parametrize("before, after", [(False, True), (True, False), (True, True)])
def test_owner_put_logs_previous_and_new(
    entity, before, after, provider, caplog
) -> None:
    _register("bob")
    path, _, set_switch = _agent_target(entity)
    set_switch(before)
    with caplog.at_level(logging.INFO, logger=widget_voice_switch.logger.name):
        response = client.put(path, headers=_login_as("bob"), json={"enabled": after})
    assert response.status_code == 200, response.text
    records = [
        r
        for r in caplog.records
        if r.getMessage() == "Widget voice switch set by owner"
    ]
    assert len(records) == 1
    record = records[0]
    assert record.actor_user_id == _user_id("bob")
    assert record.actor_is_admin is False
    assert record.previous_widget_voice_enabled is before
    assert record.widget_voice_enabled is after
    assert record.available is (True if after else None)
    assert getattr(record, "agent_id" if entity == "agent" else "workforce_id") == int(
        path.split("/")[3]
    )


@pytest.mark.parametrize("entity", ["agent", "workforce"])
def test_owner_switch_reaches_guest_auth(entity, provider) -> None:
    owner = _register("bob")
    if entity == "agent":
        agent_id = _new_agent(owner)
        _set_agent(agent_id, status=AgentStatus.PUBLISHED)
        enabled = client.put(
            f"/api/agents/{agent_id}",
            headers=owner,
            json={"widget_enabled": True, "allowed_domains": ["*"]},
        )
        assert enabled.status_code == 200, enabled.text
        db = _direct_db_session()
        try:
            key = str(db.get(Agent, agent_id).widget_key)
        finally:
            db.close()
        path = f"/api/agents/{agent_id}/widget-voice"
    else:
        workforce_id = _create_workforce("Guest auth workforce")
        key = _enable_widget(workforce_id)
        owner = _admin_headers()
        path = f"/api/workforces/{workforce_id}/widget-voice"

    def voice_enabled() -> bool:
        response = client.post(
            "/api/widget/auth", json={"guest_id": "g", "widget_key": key}
        )
        assert response.status_code == 200, response.text
        return bool(response.json()["voice_enabled"])

    assert voice_enabled() is False
    assert client.put(path, headers=owner, json={"enabled": True}).status_code == 200
    assert voice_enabled() is True
    assert client.put(path, headers=owner, json={"enabled": False}).status_code == 200
    assert voice_enabled() is False


@pytest.mark.parametrize(
    "method, entity, value",
    [
        ("get", "agent", None),
        ("put", "agent", True),
        ("put", "agent", False),
        ("get", "workforce", None),
        ("put", "workforce", True),
        ("put", "workforce", False),
    ],
)
def test_owner_voice_routes_answer_on_a_one_slot_pool(
    method, entity, value, monkeypatch
) -> None:
    _register("bob")
    path, read, set_switch = _agent_target(entity)
    set_switch(True)
    headers = _login_as("bob")

    async def resolve(owner_user_id, *, request_db):
        # Takes a connection from the one-slot pool: it only succeeds when the
        # request session has already returned its own.
        db = _direct_db_session()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()
        return AVAILABLE

    monkeypatch.setattr(
        widget_voice_switch, "resolve_widget_voice_availability", resolve
    )
    _install_one_slot_queue_pool(monkeypatch)
    if method == "get":
        response = client.get(path, headers=headers)
    else:
        response = client.put(path, headers=headers, json={"enabled": value})
    assert response.status_code == 200, response.text
    if method == "put":
        assert read() is value


@pytest.mark.parametrize(
    "route, caller",
    [
        ("agent", "team member"),
        ("agent", "owner"),
        ("workforce", "owner"),
        ("workforce", "system admin"),
    ],
)
def test_owner_voice_uses_plain_user_values(
    route, caller, provider, monkeypatch
) -> None:
    admin = _admin_headers()
    bob = _register("bob")
    dave = _register("dave")
    if route == "agent":
        agent_id = _new_agent(bob)
        _set_agent(agent_id, team_id=TEAM_ID, visibility="team")
        _team_hook(
            monkeypatch,
            {
                "bob": AgentTeamScope(team_id=TEAM_ID, is_team_admin=False),
                "dave": AgentTeamScope(team_id=TEAM_ID, is_team_admin=False),
            },
        )
        path = f"/api/agents/{agent_id}/widget-voice"
        headers = dave if caller == "team member" else bob
        read = lambda: _agent_switch(agent_id)  # noqa: E731
    else:
        workforce_id = _new_workforce("Plain values workforce", "bob")
        path = f"/api/workforces/{workforce_id}/widget-voice"
        headers = bob if caller == "owner" else admin
        read = lambda: _deployment_switch(workforce_id)  # noqa: E731

    real_release = widget_voice.release_db_connection_if_clean

    def release_then_poison(db):
        released = real_release(db)

        def refuse(*_args, **_kwargs):
            raise AssertionError("request session used after its connection was freed")

        for name in ("execute", "query", "get", "scalars", "scalar", "connection"):
            setattr(db, name, refuse)
        return released

    monkeypatch.setattr(
        widget_voice, "release_db_connection_if_clean", release_then_poison
    )
    response = client.put(path, headers=headers, json={"enabled": True})
    assert response.status_code == 200, response.text
    assert read() is True


@pytest.mark.parametrize("entity", ["agent", "workforce"])
@pytest.mark.parametrize("first", [True, False])
def test_last_write_wins(entity, first, provider, monkeypatch) -> None:
    admin = _admin_headers()
    bob = _register("bob")
    carol = _register("carol")
    if entity == "agent":
        agent_id = _new_agent(bob)
        _set_agent(agent_id, team_id=TEAM_ID, visibility="team")
        _team_hook(
            monkeypatch,
            {
                "bob": AgentTeamScope(team_id=TEAM_ID, is_team_admin=False),
                "carol": AgentTeamScope(team_id=TEAM_ID, is_team_admin=True),
            },
        )
        path = f"/api/agents/{agent_id}/widget-voice"
        read = lambda: _agent_switch(agent_id)  # noqa: E731
        callers = (bob, carol)
    else:
        workforce_id = _new_workforce("Last write workforce", "bob")
        path = f"/api/workforces/{workforce_id}/widget-voice"
        read = lambda: _deployment_switch(workforce_id)  # noqa: E731
        callers = (bob, admin)
    for headers, value in ((callers[0], first), (callers[1], not first)):
        response = client.put(path, headers=headers, json={"enabled": value})
        assert response.status_code == 200, response.text
        assert response.json()["enabled"] is value
    assert read() is (not first)
