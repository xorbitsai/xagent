"""Only the admin widget-voice API writes or returns the widget voice switch."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

from __future__ import annotations

import dataclasses
import inspect

import pytest
from pydantic import BaseModel
from sqlalchemy.exc import OperationalError

from tests.shared.widget_voice_env import clear_widget_voice_env
from xagent.web.api import admin_widget_voice
from xagent.web.api import agents as agents_api
from xagent.web.api import workforces as workforces_api
from xagent.web.api.v1 import agents as v1_agents_api
from xagent.web.models.agent import Agent
from xagent.web.models.deployment import Deployment, DeploymentOwnerType
from xagent.web.models.user import User
from xagent.web.schemas import v1 as v1_schemas
from xagent.web.services import agent_management
from xagent.web.services.agent_store import _AGENT_UPDATE_FIELDS
from xagent.web.services.widget_voice import WidgetVoiceAvailability

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _install_one_slot_queue_pool,
    _register_second_user,
    client,
)
from .test_widget_voice_asr import asr_model  # noqa: F401
from .test_widget_voice_tts import voice  # noqa: F401
from .test_workforce_widget import _create_workforce, _enable_widget

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def _widget_voice_env(monkeypatch):
    clear_widget_voice_env(monkeypatch)


COLUMN = "widget_voice_enabled"
OWNER_MODULES = (
    agents_api,
    workforces_api,
    v1_agents_api,
    v1_schemas,
    agent_management,
)


def _owner_shapes() -> list[tuple[str, set[str]]]:
    """Every request, response, and snapshot shape the owner surfaces define."""
    shapes = []
    for module in OWNER_MODULES:
        for name, value in vars(module).items():
            if not inspect.isclass(value) or value.__module__ != module.__name__:
                continue
            if issubclass(value, BaseModel):
                shapes.append((name, set(value.model_fields)))
            elif dataclasses.is_dataclass(value):
                shapes.append((name, {f.name for f in dataclasses.fields(value)}))
    return shapes


def _agent_switch(agent_id: int) -> bool:
    db = _direct_db_session()
    try:
        return bool(db.get(Agent, agent_id).widget_voice_enabled)
    finally:
        db.close()


def _deployment_switch(workforce_id: int) -> bool:
    db = _direct_db_session()
    try:
        deployment = (
            db.query(Deployment)
            .filter_by(
                owner_type=DeploymentOwnerType.WORKFORCE.value, owner_id=workforce_id
            )
            .one()
        )
        return bool(deployment.widget_voice_enabled)
    finally:
        db.close()


def test_owner_writable_fields_never_include_the_switch() -> None:
    assert COLUMN not in _AGENT_UPDATE_FIELDS
    shapes = _owner_shapes()
    assert any(name == "AgentUpdateRequest" for name, _ in shapes)
    assert any(name == "WorkforceWidgetUpdateRequest" for name, _ in shapes)
    assert [name for name, fields in shapes if COLUMN in fields] == []


def test_owner_endpoints_leave_the_switch_off() -> None:
    admin = _admin_headers()
    owner = _register_second_user()
    created = client.post(
        "/api/agents", headers=owner, json={"name": "Voice owner", COLUMN: True}
    )
    assert created.status_code == 200, created.text
    agent_id = created.json()["id"]
    assert COLUMN not in created.json()
    assert _agent_switch(agent_id) is False

    updated = client.put(
        f"/api/agents/{agent_id}",
        headers=owner,
        json={"description": "changed", "widget_enabled": True, COLUMN: True},
    )
    assert updated.status_code == 200, updated.text
    assert COLUMN not in updated.json()
    assert _agent_switch(agent_id) is False

    key = client.post("/api/me/personal-keys", headers=admin).json()["full_key"]
    v1 = {"Authorization": f"Bearer {key}"}
    v1_created = client.post(
        "/v1/agents",
        headers=v1,
        json={"name": "V1 voice", "generate_runtime_key": False, COLUMN: True},
    )
    assert v1_created.status_code == 200, v1_created.text
    assert COLUMN not in v1_created.json()["agent"]
    assert _agent_switch(v1_created.json()["agent"]["id"]) is False
    listed = client.get("/v1/agents", headers=v1)
    assert listed.status_code == 200 and all(COLUMN not in a for a in listed.json())

    workforce_id = _create_workforce("Voice workforce")
    _enable_widget(workforce_id)
    widget = client.put(
        f"/api/workforces/{workforce_id}/widget",
        headers=admin,
        json={"widget_enabled": True, COLUMN: True},
    )
    assert widget.status_code == 200, widget.text
    assert COLUMN not in widget.json()
    assert _deployment_switch(workforce_id) is False


@pytest.mark.parametrize("entity", ["agent", "workforce"])
def test_owner_update_leaves_the_switch_on(entity: str) -> None:
    admin = _admin_headers()
    if entity == "agent":
        owner = _register_second_user()
        agent_id = client.post(
            "/api/agents", headers=owner, json={"name": "Voice on"}
        ).json()["id"]
        admin_path = f"/api/admin/widget-voice/agents/{agent_id}"
        owner_path = f"/api/agents/{agent_id}"
        body = {"description": "changed", "widget_enabled": True, COLUMN: False}

        def read() -> bool:
            return _agent_switch(agent_id)

    else:
        owner = admin
        workforce_id = _create_workforce("Voice on workforce")
        _enable_widget(workforce_id)
        admin_path = f"/api/admin/widget-voice/workforces/{workforce_id}"
        owner_path = f"/api/workforces/{workforce_id}/widget"
        body = {"widget_enabled": True, COLUMN: False}

        def read() -> bool:
            return _deployment_switch(workforce_id)

    written = client.put(admin_path, headers=admin, json={"enabled": True})
    assert written.status_code == 200, written.text
    assert read() is True

    updated = client.put(owner_path, headers=owner, json=body)
    assert updated.status_code == 200, updated.text
    assert COLUMN not in updated.json()
    assert read() is True


def test_admin_api_writes_the_switch_and_reports_availability() -> None:
    admin = _admin_headers()
    owner = _register_second_user()
    agent_id = client.post("/api/agents", headers=owner, json={"name": "A"}).json()[
        "id"
    ]
    workforce_id = _create_workforce("Admin voice workforce")
    _enable_widget(workforce_id)

    for path, read in (
        (f"/api/admin/widget-voice/agents/{agent_id}", lambda: _agent_switch(agent_id)),
        (
            f"/api/admin/widget-voice/workforces/{workforce_id}",
            lambda: _deployment_switch(workforce_id),
        ),
    ):
        assert client.get(path, headers=admin).json()["enabled"] is False
        written = client.put(path, headers=admin, json={"enabled": True})
        assert written.status_code == 200, written.text
        assert read() is True
        # No voice models are configured here: on, but explained as unusable.
        assert client.get(path, headers=admin).json() == {
            "enabled": True,
            "available": False,
            "reason": "voice_models_not_configured",
        }
        assert client.put(path, headers=admin, json={"enabled": "yes"}).status_code == (
            422
        )
        assert client.get(path, headers=owner).status_code == 403
        denied = client.put(path, headers=owner, json={"enabled": False})
        assert denied.status_code == 403, denied.text
        assert read() is True


def test_admin_api_missing_targets_are_404_without_creating_rows() -> None:
    admin = _admin_headers()
    unpublished = _create_workforce("No deployment", publish=False)
    for path in (
        "/api/admin/widget-voice/agents/999999",
        "/api/admin/widget-voice/workforces/999999",
        f"/api/admin/widget-voice/workforces/{unpublished}",
    ):
        assert client.get(path, headers=admin).status_code == 404
        assert (
            client.put(path, headers=admin, json={"enabled": True}).status_code == 404
        )
    db = _direct_db_session()
    try:
        assert db.query(Deployment).filter_by(owner_id=unpublished).count() == 0
    finally:
        db.close()


def test_admin_put_leaves_the_switch_when_model_resolution_fails(monkeypatch) -> None:
    admin = _admin_headers()
    owner = _register_second_user()
    agent_id = client.post("/api/agents", headers=owner, json={"name": "B"}).json()[
        "id"
    ]
    workforce_id = _create_workforce("Failing resolution workforce")
    _enable_widget(workforce_id)

    async def broken(_owner_user_id, *, request_db):
        raise OperationalError("SELECT", {}, Exception("down"))

    monkeypatch.setattr(admin_widget_voice, "resolve_widget_voice_availability", broken)
    for path, read in (
        (f"/api/admin/widget-voice/agents/{agent_id}", lambda: _agent_switch(agent_id)),
        (
            f"/api/admin/widget-voice/workforces/{workforce_id}",
            lambda: _deployment_switch(workforce_id),
        ),
    ):
        response = client.put(path, headers=admin, json={"enabled": True})
        assert response.status_code == 500
        assert read() is False


def _make_target(entity: str, voice) -> tuple[str, int]:
    if entity == "agent":
        return f"/api/admin/widget-voice/agents/{voice.agent_id}", voice.agent_id
    workforce_id = _create_workforce("One slot workforce")
    _enable_widget(workforce_id)
    return f"/api/admin/widget-voice/workforces/{workforce_id}", workforce_id


def _switch(entity: str, target_id: int) -> bool:
    return (
        _agent_switch(target_id) if entity == "agent" else _deployment_switch(target_id)
    )


@pytest.mark.parametrize("entity", ["agent", "workforce"])
def test_admin_routes_answer_on_a_one_slot_pool(
    entity, voice, asr_model, monkeypatch
) -> None:
    path, target_id = _make_target(entity, voice)
    _install_one_slot_queue_pool(monkeypatch)
    before = _switch(entity, target_id)
    assert before is (entity == "agent")

    first = client.get(path, headers=voice.admin)
    assert first.status_code == 200, first.text
    assert first.json() == {"enabled": before, "available": True, "reason": None}

    written = client.put(path, headers=voice.admin, json={"enabled": not before})
    assert written.status_code == 200, written.text
    assert written.json() == {"enabled": not before, "available": True, "reason": None}
    assert _switch(entity, target_id) is (not before)

    after = client.get(path, headers=voice.admin)
    assert after.status_code == 200, after.text
    assert after.json()["enabled"] is (not before)


@pytest.mark.parametrize("entity", ["agent", "workforce"])
def test_admin_put_writes_nothing_when_the_target_disappears(
    entity, voice, monkeypatch
) -> None:
    path, target_id = _make_target(entity, voice)

    async def delete_target(_owner_user_id, *, request_db):
        db = _direct_db_session()
        try:
            if entity == "agent":
                db.query(Agent).filter(Agent.id == target_id).delete()
            else:
                db.query(Deployment).filter_by(
                    owner_type=DeploymentOwnerType.WORKFORCE.value, owner_id=target_id
                ).delete()
            db.commit()
        finally:
            db.close()
        return WidgetVoiceAvailability(False, "voice_models_not_configured")

    monkeypatch.setattr(
        admin_widget_voice, "resolve_widget_voice_availability", delete_target
    )
    response = client.put(path, headers=voice.admin, json={"enabled": True})
    assert response.status_code == 404, response.text
    db = _direct_db_session()
    try:
        if entity == "agent":
            assert db.query(Agent).filter(Agent.id == target_id).count() == 0
        else:
            assert db.query(Deployment).filter_by(owner_id=target_id).count() == 0
    finally:
        db.close()


async def test_admin_target_refuses_a_request_session_with_uncommitted_work() -> None:
    reads: list[int] = []

    def read() -> admin_widget_voice._Target:
        reads.append(1)
        return admin_widget_voice._Target(1, False)

    db = _direct_db_session()
    try:
        db.add(User(username="pending", password_hash="x"))
        with pytest.raises(RuntimeError):
            await admin_widget_voice._target(db, read)
        assert reads == []
    finally:
        db.rollback()
        db.close()


@pytest.mark.parametrize(
    "unset", ["XAGENT_WIDGET_VOICE_ASR_MODEL", "XAGENT_WIDGET_VOICE_TTS_MODEL"]
)
@pytest.mark.parametrize("entity", ["agent", "workforce"])
def test_admin_reports_voice_unavailable_when_either_model_is_unset(
    entity, unset, voice, asr_model, monkeypatch
) -> None:
    path, _ = _make_target(entity, voice)
    monkeypatch.delenv(unset)

    response = client.get(path, headers=voice.admin)

    assert response.status_code == 200, response.text
    assert response.json()["available"] is False
    assert response.json()["reason"] == "voice_models_not_configured"
