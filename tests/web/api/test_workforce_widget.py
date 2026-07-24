"""Tests for the workforce widget deployment channel (#948).

Covers: owner-side widget management endpoints (get / update / rotate, gated
on ``status == "active"`` for enable/rotate), widget auth resolving a
workforce deployment key (direct visit and embed ticket), the embedding
allowed-domains check, guest task creation entering through
``create_workforce_run(source="widget")``, widget-context task scoping, and
rotate/disable invalidating already-issued guest tokens.

Builds on the share-link channel patterns (#947).
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.deployment import Deployment, DeploymentOwnerType
from xagent.web.models.task import Task
from xagent.web.models.user import User
from xagent.web.models.workforce import WorkforceRun
from xagent.web.services import workforce_runs as workforce_runs_service

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")


def _user_id(username: str = "admin") -> int:
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == username).one()
        return int(user.id)
    finally:
        db.close()


def _create_published_agent(user_id: int, name: str) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=user_id,
            name=name,
            description=f"{name} description",
            instructions=f"{name} instructions",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return int(agent.id)
    finally:
        db.close()


def _create_workforce(name: str, *, publish: bool = True) -> int:
    headers = _admin_headers()
    owner_id = _user_id()
    manager_agent_id = _create_published_agent(owner_id, f"{name} Manager")
    worker_agent_id = _create_published_agent(owner_id, f"{name} Worker")
    response = client.post(
        "/api/workforces",
        headers=headers,
        json={
            "name": name,
            "description": "Coordinates widget tests",
            "manager_agent_id": manager_agent_id,
            "workers": [
                {
                    "source_type": "existing",
                    "agent_id": worker_agent_id,
                    "alias": "worker-1",
                    "assignment_instructions": "Handle everything",
                    "enabled": True,
                    "sort_order": 1,
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    workforce_id = int(response.json()["id"])
    if publish:
        published = client.post(
            f"/api/workforces/{workforce_id}/publish", headers=headers
        )
        assert published.status_code == 200, published.text
    return workforce_id


def _enable_widget(
    workforce_id: int, *, allowed_domains: list[str] | None = None
) -> str:
    """Enable the widget and return its (freshly minted) widget key."""
    response = client.put(
        f"/api/workforces/{workforce_id}/widget",
        headers=_admin_headers(),
        json={
            "widget_enabled": True,
            "allowed_domains": allowed_domains
            if allowed_domains is not None
            else ["*"],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["widget_enabled"] is True
    key = body["widget_key"]
    assert isinstance(key, str) and key
    return str(key)


def _authenticate_widget_guest_by_key(
    widget_key: str, guest_id: str = "guest_test"
) -> dict[str, str]:
    response = client.post(
        "/api/widget/auth",
        json={"guest_id": guest_id, "widget_key": widget_key},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _stub_begin_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(**_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(background_task=None)

    monkeypatch.setattr(
        workforce_runs_service.TaskTurnOrchestrator, "begin_turn", _stub
    )


# ===== Widget management endpoints =====


def test_widget_enable_requires_active_workforce() -> None:
    workforce_id = _create_workforce("Draft Widget Workforce", publish=False)

    enabled = client.put(
        f"/api/workforces/{workforce_id}/widget",
        headers=_admin_headers(),
        json={"widget_enabled": True},
    )
    assert enabled.status_code == 400, enabled.text

    rotated = client.post(
        f"/api/workforces/{workforce_id}/widget-key/rotate", headers=_admin_headers()
    )
    assert rotated.status_code == 400, rotated.text


def test_widget_enable_rotate_disable_lifecycle() -> None:
    workforce_id = _create_workforce("Lifecycle Widget Workforce")
    headers = _admin_headers()

    # Default state: widget is opt-in and starts disabled with no key.
    initial = client.get(f"/api/workforces/{workforce_id}/widget-key", headers=headers)
    assert initial.status_code == 200, initial.text
    assert initial.json() == {
        "workforce_id": workforce_id,
        "widget_enabled": False,
        "widget_key": None,
        "allowed_domains": [],
    }

    key = _enable_widget(workforce_id, allowed_domains=["example.com"])

    db = _direct_db_session()
    try:
        deployment = (
            db.query(Deployment)
            .filter(
                Deployment.owner_type == DeploymentOwnerType.WORKFORCE.value,
                Deployment.owner_id == workforce_id,
            )
            .one()
        )
        assert deployment.widget_enabled is True
        assert deployment.widget_key == key
        assert list(deployment.allowed_domains or []) == ["example.com"]
    finally:
        db.close()

    fetched = client.get(f"/api/workforces/{workforce_id}/widget-key", headers=headers)
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["widget_key"] == key
    assert fetched.json()["allowed_domains"] == ["example.com"]

    # Re-enabling keeps the existing key stable.
    assert _enable_widget(workforce_id, allowed_domains=["example.com"]) == key

    rotated = client.post(
        f"/api/workforces/{workforce_id}/widget-key/rotate", headers=headers
    )
    assert rotated.status_code == 200, rotated.text
    rotated_key = rotated.json()["widget_key"]
    assert rotated_key and rotated_key != key
    assert rotated.json()["widget_enabled"] is True

    disabled = client.put(
        f"/api/workforces/{workforce_id}/widget",
        headers=headers,
        json={"widget_enabled": False},
    )
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["widget_enabled"] is False
    # Disabling preserves the key (like agent widget); only the flag flips.
    assert disabled.json()["widget_key"] == rotated_key


def test_widget_allowed_domains_update_independent_of_enable() -> None:
    workforce_id = _create_workforce("Domains Widget Workforce")
    _enable_widget(workforce_id, allowed_domains=["a.com"])

    updated = client.put(
        f"/api/workforces/{workforce_id}/widget",
        headers=_admin_headers(),
        json={"allowed_domains": ["a.com", "b.com"]},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["allowed_domains"] == ["a.com", "b.com"]
    # widget_enabled untouched when only domains are sent.
    assert updated.json()["widget_enabled"] is True


def test_widget_requires_workforce_owner() -> None:
    workforce_id = _create_workforce("Owner Only Widget Workforce")
    other_headers = _register_second_user()

    for method, url, kwargs in (
        ("get", f"/api/workforces/{workforce_id}/widget-key", {}),
        (
            "put",
            f"/api/workforces/{workforce_id}/widget",
            {"json": {"widget_enabled": True}},
        ),
        ("post", f"/api/workforces/{workforce_id}/widget-key/rotate", {}),
    ):
        response = getattr(client, method)(url, headers=other_headers, **kwargs)
        assert response.status_code == 403, response.text


def test_widget_enable_rotate_rejected_for_archived_workforce() -> None:
    workforce_id = _create_workforce("Archived Widget Workforce")
    headers = _admin_headers()
    _enable_widget(workforce_id)

    archived = client.delete(f"/api/workforces/{workforce_id}", headers=headers)
    assert archived.status_code == 200, archived.text

    # Re-enabling re-exposes the workforce, so it stays blocked on archive
    # (400 from _ensure_active_workforce; the PUT itself uses the archived-safe
    # can_edit gate so disabling still works — see the disable test below).
    enabled = client.put(
        f"/api/workforces/{workforce_id}/widget",
        headers=headers,
        json={"widget_enabled": True},
    )
    assert enabled.status_code == 400, enabled.text
    # Rotate keeps the stricter edit gate, so it 409s on an archived workforce.
    rotated = client.post(
        f"/api/workforces/{workforce_id}/widget-key/rotate", headers=headers
    )
    assert rotated.status_code == 409, rotated.text


def test_widget_disable_allowed_for_archived_workforce() -> None:
    """Mirrors the share-link disable path: turning the widget off only removes
    access, so it stays available on an archived workforce (archived-safe
    ``can_edit_workforce`` gate)."""
    workforce_id = _create_workforce("Archived Disable Widget Workforce")
    headers = _admin_headers()
    _enable_widget(workforce_id)

    archived = client.delete(f"/api/workforces/{workforce_id}", headers=headers)
    assert archived.status_code == 200, archived.text

    disabled = client.put(
        f"/api/workforces/{workforce_id}/widget",
        headers=headers,
        json={"widget_enabled": False},
    )
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["widget_enabled"] is False


def test_widget_rotate_preserves_disabled_state() -> None:
    workforce_id = _create_workforce("Rotate Preserve Widget Workforce")
    headers = _admin_headers()
    _enable_widget(workforce_id)

    disabled = client.put(
        f"/api/workforces/{workforce_id}/widget",
        headers=headers,
        json={"widget_enabled": False},
    )
    assert disabled.status_code == 200, disabled.text

    rotated = client.post(
        f"/api/workforces/{workforce_id}/widget-key/rotate", headers=headers
    )
    assert rotated.status_code == 200, rotated.text
    body = rotated.json()
    assert body["widget_enabled"] is False  # preserved, not force-enabled
    assert body["widget_key"]  # key still rotated


# ===== Widget auth resolution =====


def test_widget_direct_visit_auth_resolves_workforce() -> None:
    workforce_id = _create_workforce("Auth Widget Workforce")
    key = _enable_widget(workforce_id)

    response = client.post(
        "/api/widget/auth",
        json={"guest_id": "guest_test", "widget_key": key},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["workforce_id"] == workforce_id
    assert body["agent_id"] is None
    assert body["agent_name"] == "Auth Widget Workforce"
    assert body["access_token"]


def test_widget_embed_ticket_and_auth_workforce() -> None:
    workforce_id = _create_workforce("Ticket Widget Workforce")
    key = _enable_widget(workforce_id, allowed_domains=["example.com"])

    ticket_resp = client.post(
        "/api/widget/embed-ticket",
        json={"widget_key": key},
        headers={"origin": "https://example.com"},
    )
    assert ticket_resp.status_code == 200, ticket_resp.text
    ticket_body = ticket_resp.json()
    assert ticket_body["workforce_id"] == workforce_id
    assert ticket_body["agent_id"] is None
    ticket = ticket_body["ticket"]
    assert ticket

    auth = client.post(
        "/api/widget/auth",
        json={"guest_id": "guest_test", "embed_ticket": ticket},
    )
    assert auth.status_code == 200, auth.text
    assert auth.json()["workforce_id"] == workforce_id


def test_widget_embed_ticket_enforces_allowed_domains() -> None:
    workforce_id = _create_workforce("Domain Gate Widget Workforce")
    key = _enable_widget(workforce_id, allowed_domains=["example.com"])

    blocked = client.post(
        "/api/widget/embed-ticket",
        json={"widget_key": key},
        headers={"origin": "https://evil.com"},
    )
    assert blocked.status_code == 403, blocked.text

    allowed = client.post(
        "/api/widget/embed-ticket",
        json={"widget_key": key},
        headers={"origin": "https://example.com"},
    )
    assert allowed.status_code == 200, allowed.text


def test_widget_auth_rejects_unknown_key() -> None:
    response = client.post(
        "/api/widget/auth",
        json={"guest_id": "guest_test", "widget_key": "no-such-key"},
    )
    assert response.status_code == 403, response.text


def test_widget_auth_rejects_disabled_widget() -> None:
    workforce_id = _create_workforce("Disabled Widget Workforce")
    key = _enable_widget(workforce_id)

    disabled = client.put(
        f"/api/workforces/{workforce_id}/widget",
        headers=_admin_headers(),
        json={"widget_enabled": False},
    )
    assert disabled.status_code == 200, disabled.text

    response = client.post(
        "/api/widget/auth",
        json={"guest_id": "guest_test", "widget_key": key},
    )
    assert response.status_code == 403, response.text


def test_widget_auth_rejects_inactive_workforce() -> None:
    workforce_id = _create_workforce("Inactive Widget Workforce")
    key = _enable_widget(workforce_id)

    unpublished = client.post(
        f"/api/workforces/{workforce_id}/unpublish", headers=_admin_headers()
    )
    assert unpublished.status_code == 200, unpublished.text

    response = client.post(
        "/api/widget/auth",
        json={"guest_id": "guest_test", "widget_key": key},
    )
    assert response.status_code == 403, response.text


def test_rotated_widget_key_invalidates_existing_guest_tokens() -> None:
    workforce_id = _create_workforce("Rotate Auth Widget Workforce")
    key = _enable_widget(workforce_id)
    guest_headers = _authenticate_widget_guest_by_key(key)

    rotated = client.post(
        f"/api/workforces/{workforce_id}/widget-key/rotate", headers=_admin_headers()
    )
    assert rotated.status_code == 200, rotated.text

    # Previously issued guest JWTs are invalidated by the key mismatch.
    response = client.post(
        "/api/widget/chat/task/create",
        headers=guest_headers,
        json={"title": "hello", "description": "hello"},
    )
    assert response.status_code == 403, response.text


def test_disabled_widget_invalidates_existing_guest_tokens() -> None:
    workforce_id = _create_workforce("Disable Invalidate Widget Workforce")
    key = _enable_widget(workforce_id)
    guest_headers = _authenticate_widget_guest_by_key(key)

    disabled = client.put(
        f"/api/workforces/{workforce_id}/widget",
        headers=_admin_headers(),
        json={"widget_enabled": False},
    )
    assert disabled.status_code == 200, disabled.text

    response = client.post(
        "/api/widget/chat/task/create",
        headers=guest_headers,
        json={"title": "hello", "description": "hello"},
    )
    assert response.status_code == 403, response.text


# ===== Guest task creation → create_workforce_run =====


def test_widget_task_create_starts_workforce_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workforce_id = _create_workforce("Guest Run Widget Workforce")
    key = _enable_widget(workforce_id)
    guest_headers = _authenticate_widget_guest_by_key(key)
    _stub_begin_turn(monkeypatch)

    response = client.post(
        "/api/widget/chat/task/create",
        headers=guest_headers,
        json={"title": "hello workforce", "description": "hello workforce"},
    )
    assert response.status_code == 200, response.text
    task_id = int(response.json()["task_id"])

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.source == "widget"
        assert bool(task.is_visible) is False
        assert int(task.user_id) == _user_id()
        assert task.channel_id is None
        assert task.agent_config.get("auth_mode") == "widget"
        assert int(task.agent_config.get("widget_workforce_id")) == workforce_id
        assert task.agent_config.get("guest_id") == "guest_test"

        run = db.query(WorkforceRun).filter(WorkforceRun.task_id == task_id).one()
        assert int(run.workforce_id) == workforce_id
        assert bool(run.is_preview) is False
    finally:
        db.close()

    # The run shows up in the owner's runs history with source=widget.
    listed = client.get(
        f"/api/workforces/{workforce_id}/runs", headers=_admin_headers()
    )
    assert listed.status_code == 200, listed.text
    items = listed.json()["items"]
    assert items and items[0]["source"] == "widget"


def test_widget_task_create_rejects_foreign_agent_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workforce_id = _create_workforce("Foreign Agent Widget Workforce")
    key = _enable_widget(workforce_id)
    guest_headers = _authenticate_widget_guest_by_key(key)
    _stub_begin_turn(monkeypatch)

    foreign_agent_id = _create_published_agent(_user_id(), "Foreign Agent")
    response = client.post(
        "/api/widget/chat/task/create",
        headers=guest_headers,
        json={
            "title": "hello",
            "description": "hello",
            "agent_id": foreign_agent_id,
        },
    )
    assert response.status_code == 403, response.text


# ===== Task-less opening-message upload =====


def test_taskless_widget_upload_enforces_file_count_cap() -> None:
    """The task-less workforce-widget upload path is reachable by any widget
    guest before a task/owner exists, so it caps files per request to blunt
    the worst storage-abuse case (mirrors the share path)."""
    from xagent.web.api.public_chat_access import MAX_TASKLESS_SHARE_UPLOAD_FILES

    workforce_id = _create_workforce("Widget Upload Cap Workforce")
    key = _enable_widget(workforce_id)
    guest_headers = _authenticate_widget_guest_by_key(key)

    over_cap = [
        ("files", (f"f{i}.txt", io.BytesIO(b"x"), "text/plain"))
        for i in range(MAX_TASKLESS_SHARE_UPLOAD_FILES + 1)
    ]
    rejected = client.post(
        "/api/widget/files/upload",
        headers=guest_headers,
        data={"task_type": "task"},
        files=over_cap,
    )
    assert rejected.status_code == 422, rejected.text
    assert str(MAX_TASKLESS_SHARE_UPLOAD_FILES) in rejected.json()["detail"]


def test_agent_widget_taskless_upload_still_requires_task_id() -> None:
    """The task-less upload relaxation is workforce-only; the agent widget
    path keeps its task_id-required contract (files ride the first WS turn)."""
    _admin_headers()  # ensure the admin owner exists before _user_id()
    owner_id = _user_id()
    agent_id = _create_published_agent(owner_id, "Agent Widget Upload")
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        agent.widget_enabled = True
        agent.widget_key = "agent-widget-key-upload-test"
        agent.allowed_domains = ["*"]
        db.commit()
    finally:
        db.close()

    auth = client.post(
        "/api/widget/auth",
        json={"guest_id": "guest_test", "widget_key": "agent-widget-key-upload-test"},
    )
    assert auth.status_code == 200, auth.text
    guest_headers = {"Authorization": f"Bearer {auth.json()['access_token']}"}

    rejected = client.post(
        "/api/widget/files/upload",
        headers=guest_headers,
        data={"task_type": "task"},
        files=[("files", ("f.txt", io.BytesIO(b"x"), "text/plain"))],
    )
    assert rejected.status_code == 400, rejected.text
    assert "task_id" in rejected.json()["detail"]


# ===== Widget task-access scoping =====


def test_widget_task_access_scoped_to_guest_and_workforce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A widget guest may only reach its OWN task on ITS workforce: another
    guest of the same workforce, or any guest of a different workforce, is
    rejected (guest_id + widget_workforce_id scoping in
    ``_get_task_for_workforce_widget_context``)."""
    _stub_begin_turn(monkeypatch)
    wf_a = _create_workforce("Scope Widget A")
    key_a = _enable_widget(wf_a)
    wf_b = _create_workforce("Scope Widget B")
    key_b = _enable_widget(wf_b)

    owner = _authenticate_widget_guest_by_key(key_a, guest_id="guest-owner")
    created = client.post(
        "/api/widget/chat/task/create",
        headers=owner,
        json={"title": "hi", "description": "hi"},
    )
    assert created.status_code == 200, created.text
    task_id = int(created.json()["task_id"])

    upload = [("files", ("f.txt", io.BytesIO(b"x"), "text/plain"))]

    # Same workforce, a different guest -> rejected.
    intruder = _authenticate_widget_guest_by_key(key_a, guest_id="guest-intruder")
    r1 = client.post(
        "/api/widget/files/upload",
        headers=intruder,
        data={"task_type": "task", "task_id": str(task_id)},
        files=upload,
    )
    assert r1.status_code == 403, r1.text

    # A guest of a different workforce (even reusing the guest id) -> rejected.
    cross_wf = _authenticate_widget_guest_by_key(key_b, guest_id="guest-owner")
    r2 = client.post(
        "/api/widget/files/upload",
        headers=cross_wf,
        data={"task_type": "task", "task_id": str(task_id)},
        files=upload,
    )
    assert r2.status_code == 403, r2.text


def test_widget_auth_accepts_legacy_agent_ticket_without_owner_type() -> None:
    """Backward-compat: agent embed tickets minted before workforce support
    carried no ``owner_type`` claim; ``_owner_from_embed_ticket`` must still
    resolve them as agent tickets."""
    from datetime import datetime, timedelta, timezone

    from jose import jwt

    from xagent.web.api.widget import EMBED_TICKET_TYPE
    from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY

    _admin_headers()
    owner_id = _user_id()
    agent_id = _create_published_agent(owner_id, "Legacy Ticket Agent")
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        agent.widget_enabled = True
        agent.widget_key = "legacy-agent-widget-key"
        agent.allowed_domains = ["example.com"]
        db.commit()
    finally:
        db.close()

    legacy_ticket = jwt.encode(
        {
            "type": EMBED_TICKET_TYPE,
            "agent_id": agent_id,
            "embed_origin": "example.com",
            # No "owner_type" -- exactly how pre-workforce tickets looked.
            "exp": datetime.now(timezone.utc) + timedelta(seconds=60),
        },
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )

    auth = client.post(
        "/api/widget/auth",
        json={"guest_id": "guest_test", "embed_ticket": legacy_ticket},
    )
    assert auth.status_code == 200, auth.text
    body = auth.json()
    assert body["agent_id"] == agent_id
    assert body.get("workforce_id") is None
