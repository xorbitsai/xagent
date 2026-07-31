"""Per-guest isolation for public share links (#973, PR1).

A single share link can be opened by many anonymous visitors. Each
``POST /api/share/auth`` mints a fresh, server-owned ``guest_id`` signed into
the guest JWT; every task created by a guest is stamped with that id, and
``get_task_for_share_context`` requires the caller's ``guest_id`` to match the
task's. This prevents guest A — holding a perfectly valid share JWT for the
same shared entity — from reading or continuing guest B's conversation.

Covers both the agent-share and workforce-share paths, plus fail-closed
rejection of legacy tokens that predate the ``guest_id`` claim.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from starlette.websockets import WebSocketDisconnect

from xagent.web.api.public_chat_access import (
    PublicChatAccessContext,
    create_public_chat_access_token,
    get_task_for_public_context,
)
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.task import Task
from xagent.web.models.user import User
from xagent.web.services import workforce_runs as workforce_runs_service
from xagent.web.services.task_runtime import (
    TASK_RUNTIME_BINDINGS_AGENT_CONFIG_KEY,
    task_extension_bindings_from_agent_config,
)
from xagent.web.services.workforce_snapshot import build_workforce_task_config

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _setup_admin,
    _share_guest_id,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")


def _user_id(username: str = "admin") -> int:
    _setup_admin()
    db = _direct_db_session()
    try:
        return int(db.query(User).filter(User.username == username).one().id)
    finally:
        db.close()


def _create_published_agent(name: str, share_token: str) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=_user_id(),
            name=name,
            description="d",
            instructions="i",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
            share_enabled=True,
            share_token=share_token,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return int(agent.id)
    finally:
        db.close()


def _create_workforce(name: str) -> int:
    headers = _admin_headers()
    manager_agent_id = _create_published_agent(f"{name} Manager", f"{name}-mgr-tok")
    worker_agent_id = _create_published_agent(f"{name} Worker", f"{name}-wrk-tok")
    response = client.post(
        "/api/workforces",
        headers=headers,
        json={
            "name": name,
            "description": "isolation tests",
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
    published = client.post(f"/api/workforces/{workforce_id}/publish", headers=headers)
    assert published.status_code == 200, published.text
    return workforce_id


def _enable_workforce_share(workforce_id: int) -> str:
    response = client.post(
        f"/api/workforces/{workforce_id}/share-link", headers=_admin_headers()
    )
    assert response.status_code == 200, response.text
    return str(response.json()["share_token"])


def _authenticate_share_guest(share_token: str) -> dict[str, str]:
    response = client.post("/api/share/auth", json={"share_token": share_token})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _stub_begin_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(**_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(background_task=None)

    monkeypatch.setattr(
        workforce_runs_service.TaskTurnOrchestrator, "begin_turn", _stub
    )


def _upload_to_task(headers: dict[str, str], task_id: int) -> Any:
    return client.post(
        "/api/share/files/upload",
        headers=headers,
        data={"task_type": "task", "task_id": str(task_id)},
        files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")},
    )


# ===== distinct guests get distinct server-minted ids =====


def test_share_auth_mints_distinct_guest_ids_per_call() -> None:
    """Two auths of the *same* link are two independent anonymous guests: the
    server mints a fresh guest id each time (never client-supplied)."""
    agent_id = _create_published_agent("Distinct Guest Agent", "distinct-tok")
    assert agent_id

    first = client.post("/api/share/auth", json={"share_token": "distinct-tok"})
    second = client.post("/api/share/auth", json={"share_token": "distinct-tok"})
    assert first.status_code == 200 and second.status_code == 200
    # Assert on the decoded ``guest_id`` claim itself — the isolation credential —
    # not merely that the two opaque tokens differ (which any varying claim, e.g.
    # a future ``jti``, would satisfy without proving guest ids are distinct).
    first_guest = _share_guest_id(first.json()["access_token"])
    second_guest = _share_guest_id(second.json()["access_token"])
    assert first_guest != second_guest


# ===== agent-share cross-guest isolation =====


def test_agent_share_guest_cannot_touch_other_guests_task() -> None:
    agent_id = _create_published_agent("Iso Agent", "iso-agent-tok")
    guest_a = _authenticate_share_guest("iso-agent-tok")
    guest_b = _authenticate_share_guest("iso-agent-tok")

    created = client.post(
        "/api/share/chat/task/create",
        headers=guest_b,
        json={"title": "b task", "description": "b task"},
    )
    assert created.status_code == 200, created.text
    task_b = int(created.json()["task_id"])

    # Guest B stamped the task; the config carries *B's* minted guest id.
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_b).one()
        assert int(task.agent_id) == agent_id
        assert task.agent_config.get("guest_id") == _share_guest_id(
            guest_b["Authorization"]
        )
    finally:
        db.close()

    # Guest A holds a valid share JWT for the same agent, but must not reach B.
    assert _upload_to_task(guest_a, task_b).status_code == 403
    # Guest B still reaches its own task.
    assert _upload_to_task(guest_b, task_b).status_code == 200


def test_agent_share_ws_connect_denies_foreign_guest() -> None:
    """The WS connect path routes through the same per-guest gate. A guest with
    a valid share JWT for the link still cannot open another guest's task over
    the socket, and the denial arrives as a 4003 close carrying the reason the
    frontend recovery flow keys on (#973).

    This also pins the accept-before-auth ordering: the endpoint accepts the
    handshake first, so the denial is a *post-accept* close and ``TestClient``
    surfaces the ``WebSocketDisconnect`` at ``receive_text()`` (a *pre-accept*
    close would instead raise at context-manager ``__enter__``, outside the
    ``pytest.raises`` scope below, turning the test red). Keeping the ``raises``
    narrowed to ``receive_text()`` is what makes a regression of the ordering
    fail here rather than pass silently.
    """
    _create_published_agent("WS Iso Agent", "ws-iso-tok")
    guest_a = _authenticate_share_guest("ws-iso-tok")
    guest_b = _authenticate_share_guest("ws-iso-tok")

    created = client.post(
        "/api/share/chat/task/create",
        headers=guest_b,
        json={"title": "b ws task", "description": "b ws task"},
    )
    assert created.status_code == 200, created.text
    task_b = int(created.json()["task_id"])

    token_a = guest_a["Authorization"].removeprefix("Bearer ")
    with client.websocket_connect(f"/api/share/chat/ws/{task_b}?token={token_a}") as ws:
        with pytest.raises(WebSocketDisconnect) as denied:
            ws.receive_text()
    assert denied.value.code == 4003
    # Guest mismatch deliberately shares the not-found detail so probing can't
    # enumerate task ids on a share link (#973).
    assert denied.value.reason == "Task not found or access denied"


def test_agent_share_task_without_guest_id_is_denied() -> None:
    """A pre-migration task whose *stored* ``agent_config`` lacks ``guest_id``
    (distinct from a legacy token) is unreachable by any guest: the strict
    inequality in the gate treats a missing id as a mismatch, so even the guest
    that created it is denied once the id is gone."""
    _create_published_agent("PreMig Agent", "premig-tok")
    guest = _authenticate_share_guest("premig-tok")
    created = client.post(
        "/api/share/chat/task/create",
        headers=guest,
        json={"title": "t", "description": "t"},
    )
    assert created.status_code == 200, created.text
    task_id = int(created.json()["task_id"])

    # Simulate a pre-#973 task by dropping guest_id from its stored config.
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        stripped = {k: v for k, v in task.agent_config.items() if k != "guest_id"}
        task.agent_config = stripped
        db.commit()
    finally:
        db.close()

    assert _upload_to_task(guest, task_id).status_code == 403


def test_build_workforce_task_config_never_emits_guest_id() -> None:
    """Workforce-share isolation threads ``guest_id`` via ``extra_agent_config``
    and depends on the snapshot config NOT carrying its own ``guest_id`` (which
    would clobber the server value in ``{**extra_agent_config, **task_config}``).
    Pin that invariant so a future snapshot field can't silently break it."""
    config = build_workforce_task_config({"workforce": {"id": 1}})
    assert "guest_id" not in config


# ===== agent-share task creation is tamper-proof against a forged guest_id =====


def test_agent_share_task_create_ignores_client_supplied_guest_id() -> None:
    """``TaskCreateRequest.agent_config`` is client-controlled, and
    ``create_share_chat_task`` relies on assigning the server keys *after*
    copying it. A guest posting a forged ``guest_id`` in the body must not be
    able to stamp its task with another guest's id — the server-minted value
    always wins. Guards against a future reorder regressing this silently."""
    agent_id = _create_published_agent("Tamper Agent", "tamper-agent-tok")
    assert agent_id
    victim = _authenticate_share_guest("tamper-agent-tok")
    attacker = _authenticate_share_guest("tamper-agent-tok")

    created = client.post(
        "/api/share/chat/task/create",
        headers=attacker,
        json={
            "title": "forged",
            "description": "forged",
            "agent_config": {"guest_id": _share_guest_id(victim["Authorization"])},
        },
    )
    assert created.status_code == 200, created.text
    task_id = int(created.json()["task_id"])

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        # The persisted id is the attacker's own minted id, never the forged
        # victim id from the request body.
        assert task.agent_config.get("guest_id") == _share_guest_id(
            attacker["Authorization"]
        )
        assert task.agent_config.get("guest_id") != _share_guest_id(
            victim["Authorization"]
        )
    finally:
        db.close()

    # And the forged value buys no access to the victim's tasks anyway.
    assert _upload_to_task(attacker, task_id).status_code == 200


def test_agent_share_task_create_drops_forged_runtime_extension_bindings() -> None:
    """Share guests are explicitly denied ``runtime_extensions`` (400), yet the
    client ``agent_config`` copy would still let a guest write the server-owned
    per-task binding record directly. Deletion dispatches by that record, so a
    forged entry naming a broken provider can wedge the owner's task."""
    assert _create_published_agent("Binding Agent", "binding-agent-tok")
    guest = _authenticate_share_guest("binding-agent-tok")

    created = client.post(
        "/api/share/chat/task/create",
        headers=guest,
        json={
            "title": "forged binding",
            "description": "forged binding",
            "agent_config": {
                TASK_RUNTIME_BINDINGS_AGENT_CONFIG_KEY: ["victim_ext"],
                "keep_me": "client value",
            },
        },
    )
    assert created.status_code == 200, created.text
    task_id = int(created.json()["task_id"])

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task_extension_bindings_from_agent_config(task.agent_config) == ()
        # Only the reserved key goes; ordinary client config and the
        # server-owned keys layered on top both survive.
        assert task.agent_config.get("keep_me") == "client value"
        assert task.agent_config.get("auth_mode") == "share"
        assert task.agent_config.get("guest_id") == _share_guest_id(
            guest["Authorization"]
        )
    finally:
        db.close()


# ===== workforce-share cross-guest isolation =====


def test_workforce_share_guest_cannot_touch_other_guests_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workforce_id = _create_workforce("Iso WF")
    token = _enable_workforce_share(workforce_id)
    guest_a = _authenticate_share_guest(token)
    guest_b = _authenticate_share_guest(token)
    _stub_begin_turn(monkeypatch)

    created = client.post(
        "/api/share/chat/task/create",
        headers=guest_b,
        json={"title": "b run", "description": "b run"},
    )
    assert created.status_code == 200, created.text
    task_b = int(created.json()["task_id"])

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_b).one()
        assert task.agent_config.get("guest_id") == _share_guest_id(
            guest_b["Authorization"]
        )
    finally:
        db.close()

    assert _upload_to_task(guest_a, task_b).status_code == 403
    assert _upload_to_task(guest_b, task_b).status_code == 200


def test_workforce_share_task_without_guest_id_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workforce-side mirror of ``test_agent_share_task_without_guest_id_is_denied``.
    Both paths route through the same ``_require_share_guest_owns_task`` gate, so
    a regression that lets a legacy task (stored ``agent_config`` missing
    ``guest_id``) slip through must fail on this side too, not only agent-share."""
    workforce_id = _create_workforce("PreMig WF")
    token = _enable_workforce_share(workforce_id)
    guest = _authenticate_share_guest(token)
    _stub_begin_turn(monkeypatch)

    created = client.post(
        "/api/share/chat/task/create",
        headers=guest,
        json={"title": "wf", "description": "wf"},
    )
    assert created.status_code == 200, created.text
    task_id = int(created.json()["task_id"])

    # Simulate a pre-#973 task by dropping guest_id from its stored config.
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        stripped = {k: v for k, v in task.agent_config.items() if k != "guest_id"}
        task.agent_config = stripped
        db.commit()
    finally:
        db.close()

    assert _upload_to_task(guest, task_id).status_code == 403


# ===== fail-closed on legacy tokens without a guest_id claim =====


def test_legacy_agent_share_token_without_guest_id_is_rejected() -> None:
    agent_id = _create_published_agent("Legacy Agent", "legacy-agent-tok")
    legacy = create_public_chat_access_token(
        {
            "sub": "admin",
            "user_id": _user_id(),
            "auth_mode": "share",
            "share_agent_id": agent_id,
            "share_token": "legacy-agent-tok",
        }
    )
    response = client.post(
        "/api/share/chat/task/create",
        headers={"Authorization": f"Bearer {legacy}"},
        json={"title": "hi", "description": "hi"},
    )
    assert response.status_code == 401, response.text


def test_share_token_with_whitespace_guest_id_is_rejected() -> None:
    """A whitespace-only guest_id is treated as absent (fail-closed): it could
    never match a server-minted token_urlsafe id and must not pass."""
    agent_id = _create_published_agent("Whitespace Agent", "whitespace-tok")
    forged = create_public_chat_access_token(
        {
            "sub": "admin",
            "user_id": _user_id(),
            "auth_mode": "share",
            "share_agent_id": agent_id,
            "share_token": "whitespace-tok",
            "guest_id": "   ",
        }
    )
    response = client.post(
        "/api/share/chat/task/create",
        headers={"Authorization": f"Bearer {forged}"},
        json={"title": "hi", "description": "hi"},
    )
    assert response.status_code == 401, response.text


def test_legacy_workforce_share_token_without_guest_id_is_rejected() -> None:
    workforce_id = _create_workforce("Legacy WF")
    token = _enable_workforce_share(workforce_id)
    legacy = create_public_chat_access_token(
        {
            "sub": "admin",
            "user_id": _user_id(),
            "auth_mode": "share",
            "share_workforce_id": workforce_id,
            "share_token": token,
        }
    )
    response = client.post(
        "/api/share/chat/task/create",
        headers={"Authorization": f"Bearer {legacy}"},
        json={"title": "hi", "description": "hi"},
    )
    assert response.status_code == 401, response.text


# ===== enumeration-oracle collapse on the widget gates (#973) =====


class _StubQuery:
    def __init__(self, task: object | None) -> None:
        self._task = task

    def filter(self, *_criteria: object) -> _StubQuery:
        return self

    def first(self) -> object | None:
        return self._task


class _StubDb:
    """Minimal Session double: every Task lookup resolves to one canned row."""

    def __init__(self, task: object | None) -> None:
        self._task = task

    def query(self, *_entities: object) -> _StubQuery:
        return _StubQuery(self._task)


@pytest.mark.parametrize(
    ("context_kwargs", "mismatch_task"),
    [
        pytest.param(
            {"widget_agent_id": 1},
            SimpleNamespace(
                agent_config={"guest_id": "guest-b"},
                agent_id=1,
                channel_id=None,
            ),
            id="widget-agent",
        ),
        pytest.param(
            {"widget_workforce_id": 7},
            SimpleNamespace(
                agent_config={
                    "auth_mode": "widget",
                    "guest_id": "guest-b",
                    "widget_workforce_id": 7,
                },
                channel_id=None,
            ),
            id="widget-workforce",
        ),
    ],
)
def test_widget_gate_guest_mismatch_matches_not_found_detail(
    context_kwargs: dict[str, Any], mismatch_task: SimpleNamespace
) -> None:
    """Widget-side mirrors of the share-gate detail assertion (#973).

    All three ownership gates collapsed their guest-mismatch 403 into the
    generic not-found detail; the share gate pins this over a real WS
    handshake above, and these pin the widget and workforce-widget gates. The
    denial for "task exists but belongs to another guest" must stay
    byte-identical to "task does not exist", or a probing visitor can
    enumerate which task ids live behind the widget key.
    """
    context = PublicChatAccessContext(
        user=SimpleNamespace(id=1),
        channel_id=None,
        guest_id="guest-a",
        **context_kwargs,
    )

    with pytest.raises(HTTPException) as mismatch:
        get_task_for_public_context(_StubDb(mismatch_task), 41, context)
    with pytest.raises(HTTPException) as missing:
        get_task_for_public_context(_StubDb(None), 41, context)

    assert mismatch.value.status_code == 403
    assert mismatch.value.detail == "Task not found or access denied"
    assert (mismatch.value.status_code, mismatch.value.detail) == (
        missing.value.status_code,
        missing.value.detail,
    )
