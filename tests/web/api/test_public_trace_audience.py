"""Widget and share guests must never receive LLM input in LLM trace frames.

``llm_call_start`` carries ``context_preview`` (the last messages sent to the
model: system prompt, uploaded-file context, the execution message). These
tests put canaries in that payload, drive the real guest websocket endpoints,
and assert no frame the guest receives carries them, live and on replay.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Iterator

import pytest
from starlette.testclient import WebSocketTestSession

from xagent.core.agent.trace import (
    TraceAction,
    TraceCategory,
    TraceEvent,
    TraceEventType,
    TraceScope,
)
from xagent.web.api.websocket import manager
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.task import TraceEvent as DatabaseTraceEvent
from xagent.web.models.user import User
from xagent.web.services.task_event_trace_handler import TaskEventTraceHandler

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")

SYSTEM_CANARY = "CANARY-SYSTEM-PROMPT-1a2b"
FILE_CANARY = "CANARY-FILE-CONTEXT-3c4d"
EXECUTION_CANARY = "CANARY-EXECUTION-MESSAGE-5e6f"
CANARIES = (SYSTEM_CANARY, FILE_CANARY, EXECUTION_CANARY)
DONE_TYPE = "public_trace_audience_done"


def _llm_start_data() -> dict[str, Any]:
    return {
        "context_messages_count": 2,
        "tools_count": 1,
        "context_preview": [
            {"role": "system", "content": f"You are helpful. {SYSTEM_CANARY}"},
            {
                "role": "user",
                "content": f"hello\n<file>{FILE_CANARY}</file>\n{EXECUTION_CANARY}",
            },
        ],
        "messages": [{"role": "system", "content": SYSTEM_CANARY}],
        "model_name": "probe-model",
        "context_tokens": 120,
        "context_threshold": 1000,
    }


def _owner_id() -> int:
    _admin_headers()
    db = _direct_db_session()
    try:
        return int(db.query(User).filter(User.username == "admin").one().id)
    finally:
        db.close()


def _create_agent(**extra: Any) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=_owner_id(),
            name=f"Trace Audience {uuid.uuid4().hex[:6]}",
            description="d",
            instructions=SYSTEM_CANARY,
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
            **extra,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return int(agent.id)
    finally:
        db.close()


def _guest_protocols(token: str) -> list[str]:
    """The guest token rides the handshake headers, never the URL."""
    return ["xagent-auth-v1", f"xagent-auth-token.{token}"]


def _widget_guest() -> tuple[str, int, list[str]]:
    agent_id = _create_agent()
    enabled = client.put(
        f"/api/agents/{agent_id}",
        headers=_admin_headers(),
        json={"widget_enabled": True, "allowed_domains": ["*"]},
    )
    assert enabled.status_code == 200, enabled.text
    db = _direct_db_session()
    try:
        widget_key = str(db.query(Agent).filter(Agent.id == agent_id).one().widget_key)
    finally:
        db.close()
    auth = client.post(
        "/api/widget/auth",
        json={"guest_id": "guest-trace-audience", "widget_key": widget_key},
    )
    assert auth.status_code == 200, auth.text
    token = str(auth.json()["access_token"])
    created = client.post(
        "/api/widget/chat/task/create",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": "hello", "description": "hello", "agent_id": agent_id},
    )
    assert created.status_code == 200, created.text
    return (
        f"/api/widget/chat/ws/{created.json()['task_id']}",
        int(created.json()["task_id"]),
        _guest_protocols(token),
    )


def _share_guest() -> tuple[str, int, list[str]]:
    share_token = f"trace-audience-{uuid.uuid4().hex[:8]}"
    _create_agent(share_enabled=True, share_token=share_token)
    auth = client.post("/api/share/auth", json={"share_token": share_token})
    assert auth.status_code == 200, auth.text
    token = str(auth.json()["access_token"])
    created = client.post(
        "/api/share/chat/task/create",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": "hello", "description": "hello"},
    )
    assert created.status_code == 200, created.text
    return (
        f"/api/share/chat/ws/{created.json()['task_id']}",
        int(created.json()["task_id"]),
        _guest_protocols(token),
    )


@pytest.fixture(params=["widget", "share"])
def guest(request: pytest.FixtureRequest) -> Iterator[tuple[str, int, list[str]]]:
    yield _widget_guest() if request.param == "widget" else _share_guest()


def _frames_until(ws: WebSocketTestSession, stop: Any) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    while True:
        text = ws.receive_text()
        for canary in CANARIES:
            assert canary not in text, f"guest received {canary}: {text[:300]}"
        frame = json.loads(text)
        frames.append(frame)
        if stop(frame):
            return frames


def test_live_llm_trace_frame_reaches_guest_without_llm_input(
    guest: tuple[str, int, list[str]],
) -> None:
    url, task_id, protocols = guest

    async def emit() -> None:
        await TaskEventTraceHandler(task_id).handle_event(
            TraceEvent(
                event_type=TraceEventType(
                    TraceScope.ACTION, TraceAction.START, TraceCategory.LLM
                ),
                task_id=str(task_id),
                step_id="step-1",
                data=_llm_start_data(),
            )
        )
        await manager.broadcast_to_task({"type": DONE_TYPE}, task_id)

    with client.websocket_connect(url, subprotocols=protocols) as ws:
        _frames_until(ws, lambda f: True)  # the socket is registered by now
        ws.portal.call(emit)
        frames = _frames_until(ws, lambda f: f.get("type") == DONE_TYPE)

    llm = [f for f in frames if f.get("event_type") == "llm_call_start"]
    assert len(llm) == 1
    data = llm[0]["data"]
    assert data["model_name"] == "probe-model"
    assert data["context_tokens"] == 120
    assert data["context_threshold"] == 1000
    assert "context_preview" not in data
    assert "messages" not in data


def test_historical_llm_trace_row_replays_to_guest_without_llm_input(
    guest: tuple[str, int, list[str]],
) -> None:
    url, task_id, protocols = guest
    db = _direct_db_session()
    try:
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id=str(uuid.uuid4()),
                event_type="llm_call_start",
                timestamp=datetime.now(timezone.utc),
                step_id="step-1",
                data=_llm_start_data(),
            )
        )
        db.commit()
    finally:
        db.close()

    with client.websocket_connect(url, subprotocols=protocols) as ws:
        frames = _frames_until(ws, lambda f: f.get("event_type") == "llm_call_start")

    data = frames[-1]["data"]
    assert data["model_name"] == "probe-model"
    assert "context_preview" not in data
    assert "messages" not in data


class _Socket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    async def close(self, code: int = 1000) -> None:
        pass


def _live_llm_frame(task_id: int) -> dict[str, Any]:
    return {
        "type": "trace_event",
        "event_id": "e-1",
        "event_type": "llm_call_start",
        "task_id": task_id,
        "data": _llm_start_data(),
    }


def _assert_owner_full_guest_projected(owner: _Socket, guest: _Socket) -> None:
    assert len(owner.sent) == len(guest.sent) == 1
    assert SYSTEM_CANARY in owner.sent[0]
    assert json.loads(owner.sent[0])["data"]["context_preview"]
    for canary in CANARIES:
        assert canary not in guest.sent[0]
    assert json.loads(guest.sent[0])["data"]["model_name"] == "probe-model"


@pytest.fixture
def isolated_manager(monkeypatch: pytest.MonkeyPatch) -> Any:
    from xagent.web.api import websocket as websocket_module

    fresh = websocket_module.ConnectionManager()
    monkeypatch.setattr(
        websocket_module, "get_shared_task_execution_enabled", lambda: False
    )
    return fresh


async def test_broadcast_projects_only_public_audience_sockets(
    isolated_manager: Any,
) -> None:
    owner, guest = _Socket(), _Socket()
    isolated_manager.register_connection(owner, 991)
    isolated_manager.register_connection(guest, 991, public_audience=True)
    frame = _live_llm_frame(991)

    await isolated_manager.broadcast_to_task(frame, 991)

    _assert_owner_full_guest_projected(owner, guest)
    assert frame["data"]["context_preview"], "the shared frame must not be mutated"


async def test_shared_delivery_projects_only_public_audience_sockets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.api import websocket as websocket_module

    monkeypatch.setattr(
        websocket_module, "get_shared_task_execution_enabled", lambda: True
    )
    shared = websocket_module.ConnectionManager()
    owner, guest = _Socket(), _Socket()
    shared.register_connection(owner, 992)
    shared.register_connection(guest, 992, public_audience=True)

    await shared.deliver_shared_event(_live_llm_frame(992), 992)
    for _ in range(20):
        if owner.sent and guest.sent:
            break
        await asyncio.sleep(0.01)

    _assert_owner_full_guest_projected(owner, guest)
    for writer in list(shared._writers.values()):
        writer.task.cancel()


async def test_personal_message_stays_projected_after_guest_disconnects(
    isolated_manager: Any,
) -> None:
    guest = _Socket()
    isolated_manager.register_connection(guest, 993, public_audience=True)
    isolated_manager.disconnect(guest)

    await isolated_manager.send_personal_message(_live_llm_frame(993), guest)

    assert len(guest.sent) == 1
    for canary in CANARIES:
        assert canary not in guest.sent[0]


def test_public_audience_frame_keeps_delegated_child_attribution() -> None:
    from xagent.web.services.public_trace_events import (
        DELEGATED_AGENT_TRACE_SOURCE,
        public_audience_frame,
    )

    attribution = {
        "source": DELEGATED_AGENT_TRACE_SOURCE,
        "worker_task_id": "child-1",
        "agent_name": "Researcher",
        "worker_alias": "worker-1",
    }
    frame = _live_llm_frame(1)
    frame["data"].update(attribution)

    data = public_audience_frame(frame)["data"]

    assert attribution.items() <= data.items()
    assert "context_preview" not in data


@pytest.mark.parametrize("scope", list(TraceScope))
@pytest.mark.parametrize("action", list(TraceAction))
def test_every_llm_trace_event_name_is_projected(
    scope: TraceScope, action: TraceAction
) -> None:
    from xagent.web.services.public_trace_events import (
        normalize_public_trace_event,
        public_audience_frame,
    )
    from xagent.web.services.task_event_trace_handler import get_event_type_mapping

    event_type, data = normalize_public_trace_event(
        get_event_type_mapping(
            TraceEvent(
                event_type=TraceEventType(scope, action, TraceCategory.LLM),
                task_id="1",
                step_id="step-1",
                data={},
            )
        ),
        _llm_start_data(),
    )
    frame = {"type": "trace_event", "event_type": event_type, "data": data}

    assert "context_preview" not in public_audience_frame(frame)["data"], event_type


def test_public_audience_frame_keeps_non_llm_frames_intact() -> None:
    from xagent.web.services.public_trace_events import public_audience_frame

    tool_frame = {
        "type": "trace_event",
        "event_type": "tool_execution_start",
        "data": {"tool_name": "calculator", "tool_params": {"x": 1}},
    }
    assert public_audience_frame(tool_frame) == tool_frame
