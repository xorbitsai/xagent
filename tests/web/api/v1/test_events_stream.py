"""Tests for the v1 SSE transport layer (``GET /v1/chat/tasks/{id}/events``).

Covers registration, the sink's frame filter and exception discipline,
the watchdog, the 1-hour cap, concurrency caps, and the ``step.*`` /
``message.*`` content projection layered on top of that transport
(classification, filtering, and the per-frame content caps). Tests
either drive ``_events_stream`` directly (fast,
deterministic -- used whenever a test needs injected short intervals or
precise control over the race between the watchdog and the deadline) or
go through the real HTTP endpoint (used for the 401/404/429/terminal-
attach shapes, where the JSON-vs-event-stream distinction actually
matters).
"""

import asyncio
import json
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from xagent.core.agent.trace import (
    ACTION_END_TOOL,
    ACTION_START_TOOL,
    AI_MESSAGE,
    TraceAction,
    TraceCategory,
)
from xagent.core.agent.trace import TraceEvent as CoreTraceEvent
from xagent.core.agent.trace import (
    TraceEventType,
    TraceScope,
)
from xagent.core.tools.adapters.vibe.connector_runtime import (
    REDACTED_RUNTIME_SECRET,
    redact_runtime_sensitive_payload,
)
from xagent.web.api.chat import chat_router
from xagent.web.api.trace_handlers import (
    DatabaseTraceHandler,
    _convert_float_to_datetime,
)
from xagent.web.api.v1 import _events_stream as es
from xagent.web.api.v1 import tasks as v1_tasks
from xagent.web.api.v1.deps import ApiKeyPrincipal, _resolve_principal_from_credentials
from xagent.web.api.v1.errors import V1ApiError, V1ErrorCode
from xagent.web.api.ws_trace_handlers import (
    WebSocketTraceHandler,
    get_event_type_mapping,
)
from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.task import Task, TaskStatus, TraceEvent

from ..conftest import (
    _admin_headers,
    _direct_db_session,
    _install_one_slot_queue_pool,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")

# ``app_for_tests`` (the shared v1 suite app in ``..conftest``) deliberately
# doesn't mount ``chat_router`` -- it's scoped to the v1 routers this suite
# actually tests. The one test here that needs the real production task-
# delete route (``DELETE /api/chat/task/{task_id}``) gets its own minimal
# app for just that router instead, mirroring the same narrow-app pattern
# ``test_chat_task_model_ids.py`` already uses. ``get_db`` needs no override
# here: the real dependency reads from the same process-global session
# factory ``_test_db`` initializes, so this app shares the exact DB rows
# the v1 SSE tests create through ``app_for_tests``.
_chat_delete_app = FastAPI()
_chat_delete_app.include_router(chat_router)
_chat_delete_client = TestClient(_chat_delete_app, raise_server_exceptions=False)


# ===== local helpers (mirrors test_tasks.py's pattern) =====


@pytest.fixture(autouse=True)
def mock_start_task():
    """Every test here creates tasks through the real POST endpoint (so
    ownership/source="sdk" scoping is authentic) but never wants an
    actual agent execution to run."""
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


@pytest.fixture(autouse=True)
def reset_principal_stream_counts():
    es.reset_principal_stream_counts_for_testing()
    yield
    es.reset_principal_stream_counts_for_testing()


def _create_agent_with_key() -> tuple[int, str]:
    headers = _admin_headers()
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "v1 events test agent",
            "description": "test",
            "instructions": "you are a test agent",
            "execution_mode": "balanced",
        },
    )
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = agent_resp.json()["id"]
    key_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    return agent_id, key_resp.json()["full_key"]


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


def _create_task(full_key: str, agent_id: int, content: str = "hello") -> int:
    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": content}},
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["task_id"]


def _principal_for(full_key: str) -> ApiKeyPrincipal:
    """Resolve a real principal through the production auth path (bcrypt
    included) rather than hand-rolling a snapshot -- keeps tests honest
    about what ``get_principal_from_api_key`` actually returns."""
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=full_key)
    return _resolve_principal_from_credentials(creds)


def _set_task_status(
    task_id: int,
    status: TaskStatus,
    *,
    output: str | None = None,
    error_message: str | None = None,
    control_state: str | None = None,
) -> None:
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.status = status
        if output is not None:
            task.output = output
        if error_message is not None:
            task.error_message = error_message
        if control_state is not None:
            task.control_state = control_state
        db.commit()
    finally:
        db.close()


def _key_prefix_for_agent(agent_id: int) -> str:
    db = _direct_db_session()
    try:
        return str(
            db.query(AgentApiKey)
            .filter(AgentApiKey.agent_id == agent_id)
            .one()
            .key_prefix
        )
    finally:
        db.close()


def _make_sink(task_id: int = 1, status: str = "running") -> es.V1EventStreamSink:
    return es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix="pfx-test", initial_status=status
    )


def _insert_trace_event(
    *,
    task_id: int,
    event_type: str,
    event_id: str,
    timestamp: datetime,
    data: dict,
    step_id: str | None = None,
    build_id: str | None = None,
) -> None:
    """Insert one ``TraceEvent`` row directly via the test DB.

    Mirrors ``test_tasks.py``'s helper of the same name -- bypasses the
    production trace handler (which runs through asyncio + a thread
    pool) so a test can set up "this step already happened before
    attach" history without spinning up the agent runtime.
    """
    db = _direct_db_session()
    try:
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id=event_id,
                event_type=event_type,
                timestamp=timestamp,
                step_id=step_id,
                build_id=build_id,
                data=data,
            )
        )
        db.commit()
    finally:
        db.close()


def _insert_question_message(task_id: int, *, content: str) -> None:
    """Write one assistant question row directly, bypassing the WS writer.

    Mirrors ``test_tasks.py``'s helper of the same name and the row
    shape ``_persist_agent_outbound_event`` (``api/websocket.py``) writes
    for ``expect_response`` outbound events: ``role='assistant'``,
    ``message_type='question'``. This is what makes
    ``task_interaction_read.get_pending_interaction_question`` -- and
    therefore ``_TaskInfoSnapshot.pending_question`` -- return
    non-``None``.
    """
    from xagent.web.models.chat_message import TaskChatMessage

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        db.add(
            TaskChatMessage(
                task_id=task_id,
                user_id=int(task.user_id),
                role="assistant",
                content=content,
                message_type="question",
                turn_id=f"question-{task_id}",
            )
        )
        db.commit()
    finally:
        db.close()


def _long_intervals(**overrides: float) -> dict[str, float]:
    """Defaults for ``build_event_stream_response``/``_generate``'s three
    tunable intervals (watchdog / absolute deadline / heartbeat): 1000s
    is long enough that none of them fire within a test's timeout unless
    a test overrides one explicitly to trigger specific timing behavior."""
    return {
        "watchdog_interval_seconds": 1000,
        "stream_max_duration_seconds": 1000,
        "heartbeat_interval_seconds": 1000,
        **overrides,
    }


# ===== error_frame code validation =====


def test_error_frame_rejects_an_unknown_code_without_reaching_the_wire():
    """``_ERROR_MESSAGES[code]`` is looked up unconditionally, so an
    unrecognized code raises ``KeyError`` instead of producing a
    ``stream.error`` frame whose ``message`` a client cannot render."""
    with pytest.raises(KeyError):
        es.error_frame("nope")

    frame = es.error_frame("task_deleted")
    data = json.loads(frame.split("data: ", 1)[1])
    assert data == {
        "code": "task_deleted",
        "message": es._ERROR_MESSAGES["task_deleted"],
    }


# ===== sink.send_text never raises =====


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "42",
        "null",
        "[]",
        json.dumps({"type": "task_completed", "task": "not-a-dict"}),
        json.dumps(
            {"type": "task_started", "task_id": "not-an-int", "status": "running"}
        ),
    ],
)
async def test_sink_send_text_never_raises_on_malformed_input(raw):
    sink = _make_sink()
    await sink.send_text(raw)  # must not raise


async def test_sink_send_text_never_raises_when_classification_throws(monkeypatch):
    sink = _make_sink()

    def _boom(message):
        raise RuntimeError("boom")

    monkeypatch.setattr(es, "_is_versioned_task_event", _boom)
    before = sink.dropped_frame_count
    await sink.send_text(
        json.dumps(
            {"type": "task_started", "task_id": sink.task_id, "status": "running"}
        )
    )  # must not raise
    assert sink.dropped_frame_count == before + 1


async def test_sink_send_text_never_raises_from_foreign_loop():
    sink = _make_sink()
    sink._owner_loop = object()  # simulate a foreign event loop
    await sink.send_text(
        json.dumps(
            {"type": "task_started", "task_id": sink.task_id, "status": "running"}
        )
    )  # must not raise
    assert sink.dropped_frame_count == 1
    assert sink.queue.empty()


# ===== unauthorized / not-owned get plain JSON, no stream bytes =====


def test_events_unauthorized_401():
    resp = client.get("/v1/chat/tasks/1/events")
    assert resp.status_code == 401
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json()["error"]["code"] == "invalid_api_key"


def test_events_not_owned_404():
    _, full_key = _create_agent_with_key()
    resp = client.get("/v1/chat/tasks/999999/events", headers=_bearer(full_key))
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json()["error"]["code"] == "task_not_found"


# ===== attach on an already-terminal task =====


def test_events_terminal_task_immediate_close():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _set_task_status(task_id, TaskStatus.COMPLETED, output="the answer")

    resp = client.get(f"/v1/chat/tasks/{task_id}/events", headers=_bearer(full_key))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert body.count("event: task.status") == 1
    assert body.count("event: task.completed") == 1
    blocks = [b for b in body.split("\n\n") if b.strip()]
    status_data = json.loads(blocks[0].split("data: ", 1)[1])
    completed_data = json.loads(blocks[1].split("data: ", 1)[1])
    assert status_data == {"status": "completed"}
    assert completed_data == {
        "status": "completed",
        "output": "the answer",
        "error": None,
    }
    # No sink is registered for the terminal fast path -- nothing to close.
    assert es.count_task_sinks(task_id) == 0


# ===== attach on a task already waiting for user input =====


@pytest.mark.timeout(10)
def test_events_waiting_for_user_attach_takes_the_fast_path():
    """The attach-time fast path for a task that's already
    ``waiting_for_user`` (and not mid-resume): emits ``task.status`` +
    ``task.input_required`` and closes immediately, instead of waiting
    out the watchdog's first cycle (up to 30s in production) to reach
    the same conclusion.

    ``client.get()`` here is a blocking call with no timeout of its own;
    the fast path not registering a sink or closing the stream is the
    only thing keeping it from hanging. A regression that fell through
    to the normal streaming path instead would make this test hang the
    whole suite rather than fail it, so it's pinned with an explicit
    timeout (``pytest-timeout``, already a dev dependency) rather than
    relying on there being no bug."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _set_task_status(task_id, TaskStatus.WAITING_FOR_USER, control_state="idle")

    resp = client.get(f"/v1/chat/tasks/{task_id}/events", headers=_bearer(full_key))
    assert resp.status_code == 200
    body = resp.text
    assert body.count("event: task.status") == 1
    assert body.count("event: task.input_required") == 1
    assert body.count("event: task.completed") == 0
    blocks = [b for b in body.split("\n\n") if b.strip()]
    status_data = json.loads(blocks[0].split("data: ", 1)[1])
    input_required_data = json.loads(blocks[1].split("data: ", 1)[1])
    assert status_data == {"status": "waiting_for_user"}
    assert input_required_data == {"task_id": task_id, "prompt": None}
    # No sink is registered for this fast path either -- nothing to close.
    assert es.count_task_sinks(task_id) == 0


async def test_events_paused_attach_does_not_take_a_fast_path():
    """A ``paused`` task is not ``waiting_for_user``, so attach must fall
    through to the normal streaming path (sink registered, watchdog
    running) instead of the fast path above -- the two tests pin the
    fast path's condition from both sides."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    _set_task_status(task_id, TaskStatus.PAUSED)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        **_long_intervals(),
    )
    first = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
    assert json.loads(first.split("data: ", 1)[1]) == {"status": "paused"}
    assert es.count_task_sinks(task_id) == 1

    await resp.body_iterator.aclose()
    assert es.count_task_sinks(task_id) == 0


# ===== all three response-construction sites disable proxy buffering =====


@pytest.mark.parametrize(
    "task_state",
    ["running_attach", "terminal_fast_path", "waiting_for_user_fast_path"],
)
async def test_sse_responses_disable_proxy_buffering(task_state):
    """``build_event_stream_response`` constructs a ``StreamingResponse``
    from three different call sites -- the running-attach path (normal
    streaming, sink + watchdog), the terminal fast path, and the
    waiting-for-user fast path -- and all three must carry the same
    anti-buffering headers. nginx buffers proxied responses by default;
    without ``X-Accel-Buffering: no`` it would hold SSE frames back
    instead of forwarding them to the client immediately."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)

    if task_state == "terminal_fast_path":
        _set_task_status(task_id, TaskStatus.COMPLETED, output="done")
    elif task_state == "waiting_for_user_fast_path":
        _set_task_status(task_id, TaskStatus.WAITING_FOR_USER, control_state="idle")

    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)
    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        **_long_intervals(),
    )
    try:
        assert resp.headers["x-accel-buffering"] == "no"
        assert resp.headers["cache-control"] == "no-cache"
    finally:
        await resp.body_iterator.aclose()


# ===== bounded outbound queue, overflow closes with resync_required =====


async def test_slow_consumer_queue_overflow_closes_with_resync_required():
    sink = _make_sink()
    for i in range(es.OUTBOUND_QUEUE_MAX_SIZE + 5):
        sink._put_or_overflow(f"frame-{i}")
    assert sink.closing is True
    # Memory doesn't grow past the cap even under sustained overflow --
    # exactly one frame (the close frame) survives.
    assert sink.queue.qsize() == 1
    assert sink.queue.get_nowait() == (es.error_frame("resync_required"), True)


def _budget_sized_frame() -> tuple[str, int]:
    """One 64 KiB frame plus how many of them fit exactly in the budget.

    Sized so the byte budget is what the tests below exercise: at 64 KiB
    a frame, the budget holds 64 of them, well under
    ``OUTBOUND_QUEUE_MAX_SIZE`` -- asserted rather than assumed, so this
    can never silently decay into another item-count test if either
    constant moves.
    """
    frame = "x" * (64 * 1024)
    fits = es.MAX_QUEUED_WIRE_BYTES // len(frame)
    assert fits * len(frame) == es.MAX_QUEUED_WIRE_BYTES, (
        "premise: the budget is a whole multiple of this frame size"
    )
    assert fits < es.OUTBOUND_QUEUE_MAX_SIZE, (
        "premise: the byte budget must bind before the item-count cap"
    )
    return frame, fits


async def test_backlog_exactly_at_the_byte_budget_does_not_close():
    """The budget is an upper bound, not an exclusive one: a backlog whose
    queued wire bytes land exactly on ``MAX_QUEUED_WIRE_BYTES`` keeps
    streaming, and only the frame that would push it over closes. Same
    boundary rule ``MAX_RAW_FRAME_TEXT_CHARS`` uses -- strictly over is
    the event, exactly equal is not."""
    sink = _make_sink()
    frame, fits = _budget_sized_frame()
    for _ in range(fits):
        sink._put_or_overflow(frame)
    assert sink.closing is False
    assert sink.queued_wire_bytes == es.MAX_QUEUED_WIRE_BYTES
    assert sink.queue.qsize() == fits


async def test_byte_budget_overflow_closes_with_resync_required_and_drains():
    """One frame past the budget closes the stream through the existing
    overflow path: the backlog is dropped, the ``resync_required`` close
    frame is the only thing left queued, and the byte accounting returns
    to zero. The item-count cap is not reached here (see
    ``_budget_sized_frame``'s premise assertions), so this closure can
    only have come from the byte budget."""
    sink = _make_sink()
    frame, fits = _budget_sized_frame()
    for _ in range(fits + 1):
        sink._put_or_overflow(frame)
    assert sink.closing is True
    assert sink.queue.qsize() == 1
    assert await sink.next_frame() == (es.error_frame("resync_required"), True)
    assert sink.queued_wire_bytes == 0


async def test_queued_wire_bytes_falls_back_as_frames_are_read_out():
    """The budget bounds what is *currently* queued, not what has ever
    been queued: a consumer that keeps up never accumulates toward it.
    Pins that ``next_frame`` -- the sink's own accounted read method --
    is what discounts a delivered frame's bytes."""
    sink = _make_sink()
    sink._put_or_overflow("a" * 100)
    sink._put_or_overflow("b" * 50)
    assert sink.queued_wire_bytes == 150
    assert await sink.next_frame() == ("a" * 100, False)
    assert sink.queued_wire_bytes == 50
    assert await sink.next_frame() == ("b" * 50, False)
    assert sink.queued_wire_bytes == 0


async def test_generator_read_path_keeps_queued_wire_bytes_at_zero_between_frames():
    """Division of labor with the two tests above: those pin
    ``next_frame``'s own body (deleting the subtraction inside it turns
    them red). This one pins the call *site* -- that ``_generate``'s
    per-frame read actually goes through ``next_frame`` rather than a
    bare ``sink.queue.get()`` that would bypass the byte accounting
    entirely. Runs a real attach, then puts and reads one
    budget-sized frame at a time -- never letting more than one frame's
    worth of backlog build up -- and checks after every read that the
    delivered frame is the one just queued (not a ``resync_required``
    close) and that ``queued_wire_bytes`` is back at zero. Precondition:
    ``fits < OUTBOUND_QUEUE_MAX_SIZE`` (asserted in
    ``_budget_sized_frame``), so the element cap never intervenes and
    every read in this loop can only be explained by the byte budget's
    own accounting.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        **_long_intervals(),
    )
    first = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
    assert "event: task.status" in first

    sink = next(
        connection
        for connection in es.manager.connections_for_task(task_id)
        if isinstance(connection, es.V1EventStreamSink)
    )
    frame, fits = _budget_sized_frame()
    for _ in range(fits + 4):
        sink._put_or_overflow(frame)
        delivered = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert delivered == frame
        assert sink.queued_wire_bytes == 0
    assert sink.closing is False

    await resp.body_iterator.aclose()


# ===== key revoked/paused closes within one watchdog cycle =====


@pytest.mark.parametrize("field", ["revoked_at", "paused_at"])
async def test_watchdog_closes_within_one_cycle_on_key_invalidation(field):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)

    db = _direct_db_session()
    try:
        key_row = db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).one()
        setattr(key_row, field, datetime.now(UTC))
        db.commit()
    finally:
        db.close()

    sink = es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix=key_prefix, initial_status="running"
    )
    closed = await es.watchdog_check_once(
        sink, task_id, principal, read_task_snapshot=v1_tasks._load_task_info_snapshot
    )
    assert closed is True
    assert sink.closing is True
    assert sink.queue.get_nowait() == (es.error_frame("unauthorized"), True)


async def test_watchdog_paused_does_not_close():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)
    _set_task_status(task_id, TaskStatus.PAUSED)

    sink = es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix=key_prefix, initial_status="running"
    )
    closed = await es.watchdog_check_once(
        sink, task_id, principal, read_task_snapshot=v1_tasks._load_task_info_snapshot
    )
    assert closed is False
    assert sink.closing is False
    assert sink.queue.get_nowait() == (es.status_frame("paused"), False)


async def test_watchdog_waiting_for_user_closes_with_null_prompt_when_no_question():
    """The watchdog is the only trigger this transport layer implements
    for input_required. Its ``prompt`` comes from
    ``_TaskInfoSnapshot.pending_question`` -- when the task has no
    persisted assistant question (this test's setup), that snapshot
    field is itself ``None``, so the frame's prompt is null too. The
    companion test below pins the opposite case."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)
    _set_task_status(task_id, TaskStatus.WAITING_FOR_USER, control_state="idle")

    sink = es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix=key_prefix, initial_status="running"
    )
    closed = await es.watchdog_check_once(
        sink, task_id, principal, read_task_snapshot=v1_tasks._load_task_info_snapshot
    )
    assert closed is True
    assert sink.closing is True
    frame_text, is_close = sink.queue.get_nowait()
    assert is_close is True
    assert frame_text.startswith("event: task.input_required\n")
    assert json.loads(frame_text.split("data: ", 1)[1]) == {
        "task_id": task_id,
        "prompt": None,
    }


async def test_watchdog_waiting_for_user_closes_with_the_pending_question_as_prompt():
    """When the task *does* have a persisted assistant question, the
    watchdog's close frame carries it as ``prompt`` -- sourced from
    ``_TaskInfoSnapshot.pending_question`` (``get_latest_waiting_question``),
    the same authoritative row read the close decision itself already
    uses. No agent_message frame sniffing is involved."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)
    _set_task_status(task_id, TaskStatus.WAITING_FOR_USER, control_state="idle")
    _insert_question_message(task_id, content="Where are you flying to?")

    sink = es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix=key_prefix, initial_status="running"
    )
    closed = await es.watchdog_check_once(
        sink, task_id, principal, read_task_snapshot=v1_tasks._load_task_info_snapshot
    )
    assert closed is True
    frame_text, is_close = sink.queue.get_nowait()
    assert is_close is True
    assert json.loads(frame_text.split("data: ", 1)[1]) == {
        "task_id": task_id,
        "prompt": "Where are you flying to?",
    }


def test_fast_path_attach_carries_the_pending_question_as_prompt():
    """Same ``prompt`` sourcing as the watchdog test above, but through
    the attach-time fast path (``_input_required_snapshot_stream``) --
    the other of the two callers ``input_required_frame`` has."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _set_task_status(task_id, TaskStatus.WAITING_FOR_USER, control_state="idle")
    _insert_question_message(task_id, content="Which city?")

    resp = client.get(f"/v1/chat/tasks/{task_id}/events", headers=_bearer(full_key))
    assert resp.status_code == 200
    blocks = [b for b in resp.text.split("\n\n") if b.strip()]
    input_required_data = json.loads(blocks[1].split("data: ", 1)[1])
    assert input_required_data == {"task_id": task_id, "prompt": "Which city?"}


async def test_watchdog_waiting_for_user_with_resume_requested_does_not_close():
    """The ``resume_requested`` carve-out: a task about to resume isn't
    "stuck waiting" even though its status hasn't flipped yet."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)
    _set_task_status(
        task_id, TaskStatus.WAITING_FOR_USER, control_state="resume_requested"
    )

    sink = es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix=key_prefix, initial_status="running"
    )
    closed = await es.watchdog_check_once(
        sink, task_id, principal, read_task_snapshot=v1_tasks._load_task_info_snapshot
    )
    assert closed is False
    assert sink.closing is False


async def test_watchdog_terminal_task_row_closes_with_completed():
    """The watchdog itself is the authoritative source for task.completed,
    independent of the attach-time fast path for an already-terminal
    task or the broadcast acceleration hint."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)
    _set_task_status(task_id, TaskStatus.FAILED, error_message="boom")

    sink = es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix=key_prefix, initial_status="running"
    )
    closed = await es.watchdog_check_once(
        sink, task_id, principal, read_task_snapshot=v1_tasks._load_task_info_snapshot
    )
    assert closed is True
    assert sink.queue.get_nowait() == (
        es.completed_frame(status="failed", output=None, error="boom"),
        True,
    )


async def test_watchdog_missing_task_row_closes_with_task_deleted():
    """A task row that vanishes out from under an open stream
    (hard-deleted) surfaces as task_deleted, not a hang or a 500."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)

    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).delete()
        db.commit()
    finally:
        db.close()

    sink = es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix=key_prefix, initial_status="running"
    )
    closed = await es.watchdog_check_once(
        sink, task_id, principal, read_task_snapshot=v1_tasks._load_task_info_snapshot
    )
    assert closed is True
    assert sink.queue.get_nowait() == (es.error_frame("task_deleted"), True)


async def test_real_delete_route_closes_stream_with_task_deleted():
    """Same ``task_deleted`` close path as the test above, but the row
    disappears through the real production delete route
    (``DELETE /api/chat/task/{task_id}``) instead of a raw DB delete --
    exercising the actual code path an operator or the SDK would trigger,
    not just the watchdog's read side of a row that's already gone."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        **_long_intervals(watchdog_interval_seconds=0.01),
    )
    first = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
    assert "event: task.status" in first

    delete_resp = _chat_delete_client.delete(
        f"/api/chat/task/{task_id}", headers=_admin_headers()
    )
    assert delete_resp.status_code == 200, delete_resp.text

    frames = []

    async def _drain() -> None:
        async for frame in resp.body_iterator:
            frames.append(frame)

    await asyncio.wait_for(_drain(), timeout=2)
    assert "event: stream.error" in frames[-1]
    assert "task_deleted" in frames[-1]
    assert es.count_task_sinks(task_id) == 0


async def test_watchdog_survives_transient_check_failure_and_still_closes():
    """A single failed watchdog cycle (e.g. a transient DB error reading
    the task row) must not silence the watchdog for the rest of the
    stream's life: an exception escaping the loop would kill its
    background task permanently, leaving nothing to close the stream
    until the 1-hour absolute cap. Each cycle therefore catches and
    logs its own failures and retries on the next tick. This injects a
    ``read_task_snapshot``
    that raises on its first call and returns a real, terminal
    snapshot on its second, and asserts the stream still reaches
    ``task.completed`` -- i.e. the watchdog retried instead of dying."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    call_count = 0

    def flaky_reader(task_id_, principal_):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("boom - transient read failure")
        return v1_tasks._load_task_info_snapshot(task_id_, principal_)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=flaky_reader,
        **_long_intervals(watchdog_interval_seconds=0.01),
    )
    first = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
    assert "event: task.status" in first

    _set_task_status(task_id, TaskStatus.COMPLETED, output="done")

    frames = []

    async def _drain():
        async for frame in resp.body_iterator:
            frames.append(frame)

    await asyncio.wait_for(_drain(), timeout=2)
    assert call_count >= 2  # the first (failing) cycle and the retry
    assert any("event: task.completed" in f for f in frames)
    assert es.count_task_sinks(task_id) == 0


# ===== a live stream never holds a connection-pool slot open =====


async def test_live_stream_holds_no_connection_pool_slot(monkeypatch):
    """A live SSE stream must not occupy a pooled DB connection for its
    lifetime: every read the stream itself does (the periodic watchdog
    checks) goes through ``run_db_io_cancellation_safe``, which checks a
    connection out and back in per read, not once for the whole stream.

    Calls the route handler ``stream_chat_task_events`` itself (not
    ``build_event_stream_response``) with ``task_id``/``principal`` passed
    directly as keyword arguments -- this runs the handler's exact body
    (its own snapshot read, then the call into ``_events_stream``) without
    going through FastAPI's HTTP/dependency-injection layer, so a session
    opened anywhere in that body, not only inside ``_events_stream``,
    would show up in the assertion below.

    Proven behaviorally, not by inspecting the endpoint's signature:
    rebind the test database onto a real one-connection pool
    (``_install_one_slot_queue_pool``), open a live stream, and -- while
    it's still open with its first frame already delivered, the moment a
    held connection would show up -- assert the pool has nothing checked
    out. A second, independent connection is then actually pulled from
    the pool to prove the one slot is genuinely free, not merely under
    some checkout-count threshold that would pass even with a stale
    reference still pinning it.

    This replaces a signature-only check that asserted no parameter was
    typed as ``Session``: true, but blind to a session opened inside the
    function body instead of injected as a dependency.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)

    engine = _install_one_slot_queue_pool(monkeypatch, pool_timeout=0.5)

    resp = await v1_tasks.stream_chat_task_events(task_id=task_id, principal=principal)
    first = await resp.body_iterator.__anext__()
    assert "event: task.status" in first
    assert es.count_task_sinks(task_id) == 1

    assert engine.pool.checkedout() == 0
    held = engine.connect()
    try:
        assert engine.pool.checkedout() == 1
    finally:
        held.close()

    await resp.body_iterator.aclose()
    assert es.count_task_sinks(task_id) == 0
    engine.dispose()


# ===== generator teardown unregisters the sink =====


async def test_sink_unregistered_after_generator_teardown():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        **_long_intervals(),
    )
    first = await resp.body_iterator.__anext__()
    assert "event: task.status" in first
    assert es.count_task_sinks(task_id) == 1

    # Simulate what Starlette does on client disconnect / response teardown.
    await resp.body_iterator.aclose()
    assert es.count_task_sinks(task_id) == 0


# ===== a generator that's built but never started leaks nothing =====


async def test_unstarted_generator_leaves_no_registration_or_reservation():
    """An async generator's body doesn't run until it's first iterated.
    Registering the sink and reserving the per-principal slot both
    happen *inside* ``_generate``'s own ``try`` (not at
    ``build_event_stream_response`` construction time) specifically so
    that a ``StreamingResponse`` that's constructed but never iterated
    -- ``__anext__`` never called even once -- leaves nothing behind:
    no sink in ``manager``, no held slot in the per-principal counter.
    Before that fix, registration/reservation ran at construction time,
    so this exact sequence (build, then ``aclose()`` with zero reads)
    leaked both forever, since the only code that ever released them
    lived inside the generator body that never got to run."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        **_long_intervals(),
    )
    # No __anext__() call at all -- the generator body has never run.
    assert es.count_task_sinks(task_id) == 0
    assert es._principal_stream_counts.get(key_prefix, 0) == 0

    await resp.body_iterator.aclose()
    assert es.count_task_sinks(task_id) == 0
    assert es._principal_stream_counts.get(key_prefix, 0) == 0


# ===== per-principal stream cap =====


async def test_try_reserve_principal_slot_caps_at_limit():
    key_prefix = "pfx-cap-unit-test"
    for _ in range(es.PER_PRINCIPAL_STREAM_CAP):
        assert es.try_reserve_principal_slot(key_prefix) is True
    assert es.try_reserve_principal_slot(key_prefix) is False
    for _ in range(es.PER_PRINCIPAL_STREAM_CAP):
        es.release_principal_slot(key_prefix)
    assert es.try_reserve_principal_slot(key_prefix) is True
    es.release_principal_slot(key_prefix)


def test_events_principal_cap_returns_429_json():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    key_prefix = _key_prefix_for_agent(agent_id)
    for _ in range(es.PER_PRINCIPAL_STREAM_CAP):
        assert es.try_reserve_principal_slot(key_prefix)

    resp = client.get(f"/v1/chat/tasks/{task_id}/events", headers=_bearer(full_key))
    assert resp.status_code == 429
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json()["error"]["code"] == "rate_limited"
    assert es.count_task_sinks(task_id) == 0  # rejected before registration


async def test_principal_slot_released_after_real_stream_teardown():
    """A real attach through ``build_event_stream_response`` (not the
    raw counter functions in isolation) must actually give its slot
    back on teardown -- otherwise every stream a key ever opens
    permanently eats one of its 32 concurrent slots."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        **_long_intervals(),
    )
    await resp.body_iterator.__anext__()
    assert es._principal_stream_counts.get(key_prefix, 0) == 1

    await resp.body_iterator.aclose()
    assert es._principal_stream_counts.get(key_prefix, 0) == 0


# ===== per-principal cap is soft: a lost reservation race doesn't 429 =====


async def test_generate_serves_stream_when_reservation_race_lost(caplog):
    """``build_event_stream_response`` only *checks* the per-principal cap
    (``principal_slot_available``, read-only) before starting the
    generator; the actual counter mutation (``try_reserve_principal_slot``)
    happens once ``_generate`` runs. Between those two moments, enough
    concurrently-racing attaches for the same key can fill the last slot
    first, so the reservation inside ``_generate`` can fail even though
    the earlier check passed. That loss must not abort an attach whose
    response has already started streaming: the stream is served anyway
    (soft cap), and because ``principal_slot_reserved`` stays False, the
    ``finally`` teardown must not release a slot this stream never held --
    doing so would erroneously free a slot actually owned by one of the
    other concurrent streams that did win the race."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    # Simulate the losing side of the race: every slot is already held by
    # other concurrent attaches by the time this stream's own generator
    # runs `try_reserve_principal_slot`.
    for _ in range(es.PER_PRINCIPAL_STREAM_CAP):
        assert es.try_reserve_principal_slot(key_prefix)
    full_count = es._principal_stream_counts[key_prefix]

    with caplog.at_level("WARNING", logger="xagent.web.api.v1._events_stream"):
        agen = es._generate(
            task_id,
            principal,
            key_prefix=key_prefix,
            initial_status=snapshot.status.value,
            read_task_snapshot=v1_tasks._load_task_info_snapshot,
            **_long_intervals(),
        )
        first = await agen.__anext__()

    # The stream is served normally despite the lost reservation race.
    assert "event: task.status" in first
    assert es.count_task_sinks(task_id) == 1
    # No extra slot was claimed -- the reservation attempt failed and
    # nothing was added on top of the already-full count.
    assert es._principal_stream_counts[key_prefix] == full_count
    assert any(
        key_prefix in record.message and "best-effort" in record.message
        for record in caplog.records
    )

    await agen.aclose()

    # Teardown must not touch the counter: this stream never held a slot,
    # so releasing one here would steal it from a stream that did.
    assert es.count_task_sinks(task_id) == 0
    assert es._principal_stream_counts.get(key_prefix, 0) == full_count
    assert es._principal_stream_counts[key_prefix] >= 0

    for _ in range(es.PER_PRINCIPAL_STREAM_CAP):
        es.release_principal_slot(key_prefix)


# ===== absolute deadline emits stream_expired before closing =====


async def test_absolute_deadline_emits_expired_then_closes():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        **_long_intervals(
            stream_max_duration_seconds=0.05, heartbeat_interval_seconds=0.01
        ),
    )
    frames = [frame async for frame in resp.body_iterator]
    # Zero or more heartbeats may land before the deadline trips, but the
    # close frame must be exactly one, and it must be last (nothing is
    # yielded after it).
    assert "event: task.status" in frames[0]
    assert frames.count(": ping\n\n") == len(frames) - 2  # everything but status+close
    assert "event: stream.error" in frames[-1]
    assert "stream_expired" in frames[-1]
    assert es.count_task_sinks(task_id) == 0


async def test_deadline_wait_budget_shrinks_below_the_heartbeat():
    """Heartbeat (5s) is deliberately much larger than the deadline
    (0.05s), so only a ``wait_budget`` that's capped at what's left
    before the deadline -- not the full heartbeat interval -- closes
    this stream quickly. The old formula (heartbeat whenever any time
    remains) would instead wait out the full 5s heartbeat before ever
    rechecking the deadline; ``wait_for(timeout=1)`` turns that into a
    hard failure instead of a slow pass."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        **_long_intervals(
            stream_max_duration_seconds=0.05, heartbeat_interval_seconds=5.0
        ),
    )

    async def _drain() -> list[str]:
        return [frame async for frame in resp.body_iterator]

    frames = await asyncio.wait_for(_drain(), timeout=1)
    assert "event: stream.error" in frames[-1]
    assert "stream_expired" in frames[-1]


# ===== watchdog vs. deadline race closes exactly once =====


async def test_enqueue_close_is_idempotent_first_writer_wins():
    sink = _make_sink()
    first = sink.enqueue_close(es.error_frame("stream_expired"))
    second = sink.enqueue_close(es.error_frame("task_deleted"))
    assert first is True
    assert second is False
    assert sink.queue.qsize() == 1
    assert sink.queue.get_nowait() == (es.error_frame("stream_expired"), True)


# ===== close frame isn't lost when it's enqueued while the generator is
# suspended mid-yield on an earlier, non-close frame =====


async def test_close_frame_delivered_when_enqueued_between_two_yields():
    """The generator can be suspended at ``yield frame_text`` (Starlette
    awaiting the socket write) when a concurrent close call lands. The
    close-ness of the *next* frame the generator delivers must come
    from that frame's own queue entry, not from re-reading
    ``sink.closing`` after the unrelated earlier yield resumes -- by
    then ``closing`` is already true even though the frame just sent
    wasn't the close frame. Driven by hand via ``asend`` to control
    exactly where the generator is suspended when the race hits."""

    async def _never_read(task_id, principal):  # pragma: no cover
        raise AssertionError("watchdog must not run in this test")

    task_id = 987_654_321  # sentinel, not a real DB row -- unused by this test
    gen = es._generate(
        task_id,
        None,
        key_prefix="pfx-close-frame-race-test",
        initial_status="running",
        read_task_snapshot=_never_read,
        watchdog_interval_seconds=10_000,
        stream_max_duration_seconds=10_000,
        heartbeat_interval_seconds=10_000,
    )
    try:
        first = await gen.asend(None)
        assert "event: task.status" in first

        # ``_generate`` builds and registers the sink internally now (fix
        # for the double-leak on an unstarted generator) -- fetch the
        # live instance via the same ``manager`` registry the generator
        # just registered it into.
        sink = next(
            c
            for c in es.manager.connections_for_task(task_id)
            if isinstance(c, es.V1EventStreamSink)
        )

        sink.enqueue_status("paused")
        second = await gen.asend(None)
        assert "paused" in second
        # The generator is now suspended right after yielding ``second``
        # -- exactly the window in which the real consumer would be
        # awaiting the socket write while other tasks run.

        won = sink.enqueue_close(
            es.completed_frame(status="completed", output="done", error=None)
        )
        assert won is True

        third = await gen.asend(None)
        assert "event: task.completed" in third
        with pytest.raises(StopAsyncIteration):
            await gen.asend(None)
    finally:
        # ``_generate``'s own ``finally`` already released the slot it
        # reserved once the close frame was delivered above; ``aclose()``
        # here only matters if an assertion failed earlier and the
        # generator is still suspended mid-stream.
        await gen.aclose()


async def test_close_exactly_once_under_watchdog_deadline_race():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)  # non-terminal at attach
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        watchdog_interval_seconds=0.001,
        stream_max_duration_seconds=0.001,
        heartbeat_interval_seconds=0.001,
    )
    # Flip terminal *after* attach so the watchdog's next tick and the
    # already-short deadline both want to close the stream.
    _set_task_status(task_id, TaskStatus.COMPLETED, output="done")

    closing_events = ("event: task.completed", "event: stream.error")
    frames = []
    async for frame in resp.body_iterator:
        frames.append(frame)
        if len(frames) > 10:
            break
    close_frames = [f for f in frames if any(name in f for name in closing_events)]
    assert len(close_frames) == 1
    assert close_frames[0] == frames[-1]
    assert es.count_task_sinks(task_id) == 0


# ===== per-task concurrency cap =====


async def test_events_per_task_cap_third_stream_429():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    responses = []
    for _ in range(es.PER_TASK_STREAM_CAP):
        resp = await es.build_event_stream_response(
            task_id=task_id,
            principal=principal,
            initial_snapshot=snapshot,
            read_task_snapshot=v1_tasks._load_task_info_snapshot,
            **_long_intervals(),
        )
        # Real delivery (Starlette) always pulls the first chunk before a
        # response can close; advance each generator once so its ``finally``
        # cleanup is reachable, same as it would be in production.
        await resp.body_iterator.__anext__()
        responses.append(resp)

    assert es.count_task_sinks(task_id) == es.PER_TASK_STREAM_CAP
    with pytest.raises(V1ApiError) as exc_info:
        await es.build_event_stream_response(
            task_id=task_id,
            principal=principal,
            initial_snapshot=snapshot,
            read_task_snapshot=v1_tasks._load_task_info_snapshot,
        )
    assert exc_info.value.code is V1ErrorCode.RATE_LIMITED
    assert exc_info.value.http_status == 429

    for resp in responses:
        await resp.body_iterator.aclose()
    assert es.count_task_sinks(task_id) == 0


# ===== the fast paths are exempt from both concurrency caps =====


@pytest.mark.parametrize(
    ("fast_path_status", "closing_event"),
    [
        (TaskStatus.COMPLETED, "event: task.completed"),
        (TaskStatus.WAITING_FOR_USER, "event: task.input_required"),
    ],
)
async def test_fast_path_attach_exempt_from_both_caps(fast_path_status, closing_event):
    """``build_event_stream_response`` checks ``_stream_close_reason``
    before either concurrency cap (see its own docstring): a task that's
    already terminal or already waiting for user input takes the
    snapshot-closing fast path and returns before the per-task and
    per-principal cap checks ever run, so it must succeed even when both
    caps are already saturated.

    Setup order matters: the per-task cap has to be filled with real
    streams *while the task is still running*, because the fast path
    never registers a sink -- once the task row is already terminal
    there is no way to fill that cap through it. Only after both caps
    are full does the task row flip to the fast-path status under test.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)
    running_snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    # Fill the per-task cap with real streams while the task is running.
    responses = []
    for _ in range(es.PER_TASK_STREAM_CAP):
        resp = await es.build_event_stream_response(
            task_id=task_id,
            principal=principal,
            initial_snapshot=running_snapshot,
            read_task_snapshot=v1_tasks._load_task_info_snapshot,
            **_long_intervals(),
        )
        await resp.body_iterator.__anext__()
        responses.append(resp)
    assert es.count_task_sinks(task_id) == es.PER_TASK_STREAM_CAP

    # Fill the rest of the per-principal cap too -- opening the streams
    # above already reserved one principal slot per stream (`_generate`
    # reserves on the same key_prefix), so only the remainder needs
    # filling here.
    already_reserved = es._principal_stream_counts.get(key_prefix, 0)
    for _ in range(es.PER_PRINCIPAL_STREAM_CAP - already_reserved):
        assert es.try_reserve_principal_slot(key_prefix)
    assert es._principal_stream_counts[key_prefix] == es.PER_PRINCIPAL_STREAM_CAP

    # Now flip the task row to the fast-path status under test.
    if fast_path_status is TaskStatus.WAITING_FOR_USER:
        _set_task_status(task_id, fast_path_status, control_state="idle")
    else:
        _set_task_status(task_id, fast_path_status, output="done")

    resp = client.get(f"/v1/chat/tasks/{task_id}/events", headers=_bearer(full_key))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert body.count("event: task.status") == 1
    assert body.count(closing_event) == 1

    # The fast path never registers a sink, so the count above -- entirely
    # made up of the cap-filling streams -- is unchanged.
    assert es.count_task_sinks(task_id) == es.PER_TASK_STREAM_CAP

    for resp in responses:
        await resp.body_iterator.aclose()
    assert es.count_task_sinks(task_id) == 0
    # Closing each stream above already released its own reserved slot;
    # release only the ones this test reserved manually.
    for _ in range(es.PER_PRINCIPAL_STREAM_CAP - already_reserved):
        es.release_principal_slot(key_prefix)
    assert es._principal_stream_counts.get(key_prefix, 0) == 0


# ===== the sink never touches the database on its per-frame path =====


async def test_sink_send_text_never_queries_db(monkeypatch):
    sink = _make_sink()
    calls: list[int] = []
    original_get_session_local = es.get_session_local

    def _tracking_get_session_local():
        calls.append(1)
        return original_get_session_local()

    monkeypatch.setattr(es, "get_session_local", _tracking_get_session_local)

    for i in range(20):
        await sink.send_text(
            json.dumps(
                {"type": "task_started", "task_id": sink.task_id, "status": f"s{i}"}
            )
        )
    await sink.send_text(
        json.dumps({"type": "task_completed", "task": {"id": sink.task_id}})
    )
    # Content frames must stay just as DB-free as the lifecycle path
    # above -- the step/message projection path reads only the broadcast
    # dict, never the database.
    await sink.send_text(
        json.dumps(
            {
                "type": "trace_event",
                "event_id": "ev-1",
                "event_type": "tool_execution_start",
                "task_id": sink.task_id,
                "timestamp": 0,
                "step_id": "step-1",
                "data": {"tool_call_id": "call-1", "tool_name": "search", "args": {}},
            }
        )
    )
    await sink.send_text(
        json.dumps(
            {
                "type": "final_answer_delta",
                "message_id": "msg-1",
                "task_id": sink.task_id,
                "delta": "hi",
            }
        )
    )
    assert calls == []


# ===== the heartbeat comment line actually gets sent while idle =====


async def test_heartbeat_actually_sent_when_idle():
    """The 15s heartbeat comment must actually be emitted while the
    stream is otherwise idle -- this is a real, non-vacuous assertion
    that at least one ``: ping`` frame arrives, not a shape check that
    would pass whether zero or many heartbeats fired (that was the bug
    in the older deadline test, which only ever asserted a frame-count
    identity that holds either way). Injects a short heartbeat interval
    with a long watchdog interval and deadline so a heartbeat is the
    only thing that can produce a frame here."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        **_long_intervals(heartbeat_interval_seconds=0.01),
    )
    pings = 0
    try:

        async def _collect() -> None:
            nonlocal pings
            async for frame in resp.body_iterator:
                if frame == ": ping\n\n":
                    pings += 1
                    return

        await asyncio.wait_for(_collect(), timeout=2)
    finally:
        await resp.body_iterator.aclose()
    assert pings >= 1


# ===== a broadcast-shaped frame reaching the sink ends up as real
# SSE output text out of the generator, not just in the internal queue =====


async def test_broadcast_frame_reaches_generator_output_as_task_status():
    """Wiring test for the full path: a real DB status change ->
    ``ConnectionManager.broadcast_to_task`` (what production code calls,
    not a hand-rolled ``sink.send_text``) -> the sink's ``send_text`` ->
    ``enqueue_status`` -> the outbound queue -> ``_generate``'s yield.
    Uses a status distinct from the initial attach status ("running") so
    the assertion can't pass merely because the *first* frame already
    says ``task.status`` -- it must be *this* broadcast that produced
    the second frame.

    The task row must be updated *before* broadcasting: ``broadcast_to_task``
    enriches any versioned event through
    ``_with_current_task_control_state``, which re-reads the task's
    current row and overwrites the event's ``status`` field with it
    whenever the event doesn't already carry a state tuple. Broadcasting
    a bare ``{"type": "task_paused"}`` while the row is still "running"
    would get its status field filled in as "running", not "paused",
    and the second frame would never arrive within the timeout below.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)
    assert snapshot.status.value == "running"

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        **_long_intervals(),
    )
    first = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
    assert json.loads(first.split("data: ", 1)[1]) == {"status": "running"}

    _set_task_status(task_id, TaskStatus.PAUSED)
    await es.manager.broadcast_to_task({"type": "task_paused"}, task_id)

    second = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
    assert "event: task.status" in second
    assert json.loads(second.split("data: ", 1)[1]) == {"status": "paused"}

    await resp.body_iterator.aclose()


# ===== the same full path, for content frames rather than lifecycle ones =====


async def test_broadcast_content_frames_reach_generator_output_as_step_and_message():
    """Composition smoke test for the content path end to end: real sink
    registration -> ``ConnectionManager.broadcast_to_task`` ->
    ``V1EventStreamSink.send_text`` -> projection -> the outbound queue ->
    ``_generate``'s yield -> the ``StreamingResponse`` body.

    Deliberately a composition test, not per-layer coverage: what each
    intermediate layer does with each frame family is already pinned by
    the direct ``send_text`` tests above. What only this test can catch is
    a break in the joins between them -- registration, the manager's
    fan-out, or body iteration -- which would leave the endpoint
    lifecycle-only while every direct test stayed green.

    Both content families go through, because they take different routes
    inside ``send_text``: a trace event folds through the projector, a
    final-answer frame maps straight to a ``message.*`` frame. The trace
    frame is built by routing a real ``CoreTraceEvent`` through the
    production conversion (``_broadcast_frame_for``) rather than
    hand-shaping a dict.

    The first frame is pulled before broadcasting for the reason the
    lifecycle test above states: the generator's body -- and therefore
    ``manager.register_connection`` -- does not run until it is first
    iterated, so a broadcast issued earlier would reach no sink.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        **_long_intervals(),
    )
    first = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
    assert "event: task.status" in first
    assert es.count_task_sinks(task_id) == 1  # the sink really did register

    start_event = CoreTraceEvent(
        ACTION_START_TOOL,
        step_id="step-1",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp(),
        data={
            "tool_call_id": "call-1",
            "tool_name": "search",
            "tool_args": {"q": "weather"},
        },
    )
    await es.manager.broadcast_to_task(
        json.loads(_broadcast_frame_for(start_event, task_id=task_id)), task_id
    )
    step_frame = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
    assert step_frame.startswith("event: step.started\n")
    step = json.loads(step_frame.split("data: ", 1)[1])["step"]
    assert step["id"] == "tool_call:call-1"
    assert step["data"]["name"] == "search"

    await es.manager.broadcast_to_task(
        {
            "type": "final_answer_delta",
            "message_id": "final_answer_e2e",
            "task_id": task_id,
            "delta": "sun",
        },
        task_id,
    )
    message_frame = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
    assert message_frame.startswith("event: message.delta\n")
    assert json.loads(message_frame.split("data: ", 1)[1]) == {
        "message_id": "final_answer_e2e",
        "text": "sun",
    }

    await resp.body_iterator.aclose()
    assert es.count_task_sinks(task_id) == 0  # teardown unregistered the sink


# ===== consecutive identical statuses only produce one frame =====


async def test_consecutive_same_status_broadcasts_are_deduped_to_one_frame():
    sink = _make_sink(task_id=1, status="running")
    for _ in range(3):
        await sink.send_text(
            json.dumps({"type": "task_paused", "task_id": 1, "status": "paused"})
        )
    assert sink.queue.qsize() == 1
    assert sink.queue.get_nowait() == (es.status_frame("paused"), False)


# ===== a `task_completed` broadcast is acceleration-only, never a status
# frame =====


async def test_task_completed_broadcast_never_enqueues_a_status_frame():
    """``task_completed`` only sets ``completion_hint`` and returns early
    -- it must never also reach ``enqueue_status``, or a ``task_completed``
    broadcast racing the watchdog's own authoritative ``task.completed``
    frame could emit a spurious extra ``task.status`` right as the stream
    is closing."""
    sink = _make_sink(task_id=1, status="running")
    await sink.send_text(json.dumps({"type": "task_completed", "task": {"id": 1}}))
    assert sink.completion_hint.is_set()
    assert sink.queue.empty()


# ===== a terminal-failure broadcast also sets the completion_hint =====


async def test_terminal_failure_broadcast_also_sets_completion_hint():
    """A failed task's broadcast (``task_error``) reaches the sink through
    the generic versioned-event branch below, not the ``task_completed``
    short-circuit above -- unlike that short-circuit, it must still do
    both things: enqueue the ``task.status`` frame *and* set
    ``completion_hint``, so the watchdog wakes early instead of waiting
    out its normal cadence to emit the authoritative ``task.completed``
    close frame."""
    sink = _make_sink(task_id=1, status="running")
    await sink.send_text(
        json.dumps({"type": "task_error", "task_id": 1, "status": "failed"})
    )
    assert sink.completion_hint.is_set()
    assert sink.queue.get_nowait() == (es.status_frame("failed"), False)


# ===== a waiting-for-user broadcast also sets the completion_hint =====


async def test_waiting_for_user_broadcast_also_sets_completion_hint():
    """A task moving to ``waiting_for_user`` reaches the sink through the
    same generic versioned-event branch as the terminal-failure case
    above, not the ``task_completed`` short-circuit -- it must also set
    ``completion_hint`` (in addition to enqueueing the ``task.status``
    frame), so the watchdog wakes early instead of waiting out its
    normal cadence to emit the authoritative ``task.input_required``
    close frame. Safe because the watchdog re-reads the authoritative
    row before closing anything, including the ``resume_requested``
    carve-out -- an early wake never closes a stream a fresh read
    wouldn't have closed anyway."""
    sink = _make_sink(task_id=1, status="running")
    await sink.send_text(
        json.dumps(
            {
                "type": "task_waiting_for_user",
                "task_id": 1,
                "status": "waiting_for_user",
            }
        )
    )
    assert sink.completion_hint.is_set()
    assert sink.queue.get_nowait() == (es.status_frame("waiting_for_user"), False)


# ===== the completion_hint wakes the watchdog early, not on its next
# periodic tick =====


async def test_completion_hint_wakes_watchdog_before_its_next_periodic_tick():
    """The ``task_completed`` broadcast hint is supposed to be an
    acceleration path -- the watchdog wakes and checks immediately
    instead of waiting out its normal interval. This pins that it's a
    genuine early wake, not something that happens to work because the
    interval is already short: the watchdog interval here is 1000s, so
    the only way this test can finish inside its 2s bound is if the
    hint actually woke it early. If the hint stopped working, this
    would hang until the bound trips and the test would fail instead of
    silently passing."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        **_long_intervals(),
    )
    await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)

    _set_task_status(task_id, TaskStatus.COMPLETED, output="done")
    sink = next(
        c
        for c in es.manager.connections_for_task(task_id)
        if isinstance(c, es.V1EventStreamSink)
    )
    await sink.send_text(
        json.dumps({"type": "task_completed", "task": {"id": task_id}})
    )

    frames = []

    async def _drain() -> None:
        async for frame in resp.body_iterator:
            frames.append(frame)

    await asyncio.wait_for(_drain(), timeout=2)
    assert any("event: task.completed" in f for f in frames)


# ===== content projection: step.* / message.* from live broadcast frames
# (see ``project_content_frames`` and its docstring for the frame-family
# classification this section pins) =====


def _trace_event_frame(
    event_type: str,
    *,
    task_id: int,
    data: dict,
    event_id: str = "ev-1",
    step_id: str | None = None,
    timestamp: float = 0.0,
) -> str:
    payload: dict = {
        "type": "trace_event",
        "event_id": event_id,
        "event_type": event_type,
        "task_id": task_id,
        "timestamp": timestamp,
        "data": data,
    }
    if step_id is not None:
        payload["step_id"] = step_id
    return json.dumps(payload)


def _broadcast_frame_for(event: "CoreTraceEvent", *, task_id: int) -> str:
    """The exact text a real broadcast of ``event`` would carry.

    Routes the core trace event through the production conversion --
    ``WebSocketTraceHandler._convert_trace_event_to_stream_event``, which
    applies ``serialize_trace_data``, ``normalize_public_trace_event``
    and ``create_stream_event`` -- and serializes it the way
    ``ConnectionManager.broadcast_to_task`` does. Constructing the
    handler touches no database (its ``__init__`` sets three attributes;
    ``_load_task_description`` is the async path and is deliberately not
    called, so no ``task_description`` is injected).
    """
    handler = WebSocketTraceHandler(task_id)
    converted = handler._convert_trace_event_to_stream_event(event)
    assert converted is not None, "fixture event must be projectable"
    return json.dumps(converted)


def _persist_core_event(event: "CoreTraceEvent", *, task_id: int) -> None:
    """Write the row the persistence path would write for ``event``.

    Reproduces what ``DatabaseTraceHandler._save_trace_event`` derives --
    ``get_event_type_mapping`` for the event type,
    ``_serialize_data_for_json`` for the data,
    ``redact_runtime_sensitive_payload`` for the ``tool_execution_*``
    family, ``event_id=str(event.id)`` and ``step_id=event.step_id`` --
    then inserts through the test's own ``_insert_trace_event`` with
    ``timestamp=_convert_float_to_datetime(event.timestamp)``.

    Deliberately does NOT reproduce ``stage_trace_event_row``'s
    ``stored_data`` rewrite or its ``build_id``/``parent_event_id``
    stamping -- the real path runs ``data = staged.stored_data`` at
    ``trace_handlers.py:709``; no fixture in this module needs either.
    """
    event_type_str = get_event_type_mapping(event)
    handler = DatabaseTraceHandler(task_id=task_id)
    data = handler._serialize_data_for_json(event.data or {})
    if event_type_str in {
        "tool_execution_start",
        "tool_execution_end",
        "tool_execution_failed",
    }:
        data = redact_runtime_sensitive_payload(data)
    _insert_trace_event(
        task_id=task_id,
        event_type=event_type_str,
        event_id=str(event.id),
        timestamp=_convert_float_to_datetime(event.timestamp),
        data=data,
        step_id=event.step_id,
    )


async def test_trace_event_tool_call_start_then_end_projects_paired_step_frames():
    """A live start/end pair for the same ``tool_call_id`` folds through
    the same ``PublicStepProjector`` pairing rule ``steps()`` uses:
    ``step.started`` (running) first, then ``step.completed`` carrying
    the tool's result, both keyed by the same public step id."""
    sink = _make_sink(task_id=42)
    await sink.send_text(
        _trace_event_frame(
            "tool_execution_start",
            task_id=42,
            step_id="step-1",
            data={
                "tool_call_id": "call-1",
                "tool_name": "search",
                "tool_args": {"query": "weather"},
            },
        )
    )
    started_text, started_close = sink.queue.get_nowait()
    assert started_close is False
    assert started_text.startswith("event: step.started\n")
    started = json.loads(started_text.split("data: ", 1)[1])["step"]
    assert started["id"] == "tool_call:call-1"
    assert started["status"] == "running"
    assert started["data"]["name"] == "search"

    await sink.send_text(
        _trace_event_frame(
            "tool_execution_end",
            task_id=42,
            step_id="step-1",
            data={"tool_call_id": "call-1", "success": True, "result": "sunny"},
        )
    )
    completed_text, completed_close = sink.queue.get_nowait()
    assert completed_close is False
    assert completed_text.startswith("event: step.completed\n")
    completed = json.loads(completed_text.split("data: ", 1)[1])["step"]
    assert completed["id"] == "tool_call:call-1"
    assert completed["status"] == "completed"
    assert completed["data"]["result"] == "sunny"


@pytest.mark.parametrize(
    ("event_id", "extra_data"),
    [
        pytest.param("persisted-event-id", {}, id="no-stream-message-id"),
        pytest.param(
            "live-final-answer",
            {"stream_message_id": "final_answer_abc"},
            id="with-stream-message-id",
        ),
    ],
)
async def test_trace_event_ai_message_projects_a_completed_message_step(
    event_id, extra_data
):
    """An ``ai_message`` folds into a public ``message`` step, always
    already ``completed`` (see ``_build_message_step``), whether or not it
    carries ``stream_message_id``.

    The second cell is the new contract. An ``ai_message`` carrying
    ``stream_message_id`` is the persisted mirror of a final answer this
    stream already delivered live as ``message.delta`` /
    ``message.completed``, so folding it too duplicates that content as a
    second delivery rather than losing it. Dropping it here -- the old
    behavior -- had no persisted counterpart: ``steps()`` shows this exact
    row as a ``message`` step regardless, so filtering only the live path
    made an already-attached stream disagree with ``steps()`` about whether
    the step exists. See
    ``test_live_projection_matches_steps_for_a_streamed_final_answer``
    below for the two-path parity this restores.
    """
    sink = _make_sink(task_id=7)
    await sink.send_text(
        _trace_event_frame(
            "ai_message",
            task_id=7,
            event_id=event_id,
            data={"content": "the answer", **extra_data},
        )
    )
    frame_text, is_close = sink.queue.get_nowait()
    assert is_close is False
    assert frame_text.startswith("event: step.completed\n")
    step = json.loads(frame_text.split("data: ", 1)[1])["step"]
    assert step["id"] == f"message:{event_id}"
    assert step["data"] == {"role": "assistant", "content": "the answer"}
    assert sink.queue.empty()


async def test_trace_event_delegated_child_source_is_filtered():
    """A trace_event whose ``data["source"]`` marks it as a
    delegated child agent's own trace is dropped, the same
    ``build_id IS NOT NULL``-equivalent exclusion ``steps()`` gets for
    free from its SQL WHERE clause."""
    sink = _make_sink(task_id=9)
    await sink.send_text(
        _trace_event_frame(
            "tool_execution_start",
            task_id=9,
            data={
                "tool_call_id": "call-1",
                "tool_name": "search",
                "tool_args": {},
                "source": "xagent-agent-tool-child",
            },
        )
    )
    assert sink.queue.empty()


async def test_trace_event_task_info_produces_no_step_frame():
    """``task_info`` short-circuits to the
    ``task.status``/``task.input_required`` lifecycle handling in
    ``send_text`` (unchanged, tested elsewhere in this file) and never
    reaches the step projector -- it isn't, and never becomes, a public
    step.

    An empty outbound queue alone doesn't pin this: ``task_info`` isn't
    one of ``_step_mapping.py``'s recognized event types either, so
    feeding it to the projector directly would *also* produce no step
    frame -- the short-circuit's queue-visible effect would be
    unchanged if it were deleted. What the short-circuit actually
    controls is whether the projector is invoked at all, which is why
    the projector's ``feed`` is spied on below and asserted never
    called -- that assertion alone would fail if the short-circuit were
    removed, even though the queue-emptiness assertion would not."""
    sink = _make_sink(task_id=11, status="running")
    feed_spy = MagicMock(wraps=sink._projector.feed)
    sink._projector.feed = feed_spy
    await sink.send_text(
        json.dumps(
            {
                "type": "trace_event",
                "event_id": "ev-1",
                "event_type": "task_info",
                "task_id": 11,
                "timestamp": 0,
                # Real broadcasts carry status/control_state both at the
                # top level (what ``_is_versioned_task_event`` reads) and
                # merged into ``data`` (``_with_task_control_state_snapshot``,
                # ``websocket.py``) -- both are set here to match.
                "status": "waiting_for_user",
                "control_state": "idle",
                "data": {
                    "status": "waiting_for_user",
                    "control_state": "idle",
                    "message": "What city?",
                },
            }
        )
    )
    # The lifecycle side still reacts (a status frame is queued and the
    # completion hint fires) -- only the *step* projection is asserted
    # absent here.
    assert sink.completion_hint.is_set()
    frame_text, _ = sink.queue.get_nowait()
    assert frame_text.startswith("event: task.status\n")
    assert sink.queue.empty()
    feed_spy.assert_not_called()


async def test_trace_event_audit_only_data_is_filtered():
    """The same ``__audit_only__`` server-side-RCA marker the WebSocket
    broadcaster already drops before fanning out (see
    ``is_audit_only_trace_data``) is honored on the live SSE path too."""
    sink = _make_sink(task_id=13)
    await sink.send_text(
        _trace_event_frame(
            "tool_execution_start",
            task_id=13,
            data={
                "tool_call_id": "call-1",
                "tool_name": "search",
                "tool_args": {},
                "__audit_only__": True,
            },
        )
    )
    assert sink.queue.empty()


async def test_trace_event_tool_call_redacts_credential_shaped_fields_on_the_wire():
    """The same runtime-secret redaction ``normalize_public_trace_event``
    already applies before a trace event reaches this stream's projector
    (see ``test_normalize_public_trace_event_redacts_tool_runtime_secrets``
    in ``tests/web/api/test_public_trace_events.py``, the ``steps()``-side
    fixture this mirrors) must also hold for a *live* SSE frame, not just
    for the function in isolation: this pins that the emitted
    ``step.started``/``step.completed`` wire text itself never carries the
    raw secret, only the shared redaction marker."""
    sink = _make_sink(task_id=17)
    await sink.send_text(
        _trace_event_frame(
            "tool_execution_start",
            task_id=17,
            data={
                "tool_call_id": "call-1",
                "tool_name": "shiftcare",
                "tool_args": {
                    "headers": {
                        "Authorization": "Bearer live-stream-secret-token",
                        "X-Account": "6185",
                    },
                    "connector_runtime": {
                        "secrets": {"authorization": "Bearer nested-live-token"},
                        "auth_selector": {"resource_owner_key": "xagent:user:owner"},
                    },
                },
            },
        )
    )
    started_text, started_close = sink.queue.get_nowait()
    assert started_close is False
    assert started_text.startswith("event: step.started\n")
    assert "live-stream-secret-token" not in started_text
    assert "nested-live-token" not in started_text
    assert "xagent:user:owner" not in started_text
    # The public step's ``data.args`` is the (already-redacted) source
    # ``tool_args``: see ``_build_tool_start`` in ``_step_mapping.py``.
    started_data = json.loads(started_text.split("data: ", 1)[1])["step"]["data"]
    assert started_data["args"]["headers"]["Authorization"] == REDACTED_RUNTIME_SECRET
    assert started_data["args"]["headers"]["X-Account"] == "6185"
    assert (
        started_data["args"]["connector_runtime"]["auth_selector"]["resource_owner_key"]
        == REDACTED_RUNTIME_SECRET
    )

    await sink.send_text(
        _trace_event_frame(
            "tool_execution_end",
            task_id=17,
            data={
                "tool_call_id": "call-1",
                "success": True,
                "result": {
                    "headers": {"Authorization": "Bearer live-stream-secret-token"}
                },
            },
        )
    )
    completed_text, completed_close = sink.queue.get_nowait()
    assert completed_close is False
    assert completed_text.startswith("event: step.completed\n")
    assert "live-stream-secret-token" not in completed_text
    completed_data = json.loads(completed_text.split("data: ", 1)[1])["step"]["data"]
    assert (
        completed_data["result"]["headers"]["Authorization"] == REDACTED_RUNTIME_SECRET
    )


@pytest.mark.parametrize("frame_type", ["final_answer_start", "final_answer_error"])
async def test_final_answer_start_and_error_produce_no_content_frame(frame_type):
    """Neither is on the public event list. A ``message.delta``
    sequence may therefore end with no ``message.completed`` -- the next
    lifecycle event is the client's abandonment signal, not a dedicated
    close event from this pair."""
    sink = _make_sink(task_id=21)
    await sink.send_text(
        json.dumps(
            {
                "type": frame_type,
                "message_id": "final_answer_abc",
                "task_id": 21,
                "error": "boom",
            }
        )
    )
    assert sink.queue.empty()


async def test_step_started_frame_text_is_unaffected_by_the_step_s_later_completion():
    """Serialization-aliasing guard: ``feed()``
    returns the projector's own live dict, which gets mutated in place
    when the step's end event later arrives. The ``step.started`` frame
    text must have been fully serialized to a JSON string at enqueue
    time -- if it instead held a reference to the projector's dict and
    serialized lazily, this test's already-queued frame would show
    ``status: "completed"`` after the second ``send_text`` call below,
    not the ``"running"`` it captured when the start event arrived."""
    sink = _make_sink(task_id=29)
    await sink.send_text(
        _trace_event_frame(
            "tool_execution_start",
            task_id=29,
            data={"tool_call_id": "call-1", "tool_name": "search", "tool_args": {}},
        )
    )
    started_text, _ = sink.queue.get_nowait()

    await sink.send_text(
        _trace_event_frame(
            "tool_execution_end",
            task_id=29,
            data={"tool_call_id": "call-1", "success": True, "result": "sunny"},
        )
    )
    # Draining the second frame too is incidental to this test's point;
    # what matters is that re-reading the *first* frame's text (captured
    # before the mutation) still says "running".
    sink.queue.get_nowait()

    assert json.loads(started_text.split("data: ", 1)[1])["step"]["status"] == "running"


# ===== raw-frame size pre-check: an oversized content frame is dropped
# before it's projected =====


def _sized_content_frame(task_id: int, total_chars: int) -> str:
    """A ``final_answer_delta`` frame whose serialized text is exactly
    ``total_chars`` long: build the envelope with an empty ``delta``,
    measure it, then pad with that many ASCII characters (each costs
    exactly one character in the serialized form)."""
    envelope: dict = {
        "type": "final_answer_delta",
        "message_id": "final_answer_abc",
        "task_id": task_id,
        "delta": "",
    }
    base_len = len(json.dumps(envelope))
    pad = total_chars - base_len
    assert pad >= 0
    envelope["delta"] = "a" * pad
    raw = json.dumps(envelope)
    assert len(raw) == total_chars
    return raw


async def test_content_frame_exactly_at_the_size_cap_is_projected():
    """Boundary case, built to land exactly on the threshold: a content
    frame whose raw text is exactly ``MAX_RAW_FRAME_TEXT_CHARS`` long is
    still parsed and projected -- the size check only rejects frames
    strictly larger. 256 KiB of delta text is itself over the 64 KiB
    per-frame content cap, so the projected frame carries
    ``truncated: true`` -- this test states that whole outcome, not
    just that the frame arrived."""
    sink = _make_sink(task_id=93)
    raw = _sized_content_frame(93, es.MAX_RAW_FRAME_TEXT_CHARS)

    await sink.send_text(raw)

    assert sink.dropped_frame_count == 0
    frame_text, _ = sink.queue.get_nowait()
    assert frame_text.startswith("event: message.delta\n")
    data = json.loads(frame_text.split("data: ", 1)[1])
    assert data["truncated"] is True


async def test_content_frame_one_char_over_the_size_cap_is_dropped():
    """One character over the same boundary is dropped and counted, so
    an off-by-one in the comparison would fail this test but not the
    one above."""
    sink = _make_sink(task_id=94)
    raw = _sized_content_frame(94, es.MAX_RAW_FRAME_TEXT_CHARS + 1)

    await sink.send_text(raw)

    assert sink.dropped_frame_count == 1
    assert sink.queue.empty()


def test_capped_text_keeps_a_string_exactly_at_the_cap():
    """``_capped_text`` measures a string's escaped-JSON wire form,
    quotes included: ``MAX_FRAME_CONTENT_BYTES - 2`` plain ASCII
    characters plus the two wrapping quotes lands exactly on the cap
    (65536 bytes), so that length must survive untouched. One character
    more must truncate, to a 65534-character result."""
    at_cap = "a" * (es.MAX_FRAME_CONTENT_BYTES - 2)
    assert es._byte_length(at_cap) == es.MAX_FRAME_CONTENT_BYTES
    text, truncated = es._capped_text(at_cap)
    assert truncated is False
    assert text == at_cap

    over_cap = "a" * (es.MAX_FRAME_CONTENT_BYTES - 1)
    text, truncated = es._capped_text(over_cap)
    assert truncated is True
    assert len(text) == 65534


async def test_oversized_task_completed_frame_is_not_dropped_by_the_raw_precheck():
    """The size check applies only to frames whose parsed ``type`` is in
    ``_CONTENT_FRAME_TYPES`` -- a ``task_completed`` broadcast returns on
    ``send_text``'s acceleration branch before the content check is ever
    reached, regardless of size. It carries its whole output twice
    (``result`` and ``output``, see ``websocket.py``), so it can cross
    ``MAX_RAW_FRAME_TEXT_CHARS`` on an ordinary large response, well
    before anything is actually wrong. A size check placed ahead of the
    parse cannot make this distinction -- it would drop a large
    ``task_completed`` whole, leaving ``completion_hint`` unset and
    delaying the watchdog's terminal close to its next scheduled cycle
    (30s in production) instead of firing immediately."""
    sink = _make_sink(task_id=97)
    duplicated_output = "x" * ((es.MAX_RAW_FRAME_TEXT_CHARS // 2) + 100)
    raw = json.dumps(
        {
            "type": "task_completed",
            "task": {"id": 97, "title": "t", "status": "completed", "description": ""},
            "result": duplicated_output,
            "output": duplicated_output,
            "success": True,
        }
    )
    assert len(raw) > es.MAX_RAW_FRAME_TEXT_CHARS

    await sink.send_text(raw)

    assert sink.dropped_frame_count == 0
    assert sink.completion_hint.is_set()


async def test_oversized_task_info_frame_is_not_dropped_by_the_raw_precheck():
    """``task_info`` is a lifecycle envelope (``_feed_trace_event``'s own
    ``event_type == "task_info"`` short-circuit), not step/message
    content, but its raw frame's own top-level ``"type"`` is
    ``"trace_event"`` -- the same type actual content frames
    (tool_call/thinking/agent_delegation/message) carry. The size check
    exempts it by its parsed ``event_type`` explicitly
    (``!= "task_info"``), after ``json.loads`` -- not by a prefix sniff.
    A check placed ahead of the parse would judge every ``trace_event``
    frame the same way regardless of its nested ``event_type``, and
    would drop a legitimately large ``task_info`` whole (its task
    description has no size bound of its own) -- losing
    ``task.status`` and ``completion_hint`` the same way it would cost
    an oversized ``task_completed`` its ``task.completed`` (see the
    companion test above)."""
    sink = _make_sink(task_id=101, status="running")
    huge_description = "x" * (es.MAX_RAW_FRAME_TEXT_CHARS + 1000)
    raw = json.dumps(
        {
            "type": "trace_event",
            "event_id": "ev-101",
            "event_type": "task_info",
            "task_id": 101,
            "timestamp": 0,
            "status": "waiting_for_user",
            "control_state": "idle",
            "data": {
                "status": "waiting_for_user",
                "control_state": "idle",
                "message": huge_description,
            },
        }
    )
    assert len(raw) > es.MAX_RAW_FRAME_TEXT_CHARS

    await sink.send_text(raw)

    assert sink.dropped_frame_count == 0
    assert sink.completion_hint.is_set()
    frame_text, _ = sink.queue.get_nowait()
    assert frame_text.startswith("event: task.status\n")
    assert json.loads(frame_text.split("data: ", 1)[1]) == {
        "status": "waiting_for_user"
    }


async def test_a_huge_task_description_does_not_drop_a_small_content_frame():
    """The size check measures the frame without its
    ``task_description``, so a long description cannot drop content.

    ``ws_trace_handlers.py``'s ``_convert_trace_event_to_stream_event``
    stamps the task's ``description`` column onto *every* trace event
    it converts, not just ``task_info``, and that column has no length
    bound. Counting it would put every ``step.*`` frame of such a task
    past ``MAX_RAW_FRAME_TEXT_CHARS`` for the life of the connection --
    a task whose first user message is long enough would receive no
    step content at all. The frame built here is over the raw cap on
    the description alone while its actual step content is a few dozen
    characters, and it must still project. The projected ``data`` is
    asserted too: the description is not merely uncounted, it never
    reaches the wire, because the step builders name their keys
    explicitly."""
    sink = _make_sink(task_id=102)
    huge_description = "x" * (es.MAX_RAW_FRAME_TEXT_CHARS + 1000)
    raw = _trace_event_frame(
        "tool_execution_start",
        task_id=102,
        step_id="step-1",
        data={
            "tool_call_id": "call-1",
            "tool_name": "search",
            "tool_args": {"query": "weather"},
            "task_description": huge_description,
        },
    )
    assert len(raw) > es.MAX_RAW_FRAME_TEXT_CHARS

    await sink.send_text(raw)

    assert sink.dropped_frame_count == 0
    frame_text, _ = sink.queue.get_nowait()
    assert frame_text.startswith("event: step.started\n")
    step = json.loads(frame_text.split("data: ", 1)[1])["step"]
    assert step["id"] == "tool_call:call-1"
    assert step["data"]["name"] == "search"
    assert "task_description" not in step["data"]


async def test_projection_consumes_the_pruned_frame_not_the_original():
    """The over-cap path hands projection the frame with
    ``task_description`` already removed, not the original.

    The wire output cannot distinguish the two -- the step builders name
    their keys explicitly, so the description never reaches a frame
    either way. What this pins is the processing boundary: the raw-frame
    check was that unbounded field's only per-frame CPU bound, and
    ``_measured_content_frame`` replaces it by pruning before projection
    rather than after, so a surviving frame's ``serialize_trace_data``
    walk never runs over the description. Asserted at the
    ``_project_and_queue`` boundary because no later observation point
    can see the difference."""
    sink = _make_sink(task_id=104)
    received: list[dict] = []
    original = sink._project_and_queue
    sink._project_and_queue = lambda frame: (
        received.append(frame),
        original(frame),
    )[-1]
    huge_description = "x" * (es.MAX_RAW_FRAME_TEXT_CHARS + 1000)
    raw = _trace_event_frame(
        "tool_execution_start",
        task_id=104,
        step_id="step-1",
        data={
            "tool_call_id": "call-1",
            "tool_name": "search",
            "tool_args": {"query": "weather"},
            "task_description": huge_description,
        },
    )

    await sink.send_text(raw)

    assert len(received) == 1
    assert "task_description" not in received[0]["data"]
    assert received[0]["data"]["tool_name"] == "search"


async def test_a_frame_still_over_the_cap_without_its_description_is_dropped():
    """The control for the test above: excluding ``task_description``
    from the measurement is not a blanket exemption for any frame that
    carries one. This frame's own tool arguments are over the raw cap
    by themselves, so it is still dropped and counted -- and nothing is
    queued, which a projection of the same frame would not have left
    (an oversized ``tool_execution_start`` still projects a
    ``step.started`` with truncated ``data``)."""
    sink = _make_sink(task_id=103)
    huge_description = "x" * (es.MAX_RAW_FRAME_TEXT_CHARS + 1000)
    huge_args = "y" * (es.MAX_RAW_FRAME_TEXT_CHARS + 1000)
    raw = _trace_event_frame(
        "tool_execution_start",
        task_id=103,
        step_id="step-2",
        data={
            "tool_call_id": "call-2",
            "tool_name": "search",
            "tool_args": {"query": huge_args},
            "task_description": huge_description,
        },
    )

    await sink.send_text(raw)

    assert sink.dropped_frame_count == 1
    assert sink.queue.empty()


async def test_unparseable_frame_is_dropped_and_counted():
    """A frame that isn't valid JSON can't be classified at all -- there
    is no parsed ``type`` to check -- so ``json.loads`` itself raises
    and the outer ``except`` in ``send_text`` drops and counts it, the
    same discipline every other drop in this method gets. Length is
    irrelevant on this path: the failure happens at the very first
    character, well before any size check would run, so a short input
    is enough to exercise it."""
    sink = _make_sink(task_id=99)
    raw = "x" * 64

    await sink.send_text(raw)

    assert sink.dropped_frame_count == 1
    assert sink.queue.empty()


async def test_oversized_non_dict_json_frame_is_ignored_without_counting():
    """A frame that IS valid JSON but not a dict (a bare array here) is
    ignored by ``send_text``'s own ``isinstance(message, dict)`` guard,
    before classification or the size check ever run -- even though it's
    larger than ``MAX_RAW_FRAME_TEXT_CHARS``. Deliberate semantic change
    from the pre-parse-sniffing era: only a frame classifiable as
    content (``_CONTENT_FRAME_TYPES``) counts as a drop now: this one is
    silently ignored, not counted."""
    sink = _make_sink(task_id=103)
    raw = json.dumps(["x"] * (es.MAX_RAW_FRAME_TEXT_CHARS // 2))
    assert len(raw) > es.MAX_RAW_FRAME_TEXT_CHARS

    await sink.send_text(raw)

    assert sink.dropped_frame_count == 0
    assert sink.queue.empty()


# ===== a frame that fails to project is dropped, not fatal, and never
# double-counted between the sink's two except layers
# (see _project_and_queue) =====


async def test_poisoned_live_frame_increments_dropped_frame_count_only_once(
    monkeypatch,
):
    """A projection failure on the live path is
    caught inside ``_project_and_queue`` and never re-raises -- if it
    did, ``send_text``'s own outer ``except`` would catch it too and
    count the same dropped frame twice."""
    sink = _make_sink(task_id=73)

    def _boom(message, projector):
        raise RuntimeError("boom live")

    monkeypatch.setattr(es, "project_content_frames", _boom)

    await sink.send_text(
        _trace_event_frame(
            "tool_execution_start",
            task_id=73,
            data={"tool_call_id": "call-1", "tool_name": "x", "tool_args": {}},
        )
    )  # must not raise

    assert sink.dropped_frame_count == 1
    assert sink.queue.empty()


# ===== single-frame content byte cap =====


@pytest.mark.parametrize(
    ("frame_type", "content_key", "wire_event", "wire_key", "oversized"),
    [
        pytest.param(
            "final_answer_delta",
            "delta",
            "message.delta",
            "text",
            False,
            id="delta-under-the-cap",
        ),
        pytest.param(
            "final_answer_delta",
            "delta",
            "message.delta",
            "text",
            True,
            id="delta-over-the-cap",
        ),
        pytest.param(
            "final_answer_end",
            "content",
            "message.completed",
            "content",
            False,
            id="completed-under-the-cap",
        ),
        pytest.param(
            "final_answer_end",
            "content",
            "message.completed",
            "content",
            True,
            id="completed-over-the-cap",
        ),
    ],
)
async def test_final_answer_frames_project_message_frames_under_the_byte_cap(
    frame_type, content_key, wire_event, wire_key, oversized
):
    """Each ``final_answer_*`` broadcast frame maps to its own
    ``message.*`` SSE event, and each carries the same per-frame byte cap
    on its own content field.

    Four cells, each catching a different error:
      - ``delta-under-the-cap`` / ``completed-under-the-cap``: the frame
        family mapping itself (``final_answer_delta`` -> ``message.delta``
        carrying ``text``, ``final_answer_end`` -> ``message.completed``
        carrying ``content``), plus the absence of a ``truncated`` key on
        content that fits. A wrong mapping, a renamed field, or a
        ``truncated`` marker on unremarkable content fails here.
      - ``delta-over-the-cap`` / ``completed-over-the-cap``: the cap is
        applied on *both* paths, not just the streamed-delta one. A
        ``_capped_text`` call missing from the final-answer *end* path
        would leave the completed cell's content whole and unmarked while
        the delta cells still passed.
    """
    sink = _make_sink(task_id=51)
    payload = (
        "x" * (es.MAX_FRAME_CONTENT_BYTES + 1000)
        if oversized
        else "just a normal chunk"
    )
    await sink.send_text(
        json.dumps(
            {
                "type": frame_type,
                "message_id": "final_answer_abc",
                "task_id": 51,
                content_key: payload,
            }
        )
    )
    frame_text, is_close = sink.queue.get_nowait()
    assert is_close is False
    assert frame_text.startswith(f"event: {wire_event}\n")
    data = json.loads(frame_text.split("data: ", 1)[1])
    assert data["message_id"] == "final_answer_abc"
    if oversized:
        assert data["truncated"] is True
        assert es._byte_length(data[wire_key]) <= es.MAX_FRAME_CONTENT_BYTES
        assert data[wire_key] != payload
    else:
        assert "truncated" not in data
        assert data == {"message_id": "final_answer_abc", wire_key: payload}


def test_capped_text_handles_a_multibyte_character_straddling_the_byte_boundary():
    """``_capped_text``'s character-slice pre-check bounds character
    *count*, not the escaped-JSON byte count the function actually caps
    against -- for an all-ASCII text (the shape every other byte-cap
    test in this module uses) that pre-slice usually still lands close
    enough to the cap to take the function's early-return branch. A
    multi-byte character (every "中" is 3 UTF-8 bytes but a 6-byte
    ``\\uXXXX`` escape on the wire) blows well past the cap even after
    that pre-slice, which is what exercises the function's other
    branch: a binary search over the character-sliced prefix for the
    longest one whose escaped bytes still fit (see the function's own
    docstring for why this replaces a raw UTF-8 byte-slice, which is
    unsafe once the measured domain is escaped bytes instead of decoded
    ones).
    """
    oversized = "中" * (es.MAX_FRAME_CONTENT_BYTES + 1000)
    capped, truncated = es._capped_text(oversized)
    assert truncated is True
    assert len(capped.encode("utf-8")) <= es.MAX_FRAME_CONTENT_BYTES
    # The cap is on the escaped wire form, so this is the assertion that
    # discriminates. A decoded-UTF-8-byte implementation returns 21845
    # characters for this input -- 65535 decoded bytes, satisfying the
    # assertion above -- while its escaped form measures 131072 bytes,
    # twice the cap.
    assert es._byte_length(capped) <= es.MAX_FRAME_CONTENT_BYTES
    # Escaped-byte-sliced inside the character-sliced prefix: strictly
    # fewer whole characters survive than a pure character-count slice
    # would keep (that would be MAX_FRAME_CONTENT_BYTES characters,
    # escaping to 6x as many wire bytes).
    assert len(capped) < es.MAX_FRAME_CONTENT_BYTES


async def test_step_data_over_the_byte_cap_collapses_to_a_truncation_marker():
    sink = _make_sink(task_id=55)
    oversized_result = "y" * (es.MAX_FRAME_CONTENT_BYTES + 1000)
    await sink.send_text(
        _trace_event_frame(
            "tool_execution_start",
            task_id=55,
            data={"tool_call_id": "call-1", "tool_name": "search", "tool_args": {}},
        )
    )
    sink.queue.get_nowait()  # drain the (small) step.started frame
    await sink.send_text(
        _trace_event_frame(
            "tool_execution_end",
            task_id=55,
            data={
                "tool_call_id": "call-1",
                "success": True,
                "result": oversized_result,
            },
        )
    )
    frame_text, _ = sink.queue.get_nowait()
    step = json.loads(frame_text.split("data: ", 1)[1])["step"]
    # "name" (the tool's identifying field, carried since the start
    # event) survives the truncation marker -- see
    # ``_STEP_DATA_IDENTIFYING_KEYS`` -- only the oversized "result"
    # content is dropped.
    assert step["data"] == {
        "truncated": True,
        "original_bytes": step["data"]["original_bytes"],
        "name": "search",
    }
    assert step["data"]["original_bytes"] > es.MAX_FRAME_CONTENT_BYTES
    # The step's own identity/status fields are untouched by the cap --
    # only its content-bearing ``data`` sub-object is affected.
    assert step["id"] == "tool_call:call-1"
    assert step["status"] == "completed"


async def test_capped_step_data_preserves_identifying_keys_per_step_type():
    """``_capped_step_data`` keeps each public step type's own
    identifying field (``phase``/``name``/``sub_agent_name``/``role``)
    through the truncation marker -- a client reading a truncated frame
    still knows *what* the step was, not just that it was too big."""
    oversized = "z" * (es.MAX_FRAME_CONTENT_BYTES + 1000)
    assert es._capped_step_data({"phase": "planning", "junk": oversized}) == {
        "truncated": True,
        "original_bytes": es._byte_length({"phase": "planning", "junk": oversized}),
        "phase": "planning",
    }
    assert es._capped_step_data(
        {"sub_agent_name": "researcher", "input": oversized}
    ) == {
        "truncated": True,
        "original_bytes": es._byte_length(
            {"sub_agent_name": "researcher", "input": oversized}
        ),
        "sub_agent_name": "researcher",
    }
    assert es._capped_step_data({"role": "assistant", "content": oversized}) == {
        "truncated": True,
        "original_bytes": es._byte_length({"role": "assistant", "content": oversized}),
        "role": "assistant",
    }
    # A key genuinely absent from this step's data is never synthesized.
    small_but_over_cap = {"role": "assistant"}
    assert "phase" not in es._capped_step_data(
        {**small_but_over_cap, "content": oversized}
    )


@pytest.mark.parametrize(
    "huge_name",
    ["n" * 70_000, "名" * 12_000],
    ids=["ascii-name", "cjk-name"],
)
def test_capped_step_data_truncates_an_oversized_identifying_value(huge_name):
    """An identifying value can itself be large enough to blow the cap
    once folded into the truncation marker (the first pass in
    ``_capped_step_data``'s docstring). The second pass must truncate it
    rather than drop it, and the result must still fit under
    ``MAX_FRAME_CONTENT_BYTES``.

    Two cells, two different bugs:
      - ``ascii-name``: truncating the value to the *full* cap and only
        then wrapping it in the marker dict overflows on the marker's own
        JSON overhead alone. This cell is what a per-key budget that
        forgets that overhead fails on.
      - ``cjk-name``: ``per_key_budget`` is computed in escaped-JSON
        bytes, the domain ``_byte_length``'s cap checks use. Before
        ``_capped_text`` measured in that same domain, handing it that
        budget for a CJK name let it keep far more *characters* than the
        escaped budget allowed (UTF-8 costs 3 bytes per "名", the escaped
        ``\\uXXXX`` wire form costs 6), so the reassembled dict was still
        over the cap on its own re-check and the identifying ``name``
        field a client needs to know which tool ran was dropped entirely
        -- not because it was genuinely too large to fit, but because the
        two functions measured in different domains.
    """
    result = es._capped_step_data({"name": huge_name, "junk": "x" * 1000})
    assert es._byte_length(result) <= es.MAX_FRAME_CONTENT_BYTES
    assert result["truncated"] is True
    assert "name" in result  # the truncation pass preserved it -- not dropped
    assert len(result["name"]) < len(huge_name)
    assert result["original_bytes"] > es.MAX_FRAME_CONTENT_BYTES


def test_capped_step_data_drops_only_the_value_it_cannot_shrink():
    """A non-string identifying value (a dict- or list-valued ``name``/
    ``role``, say) can't be truncated in place -- ``_capped_text`` only
    accepts a string. The naive fix (drop it at the final bare-marker
    step and stop) still fails: the un-shrinkable value counts toward
    the first pass's own overhead, driving every *other* key's budget
    to zero, so a survivor that should have kept real content comes
    back as ``""``. The actual fix is a bounded retry: drop only the
    largest surviving value that cannot be truncated in place and
    rebuild the marker from the original data, so the keys that survive
    get a real, non-zero budget."""
    oversized_dict = {"nested": "x" * 70_000}
    result = es._capped_step_data({"name": oversized_dict, "phase": "planning"})
    assert result["truncated"] is True
    assert "name" not in result
    assert result["phase"] == "planning"
    assert result["original_bytes"] > es.MAX_FRAME_CONTENT_BYTES
    assert es._byte_length(result) <= es.MAX_FRAME_CONTENT_BYTES

    result = es._capped_step_data({"role": ["a" * 70_000], "name": "search"})
    assert result["truncated"] is True
    assert "role" not in result
    assert result["name"] == "search"
    assert es._byte_length(result) <= es.MAX_FRAME_CONTENT_BYTES

    # Nothing else to keep once the one identifying value is dropped:
    # the honest limit of this fix, same as before it.
    result = es._capped_step_data({"name": oversized_dict})
    assert result == {
        "truncated": True,
        "original_bytes": result["original_bytes"],
    }

    # Regression guard: the string-only ladder is unchanged -- both
    # values survive, each truncated, when both can be shrunk.
    result = es._capped_step_data({"name": "n" * 70_000, "role": "r" * 70_000})
    assert result["truncated"] is True
    assert "name" in result and "role" in result
    assert 0 < len(result["name"]) < 70_000
    assert 0 < len(result["role"]) < 70_000
    assert es._byte_length(result) <= es.MAX_FRAME_CONTENT_BYTES


@pytest.mark.parametrize(
    ("data", "dropped_key", "kept_key", "kept_original"),
    [
        pytest.param(
            {"sub_agent_name": {"k": "x" * 70_000}, "role": "assistant" * 9_000},
            "sub_agent_name",
            "role",
            "assistant" * 9_000,
            id="oversized-dict-beats-oversized-string",
        ),
        pytest.param(
            {"name": {"k": "x" * 66_000}, "phase": "p" * 70_000},
            "name",
            "phase",
            "p" * 70_000,
            id="smaller-dict-still-dropped-first",
        ),
    ],
)
def test_capped_step_data_prefers_dropping_a_non_string_over_a_truncatable_one(
    data, dropped_key, kept_key, kept_original
):
    """When both a truncatable string and an un-shrinkable non-string
    survive a failed pass, the non-string is dropped first -- preferring
    to drop the salvageable string would degrade a step that could have
    kept real content down to the bare marker for no reason. Per the
    docstring, a pass over strings only always fits within the cap, so
    reaching the drop step guarantees at least one non-string survivor
    remains; falling back to the largest overall is unreachable here.

    The second cell is the one that pins "non-string first" rather than
    "largest first": there the dict is the *smaller* of the two values, so
    a size-only rule would drop the string instead.
    """
    result = es._capped_step_data(data)
    assert result["truncated"] is True
    assert dropped_key not in result
    assert kept_key in result
    assert 0 < len(result[kept_key]) < len(kept_original)
    assert es._byte_length(result) <= es.MAX_FRAME_CONTENT_BYTES


def test_capped_text_bounds_emoji_content_in_the_escaped_wire_byte_domain():
    """``_sse_frame`` serializes with ``json.dumps``'s default
    ``ensure_ascii=True``, so a non-BMP character like an emoji escapes
    to a 12-byte ``\\uD83D\\uDE00`` surrogate pair on the wire -- 3x its
    own 4-byte UTF-8 width. 16384 emoji is exactly
    ``MAX_FRAME_CONTENT_BYTES`` in UTF-8 bytes, the domain a decoded-byte
    cap check would measure -- a check in that domain would let this
    input sail through completely untruncated, while its escaped wire
    form runs to roughly 3x the cap. ``_capped_text`` measures the
    escaped form itself, so it catches this case.
    """
    oversized = "\U0001f600" * 16384
    capped, truncated = es._capped_text(oversized)
    assert truncated is True
    assert es._byte_length(capped) <= es.MAX_FRAME_CONTENT_BYTES
    assert capped != oversized


async def test_message_delta_with_cjk_content_is_capped_on_the_wire_not_decoded_bytes():
    """A CJK string long enough to fit under a decoded-UTF-8-byte cap (3
    bytes/character) but not under its escaped wire form (6 bytes/
    character via ``\\uXXXX``, ``ensure_ascii=True``) would reach the
    client whole under that measurement -- 12000 "中" characters is
    36000 UTF-8 bytes (well under ``MAX_FRAME_CONTENT_BYTES``) but
    roughly 72000 escaped wire bytes (over it). Pins that
    ``message.delta``'s ``text`` field is capped in the same domain the
    frame is actually serialized in."""
    sink = _make_sink(task_id=57)
    oversized = "中" * 12_000
    await sink.send_text(
        json.dumps(
            {
                "type": "final_answer_delta",
                "message_id": "final_answer_cjk",
                "task_id": 57,
                "delta": oversized,
            }
        )
    )
    frame_text, _ = sink.queue.get_nowait()
    data = json.loads(frame_text.split("data: ", 1)[1])
    assert data["truncated"] is True
    assert es._byte_length(data["text"]) <= es.MAX_FRAME_CONTENT_BYTES
    assert data["text"] != oversized


async def test_nonfinite_step_data_values_are_normalized_to_null_on_the_wire():
    """``PublicStep.data`` is typed ``Dict[str, Any]`` (see
    ``xagent.web.schemas.v1``), so nothing stops a tool result or
    delegation payload from carrying a raw ``float('nan')``/``inf``/
    ``-inf`` -- Python's ``json.dumps`` (used by ``_sse_frame``, the
    function every frame builder in this module funnels through)
    accepts those by default and emits the bare, non-standard tokens
    ``NaN``/``Infinity``/``-Infinity``, which a strict ``JSON.parse``
    client rejects outright.

    A ``step.*`` frame's only producer, live projection
    (``_step_content_frame``), funnels
    through ``_step_wire_frame``, which receives its dict only after a
    ``PublicStep(**step).model_dump(mode="json")`` call. Pydantic v2's
    JSON serialization mode normalizes non-finite floats to ``null`` for
    any field typed ``Any`` (verified directly below), independent of a
    field's nesting depth -- so every ``step.*`` frame this module emits
    is already immune to this failure mode with no additional guard in
    ``_step_wire_frame`` itself. This test pins that implicit
    invariant by handing three-deep nested non-finite values to
    ``_step_content_frame`` -- production's only caller of
    ``_step_wire_frame``, and where the normalization actually happens
    -- and asserting the resulting wire text is strict-JSON-clean. The
    raw dict goes in exactly as a projector produces it; normalizing it
    in test setup first would have left the assertions green even if
    ``_step_content_frame`` regressed to passing the raw dict straight
    through. It changes no production code -- if that regression ever
    happens, this test is the tripwire.
    """
    nested_nonfinite_data = {
        "name": "search",
        "args": {
            "level_one": {
                "level_two": {
                    "level_three": [float("nan"), float("inf"), float("-inf")]
                },
                "solo_nan": float("nan"),
            }
        },
        "result": float("inf"),
    }
    frame_text = es._step_content_frame(
        {
            "id": "tool_call:call-nonfinite",
            "type": "tool_call",
            "status": "completed",
            "started_at": datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
            "completed_at": datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC),
            "data": nested_nonfinite_data,
        }
    )

    assert "NaN" not in frame_text
    assert "Infinity" not in frame_text

    payload = frame_text.split("data: ", 1)[1].strip()

    def _reject_nonfinite(constant_text: str) -> None:
        pytest.fail(
            f"strict JSON parse hit a non-finite constant token: {constant_text!r}"
        )

    parsed = json.loads(payload, parse_constant=_reject_nonfinite)
    wire_args = parsed["step"]["data"]["args"]
    assert wire_args["level_one"]["level_two"]["level_three"] == [None, None, None]
    assert wire_args["level_one"]["solo_nan"] is None
    assert parsed["step"]["data"]["result"] is None


async def test_stream_projector_keeps_no_finished_steps_for_the_connection():
    """The sink's projector is built with ``retain_finished=False`` (see
    ``V1EventStreamSink.__init__``): it acts on each ``feed()`` result
    immediately by serializing it to a frame, and never calls
    ``materialized_steps()``, so retaining every finalized step would
    hold each one's untruncated ``data`` for as long as the connection
    lives. Feeds 50 tool-call pairs plus one message -- enough that the
    fold definitely ran repeatedly, not just once -- and checks what's
    left behind rather than trusting it was never accumulated.
    """
    sink = _make_sink(task_id=99)
    for i in range(50):
        await sink.send_text(
            _trace_event_frame(
                "tool_execution_start",
                task_id=99,
                step_id=f"step-{i}",
                data={
                    "tool_call_id": f"call-{i}",
                    "tool_name": "search",
                    "tool_args": {"query": f"q{i}"},
                },
            )
        )
        await sink.send_text(
            _trace_event_frame(
                "tool_execution_end",
                task_id=99,
                step_id=f"step-{i}",
                data={"tool_call_id": f"call-{i}", "success": True, "result": "ok"},
            )
        )
    await sink.send_text(
        _trace_event_frame(
            "ai_message",
            task_id=99,
            data={"content": "done"},
        )
    )
    queued = []
    while not sink.queue.empty():
        queued.append(sink.queue.get_nowait())
    assert len(queued) == 101
    assert sink._projector._finished is None
    assert sink._projector._pending == {}


# ===== two-path consistency: the same trace event sequence collapses to
# the same PublicStep whether read via steps() or via the live stream =====


async def test_live_projection_matches_steps_endpoint_for_the_same_events():
    """Same trace event sequence, same collapsed ``PublicStep`` --
    ``id``/``type``/``status``/``data``/``started_at``/``completed_at``
    -- whether read through ``steps()`` (the persisted-history path) or
    through this stream's live projection (the same
    ``PublicStepProjector``, fed the broadcast-shaped frame instead of
    the ORM row). Both surfaces derive from the same two
    ``xagent.core.agent.trace.TraceEvent`` objects through the real
    conversion pipeline -- ``_persist_core_event`` for the ``steps()``
    leg, ``_broadcast_frame_for`` for the live leg -- so a change on
    either side of that pipeline shows up here rather than being masked
    by two independently hand-built literals. Fixture data carries no
    credential-shaped keys, so both redaction call sites are a no-op and
    cannot mask a divergence. Message steps are excluded here -- see the
    divergence test below for that one, explicitly accepted difference.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    start_event = CoreTraceEvent(
        ACTION_START_TOOL,
        step_id="step-1",
        timestamp=base.timestamp(),
        data={
            "tool_call_id": "call-1",
            "tool_name": "search",
            "tool_args": {"q": "weather"},
        },
    )
    end_event = CoreTraceEvent(
        ACTION_END_TOOL,
        step_id="step-1",
        timestamp=base.timestamp(),
        data={"tool_call_id": "call-1", "success": True, "result": "sunny"},
    )
    _persist_core_event(start_event, task_id=task_id)
    _persist_core_event(end_event, task_id=task_id)

    steps_resp = client.get(
        f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
    )
    assert steps_resp.status_code == 200
    steps_via_polling = steps_resp.json()["steps"]
    assert len(steps_via_polling) == 1

    sink = _make_sink(task_id=task_id)
    await sink.send_text(_broadcast_frame_for(start_event, task_id=task_id))
    sink.queue.get_nowait()  # the step.started frame -- not under test here
    await sink.send_text(_broadcast_frame_for(end_event, task_id=task_id))
    frame_text, _ = sink.queue.get_nowait()
    step_via_live = json.loads(frame_text.split("data: ", 1)[1])["step"]

    polled = steps_via_polling[0]
    assert step_via_live["id"] == polled["id"]
    assert step_via_live["type"] == polled["type"]
    assert step_via_live["status"] == polled["status"]
    assert step_via_live["data"] == polled["data"]
    assert step_via_live["started_at"] == polled["started_at"]
    assert step_via_live["completed_at"] == polled["completed_at"]


async def test_planning_step_ids_count_per_connection_and_collide_with_steps():
    """Known divergence, not a bug, for a ``thinking:plan:``/``thinking:
    planning:`` step id -- pins the endpoint docstring's carve-out.

    Two planning cycles happen on the task: cycle 1 opens and closes
    (``dag_execution`` phase ``planning`` then ``executing``), cycle 2
    opens and is still running. ``steps()`` replays the whole persisted
    history, so it numbers them ``:1`` and ``:2`` in order. A fresh sink
    -- a client attaching mid-task -- only ever sees cycle 2's start
    broadcast (cycle 1 already happened before it attached), so its own
    projector, starting empty, numbers that as its *first* observed
    planning cycle: ``:1``. That id is not a mismatch, it's an outright
    collision -- it equals ``steps()``'s id for cycle 1's *different*
    step, not merely differing from cycle 2's own id. ``data`` and
    ``started_at`` still agree with cycle 2's row, which is what makes
    the id divergence the only thing wrong with treating the two as the
    same step.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    dag_execution_type = TraceEventType(
        TraceScope.TASK, TraceAction.UPDATE, TraceCategory.DAG
    )
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC)
    t2 = datetime(2026, 1, 1, 12, 0, 10, tzinfo=UTC)
    cycle1_start = CoreTraceEvent(
        dag_execution_type,
        task_id=str(task_id),
        timestamp=t0.timestamp(),
        data={"phase": "planning"},
    )
    cycle1_end = CoreTraceEvent(
        dag_execution_type,
        task_id=str(task_id),
        timestamp=t1.timestamp(),
        data={"phase": "executing"},
    )
    cycle2_start = CoreTraceEvent(
        dag_execution_type,
        task_id=str(task_id),
        timestamp=t2.timestamp(),
        data={"phase": "planning"},
    )
    _persist_core_event(cycle1_start, task_id=task_id)
    _persist_core_event(cycle1_end, task_id=task_id)
    _persist_core_event(cycle2_start, task_id=task_id)

    steps_resp = client.get(
        f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
    )
    assert steps_resp.status_code == 200
    thinking_steps = [s for s in steps_resp.json()["steps"] if s["type"] == "thinking"]
    assert len(thinking_steps) == 2
    step_one = next(
        s for s in thinking_steps if s["id"] == f"thinking:planning:{task_id}:1"
    )
    step_two = next(
        s for s in thinking_steps if s["id"] == f"thinking:planning:{task_id}:2"
    )
    assert step_one["started_at"] != step_two["started_at"]

    sink = _make_sink(task_id=task_id)
    await sink.send_text(_broadcast_frame_for(cycle2_start, task_id=task_id))
    frame_text, _ = sink.queue.get_nowait()
    live_step = json.loads(frame_text.split("data: ", 1)[1])["step"]

    # The live id is the collision, not merely a mismatch: it equals the
    # *other*, earlier step's steps() id rather than merely differing
    # from the step it actually is.
    assert live_step["id"] == f"thinking:planning:{task_id}:1"
    assert live_step["id"] != step_two["id"]
    assert live_step["id"] == step_one["id"]
    # Divergence is confined to the id: content and timing still agree
    # with the step this live frame actually is (cycle 2).
    assert live_step["data"] == step_two["data"]
    assert live_step["started_at"] == step_two["started_at"]


async def test_known_divergence_message_step_id_differs_between_the_two_paths():
    """Accepted divergence, not a bug: an ``ai_message``'s ``PublicStep``
    id is the persisted trace ``event_id`` via ``steps()`` (the row's own
    primary identifier) but a fresh uuid4 minted per broadcast frame via
    the live stream (``create_stream_event`` mints a new one on every
    send) -- the two paths can't share this id, so a client must not try
    to correlate a message step across ``steps()`` and the live stream by
    id. Content itself still matches -- ``data`` is the only field
    compared below; ``started_at``/``completed_at`` aren't, since this
    fixture doesn't mirror a timestamp into the live frame either.

    Both legs fork from one ``CoreTraceEvent`` through the real
    producers -- ``_persist_core_event`` for the persisted row,
    ``_broadcast_frame_for`` for the live frame -- so neither id is
    written down here. If production ever forwarded the canonical event
    id to the live frame, the inequality below fails; if uuid minting
    changed shape, the uuid assertion fails. This test takes no position
    on what the public id contract *should* be -- that is tracked
    elsewhere."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    event = CoreTraceEvent(
        AI_MESSAGE,
        task_id=str(task_id),
        timestamp=base.timestamp(),
        data={"content": "the answer"},
    )
    _persist_core_event(event, task_id=task_id)

    steps_resp = client.get(
        f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
    )
    polled = steps_resp.json()["steps"][0]
    # The persisted leg's id is the event's own id, written by the
    # persistence path as ``event_id=str(event.id)``.
    assert polled["id"] == f"message:{event.id}"

    sink = _make_sink(task_id=task_id)
    await sink.send_text(_broadcast_frame_for(event, task_id=task_id))
    frame_text, _ = sink.queue.get_nowait()
    step_via_live = json.loads(frame_text.split("data: ", 1)[1])["step"]
    # The live leg's id is minted per broadcast by ``create_stream_event``,
    # so it is a fresh uuid that is neither the event's id nor stable
    # across two conversions of the same event.
    live_id = step_via_live["id"].removeprefix("message:")
    assert uuid.UUID(live_id).version == 4
    assert live_id != str(event.id)
    assert step_via_live["id"] != polled["id"]
    assert step_via_live["data"] == polled["data"]


async def test_live_projection_matches_steps_for_a_streamed_final_answer():
    """``data``/``type``/``status`` parity for an ``ai_message`` that
    also carries ``stream_message_id`` -- the persisted mirror of a
    final answer that is ALSO delivered live via
    ``message.delta``/``message.completed`` on this same stream. The
    two-path parity tests above deliberately exclude message steps, so
    this combination gets its coverage here: the live fold projects a
    ``message`` step for it (see
    ``test_trace_event_ai_message_with_stream_message_id_also_projects_a_step``),
    and this pins that the step carries the *same* content ``steps()``
    shows for the persisted row. The live stream additionally emits the
    ``message.delta``/``message.completed`` frames for the same content
    -- duplication the contract accepts in exchange for parity -- not
    asserted again here since
    ``test_final_answer_delta_projects_message_delta_frame`` and its
    ``_completed`` counterpart already pin that half independently.
    ``id`` is excluded below for the same reason the plain-``ai_message``
    divergence test above excludes it (a fresh uuid4 per live frame);
    ``started_at``/``completed_at`` are compared, with the live frame's
    timestamp mirroring the persisted event's -- both derive from the
    same event timestamp field regardless of path (see
    ``_step_mapping.py``'s ``_ts``), so a client can rely on those two
    fields matching even though ``id`` never will."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    _insert_trace_event(
        task_id=task_id,
        event_type="ai_message",
        event_id="persisted-final-answer",
        timestamp=base,
        data={"content": "the answer", "stream_message_id": "final_answer_xyz"},
    )

    steps_resp = client.get(
        f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
    )
    assert steps_resp.status_code == 200
    steps_via_polling = steps_resp.json()["steps"]
    assert len(steps_via_polling) == 1
    polled = steps_via_polling[0]
    assert polled["data"] == {"role": "assistant", "content": "the answer"}

    sink = _make_sink(task_id=task_id)
    await sink.send_text(
        _trace_event_frame(
            "ai_message",
            task_id=task_id,
            event_id="live-final-answer-uuid",
            timestamp=base.timestamp(),
            data={"content": "the answer", "stream_message_id": "final_answer_xyz"},
        )
    )
    frame_text, _ = sink.queue.get_nowait()
    step_via_live = json.loads(frame_text.split("data: ", 1)[1])["step"]
    # Same accepted id divergence as the plain-ai_message case above --
    # data/type/status/timestamps are compared below, id is not.
    assert (
        step_via_live["data"]
        == polled["data"]
        == {
            "role": "assistant",
            "content": "the answer",
        }
    )
    assert step_via_live["type"] == polled["type"] == "message"
    assert step_via_live["status"] == polled["status"] == "completed"
    assert step_via_live["started_at"] == polled["started_at"]
    assert step_via_live["completed_at"] == polled["completed_at"]
    assert sink.queue.empty()


async def test_delegated_child_events_excluded_from_both_paths_consistently():
    """A delegated child agent's own trace events never reach a public
    step on either path -- but the two paths exclude them via different
    checks, not the same predicate: ``steps()`` excludes via its
    ``TraceEvent.build_id IS NULL`` column filter, the live path
    excludes via the ``data["source"]`` field filter (see
    ``test_trace_event_delegated_child_source_is_filtered``). They agree
    because one producer stamps both on a delegated child's events;
    nothing enforces that they keep agreeing. Verifies the two
    independent mechanisms agree in practice: the resulting step id sets
    match, not just that each mechanism works in isolation -- and
    exercises the REST-side filter for real by persisting a row with a
    non-null ``build_id`` (rather than relying on none existing to
    exclude), so ``polled_ids`` matching means the filter actually
    excluded something. Only id-set membership is compared here -- no
    per-field (let alone timestamp) comparison, since the point under
    test is exclusion, not content."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_start",
        event_id="hist-1",
        timestamp=base,
        step_id="step-1",
        data={"tool_call_id": "call-1", "tool_name": "search", "tool_args": {}},
    )
    # A delegated child's own persisted row -- excluded from steps() by
    # its non-null build_id, not by an absence of matching rows.
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_start",
        event_id="hist-child-1",
        timestamp=base,
        step_id="step-2",
        build_id="child-build-1",
        data={
            "tool_call_id": "call-child-1",
            "tool_name": "child_tool",
            "tool_args": {},
        },
    )

    steps_resp = client.get(
        f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
    )
    polled_ids = {step["id"] for step in steps_resp.json()["steps"]}
    assert polled_ids == {"tool_call:call-1"}

    sink = _make_sink(task_id=task_id)
    await sink.send_text(
        _trace_event_frame(
            "tool_execution_start",
            task_id=task_id,
            step_id="step-1",
            data={"tool_call_id": "call-1", "tool_name": "search", "tool_args": {}},
        )
    )
    started_frame_text, _ = sink.queue.get_nowait()
    live_ids = {json.loads(started_frame_text.split("data: ", 1)[1])["step"]["id"]}

    # The delegated child agent's own trace event -- same shape a real
    # broadcast for it would have, filtered by data.source before it
    # ever reaches the projector (see _feed_trace_event).
    await sink.send_text(
        _trace_event_frame(
            "tool_execution_start",
            task_id=task_id,
            step_id="step-2",
            data={
                "tool_call_id": "call-child-1",
                "tool_name": "child_tool",
                "tool_args": {},
                "source": "xagent-agent-tool-child",
            },
        )
    )
    assert sink.queue.empty()  # nothing else was projected for the child event

    assert live_ids == polled_ids


async def test_an_unclosed_dag_planning_phase_projects_a_running_thinking_step():
    """Two-path consistency for a task whose *planning* phase never
    resolves because no ``dag_execute_end`` ever arrives to close it.

    This fixture only ever sends the ``dag_execution{phase: "planning"}``
    start, with no matching ``phase: "executing"`` end and no
    ``dag_execute_end`` at all -- so on *both* paths the ``thinking``
    step is left at ``status: "running"`` forever; there is no event
    either path treats as a close for it. That's what this test pins:
    the two paths still agree with each other on the open-ended case.
    ``_step_mapping.py`` does consume ``dag_execute_end``: see
    ``test_dag_execute_end_closes_the_planning_step_as_failed``,
    directly below, for the case where that event does arrive and
    closes the step as ``failed``. This fixture is deliberately kept
    separate rather than repurposed for that outcome, since the point
    made here is specifically about the still-open, never-closed case.

    Scope of what this pins: each surface's own outcome for this event
    shape, from a literal built per surface. It is not producer-faithful
    parity coverage -- that lives in
    ``test_live_projection_matches_steps_endpoint_for_the_same_events``
    and ``test_planning_step_ids_count_per_connection_and_collide_with_steps``,
    which fork both legs from one shared ``CoreTraceEvent`` through the
    real production pipelines, so a change on either pipeline shows up
    there. ``type``/``status``/``data``/``started_at``/``completed_at``
    are compared across the two legs below; ``id`` is deliberately not
    (see the comment at the assertions).
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    _insert_trace_event(
        task_id=task_id,
        event_type="dag_execution",
        event_id="hist-1",
        timestamp=base,
        data={"pattern": "DAGPattern", "phase": "planning"},
    )
    _set_task_status(task_id, TaskStatus.FAILED, error_message="plan validation failed")

    steps_resp = client.get(
        f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
    )
    assert steps_resp.status_code == 200
    steps_via_polling = steps_resp.json()["steps"]
    assert len(steps_via_polling) == 1
    assert steps_via_polling[0]["status"] == "running"
    assert steps_via_polling[0]["data"] == {"phase": "planning"}

    sink = _make_sink(task_id=task_id)
    await sink.send_text(
        _trace_event_frame(
            "dag_execution",
            task_id=task_id,
            event_id="hist-1",
            timestamp=base.timestamp(),
            data={"pattern": "DAGPattern", "phase": "planning"},
        )
    )
    frame_text, _ = sink.queue.get_nowait()
    assert frame_text.startswith("event: step.started\n")
    step_via_live = json.loads(frame_text.split("data: ", 1)[1])["step"]

    polled = steps_via_polling[0]
    # ``id`` is not compared across the two legs on purpose: a
    # ``thinking:planning:`` id ends in the projecting side's own count
    # of planning cycles, which the endpoint docstring documents as not
    # comparable across surfaces (``test_planning_step_ids_count_per_
    # connection_and_collide_with_steps`` pins the collision). Both legs
    # here are that count's first value, so asserting them equal would
    # pin a fixture coincidence rather than the contract. ``started_at``
    # and content are what the docstring tells clients to reconcile on,
    # and they are asserted below.
    assert step_via_live["type"] == polled["type"]
    assert step_via_live["status"] == polled["status"] == "running"
    assert step_via_live["data"] == polled["data"]
    assert step_via_live["started_at"] == polled["started_at"]
    assert step_via_live["completed_at"] == polled["completed_at"]
    assert sink.queue.empty()  # no further frame -- nothing ever closes it


async def test_dag_execute_end_closes_the_planning_step_as_failed():
    """Two-path consistency for a closed planning failure: a full
    ``dag_execute_start`` -> ``dag_execution{phase: "planning"}`` ->
    ``dag_execute_end{status: "failed"}`` sequence, where the trailing
    event reaches into the still-open planning step and closes it
    as ``failed`` (see ``_step_mapping.py``'s ``dag_execute_end``
    branch). Companion to
    ``test_an_unclosed_dag_planning_phase_projects_a_running_thinking_step``
    above, which pins the still-open case this one's ``dag_execute_end``
    resolves -- that test's fixture is intentionally left alone rather
    than being turned into this one.

    ``dag_execute_start`` carries no step of its own (it only clears a
    stale open key from a prior round -- see the module docstring's
    ``dag_execute_start`` paragraph), so it is expected to project no
    frame at all on the live path; that's asserted explicitly below
    rather than just being skipped over.

    Same scope note as the companion test above: this pins each
    surface's own outcome for this event shape, from a literal built per
    surface, and ``id`` is deliberately not compared across the two
    legs. Producer-faithful parity coverage lives in the two
    shared-``CoreTraceEvent`` tests.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    plan_start = datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC)
    plan_end = datetime(2026, 1, 1, 12, 0, 2, tzinfo=UTC)
    _insert_trace_event(
        task_id=task_id,
        event_type="dag_execute_start",
        event_id="hist-0",
        timestamp=base,
        data={"pattern": "DAGPattern"},
    )
    _insert_trace_event(
        task_id=task_id,
        event_type="dag_execution",
        event_id="hist-1",
        timestamp=plan_start,
        data={"pattern": "DAGPattern", "phase": "planning"},
    )
    _insert_trace_event(
        task_id=task_id,
        event_type="dag_execute_end",
        event_id="hist-2",
        timestamp=plan_end,
        data={"status": "failed", "result": {"success": False}},
    )
    _set_task_status(task_id, TaskStatus.FAILED, error_message="plan validation failed")

    steps_resp = client.get(
        f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
    )
    assert steps_resp.status_code == 200
    steps_via_polling = steps_resp.json()["steps"]
    assert len(steps_via_polling) == 1
    assert steps_via_polling[0]["status"] == "failed"
    assert steps_via_polling[0]["data"] == {"phase": "planning"}

    sink = _make_sink(task_id=task_id)
    await sink.send_text(
        _trace_event_frame(
            "dag_execute_start",
            task_id=task_id,
            timestamp=base.timestamp(),
            data={"pattern": "DAGPattern"},
        )
    )
    assert sink.queue.empty()  # dag_execute_start projects no step of its own

    await sink.send_text(
        _trace_event_frame(
            "dag_execution",
            task_id=task_id,
            timestamp=plan_start.timestamp(),
            data={"pattern": "DAGPattern", "phase": "planning"},
        )
    )
    started_frame_text, _ = sink.queue.get_nowait()
    assert started_frame_text.startswith("event: step.started\n")

    await sink.send_text(
        _trace_event_frame(
            "dag_execute_end",
            task_id=task_id,
            timestamp=plan_end.timestamp(),
            data={"status": "failed", "result": {"success": False}},
        )
    )
    completed_frame_text, _ = sink.queue.get_nowait()
    assert completed_frame_text.startswith("event: step.completed\n")
    step_via_live = json.loads(completed_frame_text.split("data: ", 1)[1])["step"]

    polled = steps_via_polling[0]
    # ``id`` deliberately not compared across the legs -- see the
    # companion test above for why a planning id's trailing count is a
    # per-surface value.
    assert step_via_live["type"] == polled["type"]
    assert step_via_live["status"] == polled["status"] == "failed"
    assert step_via_live["data"] == polled["data"] == {"phase": "planning"}
    assert step_via_live["started_at"] == polled["started_at"]
    assert step_via_live["completed_at"] == polled["completed_at"]
    assert sink.queue.empty()  # no further frame after the close
