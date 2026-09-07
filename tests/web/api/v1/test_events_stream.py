"""Tests for the v1 SSE transport layer (``GET /v1/chat/tasks/{id}/events``).

Covers registration, the sink's frame filter and exception discipline,
the watchdog, the 1-hour cap, concurrency caps, and the ``step.*`` /
``message.*`` content projection layered on top of that transport
(classification, filtering, warm-up replay, and the fast paths' cached
step snapshot). Tests either drive ``_events_stream`` directly (fast,
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
from types import SimpleNamespace
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
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    set_cache_backend_for_testing,
)

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


def _current_watermark(task_id: int, principal: ApiKeyPrincipal) -> int:
    """The replay watermark a real attach captures right after
    registration, read the same way ``_generate`` reads it.

    ``_build_warm_up_frames`` requires one -- there is no "replay
    everything" mode -- so a test that wants the task's whole history
    admitted asks for the current cursor rather than inventing a
    sentinel that could drift from what the endpoint would have used.
    """
    return int(
        v1_tasks._load_task_steps_version_snapshot(task_id, principal).max_event_id
    )


def _make_sink(
    task_id: int = 1, status: str = "running", *, warm: bool = True
) -> es.V1EventStreamSink:
    """``warm=True`` (the default) finishes warm-up immediately with an
    empty projector, so a bare ``send_text`` call projects content
    frames right away instead of staging them -- what every content-
    projection test here wants except the ones that specifically test
    staging/warm-up itself (``warm=False``, see the warm-up section).

    ``retain_finished=False`` because ``finish_warm_up`` requires it --
    a live sink never reads a timeline back, so a retaining projector is
    rejected rather than left to accumulate one (see that method)."""
    sink = es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix="pfx-test", initial_status=status
    )
    if warm:
        sink.finish_warm_up(es.PublicStepProjector(retain_finished=False))
    return sink


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


def _parse_error_frame(body: str) -> dict:
    """Extract and parse the one ``stream.error`` frame's JSON payload out
    of a full SSE response body.

    Asserts there is exactly one -- a caller relying on the returned
    ``code``/``message`` needs the frame that actually closed the
    stream, not whichever one a substring search happened to match, and
    a body with zero or more than one ``stream.error`` frame is itself a
    sign the test's premise is wrong.
    """
    blocks = [b for b in body.split("\n\n") if b.strip()]
    error_blocks = [b for b in blocks if b.startswith("event: stream.error")]
    assert len(error_blocks) == 1, (
        f"expected exactly one stream.error frame, got {len(error_blocks)}: {body!r}"
    )
    return json.loads(error_blocks[0].split("data: ", 1)[1])


# ===== error_frame code validation =====


def test_error_frame_rejects_an_unknown_code_with_or_without_a_message():
    """``_ERROR_MESSAGES[code]`` is looked up unconditionally, so an
    unrecognized code raises ``KeyError`` whether or not a ``message``
    override was also passed -- the check is on the code, not on which
    optional argument the caller supplied."""
    with pytest.raises(KeyError):
        es.error_frame("nope")
    with pytest.raises(KeyError):
        es.error_frame("nope", message="anything")

    # A known code still honors the message override.
    frame = es.error_frame("task_deleted", message="custom text")
    data = json.loads(frame.split("data: ", 1)[1])
    assert data == {"code": "task_deleted", "message": "custom text"}


def test_error_frame_keeps_an_explicitly_empty_message():
    """``message or default_message`` used to fold an explicitly empty
    string into the code's default wording -- indistinguishable from
    the caller never having passed an override at all. Only an omitted
    override (``None``) should select the default; an explicit ``""``
    is a value the caller chose and must reach the client as-is."""
    frame = es.error_frame("task_deleted", message="")
    data = json.loads(frame.split("data: ", 1)[1])
    assert data == {"code": "task_deleted", "message": ""}


def test_error_frame_defaults_to_the_code_s_own_message():
    """The default wording is part of what a client reads; a dedup pass
    left it with no assertion of its own."""
    frame = es.error_frame("task_deleted")
    data = json.loads(frame.split("data: ", 1)[1])
    assert data == {"code": "task_deleted", "message": "The task no longer exists."}


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


@pytest.mark.parametrize(
    ("status", "output", "error_message", "expected_completed_data"),
    [
        pytest.param(
            TaskStatus.COMPLETED,
            "the answer",
            None,
            {"status": "completed", "output": "the answer", "error": None},
            id="completed",
        ),
        pytest.param(
            TaskStatus.FAILED,
            None,
            "the tool call raised",
            {"status": "failed", "output": None, "error": "the tool call raised"},
            id="failed",
        ),
    ],
)
def test_events_terminal_task_immediate_close(
    status, output, error_message, expected_completed_data
):
    """Both terminal statuses (``_TERMINAL_STATUSES``: completed and
    failed) take this fast path and their own field -- ``output`` for a
    completed task, ``error`` for a failed one -- has to actually reach
    the ``task.completed`` frame; a test pinning only the completed leg
    would miss a regression that dropped ``error`` from the frame while
    leaving ``output`` alone."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _set_task_status(task_id, status, output=output, error_message=error_message)

    resp = client.get(f"/v1/chat/tasks/{task_id}/events", headers=_bearer(full_key))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert body.count("event: task.status") == 1
    assert body.count("event: task.completed") == 1
    blocks = [b for b in body.split("\n\n") if b.strip()]
    status_data = json.loads(blocks[0].split("data: ", 1)[1])
    completed_data = json.loads(blocks[1].split("data: ", 1)[1])
    assert status_data == {"status": status.value}
    assert completed_data == expected_completed_data
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
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
        **_long_intervals(),
    )
    try:
        assert resp.headers["x-accel-buffering"] == "no"
        assert resp.headers["cache-control"] == "no-cache"
    finally:
        await resp.body_iterator.aclose()


async def test_fast_path_failure_sends_the_response_start_before_its_close_frames():
    """The fast paths' failure exits (see
    ``_fast_path_steps_read_error_frame``) rely on ``StreamingResponse``
    having already sent the 200 response start and headers before the
    first chunk is ever pulled from the generator -- that is what makes
    catching the failure and closing with a ``stream.error`` frame
    meaningful instead of redundant. Every other test in this file only
    checks the frame *text* (``resp.text`` via the real HTTP client, or
    the chunks read off ``resp.body_iterator``), never the raw ASGI
    messages the response actually sends on the wire. This is the only
    test in the suite that drives the response's ASGI interface
    directly and inspects the message sequence, to pin that the 200 and
    its headers really do arrive as the first message, before
    ``task.status`` and everything after it -- not merely that the text
    ends up assembled in the right order once it has all been read.

    ``scope["asgi"]["spec_version"]`` is set to ``"2.4"`` so
    ``StreamingResponse.__call__`` takes the branch that awaits
    ``stream_response(send)`` directly without ever touching
    ``receive``. ``receive`` here is defined but never invoked on that
    branch; it is shaped to await forever rather than return or raise,
    so that on the older branch (which awaits ``receive`` in a second
    task and cancels it once ``stream_response`` finishes) this test
    would still behave correctly instead of finishing early or
    erroring -- both branches are correct with this ``receive``, only
    the 2.4 one is actually exercised here.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    _set_task_status(task_id, TaskStatus.COMPLETED, output="done")
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    def _broken_read_task_steps_response(task_id_, principal_):
        raise RuntimeError("boom - transient read failure")

    def _unreachable_read_task_snapshot(task_id_, principal_):
        raise AssertionError(
            "the generation reread must not run when the steps read itself failed"
        )

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=_unreachable_read_task_snapshot,
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=_broken_read_task_steps_response,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
    )

    messages: list[dict] = []

    async def send(message):
        messages.append(message)

    async def receive():
        await asyncio.Event().wait()

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}}
    await resp(scope, receive, send)

    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 200
    headers = dict(messages[0]["headers"])
    assert headers[b"content-type"].startswith(b"text/event-stream")
    assert headers[b"x-accel-buffering"] == b"no"
    assert all(message["type"] == "http.response.body" for message in messages[1:])

    body = b"".join(message["body"] for message in messages[1:]).decode()
    # Conclusion-first, error-second -- the same ordering every other
    # steps-read-failure test in this file pins on the assembled text;
    # this test's own job is only the ASGI message layer around it.
    assert body.index("event: task.status") < body.index("event: task.completed")
    assert body.index("event: task.completed") < body.index("event: stream.error")
    assert "resync_required" in body

    last_message = messages[-1]
    assert last_message["body"] == b""
    assert last_message["more_body"] is False


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
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
    assert sink.abortive_close is True
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
    assert sink.abortive_close is False
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
    assert sink.abortive_close is True
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
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
            read_task_steps=v1_tasks._load_task_steps_snapshot,
            read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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

    def _empty_steps(task_id, principal):
        # Real, working (unlike ``_never_read`` above) -- the warm-up
        # read in ``_generate`` always runs, unconditionally, right
        # after the first yield. No history for this sentinel task_id,
        # so warm-up completes with zero replay frames and the test's
        # hand-driven ``asend`` sequence below still lands on the next
        # ``yield`` being the while loop's first dequeue, as before.
        return SimpleNamespace(events=())

    def _steps_cursor(task_id, principal):
        # Also real and working, for the same reason: the watermark read
        # runs unconditionally at registration, and a failure there would
        # close the stream for resync before the sequence below starts.
        return SimpleNamespace(max_event_id=0)

    task_id = 987_654_321  # sentinel, not a real DB row -- unused by this test
    gen = es._generate(
        task_id,
        None,
        key_prefix="pfx-close-frame-race-test",
        initial_status="running",
        read_task_snapshot=_never_read,
        read_task_steps=_empty_steps,
        read_task_steps_version=_steps_cursor,
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


# ===== an abortive close (unauthorized / task_deleted) cuts the warm-up
# replay short; every other close reason lets it finish first =====


async def _three_step_task() -> tuple[int, ApiKeyPrincipal]:
    """A real task with three independent, still-``running`` steps (no
    end event for any of them), so ``_build_warm_up_frames`` replays
    exactly three ``step.started`` frames in a deterministic
    ``started_at`` order -- enough to inject a close after the first one
    and still have at least one more that must (or must not) follow it."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    for index in range(3):
        _insert_trace_event(
            task_id=task_id,
            event_type="tool_execution_start",
            event_id=f"hist-{index}",
            timestamp=base.replace(second=index),
            step_id=f"step-{index}",
            data={
                "tool_call_id": f"call-{index}",
                "tool_name": "search",
                "tool_args": {},
            },
        )
    return task_id, principal


def _revoke_key_for(principal: ApiKeyPrincipal) -> None:
    """Revoke the API key this principal authenticated with -- the same
    row edit ``test_watchdog_closes_within_one_cycle_on_key_invalidation``
    makes, reachable from a principal alone."""
    db = _direct_db_session()
    try:
        key_row = (
            db.query(AgentApiKey)
            .filter(AgentApiKey.key_prefix == principal.key.key_prefix)
            .one()
        )
        key_row.revoked_at = datetime.now(UTC)
        db.commit()
    finally:
        db.close()


async def _poll_until(predicate) -> None:
    """Wait for a state the watchdog reaches on its own cadence.

    Bounded and re-checked rather than one sleep sized to a guessed
    number of cycles: each cycle does two real database reads through
    the shared thread pool, so a fixed sleep is a timing bet that gets
    thinner the more loaded the machine is.
    """
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("the watchdog did not reach the expected state in 4s")


async def test_abortive_close_during_replay_stops_further_history_frames():
    """An abortive close landing between two replayed history frames
    must stop the replay right there -- the client must never see
    history frames queued up after the point its authorization was
    revoked. This drives the close directly via ``sink.enqueue_close(...,
    abortive=True)`` -- the same frame shape (``unauthorized``) the
    watchdog would enqueue, but not through ``watchdog_check_once``
    itself; that production call site is covered separately by
    ``test_watchdog_closes_within_one_cycle_on_key_invalidation``'s own
    ``sink.abortive_close is True`` assertion. Driven by hand via
    ``__anext__`` so the close can be injected deterministically after
    exactly one history frame, the same technique
    ``test_close_frame_delivered_when_enqueued_between_two_yields`` uses
    for the non-replay case."""
    task_id, principal = await _three_step_task()
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)
    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
        **_long_intervals(),
    )
    try:
        status = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert "event: task.status" in status

        first_step = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert first_step.startswith("event: step.started\n")

        sink = next(
            c
            for c in es.manager.connections_for_task(task_id)
            if isinstance(c, es.V1EventStreamSink)
        )
        won = sink.enqueue_close(es.error_frame("unauthorized"), abortive=True)
        assert won is True

        remaining_frames = []
        async for chunk in resp.body_iterator:
            remaining_frames.append(chunk)
        remaining_body = "".join(remaining_frames)
        assert "event: step.started" not in remaining_body
        assert remaining_body.count("event: stream.error") == 1
        assert "unauthorized" in remaining_body
    finally:
        await resp.body_iterator.aclose()


async def test_abortive_close_after_a_non_abortive_one_still_stops_the_replay():
    """The revocation an abortive close carries is not first-caller-wins,
    even though the close frame is.

    A non-abortive close (here ``resync_required``, the reason a staging
    overflow enqueues) can land during the replay and claim the close
    frame. If a watchdog ``unauthorized`` close then arrives, its early
    return must not swallow the suppression: the client just lost the
    right to see this task's content, and the frames still queued behind
    the replay cursor are exactly the content that revocation covers.

    The client reads the earlier close reason on the wire -- that part
    stays first-caller-wins -- and re-attaches, where authentication
    refuses it. What it must not read is the rest of the history."""
    task_id, principal = await _three_step_task()
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)
    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
        **_long_intervals(),
    )
    try:
        status = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert "event: task.status" in status

        first_step = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert first_step.startswith("event: step.started\n")

        sink = next(
            c
            for c in es.manager.connections_for_task(task_id)
            if isinstance(c, es.V1EventStreamSink)
        )
        first_won = sink.enqueue_close(es.error_frame("resync_required"))
        assert first_won is True
        assert sink.abortive_close is False

        second_won = sink.enqueue_close(es.error_frame("unauthorized"), abortive=True)
        assert second_won is False
        assert sink.abortive_close is True

        remaining_frames = []
        async for chunk in resp.body_iterator:
            remaining_frames.append(chunk)
        remaining_body = "".join(remaining_frames)
        assert "event: step.started" not in remaining_body
        assert "event: step.completed" not in remaining_body
        assert remaining_body.count("event: stream.error") == 1
        assert "resync_required" in remaining_body
    finally:
        await resp.body_iterator.aclose()


@pytest.mark.parametrize(
    ("close_claimed_by_watchdog", "expected_close_event"),
    [
        pytest.param(False, "event: stream.error", id="close-claimed-elsewhere"),
        pytest.param(True, "event: task.completed", id="close-claimed-by-the-watchdog"),
    ],
)
async def test_watchdog_keeps_checking_authorization_across_an_ordinary_close(
    close_claimed_by_watchdog, expected_close_event
):
    """An ordinary close must not end authorization coverage.

    The warm-up replay hands its frames to the client directly, bypassing
    the outbound queue's closing guard, so a close that is not abortive
    stops nothing on that path -- only ``abortive_close`` does, set here
    by ``watchdog_check_once`` (and, on a different path, by
    ``_generate``'s own two pre-loop read failures). A watchdog that
    stops at a close therefore never observes a key revoked afterwards,
    and the replay hands that key the task's persisted history.

    The two legs differ in one variable, who claims the close, because
    the loop had two separate exits that both ended coverage: one on
    seeing ``closing`` before running a check, and one on a check of its
    own having closed the stream. A terminal task found while the
    warm-up read is still in flight reaches the second one without any
    staging overflow at all.
    """
    task_id, principal = await _three_step_task()
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)
    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
        watchdog_interval_seconds=0.05,
        stream_max_duration_seconds=600.0,
        heartbeat_interval_seconds=600.0,
    )
    try:
        status = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert "event: task.status" in status
        # Suspended on the initial status yield, which runs before the
        # warm-up read: the sink is registered and still cold, the
        # watchdog is running, and no history has been read or handed
        # out yet.
        sink = next(
            c
            for c in es.manager.connections_for_task(task_id)
            if isinstance(c, es.V1EventStreamSink)
        )
        assert sink.warm is False

        if close_claimed_by_watchdog:
            _set_task_status(task_id, TaskStatus.COMPLETED, output="done")
            await _poll_until(lambda: sink.closing)
        else:
            assert sink.enqueue_close(es.error_frame("resync_required")) is True
        assert sink.abortive_close is False

        _revoke_key_for(principal)
        await _poll_until(lambda: sink.abortive_close)

        frames = []
        async for chunk in resp.body_iterator:
            frames.append(chunk)
        body = "".join(frames)
        assert "event: step.started" not in body
        assert frames[-1].startswith(expected_close_event)
    finally:
        await resp.body_iterator.aclose()


async def test_watchdog_stops_once_the_warm_up_handoff_is_over():
    """The warm-side mirror of the test above: once the warm-up handoff
    is over, an ordinary close really does end watchdog coverage,
    because every emission from that point on goes through
    ``_put_or_overflow``'s closing guard -- there is no longer a direct
    replay path left for the watchdog to keep guarding against.

    Pulls frames until ``sink.warm`` is True. ``sink.warm`` flips inside
    the generator, between the last replayed frame and whatever it
    yields next, so the pull that observes it can block briefly on the
    generator's own wait loop -- a short ``heartbeat_interval_seconds``
    bounds that to one ``: ping`` cycle instead of the fixture's usual
    minutes-long interval.

    Then closes non-abortively and asserts the watchdog loop's actual
    exit, not only the predicate flipping: ``_watchdog_coverage_done``
    reading True proves nothing about whether the loop's own ``while``
    re-checked it, so this awaits the watchdog task itself with a
    timeout too -- a predicate that goes true without the task
    finishing would still fail this test."""
    task_id, principal = await _three_step_task()
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)
    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
        watchdog_interval_seconds=0.05,
        stream_max_duration_seconds=600.0,
        heartbeat_interval_seconds=0.05,
    )
    try:
        status = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert "event: task.status" in status
        sink = next(
            c
            for c in es.manager.connections_for_task(task_id)
            if isinstance(c, es.V1EventStreamSink)
        )
        assert sink.warm is False

        watchdog_task = next(
            t
            for t in asyncio.all_tasks()
            if t.get_coro().cr_code.co_name == "_watchdog_loop"
        )

        for _ in range(10):
            if sink.warm:
                break
            await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert sink.warm is True

        assert sink.enqueue_close(es.error_frame("resync_required")) is True
        await _poll_until(lambda: es._watchdog_coverage_done(sink))
        await asyncio.wait_for(watchdog_task, timeout=2)
        assert watchdog_task.done()
    finally:
        await resp.body_iterator.aclose()


async def test_normal_terminal_close_during_replay_lets_replay_finish_first():
    """A non-abortive close landing between two replayed history frames
    must NOT cut the replay short -- that history describes the task's
    own past and stays valid even though the stream is about to end;
    only the close frame itself is deferred to arrive after it, via the
    normal queue path. Drives the close directly via
    ``sink.enqueue_close`` with the same frame shape
    (``task.completed``) the watchdog would enqueue for a terminal task
    -- not through ``watchdog_check_once`` itself; that production call
    site is covered separately by
    ``test_watchdog_terminal_task_row_closes_with_completed``'s own
    ``sink.abortive_close is False`` assertion. Same injection point as
    the abortive test above, opposite close reason."""
    task_id, principal = await _three_step_task()
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)
    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
        **_long_intervals(),
    )
    try:
        status = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert "event: task.status" in status

        first_step = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert first_step.startswith("event: step.started\n")

        sink = next(
            c
            for c in es.manager.connections_for_task(task_id)
            if isinstance(c, es.V1EventStreamSink)
        )
        won = sink.enqueue_close(
            es.completed_frame(status="completed", output="done", error=None)
        )
        assert won is True

        remaining_frames = []
        async for chunk in resp.body_iterator:
            remaining_frames.append(chunk)
        remaining_body = "".join(remaining_frames)
        assert remaining_body.count("event: step.started") == 2  # the other two
        assert remaining_body.count("event: task.completed") == 1
        # The close frame is the very last thing delivered.
        assert remaining_frames[-1].startswith("event: task.completed\n")
    finally:
        await resp.body_iterator.aclose()


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
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
            read_task_steps=v1_tasks._load_task_steps_snapshot,
            read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
            read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
            read_task_steps=v1_tasks._load_task_steps_snapshot,
            read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
            read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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

    Also covers two properties of the fast path's step snapshot that a
    frame-count-only assertion would miss entirely: it actually emits a
    ``step.*`` frame for pre-existing history (not just the two
    lifecycle frames alone -- a regression here would have
    left ``PER_TASK_STREAM_CAP``-saturating cap coverage green while
    silently dropping the snapshot), and a second fast-path attach on
    the same task reuses the ``steps()`` cache instead of repeating the
    full trace read.
    """
    set_cache_backend_for_testing(InMemoryTTLCache())
    try:
        agent_id, full_key = _create_agent_with_key()
        task_id = _create_task(full_key, agent_id)
        principal = _principal_for(full_key)
        key_prefix = _key_prefix_for_agent(agent_id)
        running_snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)
        base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        _insert_trace_event(
            task_id=task_id,
            event_type="tool_execution_start",
            event_id="hist-1",
            timestamp=base,
            step_id="step-1",
            data={"tool_call_id": "call-1", "tool_name": "search", "tool_args": {}},
        )
        _insert_trace_event(
            task_id=task_id,
            event_type="tool_execution_end",
            event_id="hist-2",
            timestamp=base,
            step_id="step-1",
            data={"tool_call_id": "call-1", "success": True, "result": "sunny"},
        )

        # Fill the per-task cap with real streams while the task is running.
        responses = []
        for _ in range(es.PER_TASK_STREAM_CAP):
            resp = await es.build_event_stream_response(
                task_id=task_id,
                principal=principal,
                initial_snapshot=running_snapshot,
                read_task_snapshot=v1_tasks._load_task_info_snapshot,
                read_task_steps=v1_tasks._load_task_steps_snapshot,
                read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
                read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
        assert body.count("event: step.completed") == 1

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
    finally:
        set_cache_backend_for_testing(None)


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
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
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
    behavior -- had no persisted counterpart: ``steps()`` and a fresh
    attach's warm-up replay both show this exact row as a ``message``
    step regardless, so filtering only the live path made an
    already-attached stream disagree with those two about whether the
    step exists. See
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


# ===== warm-up staging: a step whose start is only in history and whose
# end arrives during the warm-up window still resolves on this stream =====


async def test_orphan_end_during_warm_up_is_rescued_by_staging():
    """The scenario ``finish_warm_up`` exists for: a step's *start* event
    is only in trace history (it happened before this connection
    attached), and its *end* event happens to broadcast live while the
    sink is still cold (registered, but the history read/replay hasn't
    finished). Fed straight to an empty projector, that end event would
    be a plain orphan end and get silently dropped -- there'd never be a
    ``step.completed`` for it on this stream. Staging it and replaying
    it through the *warmed* projector instead (which has the start
    pending, from history) rescues it."""
    sink = _make_sink(task_id=41, warm=False)

    # Arrives while the sink is still cold -- gets staged, not projected.
    await sink.send_text(
        _trace_event_frame(
            "tool_execution_end",
            task_id=41,
            data={"tool_call_id": "call-1", "success": True, "result": "sunny"},
        )
    )
    assert sink.staged_message_count == 1
    assert sink.queue.empty()

    # What a real warm-up read would have produced: a projector that has
    # already replayed the matching start event from history.
    history_start_event = {
        "id": 1,
        "task_id": 41,
        "event_id": "hist-1",
        "event_type": "tool_execution_start",
        "timestamp": 0,
        "step_id": "step-1",
        "data": {"tool_call_id": "call-1", "tool_name": "search", "tool_args": {}},
    }
    warmed = es.PublicStepProjector.from_history([history_start_event])
    # The same two calls, in the same order, that ``_build_warm_up_frames``
    # makes: read the replayed timeline out, which is also what releases
    # retention, then hand the projector over.
    warmed.take_materialized_steps()

    sink.finish_warm_up(warmed)

    assert sink.warm is True
    assert sink.staged_message_count == 0
    frame_text, is_close = sink.queue.get_nowait()
    assert is_close is False
    assert frame_text.startswith("event: step.completed\n")
    step = json.loads(frame_text.split("data: ", 1)[1])["step"]
    assert step["id"] == "tool_call:call-1"
    assert step["status"] == "completed"
    assert step["data"]["result"] == "sunny"


# ===== raw-frame size pre-check: an oversized content frame is dropped
# before it's projected, on both the warm and the staging path =====


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


# ===== the warm-up staging buffer's two bounds, and the one quantity
# that must never enter either: the task's own description stamp =====


async def test_staged_messages_overflow_closes_with_resync_required():
    """The staging list reuses the outbound queue's own count cap and
    overflow policy (``resync_required`` then close) instead of a second
    one -- this pins that reuse rather than assuming it."""
    sink = _make_sink(task_id=43, warm=False)
    for i in range(es.OUTBOUND_QUEUE_MAX_SIZE):
        await sink.send_text(
            _trace_event_frame(
                "tool_execution_start",
                task_id=43,
                data={"tool_call_id": f"call-{i}", "tool_name": "x", "tool_args": {}},
            )
        )
    assert sink.staged_message_count == es.OUTBOUND_QUEUE_MAX_SIZE
    assert sink.closing is False

    await sink.send_text(
        _trace_event_frame(
            "tool_execution_start",
            task_id=43,
            data={"tool_call_id": "overflow", "tool_name": "x", "tool_args": {}},
        )
    )

    assert sink.closing is True
    frame_text, is_close = sink.queue.get_nowait()
    assert is_close is True
    assert "resync_required" in frame_text


async def test_staged_messages_byte_budget_overflow_closes_with_resync_required():
    """``_staged_messages`` is bounded by cumulative character count on
    top of its count cap -- a run of large-but-individually-legal
    messages (each under ``MAX_RAW_FRAME_TEXT_CHARS``, so none is dropped
    by the pre-check) can still overflow the cumulative budget well
    before the count cap binds. Uses a payload just under the per-frame
    raw-size limit so few enough messages are needed to keep the test
    fast, and asserts the count cap did not bind instead."""
    sink = _make_sink(task_id=95, warm=False)
    near_cap_delta = "x" * (es.MAX_RAW_FRAME_TEXT_CHARS - 200)
    messages_needed = (es.STAGED_MESSAGES_MAX_TEXT_CHARS // len(near_cap_delta)) + 2
    assert messages_needed < es.OUTBOUND_QUEUE_MAX_SIZE, (
        "fixture assumption: the byte budget must overflow before the "
        "count cap does, or this test would re-pin the count cap instead"
    )

    for i in range(messages_needed):
        await sink.send_text(
            json.dumps(
                {
                    "type": "final_answer_delta",
                    "message_id": f"final_answer_{i}",
                    "task_id": 95,
                    "delta": near_cap_delta,
                }
            )
        )
        if sink.closing:
            break

    assert sink.closing is True
    assert sink.staged_message_count < es.OUTBOUND_QUEUE_MAX_SIZE
    frame_text, is_close = sink.queue.get_nowait()
    assert is_close is True
    assert "resync_required" in frame_text


async def test_a_cold_status_frame_leaves_the_staging_drain_no_margin():
    """The staging list and the outbound queue are capped by the *same*
    constant, so a lifecycle frame that bypassed staging while the sink
    was cold makes the warm-up drain one frame too big for the queue.

    Pins that zero margin so the two caps cannot drift apart silently:
    with the staging cap one lower, the drain would fit and this test's
    overflow would not happen. What the overflow must not become is a
    silent truncation -- the client is told to resync, which is the same
    answer every other over-full queue gets.
    """
    sink = _make_sink(task_id=97, warm=False)
    # A lifecycle frame goes straight into the outbound queue even
    # though the sink is cold -- it never touches staging.
    sink.enqueue_status("paused")
    assert sink.queue.qsize() == 1

    for i in range(es.OUTBOUND_QUEUE_MAX_SIZE):
        await sink.send_text(
            _trace_event_frame(
                "tool_execution_start",
                task_id=97,
                data={"tool_call_id": f"call-{i}", "tool_name": "x", "tool_args": {}},
            )
        )
    assert sink.staged_message_count == es.OUTBOUND_QUEUE_MAX_SIZE
    assert sink.closing is False

    sink.finish_warm_up(es.PublicStepProjector(retain_finished=False))

    # 1 lifecycle frame + 256 drained frames > 256 slots.
    assert sink.closing is True
    frame_text, is_close = sink.queue.get_nowait()
    assert is_close is True
    assert "resync_required" in frame_text


async def test_one_task_info_frame_cannot_overflow_the_staging_budget():
    """A ``task_info`` frame carries the task's ``description`` column
    under ``data["description"]``, and that column is unbounded
    (``Column(Text)``). Charged raw, one such frame crosses
    ``STAGED_MESSAGES_MAX_TEXT_CHARS`` by itself and closes the attach
    with ``resync_required`` -- and because the trigger is a property of
    the task rather than the load on it, the re-attach the close asks for
    fails at exactly the same frame, forever. The description is
    therefore excluded from the charge, which leaves this frame costing
    the few hundred characters its actual lifecycle fields occupy.

    ``task_info`` is also the family with nothing to lose by it:
    ``_feed_trace_event`` returns ``[]`` for it, so the description is
    charged against a budget it could never spend.

    The sink starts at ``status="pending"`` rather than the frame's own
    ``"running"``: ``enqueue_status`` no-ops when a status repeats the
    sink's last one, and a ``task_info`` frame's ``status`` would
    otherwise be deduped away, leaving nothing to assert on the
    lifecycle half below. ``"running"`` is not in
    ``_EARLY_WAKE_STATUS_VALUES`` either way, so nothing about the
    content half's handling changes."""
    sink = _make_sink(task_id=105, status="pending", warm=False)
    huge_description = "d" * (es.STAGED_MESSAGES_MAX_TEXT_CHARS + 1000)
    raw = json.dumps(
        {
            "type": "trace_event",
            "event_id": "ev-105",
            "event_type": "task_info",
            "task_id": 105,
            "timestamp": 0,
            "status": "running",
            "data": {
                "id": 105,
                "title": "t",
                "description": huge_description,
                "status": "running",
            },
        }
    )
    assert len(raw) > es.STAGED_MESSAGES_MAX_TEXT_CHARS

    await sink.send_text(raw)

    assert sink.closing is False
    assert sink.dropped_frame_count == 0
    assert sink.staged_message_count == 1
    assert sink._staged_text_chars < 1000
    assert "description" not in sink._staged_messages[0]["data"]
    # The lifecycle half still went out: task_info is a versioned task
    # event, and its status is enqueued before the content branch runs.
    frame_text, _ = sink.queue.get_nowait()
    assert frame_text.startswith("event: task.status\n")


async def test_a_long_task_description_cannot_overflow_the_staging_budget():
    """The same defect on the ordinary path, which is the half a
    ``task_info``-only fix does not reach.
    ``ws_trace_handlers.py``'s converter stamps the same unbounded column
    onto *every* trace event as ``data["task_description"]``, and a frame
    at or under ``MAX_RAW_FRAME_TEXT_CHARS`` used to be charged its full
    ``len(text)``. A description a little under that per-frame cap
    therefore crossed the 4 MiB staging budget after ~17 frames of a
    warm-up window, again identically on every re-attach.

    Sends more frames than the old code survived and asserts the stream
    is still open, then asserts the accumulated charge is the frames'
    own content rather than the description repeated per frame."""
    sink = _make_sink(task_id=106, status="running", warm=False)
    description = "d" * (200 * 1024)
    frames = 40

    for i in range(frames):
        await sink.send_text(
            _trace_event_frame(
                "tool_execution_start",
                task_id=106,
                step_id=f"step-{i}",
                data={
                    "tool_call_id": f"call-{i}",
                    "tool_name": "search",
                    "tool_args": {},
                    "task_description": description,
                },
            )
        )

    assert sink.closing is False
    assert sink.dropped_frame_count == 0
    assert sink.staged_message_count == frames
    assert sink._staged_text_chars < frames * 1000
    assert all(
        "task_description" not in staged["data"] for staged in sink._staged_messages
    )


async def test_the_raw_frame_drop_boundary_is_unchanged_by_the_description_exclusion():
    """Excluding the description from the *measure* must not move the
    per-frame drop boundary, which is a settled contract with tests on
    both sides of it: pruning can only lower a frame's measured length,
    and a frame already at or under the cap stays under it.

    Near-duplicate of ``test_a_frame_still_over_the_cap_without_its_description_is_dropped``
    (above): that test already pins the end-to-end drop for a frame
    whose own content -- not its description -- is what pushes it over
    the cap. What this test adds is a direct call into
    ``_measured_content_frame`` that pins the *ordering* inside it --
    prune, then compare -- rather than only the ``send_text``-level
    outcome. The complementary mutation (the drop check reading the
    *unpruned* length instead of the pruned one) is not caught here:
    this frame is over the cap on its own tool arguments regardless of
    pruning, so that mutation leaves this test green. It is caught by
    the existing
    ``test_a_huge_task_description_does_not_drop_a_small_content_frame``,
    whose frame is under the cap without its description and over it
    with -- the only shape that can tell the two checks apart."""
    sink = _make_sink(task_id=107)
    filler = "y" * (es.MAX_RAW_FRAME_TEXT_CHARS + 1000)
    raw = _trace_event_frame(
        "tool_execution_start",
        task_id=107,
        step_id="step-1",
        data={
            "tool_call_id": "call-1",
            "tool_name": "search",
            "tool_args": {"query": filler},
            "task_description": "d" * (200 * 1024),
        },
    )
    measured, pruned = es._measured_content_frame(raw, json.loads(raw))
    assert "task_description" not in pruned["data"]
    assert measured > es.MAX_RAW_FRAME_TEXT_CHARS

    await sink.send_text(raw)

    assert sink.dropped_frame_count == 1
    assert sink.queue.empty()


# ===== a frame that fails to project is dropped, not fatal, and never
# double-counted between the sink's two except layers
# (see _project_and_queue) =====


async def test_poisoned_live_frame_increments_dropped_frame_count_only_once(
    monkeypatch,
):
    """A projection failure on the live path (sink already warm) is
    caught inside ``_project_and_queue`` and never re-raises -- if it
    did, ``send_text``'s own outer ``except`` would catch it too and
    count the same dropped frame twice."""
    sink = _make_sink(task_id=73)  # warm=True default -- the live path

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


async def test_warm_up_hands_the_sink_a_projector_that_stops_retaining():
    """The live sink must never be handed a retaining projector.

    ``_build_warm_up_frames`` reads the replayed timeline out with
    ``take_materialized_steps``, which is also what releases retention,
    so the projector it returns keeps nothing from that point on. This
    pins the property on the real builder's real output rather than on a
    hand-built projector, because the release lives in the builder, not
    in the projector's constructor.

    ``finish_warm_up`` then rejects a retaining projector outright. Both
    are asserted here: what the builder produces, and what the sink
    refuses. Without the release the stream still behaves correctly --
    it just accumulates every finished step's untruncated ``data`` for
    up to an hour in a list nothing reads, which is invisible to every
    other assertion in this file."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    _insert_trace_event(
        task_id=task_id,
        event_type="ai_message",
        event_id="message-0",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        data={"content": "already finished"},
    )

    warmed, warm_up_frames = es._build_warm_up_frames(
        task_id,
        principal,
        v1_tasks._load_task_steps_snapshot,
        _current_watermark(task_id, principal),
    )

    # The history really was read out -- otherwise "retains nothing"
    # would be trivially true of a projector that folded nothing.
    assert len(warm_up_frames) == 1
    assert warmed.retains_finished is False
    with pytest.raises(RuntimeError):
        warmed.materialized_steps()

    sink = _make_sink(task_id=task_id, warm=False)
    sink.finish_warm_up(warmed)
    assert sink.warm is True

    with pytest.raises(RuntimeError):
        _make_sink(task_id=task_id, warm=False).finish_warm_up(
            es.PublicStepProjector(retain_finished=True)
        )


async def test_attach_replays_history_before_live_frames():
    """End-to-end warm-up through the real generator: a step that
    started before attach is replayed as a ``step.started`` frame,
    directly after the initial ``task.status`` and before anything else
    -- the two frames a real client sees first, in order."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_start",
        event_id="hist-1",
        timestamp=base,
        step_id="step-1",
        data={"tool_call_id": "call-1", "tool_name": "search", "tool_args": {}},
    )

    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)
    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
        **_long_intervals(),
    )
    try:
        first = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert "event: task.status" in first
        second = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert second.startswith("event: step.started\n")
        step = json.loads(second.split("data: ", 1)[1])["step"]
        assert step["id"] == "tool_call:call-1"
        assert step["status"] == "running"
    finally:
        await resp.body_iterator.aclose()


async def test_warm_up_replay_is_sorted_by_started_at_not_resolve_order():
    """``_build_warm_up_frames`` sorts its replay by ``started_at`` to
    match ``steps()``'s own ordering -- ``PublicStepProjector.
    materialized_steps()`` on its own returns *resolve* order, which
    diverges when a later-started step finishes before an earlier one.
    Step A starts first (t0) but finishes last (t3); step B starts
    second (t1) but finishes first (t2) -- resolve order is [B, A],
    started_at order is [A, B]. This pins the replay uses the latter."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC)
    t2 = datetime(2026, 1, 1, 12, 0, 2, tzinfo=UTC)
    t3 = datetime(2026, 1, 1, 12, 0, 3, tzinfo=UTC)
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_start",
        event_id="a-start",
        timestamp=t0,
        step_id="step-a",
        data={"tool_call_id": "call-a", "tool_name": "x", "tool_args": {}},
    )
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_start",
        event_id="b-start",
        timestamp=t1,
        step_id="step-b",
        data={"tool_call_id": "call-b", "tool_name": "x", "tool_args": {}},
    )
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_end",
        event_id="b-end",
        timestamp=t2,
        step_id="step-b",
        data={"tool_call_id": "call-b", "success": True, "result": "b done"},
    )
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_end",
        event_id="a-end",
        timestamp=t3,
        step_id="step-a",
        data={"tool_call_id": "call-a", "success": True, "result": "a done"},
    )

    _, warm_up_frames = es._build_warm_up_frames(
        task_id,
        principal,
        v1_tasks._load_task_steps_snapshot,
        _current_watermark(task_id, principal),
    )
    ids_in_order = [
        json.loads(frame.split("data: ", 1)[1])["step"]["id"]
        for frame in warm_up_frames
    ]
    assert ids_in_order == ["tool_call:call-a", "tool_call:call-b"]

    # And this is what steps() itself returns for the same history --
    # the replay now agrees with it instead of the resolve-order [B, A].
    steps_resp = client.get(
        f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
    )
    polled_ids = [step["id"] for step in steps_resp.json()["steps"]]
    assert polled_ids == ids_in_order


# ===== a step whose serialization raises ends the replay batch there,
# instead of failing the whole attach =====


async def test_warm_up_replay_keeps_the_prefix_before_an_unserializable_step():
    """``_build_warm_up_frames`` serializes its admitted steps one at a
    time and stops at the first one that raises, so a persisted step
    whose ``data`` cannot be turned into JSON costs the client that step
    and whatever came after it -- not the whole attach.

    The shared budget (``_snapshot_steps_within_wire_budget``) admits
    such a step on purpose: it cannot be measured, so it cannot be
    budgeted, and each caller decides what to do when the emit reaches
    it. The two attach-time fast paths report it and close; this path
    has no per-step emit loop and no conclusion frame to carry a marker,
    so it ends the batch quietly and goes live -- reason (6) in the
    endpoint's best-effort delivery list.

    Three things are asserted, because a fix that only delivers the
    prefix and then closes anyway would satisfy the first alone: the
    good step reaches the client, no ``stream.error`` follows it, and a
    frame broadcast afterwards is delivered -- which is only true if the
    sink actually finished warm-up and went live.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    class _Unserializable:
        pass

    good_row = v1_tasks._TraceEventSnapshot(
        id=1,
        task_id=task_id,
        event_id="good",
        event_type="ai_message",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        step_id=None,
        data={"content": "this step serializes fine"},
    )
    bad_row = v1_tasks._TraceEventSnapshot(
        id=2,
        task_id=task_id,
        event_id="bad",
        event_type="ai_message",
        timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC),
        step_id=None,
        data={"content": _Unserializable()},
    )

    def _poisoned_read_task_steps(task_id_, principal_):
        return SimpleNamespace(events=[good_row, bad_row])

    def _version_reader(task_id_, principal_):
        return v1_tasks._TaskStepsVersionSnapshot(
            task_id=task_id, agent_id=agent_id, max_event_id=2
        )

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        read_task_steps=_poisoned_read_task_steps,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=_version_reader,
        **_long_intervals(),
    )
    try:
        first = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert "event: task.status" in first

        # The prefix really is delivered -- the step ordered before the
        # unserializable one is not collateral damage.
        second = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert second.startswith("event: step.completed\n")
        assert "this step serializes fine" in second

        # The stream is live, not closing: a frame broadcast now is
        # projected and delivered, which only happens after
        # ``finish_warm_up`` has run.
        await es.manager.broadcast_to_task(
            {
                "type": "final_answer_delta",
                "message_id": "after_warm_up",
                "task_id": task_id,
                "delta": "live",
            },
            task_id,
        )
        third = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert third.startswith("event: message.delta\n")
        assert json.loads(third.split("data: ", 1)[1]) == {
            "message_id": "after_warm_up",
            "text": "live",
        }
    finally:
        await resp.body_iterator.aclose()


# ===== the replay's second bound: the same wire-byte budget the
# attach-time fast paths' snapshot uses =====


# ===== registration watermark: an event that lands in the gap between
# `register_connection` and the warm-up read must not be folded into
# the projector twice (once via replay, once via staged-drain) =====


async def test_replay_watermark_excludes_a_row_committed_after_registration():
    """Direct pin on ``_build_warm_up_frames``'s ``replay_watermark``
    parameter, at the same sink level ``test_orphan_end_during_warm_up_is_rescued_by_staging``
    uses for the cold-sink staging window: a trace row committed (and
    broadcast) *after* the watermark was captured -- exactly the
    registration-to-warm-up-read gap ``_generate`` now bounds replay to
    -- must be excluded from this replay entirely, since it's already
    covered by staging + ``finish_warm_up``'s drain. Without the bound
    (``replay_watermark=None``, pinned by the mutation note below), the
    same row would be replayed here AND drained from staging, producing
    two ``step.started`` frames for the same step id instead of one."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)

    # What ``_generate`` would have captured immediately after
    # ``register_connection``, before any of this task's history exists.
    watermark_at_registration = v1_tasks._load_task_steps_version_snapshot(
        task_id, principal
    ).max_event_id

    # Lands *after* that watermark -- the gap-window event -- and its
    # broadcast reaches the still-cold sink in the same window, so it's
    # staged rather than fed to a live projector.
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_start",
        event_id="gap-event",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        step_id="step-1",
        data={"tool_call_id": "call-1", "tool_name": "search", "tool_args": {}},
    )
    sink = _make_sink(task_id=task_id, warm=False)
    await sink.send_text(
        _trace_event_frame(
            "tool_execution_start",
            task_id=task_id,
            step_id="step-1",
            data={"tool_call_id": "call-1", "tool_name": "search", "tool_args": {}},
        )
    )
    assert sink.staged_message_count == 1

    warmed_projector, warm_up_frames = es._build_warm_up_frames(
        task_id,
        principal,
        v1_tasks._load_task_steps_snapshot,
        watermark_at_registration,
    )
    # The gap event's row id is > the watermark -- excluded from this
    # replay entirely, nothing to replay for a task with no history
    # before the watermark.
    assert warm_up_frames == []

    sink.finish_warm_up(warmed_projector)
    # The staged broadcast is the only source of this step's frame.
    frame_text, is_close = sink.queue.get_nowait()
    assert is_close is False
    assert frame_text.startswith("event: step.started\n")
    step = json.loads(frame_text.split("data: ", 1)[1])["step"]
    assert step["id"] == "tool_call:call-1"
    assert sink.queue.empty()  # exactly once -- no duplicate from replay


async def test_a_planning_row_folded_twice_in_the_residual_window_leaves_one_step_running():
    """Pins the accepted residual of registering before reading the
    watermark, so it cannot drift silently. Tracked as #1776.

    A row that commits inside the registration-to-watermark gap is both
    at or below the watermark (so the replay folds it) and already
    staged (so ``finish_warm_up``'s drain folds it again). This test
    builds that same end state directly: insert the row, capture a
    watermark above it, then hand the sink the row's broadcast while it
    is still cold.

    The planning family is the one that cannot absorb it. Its public
    step id comes from an in-memory counter rather than from a key the
    event carries, so two folds mint two different ids, and the row's
    single end event -- arriving live, after the handoff -- closes only
    the later one. The earlier step stays ``running`` for the rest of
    the connection.

    Asserting the duplicate is deliberate. Exactly one start with its
    matching completion is the acceptance test for #1776, and it
    belongs with the fix that carries a persisted identity onto
    broadcast frames.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)

    _insert_trace_event(
        task_id=task_id,
        event_type="dag_plan_start",
        event_id="gap-plan",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        data={},
    )
    watermark = v1_tasks._load_task_steps_version_snapshot(
        task_id, principal
    ).max_event_id

    sink = _make_sink(task_id=task_id, warm=False)
    await sink.send_text(_trace_event_frame("dag_plan_start", task_id=task_id, data={}))
    assert sink.staged_message_count == 1

    warmed_projector, warm_up_frames = es._build_warm_up_frames(
        task_id, principal, v1_tasks._load_task_steps_snapshot, watermark
    )
    replayed = json.loads(warm_up_frames[0].split("data: ", 1)[1])["step"]

    sink.finish_warm_up(warmed_projector)
    drained_text, _ = sink.queue.get_nowait()
    drained = json.loads(drained_text.split("data: ", 1)[1])["step"]

    # One row, two public ids: the counter advanced on each fold.
    assert drained["id"] != replayed["id"]
    assert drained["status"] == "running"

    await sink.send_text(_trace_event_frame("dag_plan_end", task_id=task_id, data={}))
    completed_text, _ = sink.queue.get_nowait()
    completed = json.loads(completed_text.split("data: ", 1)[1])["step"]
    # The single end event closes only the later id; the replayed one is
    # never finalized.
    assert completed["id"] == drained["id"]
    assert completed["status"] == "completed"
    assert sink.queue.empty()


async def test_generate_wires_read_task_steps_version_into_the_watermark(monkeypatch):
    """Wiring smoke test through the real generator (not just the
    ``_build_warm_up_frames`` unit above): ``read_task_steps_version``
    passed into ``_generate`` is actually called exactly once, and by
    the time it runs the sink is already visible in ``manager`` --
    proof that registration happened first, not just that the reader
    ran once -- and its result reaches ``_build_warm_up_frames`` as
    ``replay_watermark`` -- catches a signature/plumbing regression the
    sink-level test above wouldn't (it calls ``_build_warm_up_frames``
    directly, bypassing ``_generate`` entirely)."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_start",
        event_id="hist-1",
        timestamp=base,
        step_id="step-1",
        data={"tool_call_id": "call-1", "tool_name": "search", "tool_args": {}},
    )

    calls: list[str] = []
    original = v1_tasks._load_task_steps_version_snapshot

    def _tracking_version_reader(task_id_, principal_):
        # Ordering-sensitive on purpose: the sink only becomes visible
        # in the manager once ``register_connection`` has run, so a
        # reader moved ahead of registration records "unregistered"
        # here and fails the assertion below. A count alone cannot see
        # that, and the order is load-bearing -- registering second
        # turns the gap this reader bounds from a duplicated row into a
        # lost one.
        calls.append(
            "registered"
            if any(
                isinstance(c, es.V1EventStreamSink)
                for c in es.manager.connections_for_task(task_id_)
            )
            else "unregistered"
        )
        return original(task_id_, principal_)

    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)
    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=_tracking_version_reader,
        **_long_intervals(),
    )
    try:
        first = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert "event: task.status" in first
        second = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert second.startswith("event: step.started\n")
        assert calls == ["registered"]  # ran exactly once, and after registration
    finally:
        await resp.body_iterator.aclose()


def _delete_the_task_then_read_the_watermark(task_id_, principal_):
    """The race this split exists for: a supported DELETE removes the
    row while this read is in flight, so the production reader's own
    ``_resolve_task_or_404`` raises ``V1ApiError(TASK_NOT_FOUND)``. The
    delete is real and the reader is the production one -- nothing
    about the ``TASK_NOT_FOUND`` reaching the handler is synthesized."""
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id_).delete()
        db.commit()
    finally:
        db.close()
    return v1_tasks._load_task_steps_version_snapshot(task_id_, principal_)


def _fail_the_watermark_read_transiently(task_id_, principal_):
    """A failure that is not the task being gone.

    A bare ``RuntimeError`` rather than a ``V1ApiError`` carrying some
    other code, deliberately. The guard being pinned is
    ``isinstance(exc, V1ApiError) and exc.code is TASK_NOT_FOUND``, and
    this exercises the ``isinstance`` half: it is the shape a
    driver-level or connection-level database failure actually reaches
    this handler as, and the shape the comment above the handler names
    when it says "a transient DB error". The task row is untouched and
    still readable, which is exactly why ``resync_required`` -- call
    ``steps()``, re-attach -- is the recovery that works here and
    ``task_deleted`` is not."""
    raise RuntimeError("transient database failure")


def _fail_the_watermark_read_with_a_different_v1_api_error(task_id_, principal_):
    """A ``V1ApiError`` that is not ``TASK_NOT_FOUND``.

    Pins the other half of the guard: ``isinstance(exc, V1ApiError) and
    exc.code is TASK_NOT_FOUND``. The two legs above only exercise the
    ``isinstance`` half -- a bare ``RuntimeError`` is never a
    ``V1ApiError`` in the first place, so a guard loosened to just
    ``isinstance(exc, V1ApiError)`` would still pass both of them. This
    leg is the one that would catch that: a rate-limited read is a real
    ``V1ApiError``, but its code is not ``TASK_NOT_FOUND``, so this
    close must still be ``resync_required`` -- the task itself is fine,
    only this one read was throttled."""
    raise V1ApiError(V1ErrorCode.RATE_LIMITED, 429)


@pytest.mark.parametrize(
    ("read_task_steps_version", "expected_code"),
    [
        pytest.param(
            _delete_the_task_then_read_the_watermark,
            "task_deleted",
            id="task-deleted",
        ),
        pytest.param(
            _fail_the_watermark_read_transiently,
            "resync_required",
            id="transient-read-failure",
        ),
        pytest.param(
            _fail_the_watermark_read_with_a_different_v1_api_error,
            "resync_required",
            id="other-v1-api-error",
        ),
    ],
)
async def test_the_watermark_read_splits_task_deleted_from_every_other_failure(
    read_task_steps_version, expected_code
):
    """The watermark read's handler classifies by cause, and every half
    of that split is load-bearing.

    This read re-resolves the task in its own session, so a delete
    landing while it is in flight raises ``TASK_NOT_FOUND`` into the
    handler. That close is claimed before the watchdog exists, so
    nothing later can correct the code, and a client told to call
    ``steps()`` and re-attach for a row that no longer exists only gets
    a 404 back.

    The other two legs matter just as much, and are the ones a
    collapsed or loosened guard breaks silently: a failure that is not
    the task being gone must still close with ``resync_required``,
    because the task is there and resynchronizing does work -- whether
    that failure is not a ``V1ApiError`` at all, or is a ``V1ApiError``
    whose code is not ``TASK_NOT_FOUND``.

    ``_long_intervals()`` keeps the watchdog asleep for the whole test,
    so the close frame any leg observes can only be the one this
    handler queued -- not the watchdog independently discovering the
    same deleted row. Either way the warm-up block is skipped
    (``replay_watermark`` stays ``None``), so no history reaches the
    client from this attach; that is what the ``step.started``
    assertion pins."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=read_task_steps_version,
        **_long_intervals(),
    )
    frames = []
    async for chunk in asyncio_timeout_iter(resp.body_iterator, timeout=2):
        frames.append(chunk)
    body = "".join(frames)
    assert "event: task.status" in body
    assert _parse_error_frame(body)["code"] == expected_code
    assert "event: step.started" not in body


async def test_absolute_deadline_stops_the_warm_up_replay_early():
    """The warm-up replay loop re-checks the connection's own absolute
    deadline before every yield, not only the main loop below it.

    ``stream_max_duration_seconds`` is negative, so the deadline bound
    at generator start (``monotonic() + stream_max_duration_seconds``)
    is already in the past by the time the replay loop reaches its
    first frame, regardless of how long the watermark read and the
    history read themselves took. The task has three known steps
    (``_three_step_task``), so a replay that ran to completion would
    yield three ``step.started`` frames; this pins that none of them
    make it out, and that the stream instead closes with
    ``stream_expired`` once the main loop's own deadline check runs on
    its very next pass -- the deadline break needs no close of its
    own."""
    task_id, principal = await _three_step_task()
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)
    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        read_task_steps=v1_tasks._load_task_steps_snapshot,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=v1_tasks._load_task_steps_version_snapshot,
        **_long_intervals(stream_max_duration_seconds=-1.0),
    )
    try:
        frames = []
        async for chunk in asyncio_timeout_iter(resp.body_iterator, timeout=2):
            frames.append(chunk)
        body = "".join(frames)
        assert "event: task.status" in body
        assert "event: step.started" not in body
        assert _parse_error_frame(body)["code"] == "stream_expired"
        assert frames[-1].startswith("event: stream.error")
    finally:
        await resp.body_iterator.aclose()


def _fail_the_history_read_because_the_task_is_gone(task_id_, principal_):
    """The second, narrower ``TASK_NOT_FOUND`` race window
    ``_build_warm_up_frames``'s own read covers -- the task is deleted
    after the watermark read already succeeded, but before this read
    runs. Distinct from ``_delete_the_task_then_read_the_watermark``
    above, which covers the watermark read's own race instead."""
    raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)


@pytest.mark.parametrize(
    ("read_task_steps_version", "read_task_steps"),
    [
        pytest.param(
            _delete_the_task_then_read_the_watermark,
            v1_tasks._load_task_steps_snapshot,
            id="watermark-read",
        ),
        pytest.param(
            v1_tasks._load_task_steps_version_snapshot,
            _fail_the_history_read_because_the_task_is_gone,
            id="history-read",
        ),
    ],
)
async def test_generates_own_task_not_found_reads_are_abortive_and_end_watchdog_coverage(
    read_task_steps_version, read_task_steps
):
    """Both of ``_generate``'s own pre-loop reads -- the replay
    watermark and the warm-up history -- close ``task_deleted`` the
    same way ``watchdog_check_once`` does when the task row is gone,
    and on neither leg does the warm-up handoff ever run
    (``sink.warm`` stays ``False``), so only ``abortive_close`` can end
    the watchdog's own coverage on this path (see
    ``_watchdog_coverage_done``). Pins the wire-level close code, pins
    ``sink.abortive_close`` directly, and confirms the watchdog loop
    actually exits on its own -- not just that the predicate flips --
    by awaiting the watchdog task itself with a timeout. A short
    ``watchdog_interval_seconds`` is what lets that task notice and
    exit within the timeout, independent of the generator's own
    teardown (which would cancel it regardless, masking a broken
    ``abortive=True``)."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        read_task_steps=read_task_steps,
        read_task_steps_response=v1_tasks._get_chat_task_steps_sync,
        read_task_steps_version=read_task_steps_version,
        **_long_intervals(watchdog_interval_seconds=0.05),
    )
    try:
        first = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert "event: task.status" in first
        sink = next(
            c
            for c in es.manager.connections_for_task(task_id)
            if isinstance(c, es.V1EventStreamSink)
        )
        watchdog_task = next(
            t
            for t in asyncio.all_tasks()
            if t.get_coro().cr_code.co_name == "_watchdog_loop"
        )

        second = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
        assert _parse_error_frame(second)["code"] == "task_deleted"
        assert sink.abortive_close is True
        assert sink.warm is False

        await asyncio.wait_for(watchdog_task, timeout=2)
        assert watchdog_task.done()
    finally:
        await resp.body_iterator.aclose()


def test_fast_paths_without_a_steps_version_reader_is_a_call_error():
    """``read_task_steps_version`` is a required argument on all three
    attach-time fast-path entry points -- ``_terminal_snapshot_stream``,
    ``_input_required_snapshot_stream``, and the
    ``_fast_path_snapshot_stream`` body they share -- not one carrying a
    default.

    ``build_event_stream_response`` always forwards its own reader into
    whichever of the three it picks, so no caller inside this module can
    reach one of them without a reader in hand. Requiredness is what
    keeps that true for a caller outside it: the cursor baseline read
    and its recheck are the only signal that catches a trace row landing
    between the steps read and the fence, and a reader-less call would
    otherwise run the fast path with that signal silently absent. This
    pins the call itself as the failure point -- a ``TypeError`` naming
    the argument, raised before any reader runs.

    ``build_event_stream_response`` is pinned the same way. It is the
    only entry point a caller outside this module reaches, so a default
    restored there would accept reader-less attaches even with all three
    inner entry points still required. Its own missing-argument
    ``TypeError`` is raised at the call, not at the await: binding
    happens before the coroutine object exists, so nothing is left
    un-awaited by the assertion below."""

    def _unused(task_id_, principal_):
        raise AssertionError("must not run before the missing-argument TypeError")

    terminal_snapshot = SimpleNamespace(
        task_id=1, agent_id=1, status=TaskStatus.COMPLETED, output="done", error=None
    )
    with pytest.raises(TypeError, match="read_task_steps_version"):
        es._terminal_snapshot_stream(
            terminal_snapshot,
            principal=None,
            read_task_steps_response=_unused,
            read_task_snapshot=_unused,
        )

    waiting_snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=TaskStatus.WAITING_FOR_USER,
        pending_question="what next?",
    )
    with pytest.raises(TypeError, match="read_task_steps_version"):
        es._input_required_snapshot_stream(
            waiting_snapshot,
            principal=None,
            read_task_steps_response=_unused,
            read_task_snapshot=_unused,
        )

    with pytest.raises(TypeError, match="read_task_steps_version"):
        es._fast_path_snapshot_stream(
            terminal_snapshot,
            principal=None,
            read_task_steps_response=_unused,
            read_task_snapshot=_unused,
            build_conclusion=lambda snapshot_total_steps: "",
            path_name="terminal",
        )

    with pytest.raises(TypeError, match="read_task_steps_version"):
        es.build_event_stream_response(
            task_id=1,
            principal=None,
            initial_snapshot=terminal_snapshot,
            read_task_snapshot=_unused,
            read_task_steps_response=_unused,
        )


async def asyncio_timeout_iter(aiter, *, timeout: float):
    """Drive an async iterator to exhaustion with a per-item timeout, so
    a regression that makes the generator hang instead of closing fails
    this test outright rather than stalling the whole suite."""
    while True:
        try:
            item = await asyncio.wait_for(aiter.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return
        yield item


# ===== a staged frame that fails to project during warm-up doesn't
# turn an otherwise-normal stream close into a resync_required one =====


def test_fast_path_terminal_attach_includes_current_steps():
    """The attach-time fast path for an already-terminal task carries the
    task's steps (from the cached ``steps()`` read), between
    ``task.status`` and ``task.completed`` -- not just the two lifecycle
    frames alone."""
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
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_end",
        event_id="hist-2",
        timestamp=base,
        step_id="step-1",
        data={"tool_call_id": "call-1", "success": True, "result": "sunny"},
    )
    _set_task_status(task_id, TaskStatus.COMPLETED, output="done")

    resp = client.get(f"/v1/chat/tasks/{task_id}/events", headers=_bearer(full_key))
    assert resp.status_code == 200
    body = resp.text
    assert body.count("event: task.status") == 1
    assert body.count("event: step.completed") == 1
    assert body.count("event: task.completed") == 1
    blocks = [b for b in body.split("\n\n") if b.strip()]
    step = json.loads(blocks[1].split("data: ", 1)[1])["step"]
    assert step["id"] == "tool_call:call-1"
    assert step["status"] == "completed"
    assert step["data"]["result"] == "sunny"
    conclusion_data = json.loads(blocks[2].split("data: ", 1)[1])
    assert "snapshot_truncated" not in conclusion_data


def test_fast_path_step_snapshot_hits_the_steps_cache(monkeypatch):
    """The fast paths' step snapshot goes through the same
    ``max_event_id``-keyed cache the polling ``steps()`` endpoint uses --
    once that cache is warm (as it would be for a client that's been
    polling ``steps()`` and then also attaches to the stream), a fast-
    path attach doesn't repeat the full trace read
    (``_load_task_steps_snapshot``) that produced it.

    Needs a real cache backend (the default test backend is a no-op),
    or this test would pass vacuously without ever exercising a cache
    hit -- same rationale as ``test_tasks.py``'s cache tests.
    """
    set_cache_backend_for_testing(InMemoryTTLCache())
    try:
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
        _set_task_status(task_id, TaskStatus.COMPLETED, output="done")

        # Warm the steps() cache the same way a polling client would.
        steps_resp = client.get(
            f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
        )
        assert steps_resp.status_code == 200

        calls: list[int] = []
        original = v1_tasks._load_task_steps_snapshot

        def _tracking(task_id_, principal_):
            calls.append(1)
            return original(task_id_, principal_)

        monkeypatch.setattr(v1_tasks, "_load_task_steps_snapshot", _tracking)

        resp = client.get(f"/v1/chat/tasks/{task_id}/events", headers=_bearer(full_key))
        assert resp.status_code == 200
        assert "event: step.started" in resp.text
        assert calls == []  # cache hit -- the uncached full trace read never ran
    finally:
        set_cache_backend_for_testing(None)


async def test_fast_path_step_read_failure_closes_with_resync_required_not_a_bare_disconnect():
    """Both attach-time fast paths now catch a snapshot-read failure and
    close with ``stream.error {resync_required}``, matching the
    invariant ``_generate`` already enforces around its own watermark
    read (see
    ``test_the_watermark_read_splits_task_deleted_from_every_other_failure``)
    -- not a bare exception out of the generator. That distinction
    matters here specifically because ``StreamingResponse`` sends the
    HTTP response start (200, headers) before ever pulling a chunk from
    the body iterator: an uncaught raise doesn't turn into a different
    HTTP status, it just ends an already-started 200 response with no
    bytes and no close frame, indistinguishable from the client's side
    from the connection merely dropping. ``task.status`` is emitted
    first either way, the same order ``_generate`` uses around its own
    warm-up failure.

    Unlike the warm-up path, a step-read failure on either fast path
    does not cancel that path's own lifecycle conclusion: the snapshot
    that picked the fast path was already read, successfully, before
    either generator started, so ``task.completed``/``task.input_required``
    is known-good independent of the steps read below it. Both bodies
    must therefore carry their conclusion frame *and* the
    ``stream.error``, conclusion first -- a step-read failure only costs
    the client the step content, never the fact that the task already
    reached this state. This test's ``principal`` is ``None`` -- it's
    only testing the steps-read failure, not authorization.

    The message is pinned exactly, not by substring: it has to be true
    of a failure preparing the snapshot in general, not name the steps
    read specifically -- the cursor baseline read shares this same
    ``except`` (see ``_fast_path_steps_read_error_frame``) and a wording
    that only fits one of the two would misdescribe the other whenever
    it's the one that actually failed."""

    def _broken_read_task_steps_response(task_id_, principal_):
        raise RuntimeError("transient DB error reading cached steps")

    def _unused_read_task_snapshot(task_id_, principal_):
        raise AssertionError(
            "the generation reread must not run when the steps read itself failed"
        )

    def _read_task_steps_version(task_id_, principal_):
        return SimpleNamespace(max_event_id=1)

    terminal_snapshot = SimpleNamespace(
        task_id=1, agent_id=1, status=TaskStatus.COMPLETED, output="done", error=None
    )
    terminal_frames = [
        chunk
        async for chunk in es._terminal_snapshot_stream(
            terminal_snapshot,
            principal=None,
            read_task_steps_response=_broken_read_task_steps_response,
            read_task_snapshot=_unused_read_task_snapshot,
            read_task_steps_version=_read_task_steps_version,
        )
    ]
    terminal_body = "".join(terminal_frames)
    assert terminal_body.count("event: task.status") == 1
    assert terminal_body.count("event: task.completed") == 1
    terminal_error = _parse_error_frame(terminal_body)
    assert terminal_error["code"] == "resync_required"
    assert terminal_error["message"] == (
        "Preparing the task's step snapshot failed; call steps() to "
        "resync, then re-attach."
    )
    # Order matters: the conclusion frame is the authoritative one and
    # must reach the client even if the error frame that follows it is
    # somehow lost -- not the other way around.
    assert terminal_body.index("event: task.completed") < terminal_body.index(
        "event: stream.error"
    )

    waiting_snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=TaskStatus.WAITING_FOR_USER,
        pending_question="what next?",
    )
    waiting_frames = [
        chunk
        async for chunk in es._input_required_snapshot_stream(
            waiting_snapshot,
            principal=None,
            read_task_steps_response=_broken_read_task_steps_response,
            read_task_snapshot=_unused_read_task_snapshot,
            read_task_steps_version=_read_task_steps_version,
        )
    ]
    waiting_body = "".join(waiting_frames)
    assert waiting_body.count("event: task.status") == 1
    assert waiting_body.count("event: task.input_required") == 1
    waiting_error = _parse_error_frame(waiting_body)
    assert waiting_error["code"] == "resync_required"
    assert waiting_error["message"] == (
        "Preparing the task's step snapshot failed; call steps() to "
        "resync, then re-attach."
    )
    assert waiting_body.index("event: task.input_required") < waiting_body.index(
        "event: stream.error"
    )


async def test_fast_path_task_not_found_closes_with_task_deleted_not_resync_required():
    """Both attach-time fast paths must distinguish a deleted task from
    an ordinary read failure: ``read_task_steps_response`` re-resolves
    the task (see ``TaskStepsResponseReader``) after
    ``build_event_stream_response`` already resolved it once to pick
    this fast path, so a task deleted in that exact gap surfaces here as
    ``V1ApiError(TASK_NOT_FOUND)`` -- the same condition the warm-up read
    and the watchdog already report as ``task_deleted`` (see
    ``test_generates_own_task_not_found_reads_are_abortive_and_end_watchdog_coverage``),
    not ``resync_required``. Asking a client to ``steps()`` and reattach
    for a task that's gone would just 404 instead of resyncing anything.

    The conclusion frame still goes out first in this case too: the
    snapshot that picked the fast path was read before the delete, so
    it's already in hand and does not depend on the re-resolve that
    just 404ed.

    ``principal`` is ``None`` here too, the same as the companion
    read-failure test above."""

    def _deleted_read_task_steps_response(task_id_, principal_):
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)

    def _unused_read_task_snapshot(task_id_, principal_):
        raise AssertionError(
            "the generation reread must not run when the steps read itself failed"
        )

    def _read_task_steps_version(task_id_, principal_):
        return SimpleNamespace(max_event_id=1)

    terminal_snapshot = SimpleNamespace(
        task_id=1, agent_id=1, status=TaskStatus.COMPLETED, output="done", error=None
    )
    terminal_frames = [
        chunk
        async for chunk in es._terminal_snapshot_stream(
            terminal_snapshot,
            principal=None,
            read_task_steps_response=_deleted_read_task_steps_response,
            read_task_snapshot=_unused_read_task_snapshot,
            read_task_steps_version=_read_task_steps_version,
        )
    ]
    terminal_body = "".join(terminal_frames)
    assert _parse_error_frame(terminal_body)["code"] == "task_deleted"
    assert terminal_body.count("event: task.completed") == 1
    assert terminal_body.index("event: task.completed") < terminal_body.index(
        "event: stream.error"
    )

    waiting_snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=TaskStatus.WAITING_FOR_USER,
        pending_question="what next?",
    )
    waiting_frames = [
        chunk
        async for chunk in es._input_required_snapshot_stream(
            waiting_snapshot,
            principal=None,
            read_task_steps_response=_deleted_read_task_steps_response,
            read_task_snapshot=_unused_read_task_snapshot,
            read_task_steps_version=_read_task_steps_version,
        )
    ]
    waiting_body = "".join(waiting_frames)
    assert _parse_error_frame(waiting_body)["code"] == "task_deleted"
    assert waiting_body.count("event: task.input_required") == 1
    assert waiting_body.index("event: task.input_required") < waiting_body.index(
        "event: stream.error"
    )


def test_attach_step_batch_bounds_match_the_documented_endpoint_contract():
    """Pins the two literal values every attach-time batch of steps is
    bounded by -- the fast paths' one-shot snapshot and the normal
    path's history replay alike. These aren't free internal tuning:
    ``tasks.py``'s ``stream_chat_task_events`` endpoint docstring
    documents them as the literal numbers 512 and 4 MiB (not symbolic
    references to this module) for API consumers reading the ``GET
    /v1/chat/tasks/{task_id}/events`` contract, so a change to either
    constant here has to be paired with updating that docstring -- this
    test exists so such a change doesn't slip through silently.

    Pinning the two module constants alone does not do that: a constant
    can drift out of sync with the docstring's own hardcoded numbers
    without either assertion below ever noticing, since neither reads
    the docstring. The assertions on ``__doc__`` close that gap by
    pinning the authoritative passage's exact wording and requiring
    each number to appear exactly once across the whole docstring, so a
    second copy reappearing elsewhere -- the failure mode a bare
    membership check cannot see -- fails the count.
    """
    assert es.REPLAY_MAX_STEPS == 512
    assert es.MAX_SNAPSHOT_WIRE_BYTES == 4 * 1024 * 1024

    endpoint_doc = v1_tasks.stream_chat_task_events.__doc__ or ""
    assert "a step-count cap (512) keeps the most recent steps" in endpoint_doc
    assert "total-wire-bytes budget (4 MiB)" in endpoint_doc
    # Exactly one passage carries each number. A second copy is what
    # lets one of them drift while the other stays right, so the count
    # is part of the contract this test pins, not incidental.
    assert endpoint_doc.count("512") == 1
    assert endpoint_doc.count("4 MiB") == 1


@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "done", "error": None},
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_step_snapshot_is_bounded_by_replay_max_steps(
    stream_fn, status, conclusion_event, extra_snapshot
):
    """The attach-time step snapshot is the one thing this stream sends
    in a single unpaced burst -- no admission/deadline/heartbeat loop --
    so it needs its own bound. This pins it: a task with 600 public
    steps (more than ``REPLAY_MAX_STEPS`` == 512) gets only its most
    recent 512, with the ``snapshot_truncated``/``snapshot_total_steps``
    marker on the conclusion frame and on no ``step.*`` frame."""
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    total = es.REPLAY_MAX_STEPS + 88
    steps = [
        es.PublicStep(
            id=f"tool_call:call-{i}",
            type="tool_call",
            status="completed",
            started_at=base,
            completed_at=base,
            data={"name": "search", "result": "ok"},
        )
        for i in range(total)
    ]

    def _read_task_steps_response(task_id_, principal_):
        return SimpleNamespace(steps=steps)

    def _unchanged_read_task_snapshot(task_id_, principal_):
        return SimpleNamespace(run_id="run-1", state_version=1)

    def _unchanged_read_task_steps_version(task_id_, principal_):
        return SimpleNamespace(max_event_id=1)

    snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=status,
        run_id="run-1",
        state_version=1,
        **extra_snapshot,
    )
    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=None,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_unchanged_read_task_snapshot,
            read_task_steps_version=_unchanged_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    assert body.count("event: step.completed") == es.REPLAY_MAX_STEPS
    blocks = [b for b in body.split("\n\n") if b.strip()]
    first_step_data = json.loads(blocks[1].split("data: ", 1)[1])
    assert "snapshot_truncated" not in first_step_data
    # The emitted frames are the *most recent* steps -- the first one
    # here is the (total - REPLAY_MAX_STEPS)'th step, not the very first.
    assert (
        first_step_data["step"]["id"] == f"tool_call:call-{total - es.REPLAY_MAX_STEPS}"
    )
    second_step_data = json.loads(blocks[2].split("data: ", 1)[1])
    assert "snapshot_truncated" not in second_step_data
    assert body.count(f"event: {conclusion_event}") == 1
    conclusion_block = next(
        b for b in blocks if b.startswith(f"event: {conclusion_event}")
    )
    conclusion_data = json.loads(conclusion_block.split("data: ", 1)[1])
    assert conclusion_data["snapshot_truncated"] is True
    assert conclusion_data["snapshot_total_steps"] == total


@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "done", "error": None},
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_step_snapshot_applies_the_replay_cap_before_the_byte_budget(
    stream_fn, status, conclusion_event, extra_snapshot
):
    """The two admission bounds are applied in a fixed order --
    ``REPLAY_MAX_STEPS`` first (keep the newest window), the byte
    budget second (trim that window from its oldest end) -- not the
    other way around. Pinned with a history where the two orders
    produce entirely different output: 88 old steps each carrying ~48
    KiB of ``data`` (enough on their own to exhaust the byte budget),
    then 512 new, tiny ones. Applying ``REPLAY_MAX_STEPS`` first keeps
    only the 512 tiny steps -- the byte budget never binds on them, so
    all 512 go out untouched. Applying the byte budget first, over the
    full 600-step history from its oldest end the way
    ``_snapshot_steps_within_wire_budget`` walks, would instead spend
    the whole budget on roughly the first 85 large steps and never even
    reach the 512 tiny ones behind them -- a reversal this test would
    catch by the wrong count and the wrong first step id below.
    """
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    old_large = [
        es.PublicStep(
            id=f"tool_call:old-{i}",
            type="tool_call",
            status="completed",
            started_at=base,
            completed_at=base,
            data={"name": "search", "result": "x" * (48 * 1024)},
        )
        for i in range(88)
    ]
    new_tiny = [
        es.PublicStep(
            id=f"tool_call:new-{i}",
            type="tool_call",
            status="completed",
            started_at=base,
            completed_at=base,
            data={"name": "search", "result": "ok"},
        )
        for i in range(es.REPLAY_MAX_STEPS)
    ]
    steps = old_large + new_tiny
    total = len(steps)

    def _read_task_steps_response(task_id_, principal_):
        return SimpleNamespace(steps=steps)

    def _unchanged_read_task_snapshot(task_id_, principal_):
        return SimpleNamespace(run_id="run-1", state_version=1)

    def _unchanged_read_task_steps_version(task_id_, principal_):
        return SimpleNamespace(max_event_id=1)

    snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=status,
        run_id="run-1",
        state_version=1,
        **extra_snapshot,
    )
    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=None,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_unchanged_read_task_snapshot,
            read_task_steps_version=_unchanged_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    assert body.count("event: step.completed") == es.REPLAY_MAX_STEPS
    blocks = [b for b in body.split("\n\n") if b.strip()]
    first_step_data = json.loads(blocks[1].split("data: ", 1)[1])
    assert first_step_data["step"]["id"] == "tool_call:new-0"
    assert body.count(f"event: {conclusion_event}") == 1
    conclusion_block = next(
        b for b in blocks if b.startswith(f"event: {conclusion_event}")
    )
    conclusion_data = json.loads(conclusion_block.split("data: ", 1)[1])
    assert conclusion_data["snapshot_truncated"] is True
    assert conclusion_data["snapshot_total_steps"] == total


@pytest.mark.parametrize(
    ("total", "expect_truncated"),
    [
        pytest.param(es.REPLAY_MAX_STEPS, False, id="exactly-at-cap"),
        pytest.param(es.REPLAY_MAX_STEPS + 1, True, id="one-over-cap"),
    ],
)
@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "done", "error": None},
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_step_snapshot_replay_max_steps_boundary(
    stream_fn, status, conclusion_event, extra_snapshot, total, expect_truncated
):
    """``REPLAY_MAX_STEPS`` is a strict-over boundary, the same rule
    every other size cap in this module uses (see ``_put_or_overflow``
    and ``MAX_RAW_FRAME_TEXT_CHARS``'s own comments): a history of
    exactly 512 steps fits whole and is not truncated, one more tips it
    over. Expressed against ``es.REPLAY_MAX_STEPS`` rather than the
    literal 512/513 so this test keeps pinning the boundary itself even
    if the constant's value ever changes -- that value is pinned
    separately, by
    ``test_attach_step_batch_bounds_match_the_documented_endpoint_contract``.
    """
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    steps = [
        es.PublicStep(
            id=f"tool_call:call-{i}",
            type="tool_call",
            status="completed",
            started_at=base,
            completed_at=base,
            data={"name": "search", "result": "ok"},
        )
        for i in range(total)
    ]

    def _read_task_steps_response(task_id_, principal_):
        return SimpleNamespace(steps=steps)

    def _unchanged_read_task_snapshot(task_id_, principal_):
        return SimpleNamespace(run_id="run-1", state_version=1)

    def _unchanged_read_task_steps_version(task_id_, principal_):
        return SimpleNamespace(max_event_id=1)

    snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=status,
        run_id="run-1",
        state_version=1,
        **extra_snapshot,
    )
    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=None,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_unchanged_read_task_snapshot,
            read_task_steps_version=_unchanged_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    assert body.count("event: step.completed") == min(total, es.REPLAY_MAX_STEPS)
    assert body.count(f"event: {conclusion_event}") == 1
    blocks = [b for b in body.split("\n\n") if b.strip()]
    conclusion_block = next(
        b for b in blocks if b.startswith(f"event: {conclusion_event}")
    )
    conclusion_data = json.loads(conclusion_block.split("data: ", 1)[1])
    if expect_truncated:
        assert conclusion_data["snapshot_truncated"] is True
        assert conclusion_data["snapshot_total_steps"] == total
    else:
        assert "snapshot_truncated" not in conclusion_data
        assert "snapshot_total_steps" not in conclusion_data


@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "done", "error": None},
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_step_snapshot_is_bounded_by_the_wire_byte_budget(
    stream_fn, status, conclusion_event, extra_snapshot
):
    """``REPLAY_MAX_STEPS`` alone doesn't stop a snapshot from getting
    huge: 128 steps at 48 KiB of ``data`` each (well under the 64 KiB
    per-step cap, so ``_capped_step_data`` never fires) is ~6 MiB, over
    ``MAX_SNAPSHOT_WIRE_BYTES`` (4 MiB) while nowhere near
    ``REPLAY_MAX_STEPS`` (512). This pins that the byte budget -- not
    the step count -- is what cuts this particular snapshot short."""
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    total = 128
    steps = [
        es.PublicStep(
            id=f"tool_call:call-{i}",
            type="tool_call",
            status="completed",
            started_at=base,
            completed_at=base,
            data={"name": "search", "result": "x" * (48 * 1024)},
        )
        for i in range(total)
    ]

    def _read_task_steps_response(task_id_, principal_):
        return SimpleNamespace(steps=steps)

    def _unchanged_read_task_snapshot(task_id_, principal_):
        return SimpleNamespace(run_id="run-1", state_version=1)

    def _unchanged_read_task_steps_version(task_id_, principal_):
        return SimpleNamespace(max_event_id=1)

    snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=status,
        run_id="run-1",
        state_version=1,
        **extra_snapshot,
    )
    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=None,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_unchanged_read_task_snapshot,
            read_task_steps_version=_unchanged_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    step_frames = [chunk for chunk in frames if "event: step." in chunk]
    # The byte budget bound this well short of all 128 steps -- not zero
    # (some steps still fit) and not all of them (the budget did bind).
    assert 0 < len(step_frames) < total
    # The truncation marker now rides the conclusion frame, not a step
    # frame, so the measuring pass's count is exactly what the step
    # frames put on the wire -- no marker overrun left to allow for.
    assert sum(len(chunk) for chunk in step_frames) <= es.MAX_SNAPSHOT_WIRE_BYTES
    first_step_block = next(
        block for block in body.split("\n\n") if block.startswith("event: step.")
    )
    first_step_data = json.loads(first_step_block.split("data: ", 1)[1])
    assert "snapshot_truncated" not in first_step_data
    assert body.count(f"event: {conclusion_event}") == 1
    blocks = [b for b in body.split("\n\n") if b.strip()]
    conclusion_block = next(
        b for b in blocks if b.startswith(f"event: {conclusion_event}")
    )
    conclusion_data = json.loads(conclusion_block.split("data: ", 1)[1])
    assert conclusion_data["snapshot_truncated"] is True
    assert conclusion_data["snapshot_total_steps"] == total
    # A byte-budget cut is a truncated snapshot, not a stream error -- the
    # client is told via ``snapshot_truncated``, not a close frame.
    assert "event: stream.error" not in body


@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "done", "error": None},
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_snapshot_marks_truncation_when_the_budget_admits_no_steps(
    monkeypatch, stream_fn, status, conclusion_event, extra_snapshot
):
    """The byte budget can cut a snapshot to zero steps, not just to a
    shorter one: with the budget set below a single step's own wire
    length, ``_snapshot_steps_within_wire_budget`` admits none of the
    task's steps at all. No ``step.*`` frame is ever emitted in that
    case, so the truncation marker has nowhere to ride but the
    conclusion frame -- this pins that it still gets there. Production
    can reach this without an artificially tiny budget: ``PublicStep.id``
    is built from the event's own ``tool_call_id``, which nothing in
    this stream bounds the length of, so one tool call minting an
    unusually long id can by itself push that step's frame past
    ``MAX_SNAPSHOT_WIRE_BYTES``.
    """
    monkeypatch.setattr(es, "MAX_SNAPSHOT_WIRE_BYTES", 8)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    steps = [
        es.PublicStep(
            id=f"tool_call:call-{i}",
            type="tool_call",
            status="completed",
            started_at=base,
            completed_at=base,
            data={"name": "search", "result": "ok"},
        )
        for i in range(2)
    ]

    def _read_task_steps_response(task_id_, principal_):
        return SimpleNamespace(steps=steps)

    def _unused_read_task_snapshot(task_id_, principal_):
        raise AssertionError(
            "the generation reread must not run when no step was admitted"
        )

    def _read_task_steps_version(task_id_, principal_):
        return SimpleNamespace(max_event_id=1)

    snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=status,
        run_id="run-1",
        state_version=1,
        **extra_snapshot,
    )
    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=None,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_unused_read_task_snapshot,
            read_task_steps_version=_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    assert body.count("event: step.") == 0
    assert body.count(f"event: {conclusion_event}") == 1
    assert "event: stream.error" not in body
    blocks = [b for b in body.split("\n\n") if b.strip()]
    conclusion_block = next(
        b for b in blocks if b.startswith(f"event: {conclusion_event}")
    )
    conclusion_data = json.loads(conclusion_block.split("data: ", 1)[1])
    assert conclusion_data["snapshot_truncated"] is True
    assert conclusion_data["snapshot_total_steps"] == 2


async def test_fast_path_snapshot_admits_a_frame_exactly_at_the_byte_budget(
    monkeypatch,
):
    """Same boundary rule as the queue's own byte budget (see
    ``test_backlog_exactly_at_the_byte_budget_does_not_close``): strictly
    over the budget cuts the snapshot short, landing exactly on it does
    not. Drives ``_fast_path_step_snapshot`` directly with two identical
    steps and a budget set to exactly two frames' worth of wire bytes,
    then one byte short of that, so the boundary is pinned to the exact
    byte that tips it rather than one before or after."""
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    step = es.PublicStep(
        id="tool_call:call-1",
        type="tool_call",
        status="completed",
        started_at=base,
        completed_at=base,
        data={"name": "search", "result": "ok"},
    )
    frame_len = len(es._step_wire_frame(step))
    steps = [step, step]

    def _read_task_steps_response(task_id_, principal_):
        return SimpleNamespace(steps=steps)

    monkeypatch.setattr(es, "MAX_SNAPSHOT_WIRE_BYTES", frame_len * 2)
    admitted, total_steps = await es._fast_path_step_snapshot(
        1, None, _read_task_steps_response
    )
    assert len(admitted) == 2
    assert total_steps is None

    monkeypatch.setattr(es, "MAX_SNAPSHOT_WIRE_BYTES", frame_len * 2 - 1)
    admitted, total_steps = await es._fast_path_step_snapshot(
        1, None, _read_task_steps_response
    )
    assert len(admitted) == 1
    assert total_steps == 2


async def test_fast_path_snapshot_marks_truncation_when_the_window_ends_unmeasurable():
    """An unserializable step is admitted so the emit loop can reach it
    and report it (see ``_snapshot_steps_within_wire_budget``), but
    admitting it also ends the measuring pass, so any steps behind it
    fall out of the snapshot. The returned total must then mark the
    snapshot truncated -- fewer steps go out than the task has -- even
    though the step-count cap alone never bound anything here.
    """
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    class _Unserializable:
        pass

    def _step(step_id, data):
        return es.PublicStep(
            id=step_id,
            type="tool_call",
            status="completed",
            started_at=base,
            completed_at=base,
            data=data,
        )

    steps = [
        _step("tool_call:call-1", {"name": "search", "result": "ok"}),
        _step("tool_call:call-2", {"x": _Unserializable()}),
        _step("tool_call:call-3", {"name": "search", "result": "ok"}),
    ]

    def _read_task_steps_response(task_id_, principal_):
        return SimpleNamespace(steps=steps)

    admitted, total_steps = await es._fast_path_step_snapshot(
        1, None, _read_task_steps_response
    )
    assert [s.id for s in admitted] == ["tool_call:call-1", "tool_call:call-2"]
    assert total_steps == 3


# A sentinel distinguishable from any real ``ApiKeyPrincipal`` (and from
# ``None``, the placeholder every other fast-path test in this file
# passes), so the fence tests below can pin that the reread genuinely
# receives the same ``(task_id, principal)`` pair the steps read did,
# not just some value.
_FENCE_PRINCIPAL = object()


# ===== fast-path generation fence (steps read outlives the snapshot) =====


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        pytest.param("run_id", "run-new", id="run-id-moved"),
        pytest.param("state_version", 2, id="state-version-moved"),
    ],
)
@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot", "conclusion_marker"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "stale output from the superseded run", "error": None},
            "stale output from the superseded run",
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            "what next?",
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_generation_change_withholds_steps_but_still_concludes(
    stream_fn,
    status,
    conclusion_event,
    extra_snapshot,
    conclusion_marker,
    changed_field,
    changed_value,
):
    """Between the snapshot that picked a fast path and that path's own
    steps read, the task row can move to a new run (a ``POST reply``
    restarting a ``WAITING_FOR_USER`` task, or a WS ``APPEND`` restarting
    a ``COMPLETED``/``FAILED`` one), and the steps read afterward already
    belong to that new run. Simulated by handing back one step and a
    reread whose generation differs -- the shape a real restart leaves
    behind, without driving an actual restart through the DB.

    What the fence withholds is the step, not the conclusion: the
    conclusion describes ``snapshot``'s own authoritative read, the read
    that selected this path, so it goes out here exactly as it does when
    the reread fails and when the step list is empty, and
    ``stream.error(resync_required)`` follows it naming the restart.

    Four cells: both fast paths crossed with both fields the fence
    compares. Either field moving on its own is enough, so a fence that
    only compared ``run_id`` would pass the ``run-id-moved`` cells and
    fail the ``state-version-moved`` ones -- which a single-field pair
    could not distinguish.
    """
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    step = es.PublicStep(
        id="tool_call:call-1",
        type="tool_call",
        status="completed",
        started_at=base,
        completed_at=base,
        data={"name": "search", "result": "ok"},
    )
    original = {"run_id": "run-1", "state_version": 1}
    steps_calls: list[tuple[int, object]] = []
    reread_calls: list[tuple[int, object]] = []

    def _read_task_steps_response(task_id_, principal_):
        steps_calls.append((task_id_, principal_))
        return SimpleNamespace(steps=[step])

    def _reread_task_snapshot(task_id_, principal_):
        reread_calls.append((task_id_, principal_))
        return SimpleNamespace(**{**original, changed_field: changed_value})

    def _read_task_steps_version(task_id_, principal_):
        return SimpleNamespace(max_event_id=1)

    snapshot = SimpleNamespace(
        task_id=1, agent_id=1, status=status, **original, **extra_snapshot
    )
    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=_FENCE_PRINCIPAL,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_reread_task_snapshot,
            read_task_steps_version=_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    assert body.count("event: task.status") == 1
    assert body.count(f"event: {conclusion_event}") == 1
    assert "event: step.completed" not in body
    assert body.count("event: stream.error") == 1
    assert "resync_required" in body
    # The wording names what the fence actually knows -- a lifecycle
    # write -- not a claim that a new run started: an intra-run write
    # (a reply resume, a lease release) bumps state_version without
    # touching run_id at all, and this leg's own ``run-id-moved`` case
    # is the only one of the two that even could be a new run.
    assert "The task changed while this attach was reading" in body
    assert "moved to a new run" not in body
    # The conclusion precedes the error, same ordering as every other
    # failure exit on this path.
    assert body.index(f"event: {conclusion_event}") < body.index("event: stream.error")
    # The conclusion still carries ``snapshot``'s own values, not a
    # placeholder standing in for the superseded run.
    assert conclusion_marker in body
    assert steps_calls == [(snapshot.task_id, _FENCE_PRINCIPAL)]
    assert reread_calls == [(snapshot.task_id, _FENCE_PRINCIPAL)]


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        pytest.param("run_id", "run-new", id="run-id-moved"),
        pytest.param("state_version", 2, id="state-version-moved"),
    ],
)
@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "done", "error": None},
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_confirmed_generation_change_short_circuits_the_cursor_recheck(
    stream_fn, status, conclusion_event, extra_snapshot, changed_field, changed_value
):
    """The fence's cursor recheck only runs ``if not changed`` (see the
    call site in ``_fast_path_snapshot_stream``): ``changed`` is
    assigned, not OR'd, from ``_fast_path_steps_cursor_changed``'s
    return value, so that guard is a short circuit, not just an
    optimization -- without it, a generation reread that already
    confirmed a change would have its ``True`` overwritten by whatever
    the cursor recheck itself returns. A generation change can leave
    the steps cursor unmoved (a lease release, or a resume that hasn't
    re-run any tool yet), which is exactly the case pinned here: the
    cursor reader always reports the same ``max_event_id`` it gave the
    baseline, so a cursor recheck that ran anyway would report
    "unchanged" and silently flip the fence back open, sending steps
    the generation reread had already ruled stale.

    Pinned by recording every call the cursor reader receives: the
    short circuit means it must be called exactly once (the baseline,
    captured before the steps read), never a second time for the
    recheck, once the generation reread alone has already confirmed a
    change.
    """
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    step = es.PublicStep(
        id="tool_call:call-1",
        type="tool_call",
        status="completed",
        started_at=base,
        completed_at=base,
        data={"name": "search", "result": "ok"},
    )
    original = {"run_id": "run-1", "state_version": 1}

    def _read_task_steps_response(task_id_, principal_):
        return SimpleNamespace(steps=[step])

    def _reread_task_snapshot(task_id_, principal_):
        return SimpleNamespace(**{**original, changed_field: changed_value})

    version_calls: list[tuple[int, object]] = []

    def _read_task_steps_version(task_id_, principal_):
        # Constant across calls: a recheck that ran anyway would see the
        # same cursor it saw at baseline and report "unchanged" -- the
        # scenario that makes an un-short-circuited recheck dangerous.
        version_calls.append((task_id_, principal_))
        return SimpleNamespace(max_event_id=100)

    snapshot = SimpleNamespace(
        task_id=1, agent_id=1, status=status, **original, **extra_snapshot
    )
    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=_FENCE_PRINCIPAL,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_reread_task_snapshot,
            read_task_steps_version=_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    # The short circuit's own contract: exactly the baseline call, never
    # a recheck once the generation reread alone confirmed a change.
    assert version_calls == [(snapshot.task_id, _FENCE_PRINCIPAL)]
    assert body.count(f"event: {conclusion_event}") == 1
    assert "event: step.completed" not in body
    error_data = _parse_error_frame(body)
    assert error_data["code"] == "resync_required"
    assert "The task changed while this attach was reading" in error_data["message"]


@pytest.mark.parametrize("failure_exit", ["steps_read_failure", "generation_changed"])
@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "done", "error": None},
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_failure_exits_never_carry_the_truncation_marker(
    stream_fn, status, conclusion_event, extra_snapshot, failure_exit
):
    """Every failure/withhold exit on the fast paths calls
    ``build_conclusion(None)`` (see ``_fast_path_snapshot_stream``'s own
    docstring: "No snapshot was confirmed here, so the conclusion
    carries no truncation marker"). This pins that even when the
    underlying history is large enough that a *successful* read would
    have truncated it -- 600 steps, over ``REPLAY_MAX_STEPS`` -- neither
    failure exit's conclusion frame leaks ``snapshot_truncated`` /
    ``snapshot_total_steps``.

    The two exits differ in how far they get before failing:
    ``steps_read_failure`` never learns the history's size at all (the
    steps read itself raises), while ``generation_changed`` does read
    all 600 steps -- ``_fast_path_step_snapshot`` internally admits the
    most recent 512 and reports ``total_steps=600`` -- and only then
    gets withheld by the fence, so it's the one exit that actually has a
    computed truncation count in hand and still has to not send it.
    """
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    total = es.REPLAY_MAX_STEPS + 88

    if failure_exit == "steps_read_failure":

        def _read_task_steps_response(task_id_, principal_):
            raise RuntimeError("transient DB error reading cached steps")

        def _read_task_snapshot(task_id_, principal_):
            raise AssertionError(
                "the generation reread must not run when the steps read itself failed"
            )
    else:

        def _read_task_steps_response(task_id_, principal_):
            return SimpleNamespace(
                steps=[
                    es.PublicStep(
                        id=f"tool_call:call-{i}",
                        type="tool_call",
                        status="completed",
                        started_at=base,
                        completed_at=base,
                        data={"name": "search", "result": "ok"},
                    )
                    for i in range(total)
                ]
            )

        def _read_task_snapshot(task_id_, principal_):
            return SimpleNamespace(run_id="run-new", state_version=1)

    def _read_task_steps_version(task_id_, principal_):
        return SimpleNamespace(max_event_id=1)

    snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=status,
        run_id="run-1",
        state_version=1,
        **extra_snapshot,
    )
    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=None,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_read_task_snapshot,
            read_task_steps_version=_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    assert body.count(f"event: {conclusion_event}") == 1
    blocks = [b for b in body.split("\n\n") if b.strip()]
    conclusion_block = next(
        b for b in blocks if b.startswith(f"event: {conclusion_event}")
    )
    conclusion_data = json.loads(conclusion_block.split("data: ", 1)[1])
    assert "snapshot_truncated" not in conclusion_data
    assert "snapshot_total_steps" not in conclusion_data
    assert body.count("event: stream.error") == 1


@pytest.mark.parametrize(
    ("second_max_event_id", "expect_resync"),
    [
        pytest.param(101, True, id="trace-row-landed"),
        pytest.param(100, False, id="cursor-unchanged"),
    ],
)
@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "done", "error": None},
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_trace_row_landing_between_reads_forces_resync(
    stream_fn,
    status,
    conclusion_event,
    extra_snapshot,
    second_max_event_id,
    expect_resync,
):
    """``DatabaseTraceHandler`` commits trace rows through its own
    session; even on the commits that also write this task row's own
    checkpoint-pointer columns, that write never touches ``run_id`` or
    ``state_version``. So a trace row that lands after the steps read
    and before the fence's recheck moves the steps cursor
    (``max_event_id``) without moving either of those fields -- invisible
    to ``_fast_path_generation_changed`` on its own. This pins the
    second signal, ``_fast_path_steps_cursor_changed``: the cursor is
    read once before the steps read and once more after the generation
    reread passes, and a moved cursor takes the same withhold-and-resync
    exit a moved generation does. The cursor-unchanged leg pins the
    negative: two matching reads never withhold the step or close with
    an error.
    """
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    step = es.PublicStep(
        id="tool_call:call-1",
        type="tool_call",
        status="completed",
        started_at=base,
        completed_at=base,
        data={"name": "search", "result": "ok"},
    )

    def _read_task_steps_response(task_id_, principal_):
        return SimpleNamespace(steps=[step])

    def _unchanged_read_task_snapshot(task_id_, principal_):
        return SimpleNamespace(run_id="run-1", state_version=1)

    version_calls: list[tuple[int, object]] = []
    version_replies = iter([100, second_max_event_id])

    def _read_task_steps_version(task_id_, principal_):
        version_calls.append((task_id_, principal_))
        return SimpleNamespace(max_event_id=next(version_replies))

    snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=status,
        run_id="run-1",
        state_version=1,
        **extra_snapshot,
    )
    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=_FENCE_PRINCIPAL,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_unchanged_read_task_snapshot,
            read_task_steps_version=_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    assert version_calls == [
        (snapshot.task_id, _FENCE_PRINCIPAL),
        (snapshot.task_id, _FENCE_PRINCIPAL),
    ]
    assert body.count(f"event: {conclusion_event}") == 1
    if expect_resync:
        assert "event: step.completed" not in body
        assert body.count("event: stream.error") == 1
        assert "resync_required" in body
        assert body.index(f"event: {conclusion_event}") < body.index(
            "event: stream.error"
        )
    else:
        assert body.count("event: step.completed") == 1
        assert "event: stream.error" not in body


@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "done", "error": None},
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_steps_cursor_baseline_is_captured_before_the_steps_read(
    stream_fn, status, conclusion_event, extra_snapshot
):
    """The cursor baseline (``read_task_steps_version``'s first call) is
    captured before the steps read runs, not after it returns -- pinned
    here with a steps-read stub that advances the cursor as a side
    effect, the shape a trace row landing mid-read actually takes: the
    read and the row's commit interleave, rather than the row landing
    only after the read has already finished. If the baseline were
    captured after the steps read instead, it would already observe the
    advanced cursor, and the recheck that follows would see no further
    movement -- silently hiding exactly the race this second signal
    exists to catch. The trace-row-landing test above stubs the version
    reader with a fixed two-value sequence, which can't tell "captured
    before the read" from "captured after" apart; this test can, because
    the cursor's own value moves as a side effect of the steps read
    itself.
    """
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    step = es.PublicStep(
        id="tool_call:call-1",
        type="tool_call",
        status="completed",
        started_at=base,
        completed_at=base,
        data={"name": "search", "result": "ok"},
    )
    cursor_state = {"max_event_id": 100}

    def _read_task_steps_response(task_id_, principal_):
        # A trace row commits while this read is "in flight" -- the
        # cursor moves as a side effect of the steps read itself, not
        # via a separately-scripted later call.
        cursor_state["max_event_id"] = 101
        return SimpleNamespace(steps=[step])

    def _unchanged_read_task_snapshot(task_id_, principal_):
        return SimpleNamespace(run_id="run-1", state_version=1)

    version_calls: list[int] = []

    def _read_task_steps_version(task_id_, principal_):
        version_calls.append(cursor_state["max_event_id"])
        return SimpleNamespace(max_event_id=cursor_state["max_event_id"])

    snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=status,
        run_id="run-1",
        state_version=1,
        **extra_snapshot,
    )
    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=_FENCE_PRINCIPAL,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_unchanged_read_task_snapshot,
            read_task_steps_version=_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    # The baseline call observed 100 (before the steps read mutated the
    # counter); the recheck call observed 101 (after). A baseline
    # captured post-read would have observed 101 both times, and the
    # assertions below would fail.
    assert version_calls == [100, 101]
    assert "event: step.completed" not in body
    assert body.count("event: stream.error") == 1
    assert "resync_required" in body
    assert body.index(f"event: {conclusion_event}") < body.index("event: stream.error")


@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "done", "error": None},
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_steps_cursor_baseline_read_failure_closes_before_the_steps_read(
    stream_fn, status, conclusion_event, extra_snapshot
):
    """The cursor baseline read can fail the same way any other DB read
    in this module can, and it fails before the steps read is ever
    reached (see the call site in ``_fast_path_snapshot_stream``) -- so
    the steps reader must never run, and the failure is classified
    exactly like a failed steps read
    (``_fast_path_steps_read_error_frame``): a task deleted in the gap
    gets ``task_deleted``, everything else gets ``resync_required``. The
    conclusion frame (already known-good from ``snapshot``) goes out
    first either way.
    """

    def _unreachable_read_task_steps_response(task_id_, principal_):
        raise AssertionError(
            "the steps read must not run when the cursor baseline read failed"
        )

    def _unreachable_read_task_snapshot(task_id_, principal_):
        raise AssertionError(
            "the generation reread must not run when the cursor baseline read failed"
        )

    snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=status,
        run_id="run-1",
        state_version=1,
        **extra_snapshot,
    )

    def _transient_read_task_steps_version(task_id_, principal_):
        raise RuntimeError("transient DB error reading the steps cursor")

    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=None,
            read_task_steps_response=_unreachable_read_task_steps_response,
            read_task_snapshot=_unreachable_read_task_snapshot,
            read_task_steps_version=_transient_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    assert body.count(f"event: {conclusion_event}") == 1
    assert body.count("event: stream.error") == 1
    assert "resync_required" in body
    assert body.index(f"event: {conclusion_event}") < body.index("event: stream.error")

    def _deleted_read_task_steps_version(task_id_, principal_):
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)

    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=None,
            read_task_steps_response=_unreachable_read_task_steps_response,
            read_task_snapshot=_unreachable_read_task_snapshot,
            read_task_steps_version=_deleted_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    assert body.count(f"event: {conclusion_event}") == 1
    assert "task_deleted" in body
    assert "resync_required" not in body
    assert body.index(f"event: {conclusion_event}") < body.index("event: stream.error")


@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "done", "error": None},
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_steps_cursor_recheck_failure_closes_for_resync_or_deleted(
    stream_fn, status, conclusion_event, extra_snapshot
):
    """The cursor recheck (``read_task_steps_version``'s second call,
    which only runs once the run_id/state_version reread has already
    confirmed no change) can itself fail. It shares
    ``_fast_path_generation_reread_error_frame`` with the
    run_id/state_version reread's own failure -- by the time this call
    raises, the generation was already confirmed unchanged, only the
    cursor wasn't, which is exactly why that shared frame builder's
    wording no longer names "generation" specifically.
    """
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    step = es.PublicStep(
        id="tool_call:call-1",
        type="tool_call",
        status="completed",
        started_at=base,
        completed_at=base,
        data={"name": "search", "result": "ok"},
    )

    def _read_task_steps_response(task_id_, principal_):
        return SimpleNamespace(steps=[step])

    def _unchanged_read_task_snapshot(task_id_, principal_):
        return SimpleNamespace(run_id="run-1", state_version=1)

    snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=status,
        run_id="run-1",
        state_version=1,
        **extra_snapshot,
    )

    calls = {"n": 0}

    def _flaky_read_task_steps_version(task_id_, principal_):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(max_event_id=100)
        raise RuntimeError("transient DB error rereading the steps cursor")

    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=None,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_unchanged_read_task_snapshot,
            read_task_steps_version=_flaky_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    assert body.count(f"event: {conclusion_event}") == 1
    assert body.count("event: stream.error") == 1
    assert "resync_required" in body
    assert body.index(f"event: {conclusion_event}") < body.index("event: stream.error")

    calls2 = {"n": 0}

    def _flaky_read_task_steps_version_deleted(task_id_, principal_):
        calls2["n"] += 1
        if calls2["n"] == 1:
            return SimpleNamespace(max_event_id=100)
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)

    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=None,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_unchanged_read_task_snapshot,
            read_task_steps_version=_flaky_read_task_steps_version_deleted,
        )
    ]
    body = "".join(frames)
    assert body.count(f"event: {conclusion_event}") == 1
    assert "task_deleted" in body
    assert "resync_required" not in body
    assert body.index(f"event: {conclusion_event}") < body.index("event: stream.error")


@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "done", "error": None},
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_empty_steps_skips_the_generation_reread(
    stream_fn, status, conclusion_event, extra_snapshot
):
    """An empty step list carries no step content from a possibly-newer
    run, so there is nothing for the generation reread to protect
    against -- this pins that the fast path skips it entirely rather
    than paying for a read whose answer can't change the outcome."""

    steps_calls: list[tuple[int, object]] = []

    def _read_task_steps_response(task_id_, principal_):
        steps_calls.append((task_id_, principal_))
        return SimpleNamespace(steps=[])

    reread_calls: list[tuple[int, object]] = []

    def _reread_task_snapshot(task_id_, principal_):
        reread_calls.append((task_id_, principal_))
        return SimpleNamespace(run_id="run-1", state_version=1)

    def _read_task_steps_version(task_id_, principal_):
        return SimpleNamespace(max_event_id=1)

    snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=status,
        run_id="run-1",
        state_version=1,
        **extra_snapshot,
    )
    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=_FENCE_PRINCIPAL,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_reread_task_snapshot,
            read_task_steps_version=_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    assert steps_calls == [(snapshot.task_id, _FENCE_PRINCIPAL)]
    assert reread_calls == []
    assert body.count(f"event: {conclusion_event}") == 1
    assert "resync_required" not in body


@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "done", "error": None},
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_generation_reread_failure_concludes_then_closes_for_resync(
    stream_fn, status, conclusion_event, extra_snapshot
):
    """The generation reread can fail the same way any other DB read in
    this module can (a transient error) -- that answers neither
    "unchanged" nor "changed", so it is treated like the steps-read
    failure just above it: the conclusion still goes out (it was already
    known-good from ``snapshot``, independent of this reread), the step
    this path read is withheld since its generation was never confirmed,
    and the accompanying ``stream.error`` must describe what actually
    happened -- the reread itself failed -- not claim the task moved to a
    new run, which was never established either way. The raise happens
    on the run_id/state_version reread itself, before the cursor recheck
    that can also route through the same error frame ever runs (see
    ``_fast_path_snapshot_stream``'s ``if not changed:`` guard), so this
    exercises ``_fast_path_generation_reread_error_frame``'s generic
    wording for that reread on its own (see
    ``test_fast_path_steps_cursor_recheck_failure_closes_for_resync_or_deleted``
    for the cursor-recheck leg of the same error frame).
    """
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    step = es.PublicStep(
        id="tool_call:call-1",
        type="tool_call",
        status="completed",
        started_at=base,
        completed_at=base,
        data={"name": "search", "result": "ok"},
    )

    steps_calls: list[tuple[int, object]] = []
    reread_calls: list[tuple[int, object]] = []

    def _read_task_steps_response(task_id_, principal_):
        steps_calls.append((task_id_, principal_))
        return SimpleNamespace(steps=[step])

    def _raising_reread(task_id_, principal_):
        reread_calls.append((task_id_, principal_))
        raise RuntimeError("transient DB error rereading task snapshot")

    def _read_task_steps_version(task_id_, principal_):
        return SimpleNamespace(max_event_id=1)

    snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=status,
        run_id="run-1",
        state_version=1,
        **extra_snapshot,
    )
    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=_FENCE_PRINCIPAL,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_raising_reread,
            read_task_steps_version=_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    assert body.count("event: task.status") == 1
    assert body.count(f"event: {conclusion_event}") == 1
    assert "event: step.completed" not in body
    assert body.count("event: stream.error") == 1
    assert "resync_required" in body
    # The wording names what actually failed to confirm -- the reread --
    # not a claim that the task moved to a new run, which this failure
    # never established either way.
    assert "Confirming the task's steps are still current failed" in body
    assert "moved to a new run" not in body
    # The conclusion precedes the error on this exit too -- neither leg
    # asserted that before, so a reordered yield would have passed both.
    assert body.index(f"event: {conclusion_event}") < body.index("event: stream.error")
    assert steps_calls == [(snapshot.task_id, _FENCE_PRINCIPAL)]
    assert reread_calls == [(snapshot.task_id, _FENCE_PRINCIPAL)]


@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "done", "error": None},
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_generation_reread_task_deleted_closes_with_task_deleted(
    stream_fn, status, conclusion_event, extra_snapshot
):
    """A task deleted in the gap between the steps read and the
    generation reread surfaces from the reread as
    ``V1ApiError(TASK_NOT_FOUND)``, the same exception shape a task
    deleted during the steps read itself already gets classified by
    (``_fast_path_steps_read_error_frame``). The generation reread's own
    classifier must recognize it the same way rather than falling
    through to the generic ``resync_required`` every other reread
    failure gets: the task isn't merely unreadable here, it's gone, so
    ``steps()`` + reattach would just 404 instead of resyncing
    anything."""
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    step = es.PublicStep(
        id="tool_call:call-1",
        type="tool_call",
        status="completed",
        started_at=base,
        completed_at=base,
        data={"name": "search", "result": "ok"},
    )

    steps_calls: list[tuple[int, object]] = []
    reread_calls: list[tuple[int, object]] = []

    def _read_task_steps_response(task_id_, principal_):
        steps_calls.append((task_id_, principal_))
        return SimpleNamespace(steps=[step])

    def _deleted_reread(task_id_, principal_):
        reread_calls.append((task_id_, principal_))
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)

    def _read_task_steps_version(task_id_, principal_):
        return SimpleNamespace(max_event_id=1)

    snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=status,
        run_id="run-1",
        state_version=1,
        **extra_snapshot,
    )
    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=_FENCE_PRINCIPAL,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_deleted_reread,
            read_task_steps_version=_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    assert body.count("event: task.status") == 1
    assert body.count(f"event: {conclusion_event}") == 1
    assert "event: step.completed" not in body
    assert body.count("event: stream.error") == 1
    assert "task_deleted" in body
    assert "resync_required" not in body
    # The conclusion precedes the error, same ordering pinned on every
    # other failure exit on this path.
    assert body.index(f"event: {conclusion_event}") < body.index("event: stream.error")
    assert steps_calls == [(snapshot.task_id, _FENCE_PRINCIPAL)]
    assert reread_calls == [(snapshot.task_id, _FENCE_PRINCIPAL)]


# ===== fast-path step serialization failure (after the first yield) =====


@pytest.mark.parametrize(
    ("stream_fn", "status", "conclusion_event", "extra_snapshot"),
    [
        pytest.param(
            "_terminal_snapshot_stream",
            TaskStatus.COMPLETED,
            "task.completed",
            {"output": "done", "error": None},
            id="terminal",
        ),
        pytest.param(
            "_input_required_snapshot_stream",
            TaskStatus.WAITING_FOR_USER,
            "task.input_required",
            {"pending_question": "what next?"},
            id="waiting-for-user",
        ),
    ],
)
async def test_fast_path_unserializable_step_concludes_then_closes_for_resync(
    stream_fn, status, conclusion_event, extra_snapshot
):
    """The fast path's serialization loop runs after
    ``StreamingResponse`` has sent 200, and ``model_dump(mode="json")``
    raises ``PydanticSerializationError`` on a value it has no encoding
    rule for. In production the injected steps reader dumps its own
    response first, inside the steps-read guard, so this loop's guard is
    a defensive boundary; the stub reader here hands the loop a
    ``PublicStep`` that reader could not have produced, to pin what the
    boundary does. The raise happens after ``StreamingResponse`` has
    already sent 200 and the headers, so an unguarded raise would end the
    response with no close frame at all -- indistinguishable, to the
    client, from the connection dropping. Instead the conclusion goes out,
    then ``stream.error(resync_required)``, and the steps before the
    failing one are kept.

    Both legs use a ``[good, bad]`` step list so both assert the
    partial-send behavior, not just the terminal one: a guard that
    discarded the whole list on the first failure would still satisfy a
    single-step leg.
    """
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    class _Unserializable:
        pass

    good = es.PublicStep(
        id="tool_call:call-1",
        type="tool_call",
        status="completed",
        started_at=base,
        completed_at=base,
        data={"name": "search", "result": "ok"},
    )
    bad = es.PublicStep(
        id="tool_call:call-2",
        type="tool_call",
        status="completed",
        started_at=base,
        completed_at=base,
        data={"name": "search", "result": _Unserializable()},
    )

    def _read_task_steps_response(task_id_, principal_):
        return SimpleNamespace(steps=[good, bad])

    def _reread_task_snapshot(task_id_, principal_):
        return SimpleNamespace(run_id="run-1", state_version=1)

    def _read_task_steps_version(task_id_, principal_):
        return SimpleNamespace(max_event_id=1)

    snapshot = SimpleNamespace(
        task_id=1,
        agent_id=1,
        status=status,
        run_id="run-1",
        state_version=1,
        **extra_snapshot,
    )
    frames = [
        chunk
        async for chunk in getattr(es, stream_fn)(
            snapshot,
            principal=None,
            read_task_steps_response=_read_task_steps_response,
            read_task_snapshot=_reread_task_snapshot,
            read_task_steps_version=_read_task_steps_version,
        )
    ]
    body = "".join(frames)
    assert body.count("event: task.status") == 1
    assert body.count("event: step.completed") == 1  # the good step survived
    assert "tool_call:call-1" in body
    assert "tool_call:call-2" not in body
    assert body.count(f"event: {conclusion_event}") == 1
    assert body.count("event: stream.error") == 1
    assert "resync_required" in body
    assert body.index(f"event: {conclusion_event}") < body.index("event: stream.error")
    # The wording names serialization, not a read and not a restart.
    assert "Serializing the task's steps failed" in body
    assert "moved to a new run" not in body


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

    All three producers of a ``step.*`` frame -- live projection
    (``_step_content_frame``), the attach-time fast paths' cached
    snapshot, and the normal path's history replay
    (``_build_warm_up_frames``) -- funnel through
    ``_step_wire_frame``, and that function
    is where the single ``model_dump(mode="json")`` runs: it takes the
    ``PublicStep`` model, not an already-dumped dict, so neither
    producer can reach a frame builder with un-normalized values.
    Pydantic v2's JSON serialization mode normalizes non-finite floats
    to ``null`` for any field typed ``Any`` (verified directly below),
    independent of a field's nesting depth -- so every ``step.*`` frame
    this module emits is already immune to this failure mode with no
    additional guard in ``_step_wire_frame`` itself. This test pins
    that implicit invariant by handing three-deep nested non-finite
    values to ``_step_content_frame`` -- the live path's entry into
    that shared normalizer -- and asserting the resulting wire text is
    strict-JSON-clean. The raw dict goes in exactly as a projector
    produces it; normalizing it in test setup first would have left the
    assertions green even if the normalization regressed to passing the
    raw dict straight through. Because that normalization sits in
    ``_step_wire_frame``, which every producer must go through to build
    a frame at all, this one assertion covers the fast paths and the
    replay too. It changes no production code -- if that regression
    ever happens, this
    test is the tripwire.
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
