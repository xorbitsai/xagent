from __future__ import annotations

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from starlette.datastructures import State

from tests.web.services.test_task_command_terminal_events import (
    _claim_command,
    _create_running_task,
    db_session as _imported_terminal_event_db_session,
)
from xagent.web.api import websocket as websocket_api
from xagent.web.models.task_command_terminal_event import TaskCommandTerminalEvent
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import task_command_terminal_events as terminal_events_module
from xagent.web.services.task_command_terminal_events import (
    TerminalTaskEvent,
    TerminalTaskEventAccessDenied,
    TerminalTaskEventDraft,
    TerminalTaskEventHub,
    TerminalTaskEventLoopRegistry,
    TerminalTaskEventMessageCode,
    TerminalTaskEventPrincipal,
    MAX_TERMINAL_EVENT_CURSOR,
    resolve_terminal_task_event_cursor,
    terminal_task_event_payload,
)
from xagent.web.services.task_command_transport import (
    TaskCommandKind,
    fail_task_command,
)
from xagent.web.services.websocket_writer import send_websocket_text

# Pytest registers fixtures under the importing module's attribute name.
terminal_event_db_session = _imported_terminal_event_db_session


def _fail_command(
    db,
    user,
    task,
    command_id: str,
    *,
    kind: TaskCommandKind = TaskCommandKind.PAUSE,
    terminal_event: TerminalTaskEventDraft | None = None,
):
    claimed = _claim_command(db, user, task, command_id, kind=kind)
    assert fail_task_command(
        claimed.id,
        "worker-a",
        "internal detail",
        force_terminal=True,
        expected_attempt_count=claimed.attempt_count,
        terminal_event=terminal_event,
    )
    return claimed


def _event_cursor(db, command_db_id: int) -> int:
    db.expire_all()
    event = (
        db.query(TaskCommandTerminalEvent)
        .filter(TaskCommandTerminalEvent.task_command_id == command_db_id)
        .one()
    )
    return int(event.id)


async def _replay_one(user_id: int, task_id: int, *, after: int = 0):
    received = []
    ready = asyncio.Event()

    async def receive(event) -> None:
        received.append(event)
        ready.set()

    hub = TerminalTaskEventHub(poll_interval_seconds=0.01)
    subscription = await hub.attach_terminal_events(
        principal=TerminalTaskEventPrincipal(user_id=user_id, is_admin=False),
        task_id=task_id,
        sink=receive,
        after_event_id=after,
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=1)
    finally:
        await subscription.close()
        await hub.close()
    return received


@pytest.mark.asyncio
async def test_fresh_worker_hub_replays_another_workers_committed_event(
    terminal_event_db_session,
) -> None:
    db_session = terminal_event_db_session
    user, task = _create_running_task(db_session)
    claimed = _fail_command(
        db_session,
        user,
        task,
        "resume-after-worker-restart",
        kind=TaskCommandKind.RESUME,
        terminal_event=TerminalTaskEventDraft(
            message_code=TerminalTaskEventMessageCode.TASK_COMMAND_FAILED,
            resend_safe=False,
        ),
    )

    received = await _replay_one(int(user.id), int(task.id))

    assert len(received) == 1
    event = received[0]
    assert event.task_run_id == "run-1"
    assert event.task_state_version == 3
    assert event.command_id == "resume-after-worker-restart"
    assert event.outcome_version == claimed.attempt_count


@pytest.mark.asyncio
async def test_failed_sink_retries_in_order_without_reclaiming_command(
    terminal_event_db_session,
) -> None:
    db_session = terminal_event_db_session
    user, task = _create_running_task(db_session)
    first = _fail_command(db_session, user, task, "pause-1")
    _fail_command(db_session, user, task, "pause-2")
    attempts = 0
    received_ids: list[str] = []
    received_second = asyncio.Event()

    async def fail_once(event) -> None:
        nonlocal attempts
        attempts += 1
        received_ids.append(event.command_id)
        if attempts == 1:
            raise ConnectionError("socket write failed")
        if event.command_id == "pause-2":
            received_second.set()

    hub = TerminalTaskEventHub(poll_interval_seconds=0.01)
    subscription = await hub.attach_terminal_events(
        principal=TerminalTaskEventPrincipal(
            user_id=int(user.id),
            is_admin=False,
        ),
        task_id=int(task.id),
        sink=fail_once,
        after_event_id=0,
    )
    try:
        await asyncio.wait_for(received_second.wait(), timeout=1)
    finally:
        await subscription.close()
        await hub.close()

    db_session.expire_all()
    assert attempts == 3
    assert received_ids == ["pause-1", "pause-1", "pause-2"]
    assert first.attempt_count == 1


@pytest.mark.asyncio
async def test_blocked_sink_does_not_delay_another_subscription(
    terminal_event_db_session,
) -> None:
    db_session = terminal_event_db_session
    user, task = _create_running_task(db_session)
    _fail_command(db_session, user, task, "isolated-delivery")
    blocked = asyncio.Event()
    release = asyncio.Event()
    received = asyncio.Event()

    async def block(_event) -> None:
        blocked.set()
        await release.wait()

    async def receive(_event) -> None:
        received.set()

    hub = TerminalTaskEventHub(poll_interval_seconds=0.01)
    first = await hub.attach_terminal_events(
        principal=TerminalTaskEventPrincipal(user_id=int(user.id), is_admin=False),
        task_id=int(task.id),
        sink=block,
        after_event_id=0,
    )
    second = await hub.attach_terminal_events(
        principal=TerminalTaskEventPrincipal(user_id=int(user.id), is_admin=False),
        task_id=int(task.id),
        sink=receive,
        after_event_id=0,
    )
    try:
        await asyncio.wait_for(blocked.wait(), timeout=1)
        await asyncio.wait_for(received.wait(), timeout=1)
    finally:
        release.set()
        await first.close()
        await second.close()
        await hub.close()


@pytest.mark.asyncio
async def test_blocked_low_cursor_cannot_pin_healthy_sink_before_page_boundary(
    terminal_event_db_session,
) -> None:
    db_session = terminal_event_db_session
    user, task = _create_running_task(db_session)
    task_id = int(task.id)
    owner_id = int(user.id)
    created_at = datetime.utcnow()
    events = [
        TerminalTaskEvent(
            cursor=cursor,
            event_id=f"event-{cursor}",
            task_id=task_id,
            task_run_id="run-1",
            task_state_version=3,
            command_id=f"command-{cursor}",
            command_kind="pause",
            actor_user_id=owner_id,
            task_owner_user_id=owner_id,
            outcome_version=1,
            outcome="failed",
            message_code=TerminalTaskEventMessageCode.TASK_COMMAND_FAILED,
            resend_safe=False,
            include_command_identity=True,
            created_at=created_at,
        )
        for cursor in range(1, 1002)
    ]
    requested_cursors: list[int] = []

    def load_page(after_cursor_by_task: dict[int, int]) -> list[TerminalTaskEvent]:
        if task_id not in after_cursor_by_task:
            return []
        after = after_cursor_by_task[task_id]
        requested_cursors.append(after)
        return [event for event in events if event.cursor > after][:1000]

    blocked = asyncio.Event()
    release = asyncio.Event()
    received_last = asyncio.Event()

    async def block(_event) -> None:
        blocked.set()
        await release.wait()

    async def receive(event: TerminalTaskEvent) -> None:
        if event.cursor == 1001:
            received_last.set()

    hub = TerminalTaskEventHub(poll_interval_seconds=0.01)
    with patch.object(terminal_events_module, "_load_events", side_effect=load_page):
        first = await hub.attach_terminal_events(
            principal=TerminalTaskEventPrincipal(user_id=owner_id, is_admin=False),
            task_id=task_id,
            sink=block,
            after_event_id=0,
        )
        second = await hub.attach_terminal_events(
            principal=TerminalTaskEventPrincipal(user_id=owner_id, is_admin=False),
            task_id=task_id,
            sink=receive,
            after_event_id=0,
        )
        try:
            await asyncio.wait_for(blocked.wait(), timeout=1)
            await asyncio.wait_for(received_last.wait(), timeout=1)
        finally:
            release.set()
            await first.close()
            await second.close()
            await hub.close()

    assert requested_cursors[0] == 0
    assert 1000 in requested_cursors


@pytest.mark.asyncio
async def test_subscription_close_finishes_detach_before_propagating_cancellation(
    terminal_event_db_session,
) -> None:
    db_session = terminal_event_db_session
    user, task = _create_running_task(db_session)
    hub = TerminalTaskEventHub(poll_interval_seconds=0.01)
    subscription = await hub.attach_terminal_events(
        principal=TerminalTaskEventPrincipal(user_id=int(user.id), is_admin=False),
        task_id=int(task.id),
        sink=lambda _event: asyncio.sleep(0),
        after_event_id=0,
    )
    await hub._lock.acquire()
    close_task = asyncio.create_task(subscription.close())
    await asyncio.sleep(0)
    close_task.cancel()
    hub._lock.release()

    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert hub._subscribers == {}
    await subscription.close()
    await hub.close()


@pytest.mark.asyncio
async def test_registry_releases_connection_after_cancelled_detach(
    terminal_event_db_session,
) -> None:
    db_session = terminal_event_db_session
    user, task = _create_running_task(db_session)
    connection = object()
    registry = TerminalTaskEventLoopRegistry()
    await registry.attach(
        connection=connection,
        principal=TerminalTaskEventPrincipal(user_id=int(user.id), is_admin=False),
        task_id=int(task.id),
        sink=lambda _event: asyncio.sleep(0),
        after_event_id=0,
    )
    hub = registry._hubs[asyncio.get_running_loop()]
    await hub._lock.acquire()
    detach_task = asyncio.create_task(registry.detach(connection))
    await asyncio.sleep(0)
    detach_task.cancel()
    hub._lock.release()

    with pytest.raises(asyncio.CancelledError):
        await detach_task

    assert not registry.has_subscription(connection)
    assert hub._subscribers == {}
    await hub.close()


@pytest.mark.asyncio
async def test_subscription_rejects_non_owner(terminal_event_db_session) -> None:
    db_session = terminal_event_db_session
    _user, task = _create_running_task(db_session)
    stranger = User(
        username="terminal-event-stranger",
        password_hash="hash",
        is_admin=False,
    )
    db_session.add(stranger)
    db_session.commit()
    hub = TerminalTaskEventHub(poll_interval_seconds=0.01)
    try:
        with pytest.raises(TerminalTaskEventAccessDenied):
            await hub.attach_terminal_events(
                principal=TerminalTaskEventPrincipal(
                    user_id=int(stranger.id),
                    is_admin=False,
                ),
                task_id=int(task.id),
                sink=lambda _event: asyncio.sleep(0),
                after_event_id=0,
            )
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_initial_cursor_can_defer_subscription_for_missing_task(
    terminal_event_db_session,
) -> None:
    db_session = terminal_event_db_session
    user, task = _create_running_task(db_session)

    cursor = await resolve_terminal_task_event_cursor(
        principal=TerminalTaskEventPrincipal(
            user_id=int(user.id),
            is_admin=False,
        ),
        task_id=int(task.id) + 100_000,
        after_event_id=None,
        allow_missing_task=True,
    )

    assert cursor is None


@pytest.mark.asyncio
async def test_oversized_cursor_cannot_poison_another_tasks_poll(
    terminal_event_db_session,
) -> None:
    db_session = terminal_event_db_session
    user, healthy_task = _create_running_task(db_session)
    healthy_command = _fail_command(
        db_session,
        user,
        healthy_task,
        "healthy-after-hostile-cursor",
    )
    hostile_task = Task(
        user_id=int(user.id),
        title="Hostile cursor",
        description="Hostile cursor",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
    )
    db_session.add(hostile_task)
    db_session.commit()

    hostile_cursor = await resolve_terminal_task_event_cursor(
        principal=TerminalTaskEventPrincipal(user_id=int(user.id), is_admin=False),
        task_id=int(hostile_task.id),
        after_event_id=10**100,
    )
    assert hostile_cursor == MAX_TERMINAL_EVENT_CURSOR

    events = terminal_events_module._load_events(
        {
            int(hostile_task.id): hostile_cursor,
            int(healthy_task.id): 0,
        }
    )

    assert [event.cursor for event in events] == [
        _event_cursor(db_session, healthy_command.id)
    ]


@pytest.mark.asyncio
async def test_new_owner_cannot_receive_prior_owners_event(
    terminal_event_db_session,
) -> None:
    db_session = terminal_event_db_session
    original_owner, task = _create_running_task(db_session)
    _fail_command(db_session, original_owner, task, "old-owner-command")
    new_owner = User(
        username="terminal-event-new-owner",
        password_hash="hash",
        is_admin=False,
    )
    db_session.add(new_owner)
    db_session.flush()
    task.user_id = int(new_owner.id)
    db_session.commit()
    _fail_command(db_session, new_owner, task, "new-owner-command")

    received = await _replay_one(int(new_owner.id), int(task.id))

    assert [event.command_id for event in received] == ["new-owner-command"]


@pytest.mark.asyncio
async def test_database_poll_failure_retries_without_reattachment(
    terminal_event_db_session,
) -> None:
    db_session = terminal_event_db_session
    user, task = _create_running_task(db_session)
    _fail_command(db_session, user, task, "event-after-db-recovery")
    real_load_events = terminal_events_module._load_events
    attempts = 0

    def fail_once(after_cursor_by_task):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("database temporarily unavailable")
        return real_load_events(after_cursor_by_task)

    received = asyncio.Event()

    async def receive(_event) -> None:
        received.set()

    hub = TerminalTaskEventHub(poll_interval_seconds=0.01)
    with patch.object(terminal_events_module, "_load_events", side_effect=fail_once):
        subscription = await hub.attach_terminal_events(
            principal=TerminalTaskEventPrincipal(
                user_id=int(user.id),
                is_admin=False,
            ),
            task_id=int(task.id),
            sink=receive,
            after_event_id=0,
        )
        try:
            await asyncio.wait_for(received.wait(), timeout=1)
        finally:
            await subscription.close()
            await hub.close()
    assert attempts >= 2


def test_event_load_chunks_large_sqlite_subscription_sets(
    terminal_event_db_session,
) -> None:
    _ = terminal_event_db_session
    cursors = {task_id: 0 for task_id in range(10_000, 11_001)}

    assert terminal_events_module._load_events(cursors) == []


@pytest.mark.asyncio
async def test_reconnect_cursor_replays_only_newer_events(
    terminal_event_db_session,
) -> None:
    db_session = terminal_event_db_session
    user, task = _create_running_task(db_session)
    first = _fail_command(db_session, user, task, "pause-before-reconnect")
    second = _fail_command(db_session, user, task, "pause-after-reconnect")
    first_cursor = _event_cursor(db_session, first.id)
    second_cursor = _event_cursor(db_session, second.id)

    received = await _replay_one(
        int(user.id),
        int(task.id),
        after=first_cursor,
    )

    assert [event.cursor for event in received] == [second_cursor]


@pytest.mark.asyncio
async def test_fixed_initial_cursor_replays_event_committed_during_status_send(
    terminal_event_db_session,
) -> None:
    db_session = terminal_event_db_session
    user, task = _create_running_task(db_session)
    first = _fail_command(db_session, user, task, "before-initial-status")
    baseline = await resolve_terminal_task_event_cursor(
        principal=TerminalTaskEventPrincipal(
            user_id=int(user.id),
            is_admin=False,
        ),
        task_id=int(task.id),
        after_event_id=None,
    )
    assert baseline == _event_cursor(db_session, first.id)

    second = _fail_command(db_session, user, task, "during-initial-status")
    received = await _replay_one(int(user.id), int(task.id), after=baseline)

    assert [event.cursor for event in received] == [
        _event_cursor(db_session, second.id)
    ]


def test_legacy_unknown_version_cannot_project_as_current_interaction() -> None:
    event = TerminalTaskEvent(
        cursor=17,
        event_id="00000000-0000-0000-0000-000000000001",
        task_id=9,
        task_run_id="legacy-run",
        task_state_version=None,
        command_id="legacy-command",
        command_kind="cancel",
        actor_user_id=1,
        task_owner_user_id=1,
        outcome_version=1,
        outcome="failed",
        message_code=TerminalTaskEventMessageCode.TASK_COMMAND_FAILED,
        resend_safe=False,
        include_command_identity=True,
        created_at=datetime.utcnow(),
    )

    payload = terminal_task_event_payload(event)

    assert payload["type"] == "task_command_outcome"
    assert "run_id" not in payload
    assert "state_version" not in payload
    assert "message" not in payload


def test_external_projection_omits_command_identity() -> None:
    event = TerminalTaskEvent(
        cursor=18,
        event_id="00000000-0000-0000-0000-000000000002",
        task_id=9,
        task_run_id="run-9",
        task_state_version=3,
        command_id="external-command-secret",
        command_kind="cancel",
        actor_user_id=None,
        task_owner_user_id=1,
        outcome_version=1,
        outcome="failed",
        message_code=TerminalTaskEventMessageCode.EXTERNAL_CANCEL_NOT_APPLIED,
        resend_safe=False,
        include_command_identity=False,
        created_at=datetime.utcnow(),
    )

    payload = terminal_task_event_payload(event)

    assert payload["type"] == "agent_error"
    assert "command_id" not in payload
    assert "command_kind" not in payload
    assert "external-command-secret" not in repr(payload)


class _BlockedTerminalEventWebSocket:
    def __init__(self) -> None:
        self.state = State()
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()
        self.completed: list[str] = []

    async def send_text(self, data: str) -> None:
        self.send_started.set()
        await self.release_send.wait()
        self.completed.append(data)


class _RecordingTerminalEventWebSocket:
    def __init__(self) -> None:
        self.state = State()
        self.completed: list[str] = []

    async def send_text(self, data: str) -> None:
        self.completed.append(data)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state_version", "expected_type"),
    [(3, "agent_error"), (None, "task_command_outcome")],
)
async def test_terminal_send_preserves_immutable_event_correlation(
    state_version: int | None,
    expected_type: str,
) -> None:
    websocket = _RecordingTerminalEventWebSocket()
    manager = websocket_api.ConnectionManager()
    manager.register_connection(websocket, task_id=9)
    event = TerminalTaskEvent(
        cursor=20,
        event_id="00000000-0000-0000-0000-000000000004",
        task_id=9,
        task_run_id="run-at-command-acceptance",
        task_state_version=state_version,
        command_id="stale-run-command",
        command_kind="pause",
        actor_user_id=1,
        task_owner_user_id=1,
        outcome_version=1,
        outcome="failed",
        message_code=TerminalTaskEventMessageCode.TASK_COMMAND_FAILED,
        resend_safe=False,
        include_command_identity=True,
        created_at=datetime.utcnow(),
    )
    with (
        patch.object(websocket_api, "manager", manager),
        patch.object(
            websocket_api,
            "_with_current_task_control_state",
            AsyncMock(side_effect=AssertionError("terminal snapshot was relabeled")),
        ),
    ):
        await websocket_api._send_terminal_task_event(websocket, event)

    payload = json.loads(websocket.completed[0])
    assert payload["type"] == expected_type
    if state_version is None:
        assert "run_id" not in payload
        assert "state_version" not in payload
        assert "message" not in payload
    else:
        assert payload["run_id"] == "run-at-command-acceptance"
        assert payload["state_version"] == state_version


@pytest.mark.asyncio
async def test_queued_terminal_event_rechecks_task_membership_before_send() -> None:
    websocket = _BlockedTerminalEventWebSocket()
    manager = websocket_api.ConnectionManager()
    manager.register_connection(websocket, task_id=9)
    in_flight = asyncio.create_task(send_websocket_text(websocket, "in-flight"))
    await websocket.send_started.wait()
    event = TerminalTaskEvent(
        cursor=19,
        event_id="00000000-0000-0000-0000-000000000003",
        task_id=9,
        task_run_id="run-9",
        task_state_version=3,
        command_id="pause-before-move",
        command_kind="pause",
        actor_user_id=1,
        task_owner_user_id=1,
        outcome_version=1,
        outcome="failed",
        message_code=TerminalTaskEventMessageCode.TASK_COMMAND_FAILED,
        resend_safe=False,
        include_command_identity=True,
        created_at=datetime.utcnow(),
    )
    with (
        patch.object(websocket_api, "manager", manager),
        patch.object(
            websocket_api,
            "_with_current_task_control_state",
            AsyncMock(side_effect=AssertionError("terminal snapshot was relabeled")),
        ),
    ):
        terminal_send = asyncio.create_task(
            websocket_api._send_terminal_task_event(websocket, event)
        )
        await asyncio.sleep(0)
        manager.move_connection(websocket, new_task_id=10)
        websocket.release_send.set()
        await asyncio.gather(in_flight, terminal_send)

    assert websocket.completed == ["in-flight"]
