"""SDK-scope cancel: its execution core and the settlement text it relies on.

Every test here drives one invariant of the SDK cancel path: which
cancellations write the cancellation sentence, that the cancelled run's
settlement and this core leave the same row, that neither claims an outcome
it did not cause, the ownership fence, and the routing that sends only an
``sdk``-scoped command to this core.
"""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.shared.async_waits import DB_PROGRESS_TIMEOUT
from xagent.web.api import websocket as websocket_api
from xagent.web.models.agent import Agent
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.services import a2a_task_cancel as a2a_cancel_service
from xagent.web.services import sdk_task_cancel
from xagent.web.services import task_command_execution as command_execution_service
from xagent.web.services import task_coordinator_runtime, task_coordinator_service
from xagent.web.services import task_execution as task_execution_service
from xagent.web.services import task_orchestrator
from xagent.web.services.assistant_history_safety import TASK_FAILURE_MESSAGE_TYPE
from xagent.web.services.chat_history_service import (
    DELIVERY_DISPATCHED,
    DELIVERY_PENDING,
)
from xagent.web.services.client_error_messages import CLIENT_SAFE_TASK_FAILURE
from xagent.web.services.external_task_cancel import EXTERNAL_TURN_INTERRUPTED_MESSAGE
from xagent.web.services.sdk_task_cancel import (
    SDK_CANCEL_ERROR_MESSAGE,
    cancel_sdk_task_unserialized,
    is_sdk_cancelled,
    sdk_cancel_result_fact,
)
from xagent.web.services.task_command_transport import (
    ClaimedTaskCommand,
    TaskCommandKind,
    TaskCommandRejected,
)
from xagent.web.services.task_execution_controller import (
    StaleTaskRunError,
    TaskControlState,
)
from xagent.web.services.task_orchestrator import TaskTurnPayload

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")

_ROW_COLUMNS = (
    "status",
    "control_state",
    "run_id",
    "state_version",
    "output",
    "error_message",
    "runner_id",
)


def _create_agent() -> tuple[int, int]:
    response = client.post(
        "/api/agents",
        headers=_admin_headers(),
        json={
            "name": "SDK cancel agent",
            "description": "SDK cancel test agent",
            "instructions": "You are an SDK cancel test agent.",
            "execution_mode": "balanced",
        },
    )
    assert response.status_code == 200, response.text
    agent_id = int(response.json()["id"])
    db = _direct_db_session()
    try:
        owner_user_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
    finally:
        db.close()
    return agent_id, owner_user_id


def _create_task(
    *,
    source: str = "sdk",
    status: TaskStatus = TaskStatus.RUNNING,
    control_state: str = TaskControlState.RUNNING.value,
    run_id: str | None = "run-sdk",
    state_version: int = 4,
    error_message: str | None = None,
    output: str | None = None,
    storage_version: int = 1,
) -> tuple[int, int, int]:
    agent_id, owner_user_id = _create_agent()
    db = _direct_db_session()
    try:
        task = Task(
            user_id=owner_user_id,
            title="sdk cancel",
            status=status,
            control_state=control_state,
            run_id=run_id,
            state_version=state_version,
            agent_id=agent_id,
            source=source,
            is_visible=False,
            error_message=error_message,
            output=output,
            conversation_storage_version=storage_version,
        )
        db.add(task)
        db.commit()
        return int(task.id), agent_id, owner_user_id
    finally:
        db.close()


def _load_task(task_id: int) -> Task:
    db = _direct_db_session()
    try:
        return db.query(Task).filter(Task.id == task_id).one()
    finally:
        db.close()


def _row(task_id: int) -> dict[str, Any]:
    task = _load_task(task_id)
    return {column: getattr(task, column) for column in _ROW_COLUMNS}


def _assistant_rows(task_id: int) -> list[tuple[str, str]]:
    db = _direct_db_session()
    try:
        return [
            (str(row.content), str(row.message_type))
            for row in db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "assistant",
            )
            .order_by(TaskChatMessage.id)
        ]
    finally:
        db.close()


def _result_facts(task_id: int) -> list[dict[str, Any]]:
    db = _direct_db_session()
    try:
        return [
            dict(event.payload)
            for event in db.query(TaskExecutionEvent)
            .filter(
                TaskExecutionEvent.task_id == task_id,
                TaskExecutionEvent.kind == "execution_settled",
            )
            .order_by(TaskExecutionEvent.id)
        ]
    finally:
        db.close()


def _seed_pending_user_message(task_id: int, owner_user_id: int, turn_id: str) -> None:
    db = _direct_db_session()
    try:
        db.add(
            TaskChatMessage(
                task_id=task_id,
                user_id=owner_user_id,
                role="user",
                content="stop this one",
                message_type="user_message",
                turn_id=turn_id,
                delivery_status=DELIVERY_PENDING,
            )
        )
        db.commit()
    finally:
        db.close()


def _delivery_status(task_id: int, turn_id: str) -> str:
    db = _direct_db_session()
    try:
        row = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.turn_id == turn_id,
                TaskChatMessage.role == "user",
            )
            .one()
        )
        return str(row.delivery_status)
    finally:
        db.close()


def _broadcast_manager(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    manager = MagicMock()
    manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", manager)
    return manager


def _assert_cancelled(task_id: int, *, before: Task, storage_version: int) -> None:
    """The one row both writers must leave, whichever of them wrote it."""
    task = _load_task(task_id)
    assert is_sdk_cancelled(task.status, task.error_message)
    assert task.control_state == TaskControlState.FAILED.value
    assert task.error_message == SDK_CANCEL_ERROR_MESSAGE
    assert task.output is None
    assert task.run_id == before.run_id
    assert task.state_version == before.state_version + 1
    assert task.runner_id is None and task.lease_expires_at is None
    assert _assistant_rows(task_id) == [
        (CLIENT_SAFE_TASK_FAILURE, TASK_FAILURE_MESSAGE_TYPE)
    ]
    facts = _result_facts(task_id)
    if storage_version == 2:
        assert [fact["result"] for fact in facts] == [
            {"status": "cancelled", "error": SDK_CANCEL_ERROR_MESSAGE}
        ]
    else:
        assert facts == []


async def _start_blocked_run(
    monkeypatch: pytest.MonkeyPatch, *, task_id: int, owner_user_id: int, source: str
) -> asyncio.Task:
    """A real background run parked inside its execution, ready to be cancelled."""
    started = asyncio.Event()

    async def block_until_cancelled(**_kwargs: object) -> None:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        task_execution_service, "execute_task_background", block_until_cancelled
    )
    monkeypatch.setattr(
        task_orchestrator,
        "load_task_setup_snapshot_sync",
        lambda *_args, **_kwargs: SimpleNamespace(
            task=SimpleNamespace(id=task_id, conversation_storage_version=1)
        ),
    )
    monkeypatch.setattr(task_orchestrator, "_get_agent_manager", MagicMock())
    bg_task = task_orchestrator._schedule_bg(
        task_id=task_id,
        task_owner_user_id=owner_user_id,
        task_source=source,
        payload=TaskTurnPayload("stop me"),
        force_fresh=False,
        context=None,
    )
    await asyncio.wait_for(started.wait(), timeout=DB_PROGRESS_TIMEOUT)
    return bg_task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "explicit", "expected"),
    [
        ("sdk", True, SDK_CANCEL_ERROR_MESSAGE),
        ("sdk", False, "task execution cancelled"),
        ("widget", True, "task execution cancelled"),
        ("external", True, EXTERNAL_TURN_INTERRUPTED_MESSAGE),
    ],
    ids=["sdk_explicit", "sdk_shutdown", "other_source", "external"],
)
async def test_settlement_text_names_only_an_explicit_sdk_cancel(
    monkeypatch: pytest.MonkeyPatch, source: str, explicit: bool, expected: str
) -> None:
    """Only a run stopped through ``cancel_task`` settles as cancelled.

    A host shutting down cancels the same coroutine without anyone asking,
    and every other source keeps the wording it has always had.
    """
    task_id, _agent_id, owner_user_id = _create_task(
        source=source, run_id=None, state_version=0
    )
    _broadcast_manager(monkeypatch)
    bg_task = await _start_blocked_run(
        monkeypatch, task_id=task_id, owner_user_id=owner_user_id, source=source
    )
    manager = task_execution_service.background_task_manager
    try:
        if explicit:
            await manager.cancel_task(task_id, timeout_seconds=DB_PROGRESS_TIMEOUT)
        else:
            bg_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await bg_task
    finally:
        manager.running_tasks.pop(task_id, None)

    assert _load_task(task_id).error_message == expected


@pytest.mark.parametrize(
    ("source", "explicit", "expected"),
    [
        ("sdk", True, SDK_CANCEL_ERROR_MESSAGE),
        ("sdk", False, "resume execution cancelled"),
        ("widget", True, "resume execution cancelled"),
        ("external", False, EXTERNAL_TURN_INTERRUPTED_MESSAGE),
    ],
)
def test_resume_settlement_text_names_only_an_explicit_sdk_cancel(
    source: str, explicit: bool, expected: str
) -> None:
    assert (
        task_execution_service._resume_cancel_settlement_error(
            source, explicit_cancel=explicit
        )
        == expected
    )


@pytest.mark.parametrize(
    ("error_message", "expected"),
    [
        (
            SDK_CANCEL_ERROR_MESSAGE,
            {"status": "cancelled", "error": SDK_CANCEL_ERROR_MESSAGE},
        ),
        ("task execution cancelled", {"error": "task execution cancelled"}),
        (None, {"error": None}),
    ],
)
def test_result_fact_names_only_a_cancellation(error_message, expected) -> None:
    assert sdk_cancel_result_fact(error_message) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_version", [1, 2])
async def test_running_cancel_is_settled_by_the_cancelled_run(
    monkeypatch: pytest.MonkeyPatch, storage_version: int
) -> None:
    """The cancelled run's own settlement writes the row; the core adds nothing.

    A replay of the same command afterwards finds its target settled and
    leaves the row, the transcript and the facts as they are.
    """
    task_id, agent_id, owner_user_id = _create_task(
        run_id=None, state_version=0, storage_version=storage_version
    )
    manager = _broadcast_manager(monkeypatch)
    await _start_blocked_run(
        monkeypatch, task_id=task_id, owner_user_id=owner_user_id, source="sdk"
    )
    before = _load_task(task_id)
    assert before.status == TaskStatus.RUNNING

    try:
        for _attempt in range(2):
            await cancel_sdk_task_unserialized(
                task_id=task_id,
                agent_id=agent_id,
                expected_run_id=before.run_id,
                expected_state_version=before.state_version,
            )
            _assert_cancelled(task_id, before=before, storage_version=storage_version)
    finally:
        task_execution_service.background_task_manager.running_tasks.pop(task_id, None)
    # One terminal announcement per call; a version-2 task also replays the
    # turn's display events ahead of each.
    announcements = [
        call.args[0]
        for call in manager.broadcast_to_task.await_args_list
        if call.args[0].get("type") == "task_error"
    ]
    assert len(announcements) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_version", [1, 2])
async def test_running_cancel_is_finalized_by_the_core_without_a_local_run(
    monkeypatch: pytest.MonkeyPatch, storage_version: int
) -> None:
    """With no coroutine here to settle the run, the core writes the same row."""
    task_id, agent_id, owner_user_id = _create_task(
        output="stale output", storage_version=storage_version
    )
    _seed_pending_user_message(task_id, owner_user_id, "turn-running")
    _broadcast_manager(monkeypatch)
    synced: list[str] = []
    monkeypatch.setattr(
        "xagent.web.services.workforce_runtime.sync_workforce_run_status",
        lambda _db, _task, status: synced.append(f"workforce:{status.value}"),
    )
    monkeypatch.setattr(
        task_orchestrator,
        "sync_trigger_run_status",
        lambda _db, _task, status: synced.append(f"trigger:{status.value}"),
    )
    before = _load_task(task_id)

    await cancel_sdk_task_unserialized(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id=before.run_id,
        expected_state_version=before.state_version,
    )

    _assert_cancelled(task_id, before=before, storage_version=storage_version)
    assert _delivery_status(task_id, "turn-running") == DELIVERY_DISPATCHED
    assert synced == ["workforce:failed", "trigger:failed"]


@pytest.mark.asyncio
async def test_a_run_registered_after_the_first_cancel_is_stopped_after_the_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core wrote the row because no coroutine was here to settle it, so
    one that registered meanwhile is cancelled once the row is terminal."""
    task_id, agent_id, _owner = _create_task()
    _broadcast_manager(monkeypatch)
    statuses: list[TaskStatus] = []

    async def record_status(_task_id: int, **_kwargs: object) -> SimpleNamespace:
        statuses.append(_load_task(task_id).status)
        return SimpleNamespace(requested=False)

    monkeypatch.setattr(
        task_execution_service.background_task_manager, "cancel_task", record_status
    )

    await cancel_sdk_task_unserialized(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id="run-sdk",
        expected_state_version=4,
    )

    assert statuses == [TaskStatus.RUNNING, TaskStatus.FAILED]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "moved",
    [
        {"status": TaskStatus.FAILED, "error_message": "boom", "state_version": 5},
        {"status": TaskStatus.COMPLETED, "output": "done", "state_version": 5},
        {"status": TaskStatus.PAUSED, "state_version": 5},
        {"run_id": "run-next"},
        {"state_version": 6},
        {"status": TaskStatus.WAITING_FOR_USER},
    ],
    ids=["failed_itself", "completed", "paused", "new_run", "moved", "waiting"],
)
async def test_cancel_never_claims_an_outcome_it_did_not_cause(
    monkeypatch: pytest.MonkeyPatch, moved: dict[str, Any]
) -> None:
    """A target that ended or moved on its own is rejected and left as it is."""
    task_id, agent_id, _owner = _create_task()
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).update(moved)
        db.commit()
    finally:
        db.close()
    manager = _broadcast_manager(monkeypatch)
    before = _row(task_id)

    with pytest.raises(TaskCommandRejected) as raised:
        await cancel_sdk_task_unserialized(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id="run-sdk",
            expected_state_version=4,
        )

    assert raised.value.reason == "stale_run"
    assert _row(task_id) == before
    assert _assistant_rows(task_id) == []
    assert manager.broadcast_to_task.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["a2a", "external", "internal", "trigger"])
async def test_cancel_reaches_only_sdk_tasks(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    task_id, agent_id, _owner = _create_task(source=source)
    _broadcast_manager(monkeypatch)
    before = _row(task_id)

    with pytest.raises(TaskCommandRejected) as raised:
        await cancel_sdk_task_unserialized(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id="run-sdk",
            expected_state_version=4,
        )

    assert raised.value.reason == "task_not_found"
    assert _row(task_id) == before


@pytest.mark.asyncio
async def test_finalize_accepts_a_settlement_that_lands_mid_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale finalize is rechecked once, and a settled cancel then matches."""
    task_id, agent_id, _owner = _create_task()
    _broadcast_manager(monkeypatch)
    real_finalize = sdk_task_cancel._finalize_sdk_cancel_sync
    calls = 0

    def settle_then_report_stale(**kwargs: Any) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            db = _direct_db_session()
            try:
                db.query(Task).filter(Task.id == task_id).update(
                    {
                        "status": TaskStatus.FAILED,
                        "control_state": TaskControlState.FAILED.value,
                        "state_version": 5,
                        "error_message": SDK_CANCEL_ERROR_MESSAGE,
                    }
                )
                db.commit()
            finally:
                db.close()
            raise StaleTaskRunError("settled between the read and the update")
        return real_finalize(**kwargs)

    monkeypatch.setattr(
        sdk_task_cancel, "_finalize_sdk_cancel_sync", settle_then_report_stale
    )

    await cancel_sdk_task_unserialized(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id="run-sdk",
        expected_state_version=4,
    )

    assert calls == 2
    assert _load_task(task_id).state_version == 5
    assert _assistant_rows(task_id) == []


@pytest.mark.parametrize("fence", ["lock", "update"])
def test_replaced_attempt_rejects_cancel_and_preserves_task(
    monkeypatch: pytest.MonkeyPatch, fence: str
) -> None:
    """An old cancel command cannot settle a replacement lease attempt."""
    task_id, agent_id, _owner = _create_task()
    db = _direct_db_session()
    try:
        lease = task_coordinator_service.acquire_task_lease_no_commit(
            db, task_id, runner_id="worker"
        )
        db.commit()
        db.query(Task).filter(Task.id == task_id).update(
            {"lease_attempt_id": "replacement-attempt"}
        )
        db.commit()
    finally:
        db.close()
    monkeypatch.setattr(
        task_coordinator_runtime,
        "current_task_coordinator",
        lambda _task_id: SimpleNamespace(lease=lease, task_id=task_id),
    )
    if fence == "update":
        # Isolate the UPDATE predicate from the lock check ahead of it.
        monkeypatch.setattr(
            task_coordinator_service, "lock_task_lease_no_commit", lambda *_a: True
        )
    before = _row(task_id)

    with pytest.raises(StaleTaskRunError):
        sdk_task_cancel._finalize_sdk_cancel_sync(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id=before["run_id"],
            expected_state_version=before["state_version"],
            terminal_event_state={},
        )

    assert _row(task_id) == before
    assert _load_task(task_id).lease_attempt_id == "replacement-attempt"


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery_fails", [False, True])
async def test_cancel_announces_after_commit_and_survives_a_failed_broadcast(
    monkeypatch: pytest.MonkeyPatch, delivery_fails: bool
) -> None:
    task_id, agent_id, _owner = _create_task()
    manager = _broadcast_manager(monkeypatch)
    seen: list[TaskStatus] = []

    async def record_committed_status(*_args: object, **_kwargs: object) -> None:
        seen.append(_load_task(task_id).status)
        if delivery_fails:
            raise RuntimeError("socket gone")

    manager.broadcast_to_task.side_effect = record_committed_status

    await cancel_sdk_task_unserialized(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id="run-sdk",
        expected_state_version=4,
    )

    assert seen == [TaskStatus.FAILED]
    assert _load_task(task_id).error_message == SDK_CANCEL_ERROR_MESSAGE


@pytest.mark.asyncio
async def test_the_cache_is_invalidated_once_after_the_commit_and_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The invalidation is a blocking round trip with some cache backends, so
    it belongs to the finalize that runs in a worker thread."""
    task_id, agent_id, _owner = _create_task()
    _broadcast_manager(monkeypatch)
    calls: list[tuple[int, bool, TaskStatus]] = []

    def record(invalidated_task_id: int) -> None:
        try:
            asyncio.get_running_loop()
            on_event_loop = True
        except RuntimeError:
            on_event_loop = False
        calls.append((invalidated_task_id, on_event_loop, _load_task(task_id).status))

    monkeypatch.setattr(task_orchestrator, "invalidate_task_cache_best_effort", record)

    await cancel_sdk_task_unserialized(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id="run-sdk",
        expected_state_version=4,
    )

    assert calls == [(task_id, False, TaskStatus.FAILED)]


def _terminal_frame(manager: MagicMock) -> dict[str, Any]:
    frames = [call.args[0] for call in manager.broadcast_to_task.await_args_list]
    return next(frame for frame in frames if frame.get("type") == "task_error")


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_version", [1, 2])
@pytest.mark.parametrize("writer", ["core", "run_settlement"])
async def test_the_terminal_frame_carries_the_control_snapshot_of_a_v2_task(
    monkeypatch: pytest.MonkeyPatch, writer: str, storage_version: int
) -> None:
    """Without the snapshot, readers of a version-2 task's live frames never
    get the turn's last display events. Whichever writer lands the terminal
    row, the frame carries the state it left, and a version-1 task has none."""
    task_id, agent_id, owner_user_id = _create_task(
        run_id=None, state_version=0, storage_version=storage_version
    )
    _seed_pending_user_message(task_id, owner_user_id, "turn-running")
    manager = _broadcast_manager(monkeypatch)
    if writer == "run_settlement":
        await _start_blocked_run(
            monkeypatch, task_id=task_id, owner_user_id=owner_user_id, source="sdk"
        )
    before = _load_task(task_id)

    try:
        await cancel_sdk_task_unserialized(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id=before.run_id,
            expected_state_version=before.state_version,
        )
    finally:
        task_execution_service.background_task_manager.running_tasks.pop(task_id, None)

    frame = _terminal_frame(manager)
    if storage_version == 2:
        assert frame["state_version"] == before.state_version + 1
        assert frame["task"]["status"] == "failed"
        frames = [call.args[0] for call in manager.broadcast_to_task.await_args_list]
        assert frames.index(frame) > 0, "display events precede the terminal frame"
    else:
        assert "state_version" not in frame


async def _start_resume(
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_id: int,
    owner_user_id: int,
    run_id: str,
    phase: str,
    source: str = "sdk",
) -> asyncio.Task:
    """A real resume coordinator parked in ``phase``.

    It is registered the way the resume handoff registers it, in
    ``resume_tasks``, so ``cancel_task`` reaches it. ``execution`` parks it
    inside the agent's resumed execution; ``lease_acquisition`` parks it in
    the worker thread that claims the run's lease, which then claims it.
    """
    started = asyncio.Event()
    loop = asyncio.get_running_loop()
    real_acquire = task_execution_service._acquire_resume_task_lease

    def acquire_slowly(*args: Any, **kwargs: Any) -> Any:
        loop.call_soon_threadsafe(started.set)
        time.sleep(0.3)
        return real_acquire(*args, **kwargs)

    if phase == "lease_acquisition":
        monkeypatch.setattr(
            task_execution_service, "_acquire_resume_task_lease", acquire_slowly
        )

    async def block_until_cancelled(*_args: object, **_kwargs: object) -> None:
        started.set()
        await asyncio.Event().wait()

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(side_effect=block_until_cancelled)
    resume = asyncio.create_task(
        task_execution_service.execute_resume_background(
            task_id,
            agent_service,
            owner_user_id,
            expected_run_id=run_id,
            trusted_task_source=source,
        )
    )
    task_execution_service.background_task_manager.resume_tasks[task_id] = resume
    await asyncio.wait_for(started.wait(), timeout=DB_PROGRESS_TIMEOUT)
    return resume


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "explicit", "expected"),
    [
        ("execution", True, SDK_CANCEL_ERROR_MESSAGE),
        ("execution", False, "resume execution cancelled"),
        ("lease_acquisition", True, SDK_CANCEL_ERROR_MESSAGE),
        ("lease_acquisition", False, "resume cancelled during lease acquisition"),
    ],
    ids=[
        "execution-explicit_cancel",
        "execution-direct_cancel",
        "lease_acquisition-explicit_cancel",
        "lease_acquisition-direct_cancel",
    ],
)
async def test_a_cancelled_resume_settles_as_cancelled_only_when_asked(
    monkeypatch: pytest.MonkeyPatch, phase: str, explicit: bool, expected: str
) -> None:
    """A resumed SDK run takes the same two ways out as a first run: an
    explicit ``cancel_task`` settles it as cancelled, any other cancellation
    keeps the wording it has always had."""
    task_id, _agent_id, owner_user_id = _create_task(
        status=TaskStatus.WAITING_FOR_USER,
        control_state=TaskControlState.WAITING_FOR_USER.value,
    )
    _broadcast_manager(monkeypatch)
    resume = await _start_resume(
        monkeypatch,
        task_id=task_id,
        owner_user_id=owner_user_id,
        run_id="run-sdk",
        phase=phase,
    )
    manager = task_execution_service.background_task_manager
    try:
        if explicit:
            await manager.cancel_task(task_id, timeout_seconds=DB_PROGRESS_TIMEOUT)
        else:
            resume.cancel()
            with pytest.raises(asyncio.CancelledError):
                await resume
    finally:
        manager.running_tasks.pop(task_id, None)
        manager.resume_tasks.pop(task_id, None)

    row = _load_task(task_id)
    assert row.status == TaskStatus.FAILED
    assert row.error_message == expected


def _cancel_command(task_id: int, agent_id: int, **payload: Any) -> ClaimedTaskCommand:
    return ClaimedTaskCommand(
        id=5,
        task_id=task_id,
        actor_user_id=None,
        command_id=f"cancel:{task_id}:4",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": agent_id, "target_state_version": 4, **payload},
        target_run_id="run-sdk",
        attempt_count=1,
    )


@pytest.mark.asyncio
async def test_sdk_scope_routes_to_the_sdk_core_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id, agent_id, _owner = _create_task()
    _broadcast_manager(monkeypatch)
    sdk_core = AsyncMock()
    external_core = AsyncMock()
    a2a_core = AsyncMock()
    monkeypatch.setattr(
        command_execution_service, "cancel_sdk_task_unserialized", sdk_core
    )
    monkeypatch.setattr(
        command_execution_service, "cancel_external_task_unserialized", external_core
    )
    monkeypatch.setattr(a2a_cancel_service, "cancel_a2a_task", a2a_core)

    await command_execution_service._execute_durable_task_command(
        _cancel_command(task_id, agent_id, scope="sdk")
    )

    sdk_core.assert_awaited_once_with(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id="run-sdk",
        expected_state_version=4,
    )
    assert external_core.await_count == 0
    assert a2a_core.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [None, "external"], ids=["a2a", "external"])
async def test_other_scopes_never_reach_the_sdk_core(
    monkeypatch: pytest.MonkeyPatch, scope: str | None
) -> None:
    task_id, agent_id, _owner = _create_task()
    _broadcast_manager(monkeypatch)
    sdk_core = AsyncMock()
    monkeypatch.setattr(
        command_execution_service, "cancel_sdk_task_unserialized", sdk_core
    )
    monkeypatch.setattr(
        command_execution_service, "cancel_external_task_unserialized", AsyncMock()
    )
    monkeypatch.setattr(a2a_cancel_service, "cancel_a2a_task", AsyncMock())
    payload = {} if scope is None else {"scope": scope}

    await command_execution_service._execute_durable_task_command(
        _cancel_command(task_id, agent_id, **payload)
    )

    assert sdk_core.await_count == 0


def test_lease_fields_are_cleared_only_without_a_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared execution leaves ownership to the coordinator that is draining."""
    task_id, agent_id, _owner = _create_task()
    db = _direct_db_session()
    try:
        lease = task_coordinator_service.acquire_task_lease_no_commit(
            db, task_id, runner_id="worker"
        )
        db.commit()
    finally:
        db.close()
    monkeypatch.setattr(
        task_coordinator_runtime,
        "current_task_coordinator",
        lambda _task_id: SimpleNamespace(lease=lease, task_id=task_id),
    )
    before = _load_task(task_id)

    assert sdk_task_cancel._finalize_sdk_cancel_sync(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id=before.run_id,
        expected_state_version=before.state_version,
        terminal_event_state={},
    )

    after = _load_task(task_id)
    assert after.error_message == SDK_CANCEL_ERROR_MESSAGE
    assert after.runner_id == "worker"
    assert after.lease_attempt_id == lease.attempt_id
    assert after.lease_expires_at is not None
    assert after.lease_expires_at.replace(tzinfo=UTC) > datetime.now(UTC) - timedelta(
        minutes=5
    )
