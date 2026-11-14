"""SDK-scope cancel: its execution core and the settlement text it relies on.

Every test here drives one invariant of the SDK cancel path: which
cancellations write the cancellation sentence, that the cancelled run's
settlement and this core leave the same row, that neither claims an outcome
it did not cause, the ownership fence, and the routing that sends only an
``sdk``-scoped command to this core.
"""

import asyncio
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.shared.async_waits import DB_PROGRESS_TIMEOUT
from xagent.web.api import websocket as websocket_api
from xagent.web.models.agent import Agent
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_admission import TaskAdmissionBucket, TaskAdmissionTicket
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
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
    SUPERSEDED_MESSAGE_TYPE,
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
    TaskCommandDeferred,
    TaskCommandKind,
    TaskCommandRejected,
    stage_task_command,
)
from xagent.web.services.task_execution_controller import (
    StaleTaskRunError,
    TaskControlState,
)
from xagent.web.services.task_orchestrator import TaskTurnPayload
from xagent.web.services.task_start_protocol import TaskStartPayload

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
        {"status": TaskStatus.PENDING},
    ],
    ids=["failed_itself", "completed", "paused", "new_run", "moved", "pending"],
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
            expected_status=TaskStatus.RUNNING,
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
        queued_run_id=None,
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
        expected_status=TaskStatus.RUNNING,
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


# ----- a turn that was queued when its cancel was requested -----

QUEUED_RUN = "run-queued"
STOPPED_BEFORE_START = {"rejection_reason": "cancelled_before_admission"}


def _stage_start(
    task_id: int,
    owner_user_id: int,
    *,
    run_id: str = QUEUED_RUN,
    turn_id: str = "turn-queued",
    **columns: Any,
) -> int:
    """An accepted START for ``run_id`` with its pending user message.

    Its payload is the one the acceptance writes: the run and version the row
    holds now, which is what the START consumer compares the row with.
    """
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        start = TaskStartPayload(
            version=1,
            run_id=run_id,
            state_version=int(task.state_version or 0),
            expected_run_id=task.run_id,
            turn_id=turn_id,
            kind="create" if task.status == TaskStatus.PENDING else "append",
            message="hello",
        )
        staged = stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=owner_user_id,
            command_id=turn_id,
            kind=TaskCommandKind.START,
            payload=start.model_dump(mode="json"),
            target_run_id=run_id,
        )
        db.commit()
        start_id = int(staged.staged_db_id)
        if columns:
            db.query(TaskExecutionCommand).filter(
                TaskExecutionCommand.id == start_id
            ).update(columns)
            db.commit()
    finally:
        db.close()
    _seed_pending_user_message(task_id, owner_user_id, turn_id)
    return start_id


def _start(start_id: int) -> tuple[str, Any]:
    db = _direct_db_session()
    try:
        row = db.get(TaskExecutionCommand, start_id)
        return str(row.status), row.result
    finally:
        db.close()


async def _cancel_queued(task_id: int, agent_id: int) -> None:
    # The command was recorded against the row as it stood while the turn
    # was queued; the core reads the target to cancel from the queued run.
    await cancel_sdk_task_unserialized(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id="run-before",
        expected_state_version=4,
        queued_run_id=QUEUED_RUN,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "previous",
    [
        {"status": TaskStatus.COMPLETED, "control_state": "completed", "output": "x"},
        {"status": TaskStatus.FAILED, "control_state": "failed", "error_message": "e"},
        {
            "status": TaskStatus.FAILED,
            "control_state": "failed",
            "error_message": SDK_CANCEL_ERROR_MESSAGE,
        },
        {"status": TaskStatus.PAUSED, "control_state": "paused"},
    ],
    ids=["completed", "failed", "cancelled", "paused"],
)
async def test_an_appended_turn_stopped_before_it_starts_leaves_the_previous_result(
    monkeypatch: pytest.MonkeyPatch, previous: dict[str, Any]
) -> None:
    """An append's failure belongs to its command, never to the row."""
    task_id, agent_id, owner_user_id = _create_task(run_id="run-before", **previous)
    start_id = _stage_start(task_id, owner_user_id)
    manager = _broadcast_manager(monkeypatch)
    before = _row(task_id)

    for _attempt in range(2):  # the second call is the command replayed
        await _cancel_queued(task_id, agent_id)

        assert _row(task_id) == before
        assert _start(start_id) == ("failed", STOPPED_BEFORE_START)
        assert _delivery_status(task_id, "turn-queued") == "failed"
    assert _assistant_rows(task_id) == []
    assert manager.broadcast_to_task.await_count == 0


@pytest.mark.asyncio
async def test_a_new_task_cancel_applied_again_before_it_completed_is_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cancel wrote the new task's row, then its completion never
    committed and the command is applied again while the START is still
    pending. The row now holds a later version than the START, which must
    not read as a turn that ended on its own: the cancel is already done."""
    task_id, agent_id, owner_user_id = _create_task(
        status=TaskStatus.PENDING, control_state="idle", run_id=None, state_version=0
    )
    start_id = _stage_start(task_id, owner_user_id)
    _broadcast_manager(monkeypatch)

    await _cancel_queued(task_id, agent_id)
    after_first = _row(task_id)
    await _cancel_queued(task_id, agent_id)

    assert _row(task_id) == after_first
    task = _load_task(task_id)
    assert is_sdk_cancelled(task.status, task.error_message)
    assert _start(start_id)[0] == "pending"


@pytest.mark.asyncio
async def test_a_turn_appended_to_a_cancelled_new_task_is_withdrawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new task was cancelled before it ran, so its row holds no run and
    reads as cancelled; a turn appended to it is queued. Cancelling that
    turn withdraws its START: the row is not taken for the turn's own
    cancel already being done."""
    task_id, agent_id, owner_user_id = _create_task(
        status=TaskStatus.FAILED,
        control_state=TaskControlState.FAILED.value,
        run_id=None,
        state_version=1,
        error_message=SDK_CANCEL_ERROR_MESSAGE,
    )
    start_id = _stage_start(task_id, owner_user_id)
    _broadcast_manager(monkeypatch)
    before = _row(task_id)

    await _cancel_queued(task_id, agent_id)

    assert _row(task_id) == before
    assert _start(start_id) == ("failed", STOPPED_BEFORE_START)


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_version", [1, 2])
async def test_a_new_task_cancelled_before_it_starts_is_cancelled_as_a_row(
    monkeypatch: pytest.MonkeyPatch, storage_version: int
) -> None:
    """A task that never ran gets the terminal row alone: no transcript line
    and no delivery change, which its START's own settlement owns. It has no
    result fact either, so its terminal frame carries no display events,
    whatever its storage version: asking for them would fail the broadcast."""
    task_id, agent_id, owner_user_id = _create_task(
        status=TaskStatus.PENDING,
        control_state="idle",
        run_id=None,
        state_version=0,
        storage_version=storage_version,
    )
    _stage_start(task_id, owner_user_id)
    manager = _broadcast_manager(monkeypatch)

    await _cancel_queued(task_id, agent_id)

    task = _load_task(task_id)
    assert is_sdk_cancelled(task.status, task.error_message)
    assert (task.run_id, task.state_version) == (None, 1)
    assert _assistant_rows(task_id) == []
    assert _delivery_status(task_id, "turn-queued") == DELIVERY_PENDING
    assert "state_version" not in _terminal_frame(manager)


@pytest.mark.asyncio
async def test_a_queued_turn_that_started_first_is_cancelled_as_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The handoff moved the run and the version since the request; the
    queued run names the target and the version is the one read now."""
    task_id, agent_id, owner_user_id = _create_task(run_id=QUEUED_RUN, state_version=7)
    _stage_start(task_id, owner_user_id, status="completed", attempt_count=1)
    _broadcast_manager(monkeypatch)

    await _cancel_queued(task_id, agent_id)

    task = _load_task(task_id)
    assert is_sdk_cancelled(task.status, task.error_message)
    assert (task.run_id, task.state_version) == (QUEUED_RUN, 8)


@pytest.mark.asyncio
async def test_a_queued_turn_already_cancelled_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id, agent_id, owner_user_id = _create_task(
        run_id=QUEUED_RUN,
        status=TaskStatus.FAILED,
        control_state="failed",
        error_message=SDK_CANCEL_ERROR_MESSAGE,
        state_version=9,
    )
    _stage_start(task_id, owner_user_id, status="completed", attempt_count=1)
    _broadcast_manager(monkeypatch)
    before = _row(task_id)

    await _cancel_queued(task_id, agent_id)

    assert _row(task_id) == before


@pytest.mark.asyncio
async def test_a_later_queued_turn_is_never_stopped_for_an_earlier_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The aimed turn ran to completion and another one was appended since:
    the cancel is stale, and the newer START is not its target."""
    task_id, agent_id, owner_user_id = _create_task(
        run_id=QUEUED_RUN,
        status=TaskStatus.COMPLETED,
        control_state="completed",
        output="done",
    )
    _stage_start(task_id, owner_user_id, status="completed", attempt_count=1)
    later_id = _stage_start(
        task_id, owner_user_id, run_id="run-later", turn_id="turn-later"
    )
    _broadcast_manager(monkeypatch)
    before = _row(task_id)

    with pytest.raises(TaskCommandRejected) as raised:
        await _cancel_queued(task_id, agent_id)

    assert raised.value.reason == "stale_run"
    assert _row(task_id) == before
    assert _start(later_id) == ("pending", None)
    assert _delivery_status(task_id, "turn-later") == DELIVERY_PENDING


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start", "error", "reason"),
    [
        ({"status": "processing", "attempt_count": 1}, TaskCommandDeferred, None),
        ({"status": "pending", "attempt_count": 1}, TaskCommandRejected, "stale_run"),
        (
            {"status": "failed", "result": {"rejection_reason": "invalid_start"}},
            TaskCommandRejected,
            "stale_run",
        ),
        (None, TaskCommandRejected, "stale_run"),
    ],
    ids=["being_handed_off", "claimed_before", "failed_itself", "missing"],
)
async def test_a_queued_start_this_cancel_cannot_stop_is_left_as_it_is(
    monkeypatch: pytest.MonkeyPatch, start, error, reason
) -> None:
    task_id, agent_id, owner_user_id = _create_task(
        run_id="run-before", status=TaskStatus.COMPLETED, control_state="completed"
    )
    start_id = None if start is None else _stage_start(task_id, owner_user_id, **start)
    _broadcast_manager(monkeypatch)
    before = _row(task_id)
    start_before = None if start_id is None else _start(start_id)

    with pytest.raises(error) as raised:
        await _cancel_queued(task_id, agent_id)

    if reason is not None:
        assert raised.value.reason == reason
    assert _row(task_id) == before
    if start_id is not None:
        assert _start(start_id) == start_before


@pytest.mark.asyncio
async def test_a_start_is_matched_by_its_run_not_by_being_the_one_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The aimed turn failed to start and another turn is queued now: the
    queued START belongs to that other turn and stays as it is."""
    task_id, agent_id, owner_user_id = _create_task(
        run_id="run-before", status=TaskStatus.COMPLETED, control_state="completed"
    )
    _stage_start(
        task_id,
        owner_user_id,
        status="failed",
        result={"rejection_reason": "invalid_start"},
    )
    later_id = _stage_start(
        task_id, owner_user_id, run_id="run-later", turn_id="turn-later"
    )
    _broadcast_manager(monkeypatch)

    with pytest.raises(TaskCommandRejected) as raised:
        await _cancel_queued(task_id, agent_id)

    assert raised.value.reason == "stale_run"
    assert _start(later_id) == ("pending", None)


@pytest.mark.asyncio
async def test_an_sdk_cancel_of_a_running_task_keeps_the_dispatcher_run_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a cancel aimed at a queued turn bypasses the dispatcher's run
    comparison; one aimed at the running row is rejected there, as before."""
    task_id, agent_id, _owner = _create_task(run_id="run-after")
    _broadcast_manager(monkeypatch)
    sdk_core = AsyncMock()
    monkeypatch.setattr(
        command_execution_service, "cancel_sdk_task_unserialized", sdk_core
    )

    with pytest.raises(TaskCommandRejected) as raised:
        await command_execution_service._execute_durable_task_command(
            _cancel_command(task_id, agent_id, scope="sdk")
        )

    assert raised.value.reason == "stale_run"
    sdk_core.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_queued_cancel_read_without_its_queued_run_is_rejected_unwritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process that does not read ``queued_run_id`` sees a cancel aimed at
    the queued turn's run while the row still holds an earlier, cancelled
    run. The run comparison rejects it before any core, so that earlier
    run is never reported as this cancel's outcome and the queued START is
    left alone."""
    task_id, agent_id, owner_user_id = _create_task(
        run_id="run-before",
        status=TaskStatus.FAILED,
        control_state="failed",
        error_message=SDK_CANCEL_ERROR_MESSAGE,
    )
    start_id = _stage_start(task_id, owner_user_id)
    _broadcast_manager(monkeypatch)
    sdk_core = AsyncMock()
    monkeypatch.setattr(
        command_execution_service, "cancel_sdk_task_unserialized", sdk_core
    )
    before = _row(task_id)
    command = replace(
        _cancel_command(task_id, agent_id, scope="sdk"), target_run_id=QUEUED_RUN
    )

    with pytest.raises(TaskCommandRejected) as raised:
        await command_execution_service._execute_durable_task_command(command)

    assert raised.value.reason == "stale_run"
    sdk_core.assert_not_awaited()
    assert _row(task_id) == before
    assert _start(start_id) == ("pending", None)


@pytest.mark.asyncio
async def test_an_sdk_cancel_aimed_at_a_queued_turn_reaches_its_core_past_the_row_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row holds the run before the queued one while the command targets
    the queued turn's run, as ``_enqueue_cancel`` records it, so the
    dispatcher's own run comparison must not reject it."""
    task_id, agent_id, _owner = _create_task(run_id="run-before")
    _broadcast_manager(monkeypatch)
    sdk_core = AsyncMock()
    monkeypatch.setattr(
        command_execution_service, "cancel_sdk_task_unserialized", sdk_core
    )
    command = replace(
        _cancel_command(task_id, agent_id, scope="sdk", queued_run_id=QUEUED_RUN),
        target_run_id=QUEUED_RUN,
    )

    await command_execution_service._execute_durable_task_command(command)

    sdk_core.assert_awaited_once_with(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id=QUEUED_RUN,
        expected_state_version=4,
        queued_run_id=QUEUED_RUN,
    )


def _update_row(task_id: int, **values: Any) -> None:
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).update(values)
        db.commit()
    finally:
        db.close()


def _wait_for_admission(start_id: int, task_id: int) -> None:
    """Give the START an unclaimed admission ticket: it waits for capacity."""
    db = _direct_db_session()
    try:
        if db.get(TaskAdmissionBucket, "test-bucket") is None:
            db.add(TaskAdmissionBucket(key="test-bucket", capacity=1, max_pending=10))
            db.flush()
        db.add(
            TaskAdmissionTicket(
                command_id=start_id, task_id=task_id, bucket_key="test-bucket"
            )
        )
        db.commit()
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("attempts", "ticket", "stopped"),
    [
        # Sent back to the admission queue after an attempt: still waiting.
        (1, True, True),
        # Held below the capacity cap before any attempt.
        (0, False, True),
        # Claimed before and holding no ticket: it may already have had an
        # effect, and nothing says it still waits.
        (1, False, False),
    ],
    ids=["returned_to_the_admission_queue", "never_attempted", "claimed_before"],
)
async def test_a_queued_start_is_stopped_only_while_a_cancel_can_overtake_it(
    monkeypatch: pytest.MonkeyPatch, attempts: int, ticket: bool, stopped: bool
) -> None:
    task_id, agent_id, owner_user_id = _create_task(
        run_id="run-before", status=TaskStatus.COMPLETED, control_state="completed"
    )
    start_id = _stage_start(task_id, owner_user_id, attempt_count=attempts)
    if ticket:
        _wait_for_admission(start_id, task_id)
    _broadcast_manager(monkeypatch)
    before = _row(task_id)

    if stopped:
        await _cancel_queued(task_id, agent_id)
        assert _start(start_id) == ("failed", STOPPED_BEFORE_START)
    else:
        with pytest.raises(TaskCommandRejected) as raised:
            await _cancel_queued(task_id, agent_id)
        assert raised.value.reason == "stale_run"
        assert _start(start_id) == ("pending", None)
    assert _row(task_id) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "moved_row",
    [
        {
            "status": TaskStatus.RUNNING,
            "control_state": "running",
            "run_id": "run-first",
            "state_version": 5,
        },
        {
            "status": TaskStatus.WAITING_FOR_USER,
            "control_state": "waiting_for_user",
            "run_id": "run-first",
            "state_version": 5,
        },
        # The same run, one version on: the START was accepted at the older one.
        {
            "status": TaskStatus.COMPLETED,
            "control_state": "completed",
            "state_version": 5,
        },
    ],
    ids=["running_on_another_run", "waiting_on_another_run", "version_moved"],
)
async def test_a_queued_start_that_can_no_longer_hand_off_is_not_withdrawn(
    monkeypatch: pytest.MonkeyPatch, moved_row: dict[str, Any]
) -> None:
    """The row moved after the START was accepted, so the consumer would
    reject the START and the turn never starts. Withdrawing it would answer
    that the turn was cancelled while the run the row holds goes on: the
    command is rejected, with the START and the row as they were."""
    task_id, agent_id, owner_user_id = _create_task(
        run_id="run-before",
        state_version=4,
        status=TaskStatus.COMPLETED,
        control_state="completed",
    )
    start_id = _stage_start(task_id, owner_user_id)
    _update_row(task_id, **moved_row)
    _broadcast_manager(monkeypatch)
    before = _row(task_id)

    with pytest.raises(TaskCommandRejected) as raised:
        await _cancel_queued(task_id, agent_id)

    assert raised.value.reason == "stale_run"
    assert _row(task_id) == before
    assert _start(start_id) == ("pending", None)
    assert _delivery_status(task_id, "turn-queued") == DELIVERY_PENDING


@pytest.mark.asyncio
async def test_a_new_task_whose_row_moved_is_not_cancelled_for_its_queued_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id, agent_id, owner_user_id = _create_task(
        status=TaskStatus.PENDING,
        control_state="idle",
        run_id=None,
        state_version=0,
    )
    start_id = _stage_start(task_id, owner_user_id)
    _update_row(task_id, state_version=1)
    _broadcast_manager(monkeypatch)
    before = _row(task_id)

    with pytest.raises(TaskCommandRejected) as raised:
        await _cancel_queued(task_id, agent_id)

    assert raised.value.reason == "stale_run"
    assert _row(task_id) == before
    assert _start(start_id) == ("pending", None)


def test_a_start_handed_off_while_it_was_being_stopped_is_cancelled_as_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The START is handed off between the read and the write that stops it:
    the second pass finds the turn running on the queued run and targets it
    at the version read then."""
    task_id, agent_id, owner_user_id = _create_task(
        run_id="run-before", status=TaskStatus.COMPLETED, control_state="completed"
    )
    start_id = _stage_start(task_id, owner_user_id)
    stops: list[int] = []

    def handed_off_first(_db: Any, _task_id: int, start: Any) -> bool:
        stops.append(int(start.id))
        other = _direct_db_session()
        try:
            other.query(TaskExecutionCommand).filter(
                TaskExecutionCommand.id == start.id
            ).update({"status": "completed", "attempt_count": 1})
            other.query(Task).filter(Task.id == task_id).update(
                {
                    "status": TaskStatus.RUNNING,
                    "control_state": "running",
                    "run_id": QUEUED_RUN,
                    "state_version": 7,
                }
            )
            other.commit()
        finally:
            other.close()
        return False

    monkeypatch.setattr(
        sdk_task_cancel, "_stop_queued_start_no_commit", handed_off_first
    )

    target = sdk_task_cancel._resolve_queued_cancel_target_sync(
        task_id=task_id, agent_id=agent_id, queued_run_id=QUEUED_RUN
    )

    assert stops == [start_id]
    assert target == sdk_task_cancel._CancelTarget(QUEUED_RUN, 7, TaskStatus.RUNNING)


@pytest.mark.parametrize(
    ("status", "queued"),
    [("pending", True), ("processing", True), ("completed", False), ("failed", False)],
)
def test_the_queued_start_lookup_and_the_task_read_share_one_definition(
    status: str, queued: bool
) -> None:
    from xagent.web.services.task_start_protocol import has_queued_start

    task_id, _agent_id, owner_user_id = _create_task(
        run_id="run-before", status=TaskStatus.COMPLETED, control_state="completed"
    )
    _stage_start(task_id, owner_user_id, status=status)
    db = _direct_db_session()
    try:
        assert has_queued_start(db, task_id) is queued
        assert (sdk_task_cancel._queued_start(db, task_id) is not None) is queued
    finally:
        db.close()


def test_the_stopped_before_start_wording_has_one_definition() -> None:
    """The cancel stops a START with the wording every other control uses, so
    it imports it from where the others write it."""
    import ast
    import pathlib

    from xagent.web.services import task_execution_admission

    assert (
        task_execution_admission.STOPPED_BEFORE_START_REASON
        == "cancelled_before_admission"
    )
    tree = ast.parse(pathlib.Path(sdk_task_cancel.__file__).read_text())
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert not assigned & {"STOPPED_BEFORE_START", "STOPPED_BEFORE_START_REASON"}
    assert (
        sdk_task_cancel.STOPPED_BEFORE_START_REASON
        == task_execution_admission.STOPPED_BEFORE_START_REASON
    )


# ----- a run that is waiting for an answer -----


def _add_active_interaction(task_id: int, run_id: str, key: str) -> int:
    """A native interaction holding the task's one answer slot for ``run_id``."""
    db = _direct_db_session()
    try:
        anchor = TraceEvent(
            task_id=task_id,
            event_id=f"anchor-{key}",
            event_type="agent_execution_checkpoint",
            timestamp=datetime.now(UTC),
            data={},
        )
        db.add(anchor)
        db.flush()
        row = TaskInteractionRequest(
            task_id=task_id,
            run_id=run_id,
            kind="clarification",
            protocol_version=1,
            status="active",
            active_slot=1,
            origin="sdk",
            request_payload={
                "message": "Continue?",
                "interactions": [
                    {"type": "text_input", "field": "response", "label": "Answer"}
                ],
            },
            request_idempotency_key=key,
            resume_trace_event_id=int(anchor.id),
            resume_event_id=f"resume-event-{key}",
            resume_execution_id=f"resume-execution-{key}",
            resume_locator_format="trace_event_pk_v1",
            resume_checkpoint_type="agent_execution_checkpoint",
            resume_run_partition=run_id,
            expires_at=datetime.now(UTC) + timedelta(minutes=15),
        )
        db.add(row)
        db.query(Task).filter(Task.id == task_id).update(
            {"interaction_protocol_version": 1}
        )
        db.commit()
        return int(row.id)
    finally:
        db.close()


def _interaction(row_id: int) -> tuple[str, int | None]:
    db = _direct_db_session()
    try:
        row = db.get(TaskInteractionRequest, row_id)
        return str(row.status), row.active_slot
    finally:
        db.close()


def _question_traces(task_id: int) -> list[str]:
    db = _direct_db_session()
    try:
        rows = (
            db.query(TraceEvent)
            .filter(
                TraceEvent.task_id == task_id,
                TraceEvent.event_type == "agent_message",
            )
            .all()
        )
        return [str(row.data.get("message")) for row in rows]
    finally:
        db.close()


def _waiting_task(**overrides: Any) -> tuple[int, int, int]:
    values = {
        "status": TaskStatus.WAITING_FOR_USER,
        "control_state": TaskControlState.WAITING_FOR_USER.value,
        "output": "stale output",
    }
    values.update(overrides)
    task_id, agent_id, owner_user_id = _create_task(**values)
    db = _direct_db_session()
    try:
        db.add(
            TaskChatMessage(
                task_id=task_id,
                user_id=owner_user_id,
                role="assistant",
                content="Continue?",
                message_type="question",
                turn_id="turn-question",
            )
        )
        db.commit()
    finally:
        db.close()
    _add_active_interaction(task_id, "run-sdk", "question-1")
    return task_id, agent_id, owner_user_id


def _active_interaction_of(task_id: int) -> int:
    db = _direct_db_session()
    try:
        return int(
            db.query(TaskInteractionRequest.id)
            .filter(TaskInteractionRequest.task_id == task_id)
            .one()[0]
        )
    finally:
        db.close()


async def _cancel_as_read(task_id: int, agent_id: int) -> None:
    await cancel_sdk_task_unserialized(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id="run-sdk",
        expected_state_version=4,
    )


@pytest.mark.asyncio
async def test_a_waiting_run_is_not_waited_on_before_its_cancel_is_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run waiting for an answer has no coroutine to wait for, so the core
    writes the row without first stopping one: a resume that started after
    the target was read must not be cancelled as if it were the aimed run.
    Only the stop after the write, for a coroutine registered since, runs."""
    task_id, agent_id, _owner = _waiting_task()
    _broadcast_manager(monkeypatch)
    statuses: list[TaskStatus] = []

    async def record_status(_task_id: int, **_kwargs: object) -> SimpleNamespace:
        statuses.append(_load_task(task_id).status)
        return SimpleNamespace(requested=False)

    monkeypatch.setattr(
        task_execution_service.background_task_manager, "cancel_task", record_status
    )

    await _cancel_as_read(task_id, agent_id)

    assert statuses == [TaskStatus.FAILED]


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_version", [1, 2])
async def test_a_waiting_run_is_cancelled_and_its_question_closed(
    monkeypatch: pytest.MonkeyPatch, storage_version: int
) -> None:
    task_id, agent_id, _owner = _waiting_task(storage_version=storage_version)
    interaction_id = _active_interaction_of(task_id)
    _broadcast_manager(monkeypatch)

    await _cancel_as_read(task_id, agent_id)

    task = _load_task(task_id)
    assert is_sdk_cancelled(task.status, task.error_message)
    assert (task.run_id, task.state_version, task.output) == ("run-sdk", 5, None)
    # The question no longer reads as pending; the failure line follows it.
    assert _assistant_rows(task_id) == [
        ("Continue?", SUPERSEDED_MESSAGE_TYPE),
        (CLIENT_SAFE_TASK_FAILURE, TASK_FAILURE_MESSAGE_TYPE),
    ]
    # The answer slot is released, so a later run of the task can ask again.
    assert _interaction(interaction_id) == ("terminated", None)
    later = _add_active_interaction(task_id, "run-later", "question-2")
    assert _interaction(later) == ("active", 1)
    # A version-1 question with no trace row of its own is given one, so the
    # task's steps still show what was asked.
    assert _question_traces(task_id) == (["Continue?"] if storage_version == 1 else [])
    assert [fact["result"] for fact in _result_facts(task_id)] == (
        [{"status": "cancelled", "error": SDK_CANCEL_ERROR_MESSAGE}]
        if storage_version == 2
        else []
    )


@pytest.mark.parametrize("claimed_by", ["control_state", "queued_command"])
def test_a_waiting_run_an_answer_already_claimed_is_not_cancelled(
    claimed_by: str,
) -> None:
    """The answer owns the waiting run until it resumes it."""
    task_id, agent_id, owner_user_id = _waiting_task()
    db = _direct_db_session()
    try:
        if claimed_by == "control_state":
            db.query(Task).filter(Task.id == task_id).update(
                {"control_state": TaskControlState.RESUME_REQUESTED.value}
            )
        else:
            stage_task_command(
                db,
                task_id=task_id,
                actor_user_id=owner_user_id,
                command_id="reply-1",
                kind=TaskCommandKind.RESUME_INPUT,
                payload={},
            )
        db.commit()
    finally:
        db.close()
    before = _row(task_id)

    with pytest.raises(StaleTaskRunError):
        sdk_task_cancel._finalize_sdk_cancel_sync(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id="run-sdk",
            expected_state_version=4,
            expected_status=TaskStatus.WAITING_FOR_USER,
            terminal_event_state={},
        )

    assert _row(task_id) == before
    assert _assistant_rows(task_id) == [("Continue?", "question")]


@pytest.mark.asyncio
async def test_a_paused_run_is_cancelled_without_touching_a_later_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paused task accepts an append, so a pending user message may belong
    to a later turn: the cancelled run's settlement leaves it alone."""
    task_id, agent_id, owner_user_id = _create_task(
        status=TaskStatus.PAUSED, control_state=TaskControlState.PAUSED.value
    )
    _seed_pending_user_message(task_id, owner_user_id, "turn-appended")
    _broadcast_manager(monkeypatch)

    await _cancel_as_read(task_id, agent_id)

    task = _load_task(task_id)
    assert is_sdk_cancelled(task.status, task.error_message)
    assert task.state_version == 5
    assert _delivery_status(task_id, "turn-appended") == DELIVERY_PENDING
    assert _assistant_rows(task_id) == [
        (CLIENT_SAFE_TASK_FAILURE, TASK_FAILURE_MESSAGE_TYPE)
    ]


@pytest.mark.parametrize("moved", ["run", "state_version"])
def test_a_paused_run_that_moved_before_its_lock_keeps_its_automatic_resume(
    monkeypatch: pytest.MonkeyPatch, moved: str
) -> None:
    """The row read under the lock is checked against the run and version
    the cancel was aimed at before the automatic resume is stopped: a row
    that moved since is stale, and nothing is stopped or written."""
    from xagent.web.services import task_auto_resume

    task_id, _agent_id, _owner = _create_task(
        status=TaskStatus.PAUSED, control_state=TaskControlState.PAUSED.value
    )
    stop = MagicMock()
    monkeypatch.setattr(task_auto_resume, "cancel_pending_auto_resume_no_commit", stop)
    before = _row(task_id)
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        with pytest.raises(StaleTaskRunError):
            sdk_task_cancel._stop_auto_resume_no_commit(
                db,
                task,
                expected_run_id="run-other" if moved == "run" else "run-sdk",
                expected_state_version=3 if moved == "state_version" else 4,
            )
    finally:
        db.close()

    stop.assert_not_called()
    assert _row(task_id) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_version", [1, 2])
@pytest.mark.parametrize("state", ["waiting", "paused"])
async def test_the_terminal_frame_of_a_waiting_or_paused_run_carries_its_snapshot(
    monkeypatch: pytest.MonkeyPatch, state: str, storage_version: int
) -> None:
    """The core writes a waiting or paused run's result fact itself, so a
    version-2 frame carries the turn's display events, on the write and on
    a replay of the same command alike."""
    if state == "waiting":
        task_id, agent_id, _owner = _waiting_task(storage_version=storage_version)
    else:
        task_id, agent_id, _owner = _create_task(
            status=TaskStatus.PAUSED,
            control_state=TaskControlState.PAUSED.value,
            storage_version=storage_version,
        )
    manager = _broadcast_manager(monkeypatch)

    for _replay in range(2):
        manager.broadcast_to_task.reset_mock()
        await _cancel_as_read(task_id, agent_id)

        frame = _terminal_frame(manager)
        if storage_version == 2:
            assert frame["state_version"] == 5
            assert frame["task"]["status"] == "failed"
        else:
            assert "state_version" not in frame


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        {
            "status": TaskStatus.WAITING_FOR_USER,
            "control_state": TaskControlState.WAITING_FOR_USER.value,
        },
        {"status": TaskStatus.PAUSED, "control_state": TaskControlState.PAUSED.value},
    ],
    ids=["waiting", "paused"],
)
async def test_a_queued_turn_that_started_and_now_waits_is_cancelled_as_it_is(
    monkeypatch: pytest.MonkeyPatch, state: dict[str, Any]
) -> None:
    """The queued run started and is waiting for an answer or paused by the
    time the command applies: it is cancelled the way its state is."""
    task_id, agent_id, owner_user_id = _create_task(
        run_id=QUEUED_RUN, state_version=7, **state
    )
    _stage_start(task_id, owner_user_id, status="completed", attempt_count=1)
    _broadcast_manager(monkeypatch)

    await _cancel_queued(task_id, agent_id)

    task = _load_task(task_id)
    assert is_sdk_cancelled(task.status, task.error_message)
    assert (task.run_id, task.state_version) == (QUEUED_RUN, 8)
