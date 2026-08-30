"""External-scope cancel: its own execution core, its own wording.

Every test here drives one invariant of the external cancel path: the
terminal event the visitor needs, the two texts the two writers own, the
replay judgement, the delivery row the finalize closes, the wait constant
that must not travel to the A2A path, the failure classification, and the
routing that keeps a scopeless cancel on the A2A core.
"""

import asyncio
import inspect
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.web.api import a2a as a2a_api
from xagent.web.api import websocket as websocket_api
from xagent.web.models.agent import Agent
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import external_task_cancel, task_orchestrator
from xagent.web.services.assistant_history_safety import (
    CLIENT_SAFE_FAILURE_MESSAGE_TYPE,
)
from xagent.web.services.chat_history_service import (
    DELIVERY_DISPATCHED,
    DELIVERY_PENDING,
)
from xagent.web.services.external_task_cancel import (
    EXTERNAL_CANCEL_ERROR_MESSAGE,
    EXTERNAL_CANCEL_NOT_APPLIED_MESSAGE,
    EXTERNAL_CANCEL_WAIT_SECONDS,
    EXTERNAL_TURN_INTERRUPTED_MESSAGE,
    cancel_external_task_unserialized,
)
from xagent.web.services.task_command_transport import (
    MAX_COMMAND_FAILURES,
    ClaimedTaskCommand,
    TaskCommandKind,
    TaskCommandRejected,
)
from xagent.web.services.task_command_terminal_events import (
    render_terminal_task_event_message,
    terminal_event_draft_for_error,
)
from xagent.web.services.task_execution_controller import TaskControlState
from xagent.web.services.task_orchestrator import TaskTurnPayload

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


def _create_agent() -> tuple[int, int]:
    """One agent plus its owner id, the pair every task row here needs."""
    headers = _admin_headers()
    response = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "External cancel agent",
            "description": "External cancel test agent",
            "instructions": "You are an external cancel test agent.",
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
    agent_id: int,
    owner_user_id: int,
    title: str,
    source: str = "external",
    status: TaskStatus = TaskStatus.RUNNING,
    control_state: str = TaskControlState.RUNNING.value,
    run_id: str | None = "run-external",
    state_version: int = 4,
    error_message: str | None = None,
    runner_id: str | None = None,
    agent_config: dict[str, Any] | None = None,
) -> int:
    db = _direct_db_session()
    try:
        task = Task(
            user_id=owner_user_id,
            title=title,
            status=status,
            control_state=control_state,
            run_id=run_id,
            state_version=state_version,
            runner_id=runner_id,
            lease_attempt_id="attempt-external" if runner_id else None,
            lease_expires_at=(
                datetime.now(UTC) + timedelta(minutes=1) if runner_id else None
            ),
            last_heartbeat_at=datetime.now(UTC) if runner_id else None,
            agent_id=agent_id,
            source=source,
            is_visible=False,
            error_message=error_message,
            agent_config=agent_config,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return int(task.id)
    finally:
        db.close()


def _load_task(task_id: int) -> Task:
    db = _direct_db_session()
    try:
        return db.query(Task).filter(Task.id == task_id).one()
    finally:
        db.close()


def _seed_pending_user_message(
    *, task_id: int, owner_user_id: int, turn_id: str
) -> None:
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


def _delivery_status(task_id: int, turn_id: str) -> str | None:
    db = _direct_db_session()
    try:
        message = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.turn_id == turn_id,
            )
            .one()
        )
        return str(message.delivery_status)
    finally:
        db.close()


def _interruption_transcript_count(task_id: int) -> int:
    db = _direct_db_session()
    try:
        return (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "assistant",
                TaskChatMessage.content == EXTERNAL_TURN_INTERRUPTED_MESSAGE,
            )
            .count()
        )
    finally:
        db.close()


def _broadcast_manager(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    manager = MagicMock()
    manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", manager)
    return manager


def _broadcast_payloads(manager: MagicMock) -> list[dict]:
    return [call.args[0] for call in manager.broadcast_to_task.await_args_list]


@pytest.mark.asyncio
async def test_external_cancel_broadcasts_terminal_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The visitor learns the turn ended from the core, not from settlement.

    The cancelled coroutine's settlement broadcasts only for setup/run
    errors, so without this frame the widget keeps waiting forever.
    """
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="external cancel broadcast",
    )
    manager = _broadcast_manager(monkeypatch)
    monkeypatch.setattr(
        websocket_api.background_task_manager,
        "cancel_task",
        AsyncMock(return_value=MagicMock(requested=False)),
    )

    await cancel_external_task_unserialized(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id="run-external",
        expected_state_version=4,
    )

    payloads = _broadcast_payloads(manager)
    assert [payload["type"] for payload in payloads] == ["task_error"]
    assert payloads[0]["message"] == EXTERNAL_TURN_INTERRUPTED_MESSAGE
    assert payloads[0]["task"]["status"] == TaskStatus.FAILED.value
    cancelled = _load_task(task_id)
    assert cancelled.status == TaskStatus.FAILED
    assert cancelled.control_state == TaskControlState.FAILED.value
    assert cancelled.state_version == 5
    assert cancelled.error_message == EXTERNAL_CANCEL_ERROR_MESSAGE
    assert cancelled.runner_id is None
    assert cancelled.lease_expires_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task_source", "expected_error"),
    [
        ("external", EXTERNAL_TURN_INTERRUPTED_MESSAGE),
        ("widget", "task execution cancelled"),
    ],
)
async def test_settlement_text_by_task_source(
    monkeypatch: pytest.MonkeyPatch,
    task_source: str,
    expected_error: str,
) -> None:
    """Settlement text is derived from the task source, for that audience.

    The settlement of a cancelled run writes its text to the durable row and
    to the transcript, so an external visitor must not read the operator
    wording every other source keeps.
    """
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="settlement text",
        source=task_source,
        run_id=None,
        state_version=0,
    )
    _broadcast_manager(monkeypatch)
    execution_started = asyncio.Event()

    async def block_until_cancelled(**_kwargs: object) -> None:
        execution_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(websocket_api, "execute_task_background", block_until_cancelled)
    monkeypatch.setattr(
        task_orchestrator,
        "load_task_setup_snapshot_sync",
        lambda *_args, **_kwargs: SimpleNamespace(task=SimpleNamespace(id=task_id)),
    )
    monkeypatch.setattr(task_orchestrator, "_get_agent_manager", MagicMock())

    bg_task = task_orchestrator._schedule_bg(
        task_id=task_id,
        task_owner_user_id=owner_user_id,
        task_source=task_source,
        payload=TaskTurnPayload("stop me"),
        force_fresh=False,
        context=None,
    )
    try:
        await asyncio.wait_for(execution_started.wait(), timeout=5)
        bg_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await bg_task
    finally:
        websocket_api.background_task_manager.running_tasks.pop(task_id, None)

    assert _load_task(task_id).error_message == expected_error


@pytest.mark.asyncio
async def test_interrupted_transcript_finalize_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stopped turn gets exactly one AI-side explanation line.

    When this finalize's fenced UPDATE wins the race, the settlement's own
    UPDATE requires ``status == RUNNING`` and matches no row, so it skips
    its own write. This finalize is then the only writer, and it must not
    leave the reader without any explanation.
    """
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="interruption transcript finalize wins",
    )
    _broadcast_manager(monkeypatch)
    monkeypatch.setattr(
        websocket_api.background_task_manager,
        "cancel_task",
        AsyncMock(return_value=MagicMock(requested=False)),
    )

    await cancel_external_task_unserialized(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id="run-external",
        expected_state_version=4,
    )

    assert _interruption_transcript_count(task_id) == 1
    db = _direct_db_session()
    try:
        row = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "assistant",
                TaskChatMessage.content == EXTERNAL_TURN_INTERRUPTED_MESSAGE,
            )
            .one()
        )
        assert row.message_type == CLIENT_SAFE_FAILURE_MESSAGE_TYPE
        assert row.turn_id is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_interrupted_transcript_settlement_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A settlement that beats this finalize still leaves exactly one line.

    Reuses ``test_settlement_text_by_task_source``'s driving shape: a real
    background run is cancelled and its own settlement writes the
    interruption line before this core is ever asked to finalize anything.
    The exact target this core is handed afterwards is the run/version pair
    the settlement raced against, so the replay judgement must recognise it
    as already settled and add nothing.
    """
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="interruption transcript settlement wins",
        source="external",
        run_id=None,
        state_version=0,
    )
    manager = _broadcast_manager(monkeypatch)
    execution_started = asyncio.Event()

    async def block_until_cancelled(**_kwargs: object) -> None:
        execution_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(websocket_api, "execute_task_background", block_until_cancelled)
    monkeypatch.setattr(
        task_orchestrator,
        "load_task_setup_snapshot_sync",
        lambda *_args, **_kwargs: SimpleNamespace(task=SimpleNamespace(id=task_id)),
    )
    monkeypatch.setattr(task_orchestrator, "_get_agent_manager", MagicMock())

    bg_task = task_orchestrator._schedule_bg(
        task_id=task_id,
        task_owner_user_id=owner_user_id,
        task_source="external",
        payload=TaskTurnPayload("stop me"),
        force_fresh=False,
        context=None,
    )
    try:
        await asyncio.wait_for(execution_started.wait(), timeout=5)
        before_cancel = _load_task(task_id)
        bg_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await bg_task
    finally:
        websocket_api.background_task_manager.running_tasks.pop(task_id, None)

    settled = _load_task(task_id)
    assert settled.error_message == EXTERNAL_TURN_INTERRUPTED_MESSAGE
    assert _interruption_transcript_count(task_id) == 1

    await cancel_external_task_unserialized(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id=before_cancel.run_id,
        expected_state_version=before_cancel.state_version,
    )

    assert _interruption_transcript_count(task_id) == 1
    assert _load_task(task_id).error_message == EXTERNAL_TURN_INTERRUPTED_MESSAGE
    assert manager.broadcast_to_task.await_count == 1
    db = _direct_db_session()
    try:
        row = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "assistant",
            )
            .one()
        )
        assert row.message_type == CLIENT_SAFE_FAILURE_MESSAGE_TYPE
    finally:
        db.close()


@pytest.mark.asyncio
async def test_interrupted_transcript_replayed_after_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A redelivery of the same command after its own finalize adds nothing."""
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="interruption transcript replayed",
    )
    _broadcast_manager(monkeypatch)
    monkeypatch.setattr(
        websocket_api.background_task_manager,
        "cancel_task",
        AsyncMock(return_value=MagicMock(requested=False)),
    )

    for _ in range(2):
        await cancel_external_task_unserialized(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id="run-external",
            expected_state_version=4,
        )

    assert _interruption_transcript_count(task_id) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expected_state_version",
    [5, 4],
    ids=["same_version_terminal", "one_version_past"],
)
@pytest.mark.parametrize(
    "settled_by_the_run",
    [EXTERNAL_CANCEL_ERROR_MESSAGE, EXTERNAL_TURN_INTERRUPTED_MESSAGE],
    ids=["own_finalize_message", "settlement_message"],
)
async def test_external_cancel_finalize_replay_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    settled_by_the_run: str,
    expected_state_version: int,
) -> None:
    """A replay reads the state tuple and the settling sentence, and writes nothing.

    The same command redelivered after its own finalize finds its own state
    version; the settlement of the run it cancelled leaves the same terminal
    tuple one version further on. Both are the outcome the command asked
    for, so neither may rewrite the row - but only when the row also
    carries one of the two sentences a cancellation writes, not any text.
    """
    agent_id, owner_user_id = _create_agent()
    manager = _broadcast_manager(monkeypatch)
    cancel_task = AsyncMock(return_value=MagicMock(requested=False))
    monkeypatch.setattr(
        websocket_api.background_task_manager, "cancel_task", cancel_task
    )
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="settled cancel target",
        status=TaskStatus.FAILED,
        control_state=TaskControlState.FAILED.value,
        state_version=5,
        error_message=settled_by_the_run,
    )

    await cancel_external_task_unserialized(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id="run-external",
        expected_state_version=expected_state_version,
    )

    replayed = _load_task(task_id)
    assert replayed.status == TaskStatus.FAILED
    assert replayed.state_version == 5
    assert replayed.error_message == settled_by_the_run
    assert cancel_task.await_count == 0
    # A settled target is not cancelled a second time, but the terminal
    # event still goes out: the attempt that settled it may have died
    # before broadcasting.
    assert len(_broadcast_payloads(manager)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expected_state_version",
    [4, 5],
    ids=["version_moved", "same_version_terminal"],
)
async def test_replay_judgement_ignores_a_genuine_failure(
    monkeypatch: pytest.MonkeyPatch,
    expected_state_version: int,
) -> None:
    """A tuple match with an unrelated failure text is not a settled cancel.

    A run that failed on its own - a provider timeout, a setup error -
    leaves the same durable tuple a cancellation would. Without checking
    the text, this command would report someone else's failure as a
    cancellation it performed and broadcast a false terminal event.
    """
    agent_id, owner_user_id = _create_agent()
    manager = _broadcast_manager(monkeypatch)
    cancel_task = AsyncMock(return_value=MagicMock(requested=False))
    monkeypatch.setattr(
        websocket_api.background_task_manager, "cancel_task", cancel_task
    )
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="genuinely failed task",
        status=TaskStatus.FAILED,
        control_state=TaskControlState.FAILED.value,
        state_version=5,
        error_message="setup/run error: provider timeout",
    )

    with pytest.raises(TaskCommandRejected) as raised:
        await cancel_external_task_unserialized(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id="run-external",
            expected_state_version=expected_state_version,
        )

    assert raised.value.reason == "stale_run"
    untouched = _load_task(task_id)
    assert untouched.status == TaskStatus.FAILED
    assert untouched.state_version == 5
    assert untouched.error_message == "setup/run error: provider timeout"
    assert cancel_task.await_count == 0
    assert len(_broadcast_payloads(manager)) == 0
    assert _interruption_transcript_count(task_id) == 0


@pytest.mark.asyncio
async def test_external_cancel_marks_delivery_dispatched_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finalize that outruns the settlement closes the delivery row itself.

    The delivery row is closed by whichever coroutine settles the turn. When
    the wait expires and this finalize wins, nothing else will ever move the
    row off ``pending`` and every resend of that client message id would be
    refused forever.
    """
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="delivery row after timeout",
    )
    _seed_pending_user_message(
        task_id=task_id, owner_user_id=owner_user_id, turn_id="turn-timeout"
    )
    _broadcast_manager(monkeypatch)
    monkeypatch.setattr(external_task_cancel, "EXTERNAL_CANCEL_WAIT_SECONDS", 0.05)

    async def unwind_past_the_wait() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.2)

    unwinding = asyncio.create_task(unwind_past_the_wait())
    websocket_api.background_task_manager.register_task(task_id, unwinding)
    try:
        await cancel_external_task_unserialized(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id="run-external",
            expected_state_version=4,
        )
    finally:
        websocket_api.background_task_manager.running_tasks.pop(task_id, None)

    assert _load_task(task_id).status == TaskStatus.FAILED
    assert _delivery_status(task_id, "turn-timeout") == DELIVERY_DISPATCHED


@pytest.mark.asyncio
async def test_external_cancel_closes_the_named_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command carrying a turn id closes exactly that row, not the newest.

    Without a turn id the fallback closes the newest pending user row, which
    only happens to be the running turn's row because a task cannot take a
    new turn while it is running. A caller that does know the turn must not
    be at the mercy of that fallback.
    """
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="named turn delivery row",
    )
    _seed_pending_user_message(
        task_id=task_id, owner_user_id=owner_user_id, turn_id="turn-older"
    )
    _seed_pending_user_message(
        task_id=task_id, owner_user_id=owner_user_id, turn_id="turn-newer"
    )
    _broadcast_manager(monkeypatch)
    monkeypatch.setattr(
        websocket_api.background_task_manager,
        "cancel_task",
        AsyncMock(return_value=MagicMock(requested=False)),
    )

    await cancel_external_task_unserialized(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id="run-external",
        expected_state_version=4,
        turn_id="turn-older",
    )

    assert _delivery_status(task_id, "turn-older") == DELIVERY_DISPATCHED
    assert _delivery_status(task_id, "turn-newer") == DELIVERY_PENDING


@pytest.mark.asyncio
async def test_cancel_command_turn_id_reaches_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatcher threads a payload ``turn_id`` through to the core.

    ``_execute_durable_task_command`` is the only caller of the external
    core, so this is the site that must translate the payload's turn id
    into the keyword argument the core closes a delivery row with.
    """
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="turn id threaded through the dispatcher",
    )
    _broadcast_manager(monkeypatch)
    observed: list[dict[str, Any]] = []

    async def record_target(**kwargs: Any) -> None:
        observed.append(kwargs)

    monkeypatch.setattr(
        websocket_api, "cancel_external_task_unserialized", record_target
    )

    await websocket_api._execute_durable_task_command(
        ClaimedTaskCommand(
            id=4,
            task_id=task_id,
            actor_user_id=None,
            command_id=f"cancel:{task_id}:4",
            kind=TaskCommandKind.CANCEL,
            payload={
                "agent_id": agent_id,
                "target_state_version": 4,
                "scope": "external",
                "turn_id": "turn-older",
            },
            target_run_id="run-external",
            attempt_count=1,
        )
    )

    assert len(observed) == 1
    assert observed[0]["turn_id"] == "turn-older"


@pytest.mark.asyncio
async def test_cancel_wait_constants_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """The longer external wait stays on the external path.

    The wait occupies a dispatcher slot per handle, so leaking the external
    value onto the A2A path would multiply every A2A cancel's hold time.
    """
    agent_id, owner_user_id = _create_agent()
    external_task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="external wait",
    )
    a2a_task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="a2a wait",
        source="a2a",
        agent_config={"a2a_context_id": "ctx-external-wait"},
    )
    _broadcast_manager(monkeypatch)
    waits: list[float | None] = []

    async def record_wait(
        _task_id: int, timeout_seconds: float | None = None
    ) -> MagicMock:
        waits.append(timeout_seconds)
        return MagicMock(requested=False)

    monkeypatch.setattr(
        websocket_api.background_task_manager, "cancel_task", record_wait
    )

    await cancel_external_task_unserialized(
        task_id=external_task_id,
        agent_id=agent_id,
        expected_run_id="run-external",
        expected_state_version=4,
    )
    await a2a_api._cancel_task_unserialized(
        task_id=a2a_task_id,
        agent_id=agent_id,
        expected_run_id="run-external",
        expected_state_version=4,
    )

    a2a_default = inspect.signature(
        websocket_api.BackgroundTaskManager.cancel_task
    ).parameters["timeout_seconds"]
    assert waits == [EXTERNAL_CANCEL_WAIT_SECONDS, None]
    assert EXTERNAL_CANCEL_WAIT_SECONDS == 5.0
    assert a2a_default.default == 0.5


@pytest.mark.asyncio
async def test_external_cancel_invalidates_the_task_cache_on_finalize_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finalize that actually writes the terminal row invalidates the cache.

    The invalidation must run only after the write it is protecting against
    staleness has committed, so a reader racing the cache never sees a
    pre-cancel projection after the row is already durably FAILED.
    """
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="cache invalidated on finalize commit",
    )
    _broadcast_manager(monkeypatch)
    monkeypatch.setattr(
        websocket_api.background_task_manager,
        "cancel_task",
        AsyncMock(return_value=MagicMock(requested=False)),
    )
    invalidated = MagicMock()
    monkeypatch.setattr(
        task_orchestrator, "invalidate_task_cache_best_effort", invalidated
    )

    await cancel_external_task_unserialized(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id="run-external",
        expected_state_version=4,
    )

    invalidated.assert_called_once_with(task_id)


@pytest.mark.asyncio
async def test_external_cancel_replay_short_circuit_skips_cache_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finalize that discovers the target already settled writes nothing.

    The race is settled between the two reads: the initial check sees the
    task still RUNNING, but by the time finalize re-checks (after awaiting
    the cancel), the target has already reached its cancel outcome
    elsewhere. Nothing this finalize writes, so no cache invalidation is
    owed either - only a write this core performed should trigger one.
    """
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="settled during the wait",
    )
    _broadcast_manager(monkeypatch)
    invalidated = MagicMock()
    monkeypatch.setattr(
        task_orchestrator, "invalidate_task_cache_best_effort", invalidated
    )

    async def settle_during_the_wait(*_args: object, **_kwargs: object) -> MagicMock:
        db = _direct_db_session()
        try:
            db.query(Task).filter(Task.id == task_id).update(
                {
                    "status": TaskStatus.FAILED,
                    "control_state": TaskControlState.FAILED.value,
                    "state_version": 5,
                    "error_message": EXTERNAL_CANCEL_ERROR_MESSAGE,
                    "run_id": "run-external",
                }
            )
            db.commit()
        finally:
            db.close()
        return MagicMock(requested=False)

    monkeypatch.setattr(
        websocket_api.background_task_manager,
        "cancel_task",
        settle_during_the_wait,
    )

    await cancel_external_task_unserialized(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id="run-external",
        expected_state_version=4,
    )

    invalidated.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expected_run_id", "expected_state_version", "wrong_agent"),
    [
        ("run-external", 4, True),
        ("run-rotated", 4, False),
        ("run-external", 9, False),
    ],
    ids=["not_this_agents_task", "run_rotated", "state_version_moved"],
)
async def test_external_cancel_failures_classified_rejected(
    monkeypatch: pytest.MonkeyPatch,
    expected_run_id: str,
    expected_state_version: int,
    wrong_agent: bool,
) -> None:
    """Every expected failure leaves the core as a rejection.

    A rejection is terminal without an error frame, which is the outcome an
    anonymous visitor should get for a stop whose target is gone. A plain
    exception would instead retry and then render as an error bubble.
    """
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="rejected cancel",
    )
    _broadcast_manager(monkeypatch)
    monkeypatch.setattr(
        websocket_api.background_task_manager,
        "cancel_task",
        AsyncMock(return_value=MagicMock(requested=False)),
    )

    with pytest.raises(TaskCommandRejected):
        await cancel_external_task_unserialized(
            task_id=task_id,
            agent_id=agent_id + 1 if wrong_agent else agent_id,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        )

    untouched = _load_task(task_id)
    assert untouched.status == TaskStatus.RUNNING
    assert untouched.state_version == 4
    assert untouched.error_message is None


@pytest.mark.asyncio
async def test_terminal_command_error_text_external() -> None:
    """An exhausted external cancel stages the neutral sentence code.

    Driven by a deleted task row, which is a real raise site outside the
    external core: the failure budget runs out and the broadcast reaches the
    visitor's conversation, where the default wording would carry command
    and exception detail. The row being gone means the status read cannot
    answer, so the wording is the conservative "didn't go through" sentence,
    not the terminal one - the deletion is not proof the turn is over.
    """
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="deleted before cancel",
    )
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).delete()
        db.commit()
    finally:
        db.close()
    command = ClaimedTaskCommand(
        id=1,
        task_id=task_id,
        actor_user_id=None,
        command_id=f"cancel:{task_id}:4",
        kind=TaskCommandKind.CANCEL,
        payload={
            "agent_id": agent_id,
            "target_state_version": 4,
            "scope": "external",
            "end_user_id": "visitor-1",
        },
        target_run_id="run-external",
        attempt_count=MAX_COMMAND_FAILURES,
        failure_count=MAX_COMMAND_FAILURES - 1,
    )

    with pytest.raises(ValueError) as raised:
        await websocket_api.execute_durable_task_command(command)

    draft = terminal_event_draft_for_error(raised.value)
    assert draft is not None
    assert draft.include_command_identity is False
    message = render_terminal_task_event_message(
        SimpleNamespace(
            message_code=draft.message_code,
            command_kind=command.kind.value,
        )
    )
    assert message == EXTERNAL_CANCEL_NOT_APPLIED_MESSAGE
    assert str(raised.value) not in str(message)
    assert command.command_id not in str(message)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task_status", "expected_message"),
    [
        (TaskStatus.RUNNING, EXTERNAL_CANCEL_NOT_APPLIED_MESSAGE),
        (TaskStatus.FAILED, EXTERNAL_TURN_INTERRUPTED_MESSAGE),
        (TaskStatus.COMPLETED, EXTERNAL_TURN_INTERRUPTED_MESSAGE),
        (TaskStatus.PAUSED, EXTERNAL_CANCEL_NOT_APPLIED_MESSAGE),
    ],
    ids=[
        "still_running",
        "already_failed",
        "already_completed",
        "paused_is_still_alive",
    ],
)
async def test_terminal_command_error_text_follows_the_task_state(
    task_status: TaskStatus,
    expected_message: str,
) -> None:
    """The exhausted-cancel wording asserts only what the task's state proves.

    ``PAUSED`` looks stopped to the frontend's spinner logic, but the run
    behind it can still produce more output on resume, so the terminal
    sentence would be false there just as it would be on ``RUNNING``.
    """
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="terminal wording by task state",
        status=task_status,
        control_state=TaskControlState.RUNNING.value,
    )
    command = ClaimedTaskCommand(
        id=1,
        task_id=task_id,
        actor_user_id=None,
        command_id=f"cancel:{task_id}:4",
        kind=TaskCommandKind.CANCEL,
        payload={
            "agent_id": agent_id,
            "target_state_version": 4,
            "scope": "external",
        },
        target_run_id="run-external",
        attempt_count=1,
    )

    draft = await websocket_api._terminal_command_event_draft(
        command, RuntimeError(f"lease lost at {task_id}")
    )

    assert (
        render_terminal_task_event_message(
            SimpleNamespace(
                message_code=draft.message_code,
                command_kind=command.kind.value,
            )
        )
        == expected_message
    )


@pytest.mark.asyncio
async def test_cancel_payload_audit_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """``end_user_id`` is audit data: nothing on this path reads it.

    It is a value the producer reports about itself, not an authorization
    fact, so the same command with and without it must hand the core exactly
    the same target.
    """
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="audit field only",
    )
    _broadcast_manager(monkeypatch)
    observed: list[dict[str, Any]] = []

    async def record_target(**kwargs: Any) -> None:
        observed.append(kwargs)

    monkeypatch.setattr(
        websocket_api, "cancel_external_task_unserialized", record_target
    )
    payloads: list[dict[str, Any]] = [
        {"agent_id": agent_id, "target_state_version": 4, "scope": "external"},
        {
            "agent_id": agent_id,
            "target_state_version": 4,
            "scope": "external",
            "end_user_id": "visitor-1",
        },
    ]

    for payload in payloads:
        await websocket_api._execute_durable_task_command(
            ClaimedTaskCommand(
                id=2,
                task_id=task_id,
                actor_user_id=None,
                command_id=f"cancel:{task_id}:4",
                kind=TaskCommandKind.CANCEL,
                payload=payload,
                target_run_id="run-external",
                attempt_count=1,
            )
        )

    assert len(observed) == len(payloads)
    assert observed[0] == observed[1]


@pytest.mark.asyncio
async def test_cancel_without_scope_stays_on_the_a2a_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A payload with no scope reaches the A2A core exactly as before."""
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="scopeless cancel",
        source="a2a",
        agent_config={"a2a_context_id": "ctx-scopeless"},
    )
    _broadcast_manager(monkeypatch)
    monkeypatch.setattr(
        websocket_api.background_task_manager,
        "cancel_task",
        AsyncMock(return_value=MagicMock(requested=False)),
    )
    external_core = AsyncMock()
    monkeypatch.setattr(
        websocket_api, "cancel_external_task_unserialized", external_core
    )

    await websocket_api._execute_durable_task_command(
        ClaimedTaskCommand(
            id=3,
            task_id=task_id,
            actor_user_id=owner_user_id,
            command_id=f"cancel:{task_id}:4",
            kind=TaskCommandKind.CANCEL,
            payload={"agent_id": agent_id, "target_state_version": 4},
            target_run_id="run-external",
            attempt_count=1,
        )
    )

    cancelled = _load_task(task_id)
    assert external_core.await_count == 0
    assert cancelled.status == TaskStatus.FAILED
    assert cancelled.error_message == "Task canceled by A2A client."
    assert cancelled.agent_config["a2a_state"] == "TASK_STATE_CANCELED"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    ["foo", "", 123, None, {"scope": "external"}],
    ids=[
        "unknown_string",
        "empty_string",
        "non_string",
        "explicit_none",
        "unhashable",
    ],
)
async def test_cancel_rejects_an_unknown_scope(
    monkeypatch: pytest.MonkeyPatch,
    scope: Any,
) -> None:
    """A scope naming neither core is rejected before either core runs.

    A payload without the ``scope`` key at all keeps the pre-external A2A
    shape; every other value - including an explicit ``None`` - names a
    core that does not exist, and silently defaulting to the A2A core would
    cancel nothing while reporting success.
    """
    agent_id, owner_user_id = _create_agent()
    task_id = _create_task(
        agent_id=agent_id,
        owner_user_id=owner_user_id,
        title="unknown scope cancel",
    )
    _broadcast_manager(monkeypatch)
    external_core = AsyncMock()
    monkeypatch.setattr(
        websocket_api, "cancel_external_task_unserialized", external_core
    )
    a2a_core = AsyncMock()
    monkeypatch.setattr(a2a_api, "_cancel_task_unserialized", a2a_core)

    with pytest.raises(TaskCommandRejected) as raised:
        await websocket_api._execute_durable_task_command(
            ClaimedTaskCommand(
                id=5,
                task_id=task_id,
                actor_user_id=None,
                command_id=f"cancel:{task_id}:4",
                kind=TaskCommandKind.CANCEL,
                payload={
                    "agent_id": agent_id,
                    "target_state_version": 4,
                    "scope": scope,
                },
                target_run_id="run-external",
                attempt_count=1,
            )
        )

    assert raised.value.reason == "unsupported_scope"
    assert external_core.await_count == 0
    assert a2a_core.await_count == 0
    untouched = _load_task(task_id)
    assert untouched.status == TaskStatus.RUNNING
    assert untouched.state_version == 4
    assert untouched.error_message is None
