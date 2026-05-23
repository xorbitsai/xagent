"""Tests for the turn-lifecycle API in ``task_orchestrator``.

Covers:

  - ``TaskTurnPayload`` dual-message channel
  - ``TurnKind`` + ``force_fresh`` orthogonal kind/flag
  - ``begin_turn`` atomic claim + persist + bg schedule
  - ``finish_turn`` symmetric terminal-field writer + lease ownership guard
  - ``_schedule_bg`` lease lifecycle wrapper

Tests use SQLite in-memory + direct ORM, mocking only the bits that
require an actual agent runtime (``execute_task_background``).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.task_lease_service import get_runner_id
from xagent.web.services.task_orchestrator import (
    TaskTurnError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
    _schedule_bg,
    finish_turn,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'orchestrator.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _create_user(db) -> User:
    user = User(username="orch-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_task(
    db,
    user_id: int,
    *,
    status: TaskStatus = TaskStatus.PENDING,
    input_: str | None = None,
    output: str | None = None,
    error_message: str | None = None,
) -> Task:
    task = Task(
        user_id=user_id,
        title="Orchestrator test",
        description="test",
        status=status,
        execution_mode="auto",
        input=input_,
        output=output,
        error_message=error_message,
        source="sdk",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@pytest.fixture()
def mock_schedule_bg():
    """Stub the bg coroutine spawn so begin_turn tests don't actually run
    an agent. Opt-in: tests that drive ``_schedule_bg`` directly skip
    this fixture and patch deeper layers themselves.

    Uses ``AsyncMock()`` without an explicit ``return_value`` —
    instantiating ``asyncio.Future()`` at fixture-setup time needs a
    running event loop, which pytest-asyncio doesn't provide during
    fixture collection in CI. The default ``AsyncMock`` return is a
    plain ``MagicMock``, which begin_turn ignores anyway.
    """
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=AsyncMock(),
    ) as mocked:
        yield mocked


@pytest.fixture(autouse=True)
def _clear_bg_manager():
    """Reset the global bg manager between tests so _refuse_if_bg_inflight
    sees a clean slate."""
    from xagent.web.api.websocket import background_task_manager

    background_task_manager.running_tasks.clear()
    yield
    background_task_manager.running_tasks.clear()


# ---------------------------------------------------------------------------
# TaskTurnPayload
# ---------------------------------------------------------------------------


def test_payload_for_agent_falls_back_to_transcript() -> None:
    p = TaskTurnPayload(transcript_message="hi")
    assert p.for_agent == "hi"


def test_payload_uses_execution_when_provided() -> None:
    p = TaskTurnPayload(
        transcript_message="summarize this",
        execution_message="summarize this\n\n[file context]",
    )
    assert p.for_agent == "summarize this\n\n[file context]"


# ---------------------------------------------------------------------------
# begin_turn — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_begin_turn_create_clears_no_terminal_fields_when_pending(
    db_session,
    mock_schedule_bg,
) -> None:
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)

    await TaskTurnOrchestrator.begin_turn(
        task=task,
        payload=TaskTurnPayload("first turn"),
        user=user,
        db=db_session,
        kind=TurnKind.CREATE,
        force_fresh=False,
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.input == "first turn"
    assert task.output is None
    assert task.error_message is None


@pytest.mark.asyncio
async def test_begin_turn_append_clears_stale_output_and_error(
    db_session,
    mock_schedule_bg,
) -> None:
    """Latest-turn snapshot invariant: appending a new turn must reset
    output / error_message from the previous turn so GET returns a
    coherent latest-turn snapshot.
    """
    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user.id,
        status=TaskStatus.COMPLETED,
        input_="first question",
        output="first answer",
        error_message=None,
    )

    await TaskTurnOrchestrator.begin_turn(
        task=task,
        payload=TaskTurnPayload("second question"),
        user=user,
        db=db_session,
        kind=TurnKind.APPEND,
        force_fresh=False,
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.input == "second question"
    assert task.output is None, "stale first-turn output must be cleared"
    assert task.error_message is None


@pytest.mark.asyncio
async def test_begin_turn_append_clears_stale_error_message(
    db_session,
    mock_schedule_bg,
) -> None:
    """Latest-turn snapshot invariant (FAILED side): appending after a
    failed turn must also clear the prior turn's error_message."""
    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user.id,
        status=TaskStatus.FAILED,
        input_="first",
        output=None,
        error_message="first turn blew up",
    )

    await TaskTurnOrchestrator.begin_turn(
        task=task,
        payload=TaskTurnPayload("second"),
        user=user,
        db=db_session,
        kind=TurnKind.APPEND,
        force_fresh=False,
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.input == "second"
    assert task.error_message is None
    assert task.output is None


@pytest.mark.asyncio
async def test_begin_turn_append_accepts_paused_task_as_new_turn(
    db_session,
    mock_schedule_bg,
) -> None:
    """A message sent after pause starts the next turn, not a checkpoint resume."""
    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user.id,
        status=TaskStatus.PAUSED,
        input_="previous request",
        output="stale partial output",
        error_message="stale pause detail",
    )

    payload = TaskTurnPayload("new request after pause")
    await TaskTurnOrchestrator.begin_turn(
        task=task,
        payload=payload,
        user=user,
        db=db_session,
        kind=TurnKind.APPEND,
        force_fresh=False,
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.input == "new request after pause"
    assert task.output is None
    assert task.error_message is None

    persisted = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == int(task.id), TaskChatMessage.role == "user")
        .one()
    )
    assert persisted.content == "new request after pause"
    assert persisted.turn_id == payload.turn_id

    mock_schedule_bg.assert_awaited_once()


@pytest.mark.asyncio
async def test_begin_turn_passes_force_fresh_through_to_schedule_bg(
    db_session,
    mock_schedule_bg,
) -> None:
    """Dual-channel payload + force_fresh forwarding: begin_turn forwards
    the full ``TaskTurnPayload`` and ``force_fresh`` flag to
    ``_schedule_bg`` so the execution side receives both message
    channels and the right reconstruct-state mode."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.COMPLETED)

    payload = TaskTurnPayload(
        transcript_message="show me",
        execution_message="show me\n\n[file: foo.pdf]",
    )
    await TaskTurnOrchestrator.begin_turn(
        task=task,
        payload=payload,
        user=user,
        db=db_session,
        kind=TurnKind.APPEND,
        force_fresh=True,
    )

    mock_schedule_bg.assert_awaited_once()
    kwargs = mock_schedule_bg.await_args.kwargs
    assert kwargs["payload"] is payload
    assert kwargs["force_fresh"] is True

    persisted = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == int(task.id), TaskChatMessage.role == "user")
        .one()
    )
    assert persisted.turn_id == payload.turn_id


# ---------------------------------------------------------------------------
# begin_turn — failure modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_begin_turn_rejects_create_with_force_fresh(
    db_session,
    mock_schedule_bg,
) -> None:
    """Invalid kind + flag combo: CREATE + force_fresh has no meaning."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)

    with pytest.raises(ValueError, match="force_fresh has no meaning"):
        await TaskTurnOrchestrator.begin_turn(
            task=task,
            payload=TaskTurnPayload("x"),
            user=user,
            db=db_session,
            kind=TurnKind.CREATE,
            force_fresh=True,
        )


@pytest.mark.asyncio
async def test_begin_turn_asserts_session_clean_precondition(
    db_session,
    mock_schedule_bg,
) -> None:
    """Caller contract: db session must be clean of uncommitted state."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)

    # Stage an uncommitted user without committing — simulates a caller
    # that forgot to commit before calling begin_turn.
    stray = User(username="stray", password_hash="hash")
    db_session.add(stray)
    # NOT committing on purpose

    with pytest.raises(ValueError, match="clean db session"):
        await TaskTurnOrchestrator.begin_turn(
            task=task,
            payload=TaskTurnPayload("x"),
            user=user,
            db=db_session,
            kind=TurnKind.CREATE,
        )


@pytest.mark.asyncio
async def test_begin_turn_refuses_when_bg_inflight(
    db_session,
    mock_schedule_bg,
) -> None:
    from xagent.web.api.websocket import background_task_manager

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.COMPLETED)

    # Plant a fake "still-running" entry in the bg manager registry.
    # ``_refuse_if_bg_inflight`` only checks ``.done() is False``, so a
    # MagicMock with that one attribute is enough — we don't need a
    # real asyncio.Task (and creating one would require an extra event
    # loop, which trips up pytest-asyncio's fixture machinery in CI).
    fake_inflight = MagicMock(spec=asyncio.Task)
    fake_inflight.done.return_value = False
    background_task_manager.running_tasks[int(task.id)] = fake_inflight

    try:
        with pytest.raises(TaskTurnError) as excinfo:
            await TaskTurnOrchestrator.begin_turn(
                task=task,
                payload=TaskTurnPayload("x"),
                user=user,
                db=db_session,
                kind=TurnKind.APPEND,
            )
        assert excinfo.value.reason == "bg_inflight"

        # Critical: the DB row must NOT have been mutated
        db_session.refresh(task)
        assert task.status == TaskStatus.COMPLETED  # unchanged
        assert task.input is None  # unchanged
    finally:
        background_task_manager.running_tasks.pop(int(task.id), None)


@pytest.mark.asyncio
async def test_begin_turn_refuses_create_against_terminal_task(
    db_session,
    mock_schedule_bg,
) -> None:
    """kind=CREATE filters status==PENDING; a COMPLETED task must reject."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.COMPLETED)

    with pytest.raises(TaskTurnError) as excinfo:
        await TaskTurnOrchestrator.begin_turn(
            task=task,
            payload=TaskTurnPayload("x"),
            user=user,
            db=db_session,
            kind=TurnKind.CREATE,
        )
    assert excinfo.value.reason == "busy"


@pytest.mark.asyncio
async def test_begin_turn_refuses_append_against_pending_task(
    db_session,
    mock_schedule_bg,
) -> None:
    """kind=APPEND filters status IN TERMINAL; a PENDING task must reject."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)

    with pytest.raises(TaskTurnError) as excinfo:
        await TaskTurnOrchestrator.begin_turn(
            task=task,
            payload=TaskTurnPayload("x"),
            user=user,
            db=db_session,
            kind=TurnKind.APPEND,
        )
    assert excinfo.value.reason == "busy"


# ---------------------------------------------------------------------------
# finish_turn
# ---------------------------------------------------------------------------


def test_finish_turn_completed_writes_output_clears_error(db_session) -> None:
    from xagent.web.models.chat_message import TaskChatMessage

    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user.id,
        status=TaskStatus.COMPLETED,
        error_message="stale",
    )
    msg = TaskChatMessage(
        task_id=task.id,
        user_id=user.id,
        role="assistant",
        content="hello world",
        message_type="assistant_message",
    )
    db_session.add(msg)
    db_session.commit()

    finish_turn(db_session, int(task.id))

    db_session.refresh(task)
    assert task.output == "hello world"
    assert task.error_message is None


def test_finish_turn_failed_writes_error_clears_stale_output(db_session) -> None:
    """Latest-turn snapshot invariant (FAILED side): a FAILED turn
    must clear the prior turn's stale ``output`` so the GET response
    doesn't show ``status='failed' + output='prior answer'``."""
    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user.id,
        status=TaskStatus.FAILED,
        output="prior successful output",
        error_message=None,
    )

    finish_turn(db_session, int(task.id))

    db_session.refresh(task)
    assert task.error_message is not None
    assert "Task execution failed" in task.error_message
    assert task.output is None  # latest-turn snapshot invariant


def test_finish_turn_running_skips_when_other_worker_holds_live_lease(
    db_session,
) -> None:
    """Lease ownership guard: when another worker actively holds the
    lease, finish_turn must leave the row alone and not flip RUNNING
    to FAILED."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    # Plant a live lease held by a different runner
    task.runner_id = "other-worker"
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    task.output = "other worker's in-progress output"
    db_session.commit()

    finish_turn(db_session, int(task.id))

    db_session.refresh(task)
    # No change: status stays RUNNING, output untouched, no error injected
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == "other-worker"
    assert task.output == "other worker's in-progress output"
    assert task.error_message is None


def test_finish_turn_running_flips_failed_when_no_live_lease(db_session) -> None:
    """RUNNING + no live lease elsewhere → genuine stuck task → flip FAILED."""
    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user.id,
        status=TaskStatus.RUNNING,
        output="stale partial output",
    )
    # No runner_id / lease — task is stuck
    db_session.commit()

    finish_turn(db_session, int(task.id))

    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    assert task.error_message is not None
    assert task.output is None  # latest-turn snapshot invariant


def test_finish_turn_running_flips_failed_when_lease_expired(db_session) -> None:
    """RUNNING + lease present but expired → still flip FAILED."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task.runner_id = "other-worker"
    task.lease_expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db_session.commit()

    finish_turn(db_session, int(task.id))

    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED


def test_finish_turn_running_flips_failed_when_we_own_lease(db_session) -> None:
    """RUNNING + we own the lease ourselves → still our bug to finalize."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task.runner_id = get_runner_id()  # our own process
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    db_session.commit()

    finish_turn(db_session, int(task.id))

    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED


# ---------------------------------------------------------------------------
# _schedule_bg lease lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_bg_skips_finish_turn_when_lease_acquire_fails(
    db_session,
) -> None:
    """Running-elsewhere short-circuit: lease taken by another worker
    → never call execute_task_background or finish_turn; bg coroutine
    returns clean."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)

    from xagent.web.api.websocket import background_task_manager

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease",
            return_value=None,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(),
        ) as mock_exec,
        patch(
            "xagent.web.services.task_orchestrator.finish_turn",
        ) as mock_finish,
        patch.object(background_task_manager, "register_task"),
    ):
        # Note: this test does NOT use the mock_schedule_bg fixture
        # because we're testing _schedule_bg itself. The real
        # function runs with the deeper layers patched.
        bg_task = await _schedule_bg(
            task=task,
            user=user,
            payload=TaskTurnPayload("x"),
            force_fresh=False,
            context=None,
        )
        await bg_task

    mock_exec.assert_not_awaited()
    mock_finish.assert_not_called()


@pytest.mark.asyncio
async def test_schedule_bg_releases_lease_on_execute_task_background_exception(
    db_session,
) -> None:
    """Lease must not leak when execute_task_background raises — _runner.finally
    must still call release_current_runner_task_lease."""
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.services.task_lease_service import TaskLease

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    fake_lease = TaskLease(task_id=int(task.id), runner_id="test-runner")

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
        patch(
            "xagent.web.services.task_orchestrator.release_current_runner_task_lease",
        ) as mock_release,
        patch(
            "xagent.web.services.task_orchestrator.finish_turn",
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = await _schedule_bg(
            task=task,
            user=user,
            payload=TaskTurnPayload("x"),
            force_fresh=False,
            context=None,
        )
        # Wait for the inner _runner to finish (which raises internally
        # but the wrapping create_task absorbs it). The release should
        # still have been called in _runner.finally.
        try:
            await bg_task
        except RuntimeError:
            pass

    mock_release.assert_called_once()


@pytest.mark.asyncio
async def test_schedule_bg_forwards_execution_message_to_execute_task_background(
    db_session,
) -> None:
    """Dual-channel payload propagation through the scheduler:
    ``_schedule_bg`` must pass ``payload.execution_message`` to
    ``execute_task_background``'s ``llm_user_message=`` parameter so
    the LLM-facing variant of the turn input survives the orchestrator
    boundary.

    Together with the ``begin_turn → _schedule_bg`` test above this
    locks in the full payload chain
    (begin_turn → _schedule_bg → execute_task_background) at the
    type-signature level, so a future refactor can't silently collapse
    transcript and execution into a single string.
    """
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.services.task_lease_service import TaskLease

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    fake_lease = TaskLease(task_id=int(task.id), runner_id="test-runner")

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(),
        ) as mock_exec,
        patch(
            "xagent.web.services.task_orchestrator.release_current_runner_task_lease",
        ),
        patch(
            "xagent.web.services.task_orchestrator.finish_turn",
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        payload = TaskTurnPayload(
            transcript_message="summarize this",
            execution_message="summarize this\n\n[uploaded file: secret.txt]",
        )
        bg_task = await _schedule_bg(
            task=task,
            user=user,
            payload=payload,
            force_fresh=False,
            context={"turn_id": "caller-turn", "existing": "value"},
        )
        await bg_task

    mock_exec.assert_awaited_once()
    kwargs = mock_exec.await_args.kwargs
    # Dual-channel payload contract: transcript and LLM-facing channels are both
    # forwarded explicitly so execute_task_background can pick the
    # right one for the agent input.
    assert kwargs["user_message"] == "summarize this", (
        "transcript_message must reach execute_task_background.user_message"
    )
    assert (
        kwargs["llm_user_message"] == "summarize this\n\n[uploaded file: secret.txt]"
    ), "execution_message must reach execute_task_background.llm_user_message"
    assert kwargs["context"]["turn_id"] == payload.turn_id
    assert kwargs["context"]["existing"] == "value"
