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
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from threading import Barrier, Event, get_ident
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from tests.shared.db_teardown import drop_all_tables
from tests.web.pool_contention_shared import (
    CONTENTION_POOL_TIMEOUT,
    EXHAUSTION_POOL_TIMEOUT,
    GUARD_TIMEOUT,
    LOOP_LIVENESS_TICKS,
    assert_pool_checkout_off_loop,
    gated_pool_checkout,
    wait_for_ticks,
)
from xagent.core.agent.checkpoint import CHECKPOINT_TYPE
from xagent.core.tools.adapters.vibe.config import RequiredMCPUnavailableError
from xagent.core.tools.adapters.vibe.connector_runtime import (
    ConnectorRef,
    ConnectorRuntimeError,
)
from xagent.web.models import database as database_module
from xagent.web.models.agent import Agent
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerRun,
    TriggerRunStatus,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services import task_orchestrator as task_orchestrator_module
from xagent.web.services.assistant_history_safety import (
    CLIENT_SAFE_FAILURE_MESSAGE_TYPE,
    TASK_FAILURE_MESSAGE_TYPE,
)
from xagent.web.services.chat_history_service import (
    DELIVERY_COMPLETED,
    DELIVERY_DISPATCHED,
    DELIVERY_FAILED,
    DELIVERY_PENDING,
    claim_user_message_delivery,
    inspect_user_message_delivery,
    mark_user_message_delivery,
)
from xagent.web.services.client_error_messages import (
    CLIENT_SAFE_AUTO_MODEL_UNAVAILABLE,
    CLIENT_SAFE_TASK_FAILURE,
)
from xagent.web.services.connector_runtime import (
    get_ephemeral_runtime_values,
    pop_ephemeral_runtime_values,
    store_ephemeral_runtime_values,
)
from xagent.web.services.llm_utils import AutoModelUnavailableError
from xagent.web.services.mcp_runtime import (
    MCPBuiltinOAuthActorPolicy,
    MCPBuiltinOAuthActorPolicyRequiredError,
)
from xagent.web.services.task_execution_controller import task_execution_controller
from xagent.web.services.task_lease_service import (
    TaskLease,
    TaskLeaseHeartbeatOutcome,
    TaskLeaseRefreshState,
    acquire_task_lease_isolated,
    get_runner_id,
)
from xagent.web.services.task_orchestrator import (
    TaskTurnCommitOutcomeUnknown,
    TaskTurnError,
    TaskTurnNotFoundError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
    _begin_turn_atomic_sync,
    _ClaimedTurn,
    _reconcile_finalized_turn_delivery,
    _schedule_bg,
    finish_turn,
    settle_task_lease_isolated,
)
from xagent.web.services.task_runtime import (
    MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY,
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
        drop_all_tables(get_engine())


@pytest.fixture()
def queue_pool_runtime_db_factory(tmp_path):
    """Build one-slot QueuePool engines with a caller-chosen checkout timeout."""
    engines = []

    def _make(*, pool_timeout: float = EXHAUSTION_POOL_TIMEOUT):
        db_path = tmp_path / f"orchestrator-queue-pool-{len(engines)}.db"
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            poolclass=QueuePool,
            pool_size=1,
            max_overflow=0,
            pool_timeout=pool_timeout,
        )
        engines.append(engine)
        Base.metadata.create_all(bind=engine)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return engine, SessionLocal

    try:
        yield _make
    finally:
        for engine in engines:
            engine.dispose()


@pytest.fixture()
def queue_pool_runtime_db(queue_pool_runtime_db_factory):
    """A real one-slot QueuePool used to exercise checkout exhaustion."""
    return queue_pool_runtime_db_factory()


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


def _store_runtime_secret_for_turn(turn_id: str) -> None:
    store_ephemeral_runtime_values(
        turn_id,
        {
            ConnectorRef("mcp", 1): {
                "secrets": {"authorization": "Bearer cleanup-token"}
            }
        },
    )


def test_channel_and_web_claims_are_cross_process_exclusive(db_session) -> None:
    user = _create_user(db_session)
    task = _create_task(db_session, int(user.id))
    task_id = int(task.id)
    barrier = Barrier(2)

    def claim_channel() -> str | None:
        barrier.wait()
        lease = acquire_task_lease_isolated(
            task_id,
            runner_id="channel-runner",
            new_run=True,
        )
        return lease.run_id if lease is not None else None

    def claim_web() -> str | None:
        barrier.wait()
        try:
            claimed = _begin_turn_atomic_sync(
                task_id,
                int(user.id),
                payload=TaskTurnPayload("web turn"),
                kind=TurnKind.CREATE,
            )
        except TaskTurnError:
            return None
        return claimed.run_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        channel_result = executor.submit(claim_channel)
        web_result = executor.submit(claim_web)
        winners = [
            run_id
            for run_id in (channel_result.result(), web_result.result())
            if run_id is not None
        ]

    assert len(winners) == 1
    db_session.expire_all()
    stored = db_session.query(Task).filter(Task.id == task_id).one()
    assert stored.status == TaskStatus.RUNNING
    assert stored.run_id == winners[0]


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
        new=MagicMock(),
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


def test_execution_context_carries_path_stripped_turn_attachments() -> None:
    files = [{"file_id": "image-id", "name": "screen.png", "type": "image/png"}]

    context = task_orchestrator_module._execution_context_with_turn_id(
        {"existing": True},
        "turn-1",
        files=files,
    )

    assert context == {
        "existing": True,
        "turn_id": "turn-1",
        "files": files,
    }
    assert context["files"] is not files


def test_execution_context_preserves_existing_turn_attachments() -> None:
    existing_files = [
        {"file_id": "existing", "name": "existing.png", "type": "image/png"}
    ]

    context = task_orchestrator_module._execution_context_with_turn_id(
        {"files": existing_files},
        "turn-2",
        files=[{"file_id": "new", "name": "new.png", "type": "image/png"}],
    )

    assert context["files"] is existing_files


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
    payload = TaskTurnPayload("first turn")

    started = await TaskTurnOrchestrator.begin_turn(
        task_id=int(task.id),
        payload=payload,
        task_owner_user_id=int(user.id),
        kind=TurnKind.CREATE,
        force_fresh=False,
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.input == "first turn"
    assert task.output is None
    assert task.error_message is None
    assert task.runner_id == get_runner_id()
    assert task.lease_expires_at is not None
    assert task.run_id == started.run_id
    assert task.state_version == started.state_version == 1
    assert task.control_state == started.control_state == "running"
    delivery = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == payload.turn_id)
        .one()
    )
    assert delivery.delivery_status == DELIVERY_DISPATCHED


@pytest.mark.asyncio
async def test_begin_turn_projects_workforce_running_in_claim_transaction(
    db_session,
    mock_schedule_bg,
) -> None:
    user = _create_user(db_session)
    manager = Agent(user_id=user.id, name="claim projection manager")
    db_session.add(manager)
    db_session.flush()
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name="claim projection workforce",
        manager_agent_id=manager.id,
        status="active",
    )
    db_session.add(workforce)
    db_session.flush()
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)
    run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="failed",
        snapshot={"version": 1},
        completed_at=datetime.now(timezone.utc),
    )
    db_session.add(run)
    db_session.flush()
    task.agent_config = {"workforce_run_id": int(run.id)}
    db_session.commit()

    await TaskTurnOrchestrator.begin_turn(
        task_id=int(task.id),
        task_owner_user_id=int(user.id),
        payload=TaskTurnPayload("retry"),
        kind=TurnKind.CREATE,
    )

    db_session.expire_all()
    persisted_run = db_session.get(WorkforceRun, int(run.id))
    assert persisted_run.status == "running"
    assert persisted_run.completed_at is None


@pytest.mark.asyncio
async def test_begin_turn_claim_replaces_stale_lease_and_checkpoint_pointer(
    db_session,
    mock_schedule_bg,
) -> None:
    """Append claim must reset stale lease columns so the bg runner can
    acquire after a crashed worker left runner_id behind."""
    from xagent.web.services.task_lease_service import utc_now

    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user.id,
        status=TaskStatus.COMPLETED,
        input_="done",
        output="answer",
    )
    task.runner_id = "dead-runner-from-crash"
    task.lease_expires_at = utc_now() + timedelta(hours=1)
    task.last_checkpoint_event_id = "previous-run-checkpoint"
    stale_checkpoint = TraceEvent(
        task_id=task.id,
        event_id="previous-run-checkpoint",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={"checkpoint_type": CHECKPOINT_TYPE, "snapshot": {"type": "checkpoint"}},
    )
    db_session.add(stale_checkpoint)
    db_session.flush()
    task.last_checkpoint_trace_event_id = stale_checkpoint.id
    db_session.commit()

    await TaskTurnOrchestrator.begin_turn(
        task_id=int(task.id),
        payload=TaskTurnPayload("follow up"),
        task_owner_user_id=int(user.id),
        kind=TurnKind.APPEND,
        force_fresh=False,
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == get_runner_id()
    assert task.lease_expires_at is not None
    assert task.last_checkpoint_event_id is None
    # Both pointer columns clear together on a turn claim -- see
    # lease_checkpoint_trace_event_id_case's docstring in task_lease_service
    # for why a single column left behind would desync the recovery CAS
    # fence.
    assert task.last_checkpoint_trace_event_id is None


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
        task_id=int(task.id),
        payload=TaskTurnPayload("second question"),
        task_owner_user_id=int(user.id),
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
        task_id=int(task.id),
        payload=TaskTurnPayload("second"),
        task_owner_user_id=int(user.id),
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
        task_id=int(task.id),
        payload=payload,
        task_owner_user_id=int(user.id),
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

    mock_schedule_bg.assert_called_once()


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
        task_id=int(task.id),
        payload=payload,
        task_owner_user_id=int(user.id),
        kind=TurnKind.APPEND,
        force_fresh=True,
    )

    mock_schedule_bg.assert_called_once()
    kwargs = mock_schedule_bg.call_args.kwargs
    assert kwargs["payload"] is payload
    assert kwargs["force_fresh"] is True

    persisted = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == int(task.id), TaskChatMessage.role == "user")
        .one()
    )
    assert persisted.turn_id == payload.turn_id


def test_begin_turn_reconciles_a_commit_acknowledgement_failure(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)
    payload = TaskTurnPayload("acknowledged after disconnect")
    original_commit = Session.commit
    commit_raised = False

    def acknowledge_then_disconnect(session: Session) -> None:
        nonlocal commit_raised
        original_commit(session)
        if session is not db_session and not commit_raised:
            commit_raised = True
            raise ConnectionError("commit acknowledgement lost")

    monkeypatch.setattr(Session, "commit", acknowledge_then_disconnect)

    claimed = _begin_turn_atomic_sync(
        int(task.id),
        int(user.id),
        payload=payload,
        kind=TurnKind.CREATE,
    )

    assert commit_raised is True
    assert claimed.run_id
    db_session.expire_all()
    stored = db_session.query(TaskChatMessage).filter_by(turn_id=payload.turn_id).one()
    assert stored.content == "acknowledged after disconnect"


def test_begin_turn_retires_failed_session_before_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    commit_error = ConnectionError("commit acknowledgement lost")
    lease = MagicMock(runner_id="runner", run_id="run")
    claimed = _ClaimedTurn(
        task_lease=lease,
        status=TaskStatus.RUNNING,
        updated_at=None,
        before_message_id=None,
        task_source="sdk",
        run_id="run",
    )

    class FailedSession:
        def flush(self) -> None:
            events.append("flush")

        def commit(self) -> None:
            events.append("commit")
            raise commit_error

        def rollback(self) -> None:
            events.append("rollback")

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(
        database_module,
        "get_session_local",
        lambda: lambda: FailedSession(),
    )
    monkeypatch.setattr(
        task_orchestrator_module,
        "_claim_turn_no_commit",
        MagicMock(return_value=claimed),
    )

    def reconcile(**_kwargs) -> bool:
        events.append("reconcile")
        assert events.index("close") < events.index("reconcile")
        return True

    monkeypatch.setattr(
        task_orchestrator_module,
        "_reconcile_claimed_turn_after_commit_ack_failure",
        reconcile,
    )

    result = _begin_turn_atomic_sync(
        123,
        456,
        payload=TaskTurnPayload("accepted", turn_id="retire-before-reconcile"),
        kind=TurnKind.CREATE,
    )

    assert result is claimed
    assert events[:4] == ["flush", "commit", "close", "reconcile"]


def test_commit_reconciliation_accepts_a_late_visible_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = MagicMock(runner_id="runner", run_id="run")
    claimed = _ClaimedTurn(
        task_lease=lease,
        status=TaskStatus.RUNNING,
        updated_at=None,
        before_message_id=None,
        task_source="sdk",
        run_id="run",
    )
    payload = TaskTurnPayload("accepted", turn_id="late-visible-turn")
    sessions = [MagicMock(), MagicMock(), MagicMock()]
    sessions[0].query.return_value.filter.return_value.first.return_value = None
    sessions[1].query.return_value.filter.return_value.first.return_value = None
    task_query = MagicMock()
    task_query.filter.return_value.first.return_value = MagicMock()
    message_query = MagicMock()
    message_query.filter.return_value.first.return_value = MagicMock()
    third_queries = [task_query, message_query]
    sessions[2].query.side_effect = third_queries
    created_sessions: list[MagicMock] = []

    def session_factory() -> MagicMock:
        session = sessions[len(created_sessions)]
        created_sessions.append(session)
        return session

    monkeypatch.setattr(
        database_module,
        "get_session_local",
        lambda: session_factory,
    )
    monkeypatch.setattr(task_orchestrator_module.time, "sleep", MagicMock())

    assert (
        task_orchestrator_module._reconcile_claimed_turn_after_commit_ack_failure(
            task_id=123,
            task_owner_user_id=456,
            payload=payload,
            claimed=claimed,
        )
        is True
    )
    assert len(created_sessions) == 3
    assert all(session.close.called for session in created_sessions)


def test_successful_commit_is_not_rejected_when_session_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    claimed = _ClaimedTurn(
        task_lease=MagicMock(runner_id="runner", run_id="run"),
        status=TaskStatus.RUNNING,
        updated_at=None,
        before_message_id=None,
        task_source="sdk",
        run_id="run",
    )

    class CloseFailingSession:
        def flush(self) -> None:
            events.append("flush")

        def commit(self) -> None:
            events.append("commit")

        def rollback(self) -> None:
            events.append("rollback")

        def close(self) -> None:
            events.append("close")
            raise RuntimeError("close failed")

        def invalidate(self) -> None:
            events.append("invalidate")

    monkeypatch.setattr(
        database_module,
        "get_session_local",
        lambda: lambda: CloseFailingSession(),
    )
    monkeypatch.setattr(
        task_orchestrator_module,
        "_claim_turn_no_commit",
        MagicMock(return_value=claimed),
    )
    invalidate_cache = MagicMock()
    monkeypatch.setattr(
        task_orchestrator_module,
        "invalidate_task_cache_best_effort",
        invalidate_cache,
    )

    result = _begin_turn_atomic_sync(
        123,
        456,
        payload=TaskTurnPayload("accepted", turn_id="close-failure-turn"),
        kind=TurnKind.CREATE,
    )

    assert result is claimed
    assert events == ["flush", "commit", "close", "invalidate"]
    invalidate_cache.assert_called_once_with(123)


def test_reconciliation_read_failure_remains_commit_outcome_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit_error = ConnectionError("commit acknowledgement lost")
    claimed = _ClaimedTurn(
        task_lease=MagicMock(runner_id="runner", run_id="run"),
        status=TaskStatus.RUNNING,
        updated_at=None,
        before_message_id=None,
        task_source="sdk",
        run_id="run",
    )
    factory_calls = 0

    class FailedCommitSession:
        def flush(self) -> None:
            return None

        def commit(self) -> None:
            raise commit_error

        def rollback(self) -> None:
            return None

        def close(self) -> None:
            return None

    def session_factory():
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            return FailedCommitSession()
        raise RuntimeError("reconciliation database unavailable")

    monkeypatch.setattr(
        database_module,
        "get_session_local",
        lambda: session_factory,
    )
    monkeypatch.setattr(
        task_orchestrator_module,
        "_claim_turn_no_commit",
        MagicMock(return_value=claimed),
    )
    monkeypatch.setattr(task_orchestrator_module.time, "sleep", MagicMock())

    with pytest.raises(TaskTurnCommitOutcomeUnknown) as exc_info:
        _begin_turn_atomic_sync(
            123,
            456,
            payload=TaskTurnPayload(
                "accepted",
                turn_id="reconciliation-read-failure",
            ),
            kind=TurnKind.CREATE,
        )

    assert exc_info.value.__cause__ is commit_error
    assert factory_calls == 4


@pytest.mark.asyncio
async def test_begin_turn_actor_is_audit_only_not_runtime(
    db_session,
    mock_schedule_bg,
    caplog,
) -> None:
    """``actor_user_id`` records who initiated the turn (an admin acting on
    another user's task) for audit/logging only. It must never reach the
    claim or the bg schedule -- those run as the OWNER -- and it must appear
    in the audit log line."""
    owner = _create_user(db_session)
    task = _create_task(db_session, owner.id, status=TaskStatus.COMPLETED)
    actor_id = int(owner.id) + 999  # a different principal (e.g. an admin)

    with caplog.at_level(logging.INFO, logger="xagent.web.services.task_orchestrator"):
        await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            payload=TaskTurnPayload("follow-up"),
            task_owner_user_id=int(owner.id),
            actor_user_id=actor_id,
            kind=TurnKind.APPEND,
        )

    # The bg schedule runs as the owner; the actor never leaks into it.
    kwargs = mock_schedule_bg.call_args.kwargs
    assert kwargs["task_owner_user_id"] == int(owner.id)
    assert actor_id not in kwargs.values()

    # The persisted user message is attributed to the owner, not the actor.
    persisted = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == int(task.id), TaskChatMessage.role == "user")
        .one()
    )
    assert persisted.user_id == int(owner.id)

    # The actor is captured in the audit log.
    assert "turn started" in caplog.text
    assert f"owner={int(owner.id)}" in caplog.text
    assert f"actor={actor_id}" in caplog.text


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
            task_id=int(task.id),
            payload=TaskTurnPayload("x"),
            task_owner_user_id=int(user.id),
            kind=TurnKind.CREATE,
            force_fresh=True,
        )


@pytest.mark.asyncio
async def test_begin_turn_rejects_task_not_owned_by_user(
    db_session,
    mock_schedule_bg,
) -> None:
    """Ownership is folded into the atomic claim predicate. A ``user_id``
    that does not own the task → ``TaskTurnNotFoundError`` (404), NOT
    ``TaskTurnError`` (409), and no row is mutated. Passing a *different*
    user id (not ``task.user_id``) proves the predicate actually guards."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)

    with pytest.raises(TaskTurnNotFoundError):
        await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            task_owner_user_id=int(user.id) + 9999,
            payload=TaskTurnPayload("x"),
            kind=TurnKind.CREATE,
        )

    db_session.refresh(task)
    assert task.status == TaskStatus.PENDING, "rejected claim must not mutate"
    mock_schedule_bg.assert_not_called()


@pytest.mark.asyncio
async def test_begin_turn_marks_failed_when_schedule_raises(
    db_session,
) -> None:
    """Post-commit invariant: once the claim commits (RUNNING) but
    ``_schedule_bg`` raises, the task must be forced FAILED so it is never
    left RUNNING with no bg worker (zombie)."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)
    payload = TaskTurnPayload("x")

    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(side_effect=RuntimeError("schedule boom")),
    ):
        with pytest.raises(RuntimeError, match="schedule boom"):
            await TaskTurnOrchestrator.begin_turn(
                task_id=int(task.id),
                task_owner_user_id=int(user.id),
                payload=payload,
                kind=TurnKind.CREATE,
            )

    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    delivery = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == payload.turn_id)
        .one()
    )
    assert delivery.delivery_status == DELIVERY_FAILED


def test_schedule_failure_compensation_does_not_fail_foreign_live_lease(
    db_session,
) -> None:
    """Compensation only owns the still-unleased claim it just committed.

    Another scheduler may acquire the same run before the failed scheduler's
    compensation worker reaches the database.  Matching the run alone must not
    let that stale compensation overwrite the live runner's execution.
    """
    user = _create_user(db_session)
    manager = Agent(user_id=user.id, name="compensation manager")
    db_session.add(manager)
    db_session.flush()
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name="compensation workforce",
        manager_agent_id=manager.id,
        status="active",
    )
    db_session.add(workforce)
    db_session.flush()
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="running",
        snapshot={"version": 1},
    )
    db_session.add(run)
    db_session.flush()
    task.agent_config = {"workforce_run_id": int(run.id)}
    task.run_id = "claimed-run"
    task.runner_id = "replacement-runner"
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    task.error_message = "replacement still running"
    db_session.commit()

    assert (
        settle_task_lease_isolated(
            TaskLease(
                task_id=int(task.id),
                runner_id="stale-runner",
                run_id="claimed-run",
            ),
            error_message="stale schedule failure",
        )
        is False
    )

    db_session.expire_all()
    persisted = db_session.query(Task).filter(Task.id == task.id).one()
    assert persisted.status == TaskStatus.RUNNING
    assert persisted.run_id == "claimed-run"
    assert persisted.runner_id == "replacement-runner"
    assert persisted.error_message == "replacement still running"
    persisted_run = (
        db_session.query(WorkforceRun).filter(WorkforceRun.id == run.id).one()
    )
    assert persisted_run.status == "running"


def test_error_settlement_releases_terminal_lease_without_reporting_failure(
    db_session,
) -> None:
    """A post-commit presentation error must not rewrite or announce COMPLETED."""

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.COMPLETED)
    task.run_id = "completed-run"
    task.runner_id = "completed-runner"
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(user.id),
            role="assistant",
            content="durable result",
            message_type="assistant_message",
        )
    )
    db_session.commit()

    error_committed = settle_task_lease_isolated(
        TaskLease(
            task_id=int(task.id),
            runner_id="completed-runner",
            run_id="completed-run",
        ),
        error_message="completion broadcast failed",
    )

    assert error_committed is False
    db_session.expire_all()
    persisted = db_session.query(Task).filter(Task.id == task.id).one()
    assert persisted.status == TaskStatus.COMPLETED
    assert persisted.output == "durable result"
    assert persisted.error_message is None
    assert persisted.runner_id is None
    assert persisted.lease_expires_at is None


def test_error_settlement_persists_client_safe_history(db_session) -> None:
    secret = "orchestrator-provider-secret"
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task_id = int(task.id)
    lease = acquire_task_lease_isolated(task_id)
    assert lease is not None

    assert (
        settle_task_lease_isolated(
            lease,
            error_message=f"setup/run failed: {secret}",
        )
        is True
    )

    db_session.expire_all()
    persisted = db_session.get(Task, task_id)
    assert persisted is not None
    assert persisted.error_message == f"setup/run failed: {secret}"
    messages = (
        db_session.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == task_id,
            TaskChatMessage.role == "assistant",
        )
        .all()
    )
    assert len(messages) == 1
    assert messages[0].content == "Task execution failed."
    assert secret not in messages[0].content


@pytest.mark.asyncio
async def test_begin_turn_dispatch_timeout_does_not_reject_scheduled_turn(
    db_session,
    mock_schedule_bg,
    caplog,
) -> None:
    """A post-schedule delivery write is best-effort under pool exhaustion.

    The claim is already committed and the background task is already
    scheduled.  Reporting the checkout timeout to the API would tell the
    caller that a turn which is actually running was rejected.
    """
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)
    payload = TaskTurnPayload("start once")

    caplog.set_level(logging.ERROR, logger="xagent.web.services.task_orchestrator")
    with patch(
        "xagent.web.services.task_orchestrator.mark_user_message_delivery_sync",
        side_effect=SQLAlchemyTimeoutError("delivery pool exhausted"),
    ) as mark_delivery:
        started = await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            payload=payload,
            kind=TurnKind.CREATE,
        )

    assert started.task_id == int(task.id)
    mock_schedule_bg.assert_called_once()
    mark_delivery.assert_called_once_with(
        int(task.id), payload.turn_id, DELIVERY_DISPATCHED
    )
    db_session.expire_all()
    stored_task = db_session.query(Task).filter(Task.id == task.id).one()
    assert stored_task.status == TaskStatus.RUNNING
    delivery = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == payload.turn_id)
        .one()
    )
    assert delivery.delivery_status == DELIVERY_PENDING
    assert "component=turn-delivery database pool checkout timed out" in caplog.text


@pytest.mark.asyncio
async def test_begin_turn_dispatch_projection_failure_keeps_scheduled_turn(
    db_session: Session,
    mock_schedule_bg: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed delivery projection cannot undo a successful schedule.

    The durable task claim and background handle are already committed facts.
    A non-pool failure while projecting ``dispatched`` must leave the turn
    running, return the original handle, and avoid compensation or reschedule.
    """
    user = _create_user(db_session)
    task = _create_task(db_session, int(user.id), status=TaskStatus.PENDING)
    payload = TaskTurnPayload("start exactly once")

    caplog.set_level(logging.ERROR, logger="xagent.web.services.task_orchestrator")
    with (
        patch(
            "xagent.web.services.task_orchestrator.mark_user_message_delivery_sync",
            side_effect=RuntimeError("delivery projection failed"),
        ) as mark_delivery,
        patch(
            "xagent.web.services.task_orchestrator.settle_task_lease_isolated"
        ) as compensate,
    ):
        started = await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            payload=payload,
            kind=TurnKind.CREATE,
        )

    assert started.background_task is mock_schedule_bg.return_value
    mock_schedule_bg.assert_called_once()
    mark_delivery.assert_called_once_with(
        int(task.id), payload.turn_id, DELIVERY_DISPATCHED
    )
    compensate.assert_not_called()
    db_session.expire_all()
    stored_task = db_session.query(Task).filter(Task.id == task.id).one()
    assert stored_task.status == TaskStatus.RUNNING
    delivery = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == payload.turn_id)
        .one()
    )
    assert delivery.delivery_status == DELIVERY_PENDING
    assert "component=turn-delivery projection failed after scheduling" in caplog.text


@pytest.mark.asyncio
async def test_schedule_failure_pool_timeout_does_not_checkout_delivery_again(
    db_session,
    caplog,
) -> None:
    """A failed terminal write is the last checkout after schedule failure."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)
    payload = TaskTurnPayload("cannot schedule")
    mark_delivery = MagicMock()
    settle = MagicMock(side_effect=SQLAlchemyTimeoutError("terminal pool exhausted"))

    caplog.set_level(logging.ERROR, logger="xagent.web.services.task_orchestrator")
    with (
        patch(
            "xagent.web.services.task_orchestrator._schedule_bg",
            side_effect=RuntimeError("schedule boom"),
        ),
        patch(
            "xagent.web.services.task_orchestrator.settle_task_lease_isolated",
            settle,
        ),
        patch(
            "xagent.web.services.task_orchestrator.mark_user_message_delivery_sync",
            mark_delivery,
        ),
        pytest.raises(RuntimeError, match="schedule boom"),
    ):
        await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            payload=payload,
            kind=TurnKind.CREATE,
        )

    settle.assert_called_once()
    mark_delivery.assert_not_called()
    assert (
        "component=turn-schedule-terminal database pool checkout timed out"
        in caplog.text
    )


@pytest.mark.asyncio
async def test_begin_turn_schedules_even_when_caller_cancelled(db_session) -> None:
    """Cancellation safety: if begin_turn's caller is cancelled while the
    off-loop claim is in flight (which commits RUNNING in a worker thread),
    the owned claim+schedule task must settle before cancellation propagates,
    so a committed RUNNING task is never left with no scheduled worker."""
    import threading

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)

    claim_started = threading.Event()
    release_claim = threading.Event()

    def slow_claim(task_id, task_owner_user_id, *, payload, kind):
        claim_started.set()
        assert release_claim.wait(timeout=GUARD_TIMEOUT)
        return _ClaimedTurn(
            task_lease=TaskLease(
                task_id=task_id,
                runner_id=get_runner_id(),
                run_id="slow-claim-run",
            ),
            status=TaskStatus.RUNNING,
            updated_at=datetime.now(timezone.utc),
            before_message_id=1,
            task_source="sdk",
            run_id="slow-claim-run",
        )

    sched = MagicMock(return_value=MagicMock())
    with (
        patch(
            "xagent.web.services.task_orchestrator._begin_turn_atomic_sync",
            new=slow_claim,
        ),
        patch(
            "xagent.web.services.task_orchestrator._schedule_bg",
            new=sched,
        ),
    ):
        t = asyncio.create_task(
            TaskTurnOrchestrator.begin_turn(
                task_id=int(task.id),
                task_owner_user_id=int(user.id),
                payload=TaskTurnPayload("x"),
                kind=TurnKind.CREATE,
            )
        )
        try:
            assert await asyncio.to_thread(claim_started.wait, GUARD_TIMEOUT)
            t.cancel()
        finally:
            release_claim.set()
            await asyncio.wait_for(
                asyncio.gather(t, return_exceptions=True), timeout=GUARD_TIMEOUT
            )
        with pytest.raises(asyncio.CancelledError):
            t.result()

    sched.assert_called_once()  # scheduled despite the cancellation


@pytest.mark.asyncio
async def test_repeated_cancellation_keeps_turn_command_gate_until_claim_settles(
    db_session,
) -> None:
    """A second cancellation cannot release the per-task command gate while
    the owned claim is still committing in its worker thread."""
    import threading

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)
    claim_started = threading.Event()
    release_claim = threading.Event()
    contender_entered = asyncio.Event()

    def blocked_claim(task_id, task_owner_user_id, *, payload, kind):
        claim_started.set()
        assert release_claim.wait(timeout=2)
        return _ClaimedTurn(
            task_lease=TaskLease(
                task_id=task_id,
                runner_id=get_runner_id(),
                run_id="blocked-claim-run",
            ),
            status=TaskStatus.RUNNING,
            updated_at=datetime.now(timezone.utc),
            before_message_id=1,
            task_source="sdk",
            run_id="blocked-claim-run",
        )

    async def contender() -> None:
        async with task_execution_controller.command(int(task.id)):
            contender_entered.set()

    sched = MagicMock(return_value=MagicMock())
    with (
        patch(
            "xagent.web.services.task_orchestrator._begin_turn_atomic_sync",
            new=blocked_claim,
        ),
        patch("xagent.web.services.task_orchestrator._schedule_bg", new=sched),
    ):
        turn = asyncio.create_task(
            TaskTurnOrchestrator.begin_turn(
                task_id=int(task.id),
                task_owner_user_id=int(user.id),
                payload=TaskTurnPayload("x"),
                kind=TurnKind.CREATE,
            )
        )
        while not claim_started.is_set():
            await asyncio.sleep(0)

        turn.cancel()
        await asyncio.sleep(0.01)
        turn.cancel()
        contender_task = asyncio.create_task(contender())
        await asyncio.sleep(0.05)

        assert not turn.done()
        assert not contender_entered.is_set()

        release_claim.set()
        with pytest.raises(asyncio.CancelledError):
            await turn
        await asyncio.wait_for(contender_task, timeout=1)

    assert contender_entered.is_set()
    sched.assert_called_once()


@pytest.mark.asyncio
async def test_schedule_claimed_create_turn_offloads_cache_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committed domain claim must not invalidate Redis on the event loop."""
    task_id = 987654
    event_loop_thread = get_ident()
    invalidations: list[tuple[int, int]] = []
    entered = Event()
    release = Event()

    def slow_invalidate(observed_task_id: int) -> None:
        invalidations.append((observed_task_id, get_ident()))
        entered.set()
        assert get_ident() != event_loop_thread
        assert release.wait(timeout=30), "cache invalidation was never released"

    async def fake_schedule(**_kwargs):
        async def done() -> None:
            return None

        return asyncio.create_task(done())

    monkeypatch.setattr(
        task_orchestrator_module,
        "invalidate_task_cache",
        slow_invalidate,
    )
    monkeypatch.setattr(
        task_orchestrator_module,
        "_schedule_committed_turn",
        fake_schedule,
    )
    claimed = _ClaimedTurn(
        task_lease=TaskLease(
            task_id=task_id,
            runner_id=get_runner_id(),
            run_id="committed-run",
        ),
        status=TaskStatus.RUNNING,
        updated_at=None,
        before_message_id=None,
        task_source="internal",
        run_id="committed-run",
    )

    startup = asyncio.create_task(
        TaskTurnOrchestrator.schedule_claimed_create_turn(
            task_id=task_id,
            task_owner_user_id=1,
            actor_user_id=1,
            payload=TaskTurnPayload("start"),
            claimed=claimed,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 30)
        assert not startup.done()
    finally:
        release.set()
        started = await asyncio.wait_for(startup, timeout=30)
    await started.background_task

    assert len(invalidations) == 1
    assert invalidations[0][0] == task_id
    assert invalidations[0][1] != event_loop_thread


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
                task_id=int(task.id),
                payload=TaskTurnPayload("x"),
                task_owner_user_id=int(user.id),
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
            task_id=int(task.id),
            payload=TaskTurnPayload("x"),
            task_owner_user_id=int(user.id),
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
            task_id=int(task.id),
            payload=TaskTurnPayload("x"),
            task_owner_user_id=int(user.id),
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


def test_finish_turn_does_not_touch_a_new_run_owned_by_same_process(
    db_session,
) -> None:
    """The concrete lease, not the process-global runner id, fences cleanup.

    A worker process can finish an old coroutine after the same process has
    already claimed a newer run.  Matching only ``get_runner_id()`` would let
    the stale coroutine fail the new run because both leases share that value.
    """
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task.runner_id = get_runner_id()
    task.run_id = "new-run"
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    db_session.commit()

    stale_lease = TaskLease(
        task_id=int(task.id),
        runner_id=get_runner_id(),
        run_id="old-run",
    )
    finish_turn(db_session, int(task.id), task_lease=stale_lease)

    db_session.expire_all()
    persisted = db_session.query(Task).filter(Task.id == task.id).one()
    assert persisted.status == TaskStatus.RUNNING
    assert persisted.runner_id == get_runner_id()
    assert persisted.run_id == "new-run"
    assert persisted.error_message is None


def test_finish_turn_cache_invalidation_failure_is_non_fatal_after_release(
    db_session,
) -> None:
    """A committed exact-lease settlement must not be reported as failed."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.FAILED)
    task.runner_id = "settlement-runner"
    task.run_id = "settlement-run"
    task.error_message = "execution failed"
    db_session.commit()
    lease = TaskLease(
        task_id=int(task.id),
        runner_id="settlement-runner",
        run_id="settlement-run",
    )

    with patch(
        "xagent.web.services.task_orchestrator.invalidate_task_cache",
        side_effect=RuntimeError("cache backend unavailable"),
    ):
        assert finish_turn(db_session, int(task.id), task_lease=lease) is True

    db_session.expire_all()
    persisted = db_session.query(Task).filter(Task.id == task.id).one()
    assert persisted.status == TaskStatus.FAILED
    assert persisted.runner_id is None
    assert persisted.lease_expires_at is None


def test_finish_turn_mirrors_failed_task_to_trigger_run(db_session) -> None:
    """The existing terminal-state owner mirrors setup failures to TriggerRun."""
    user = _create_user(db_session)
    agent = Agent(user_id=user.id, name="trigger agent")
    db_session.add(agent)
    db_session.flush()
    trigger = AgentTrigger(
        user_id=user.id,
        agent_id=agent.id,
        type=TriggerType.SCHEDULED.value,
        name="scheduled test",
        config={},
    )
    db_session.add(trigger)
    db_session.flush()
    task = _create_task(
        db_session,
        user.id,
        status=TaskStatus.FAILED,
        error_message="Required MCP servers are unavailable.",
    )
    task.source = "trigger"
    run = TriggerRun(
        trigger_id=trigger.id,
        task_id=task.id,
        status=TriggerRunStatus.RUNNING.value,
        idempotency_key="strict-mcp-setup-failure",
    )
    db_session.add_all([task, run])
    db_session.commit()

    finish_turn(db_session, int(task.id))

    db_session.refresh(run)
    assert run.status == TriggerRunStatus.FAILED.value
    assert run.error_message == "Required MCP servers are unavailable."


# ---------------------------------------------------------------------------
# _schedule_bg lease lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_bg_cancels_execution_after_heartbeat_loses_lease(
    db_session,
) -> None:
    from xagent.web.api.websocket import background_task_manager

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task.runner_id = "test-runner"
    task.run_id = "run-a"
    db_session.commit()
    lease = TaskLease(int(task.id), "test-runner", "run-a")
    execution_started = asyncio.Event()
    execution_cancelled = asyncio.Event()

    async def heartbeat(*_args, **_kwargs) -> TaskLeaseHeartbeatOutcome:
        await execution_started.wait()
        return TaskLeaseHeartbeatOutcome(lease_lost=True)

    async def execute(**_kwargs) -> None:
        execution_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            execution_cancelled.set()

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=heartbeat,
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            return_value=MagicMock(),
        ),
        patch.object(
            task_orchestrator_module,
            "resolve_execution_scope",
            return_value=None,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=execute,
        ),
        patch(
            "xagent.web.services.task_orchestrator.settle_task_lease_isolated",
        ) as settle,
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        await _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            run_id="run-a",
            payload=TaskTurnPayload("hello"),
            force_fresh=False,
            context=None,
        )

    assert execution_cancelled.is_set()
    settle.assert_not_called()


@pytest.mark.asyncio
async def test_delayed_preclaimed_scheduler_does_not_resurrect_recovered_task(
    db_session,
) -> None:
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.services.task_lease_recovery import (
        recover_expired_task_leases_until_cutoff,
    )
    from xagent.web.services.task_lease_service import utc_now

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)
    payload = TaskTurnPayload("claimed before scheduler delay")
    claimed = _begin_turn_atomic_sync(
        int(task.id),
        int(user.id),
        payload=payload,
        kind=TurnKind.CREATE,
    )
    db_session.expire_all()
    persisted = db_session.get(Task, int(task.id))
    persisted.lease_expires_at = utc_now() - timedelta(seconds=1)
    db_session.commit()

    assert (
        await recover_expired_task_leases_until_cutoff(
            cutoff=utc_now(),
            batch_size=10,
        )
        == 1
    )

    with (
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(),
        ) as execute,
        patch(
            "xagent.web.services.task_orchestrator.settle_task_lease_isolated",
        ) as settle,
        patch.object(background_task_manager, "register_task"),
    ):
        await _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            run_id=claimed.run_id,
            task_lease=claimed.task_lease,
            payload=payload,
            force_fresh=False,
            context=None,
        )

    execute.assert_not_awaited()
    settle.assert_not_called()
    db_session.expire_all()
    assert db_session.get(Task, int(task.id)).status == TaskStatus.FAILED


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

    payload = TaskTurnPayload("x")
    _store_runtime_secret_for_turn(payload.turn_id)
    assert get_ephemeral_runtime_values(payload.turn_id) is not None

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
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
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=payload,
            force_fresh=False,
            context=None,
        )
        await bg_task

    mock_exec.assert_not_awaited()
    mock_finish.assert_not_called()
    assert get_ephemeral_runtime_values(payload.turn_id) is None
    assert pop_ephemeral_runtime_values(payload.turn_id) is None


@pytest.mark.asyncio
async def test_schedule_bg_resolves_scope_off_loop_before_execution(
    db_session,
    queue_pool_runtime_db_factory,
) -> None:
    """A contended scope lookup must not block the main event loop."""
    from xagent.web.api.websocket import background_task_manager

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task_id = int(task.id)
    fake_lease = TaskLease(task_id=task_id, runner_id="test-runner")
    # The scope lookup must wait for the slot, never give up on it.
    engine, SessionLocal = queue_pool_runtime_db_factory(
        pool_timeout=CONTENTION_POOL_TIMEOUT
    )
    events: list[str] = []
    resolver_threads: list[int] = []
    loop_thread = get_ident()

    def load_snapshot(*_args, **_kwargs):
        events.append("snapshot")
        return MagicMock()

    def resolve_scope(resolved_task_id):
        assert resolved_task_id == task_id
        resolver_threads.append(get_ident())
        with SessionLocal() as scope_db:
            scope_db.execute(text("SELECT 1")).scalar()
        events.append("scope")
        return None

    async def execute(**kwargs):
        events.append("execute")
        assert "resolved_execution_scope" in kwargs
        assert kwargs["resolved_execution_scope"] is None

    held_connection = engine.connect()
    ticker_stop = asyncio.Event()
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while not ticker_stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    with (
        gated_pool_checkout(engine) as gate,
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            side_effect=load_snapshot,
        ),
        patch.object(
            task_orchestrator_module,
            "resolve_execution_scope",
            side_effect=resolve_scope,
            create=True,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=execute,
        ),
        patch(
            "xagent.web.services.task_orchestrator.settle_task_lease_isolated",
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=task_id,
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=TaskTurnPayload("hello"),
            force_fresh=False,
            context=None,
        )
        ticker_task = asyncio.create_task(ticker())
        try:
            await gate.wait_until_contending()
            observed = await wait_for_ticks(lambda: ticks)
            assert observed >= LOOP_LIVENESS_TICKS, (
                "scope QueuePool checkout blocked the event loop"
            )
            assert not bg_task.done(), "execution started before scope resolution"
        finally:
            held_connection.close()
            gate.let_through()
            await asyncio.wait_for(bg_task, timeout=GUARD_TIMEOUT)
            ticker_stop.set()
            await ticker_task

    assert events == ["snapshot", "scope", "execute"]
    assert resolver_threads and resolver_threads[0] != loop_thread


@pytest.mark.parametrize("failure_point", ["missing_snapshot", "execute_error"])
@pytest.mark.asyncio
async def test_schedule_bg_persists_setup_failures_off_loop(
    db_session,
    failure_point,
) -> None:
    from xagent.web.api.websocket import background_task_manager

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task_id = int(task.id)
    fake_lease = TaskLease(task_id=task_id, runner_id="test-runner")
    loop_thread = get_ident()
    marker_threads: list[int] = []
    snapshot = None if failure_point == "missing_snapshot" else MagicMock()
    execute = AsyncMock(
        side_effect=(
            RuntimeError("simulated execute failure")
            if failure_point == "execute_error"
            else None
        )
    )

    def record_failure(*_args, **_kwargs) -> None:
        marker_threads.append(get_ident())

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            return_value=snapshot,
        ),
        patch.object(
            task_orchestrator_module,
            "resolve_execution_scope",
            return_value=None,
            create=True,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=execute,
        ),
        patch(
            "xagent.web.services.task_orchestrator.settle_task_lease_isolated",
            side_effect=record_failure,
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        await _schedule_bg(
            task_id=task_id,
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=TaskTurnPayload("hello"),
            force_fresh=False,
            context=None,
        )

    assert marker_threads and marker_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_schedule_bg_pool_timeout_defers_settlement_to_lease_recovery(
    queue_pool_runtime_db,
    caplog,
) -> None:
    """One exhausted checkout must not trigger an immediate second checkout."""
    from xagent.web.api.websocket import background_task_manager

    engine, SessionLocal = queue_pool_runtime_db
    with SessionLocal() as seed_db:
        user = _create_user(seed_db)
        task = _create_task(seed_db, user.id, status=TaskStatus.RUNNING)
        task_id = int(task.id)
        user_id = int(user.id)
        task_source = task.source
        task.runner_id = "test-runner"
        task.run_id = "run-a"
        seed_db.commit()

    lease = TaskLease(task_id=task_id, runner_id="test-runner", run_id="run-a")

    def load_snapshot_from_contended_pool(*_args, **_kwargs):
        with SessionLocal() as snapshot_db:
            snapshot_db.execute(text("SELECT 1")).scalar()
        return MagicMock()

    held_connection = engine.connect()
    caplog.set_level(
        logging.ERROR,
        logger="xagent.web.services.task_orchestrator",
    )
    try:
        with (
            assert_pool_checkout_off_loop(engine),
            patch(
                "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
                return_value=lease,
            ),
            patch(
                "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.services.task_orchestrator.stop_task_lease_heartbeat",
                new=AsyncMock(),
            ) as mock_stop_heartbeat,
            patch(
                "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
                side_effect=load_snapshot_from_contended_pool,
            ),
            patch(
                "xagent.web.api.websocket.execute_task_background",
                new=AsyncMock(),
            ) as mock_execute,
            patch(
                "xagent.web.services.task_orchestrator.settle_task_lease_isolated",
            ) as mock_settle,
            patch.object(background_task_manager, "register_task"),
            patch(
                "xagent.web.services.task_orchestrator._get_agent_manager",
                return_value=MagicMock(),
            ),
        ):
            await _schedule_bg(
                task_id=task_id,
                task_owner_user_id=user_id,
                task_source=task_source,
                run_id="run-a",
                payload=TaskTurnPayload("hello"),
                force_fresh=False,
                context=None,
            )

        mock_execute.assert_not_awaited()
        mock_stop_heartbeat.assert_awaited_once()
        mock_settle.assert_not_called()
    finally:
        held_connection.close()

    with SessionLocal() as verify_db:
        retained = verify_db.query(Task).filter(Task.id == task_id).one()
        assert retained.status == TaskStatus.RUNNING
        assert retained.runner_id == lease.runner_id
        assert retained.run_id == lease.run_id

    assert f"task_id={task_id}" in caplog.text
    assert "component=setup/run" in caplog.text
    assert "retaining lease for TTL recovery" in caplog.text


@pytest.mark.asyncio
async def test_schedule_bg_heartbeat_pool_timeout_skips_settlement(
    db_session,
) -> None:
    """A heartbeat checkout timeout must not trigger a cleanup checkout."""
    from xagent.web.api.websocket import background_task_manager

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task_id = int(task.id)
    lease = TaskLease(task_id=task_id, runner_id="test-runner", run_id="run-a")
    heartbeat_timeout = SQLAlchemyTimeoutError("heartbeat pool timeout")

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.stop_task_lease_heartbeat",
            new=AsyncMock(
                return_value=TaskLeaseHeartbeatOutcome(pool_timeout=heartbeat_timeout)
            ),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            return_value=MagicMock(),
        ),
        patch.object(
            task_orchestrator_module,
            "resolve_execution_scope",
            return_value=None,
            create=True,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.settle_task_lease_isolated",
        ) as settle,
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        await _schedule_bg(
            task_id=task_id,
            task_owner_user_id=int(user.id),
            task_source=task.source,
            run_id="run-a",
            payload=TaskTurnPayload("hello"),
            force_fresh=False,
            context=None,
        )

    settle.assert_not_called()


@pytest.mark.asyncio
async def test_schedule_bg_settles_owned_terminal_task_observed_by_heartbeat(
    db_session,
    monkeypatch,
) -> None:
    """A terminal commit is settlement-ready, not evidence of lease loss."""
    from xagent.web.api.websocket import background_task_manager

    user = _create_user(db_session)
    agent = Agent(user_id=user.id, name="terminal heartbeat agent")
    db_session.add(agent)
    db_session.flush()
    trigger = AgentTrigger(
        user_id=user.id,
        agent_id=agent.id,
        type=TriggerType.SCHEDULED.value,
        name="terminal heartbeat trigger",
        config={},
    )
    db_session.add(trigger)
    db_session.flush()
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task_id = int(task.id)
    task.source = "trigger"
    task.runner_id = "terminal-runner"
    task.run_id = "terminal-run"
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    run = TriggerRun(
        trigger_id=trigger.id,
        task_id=task.id,
        status=TriggerRunStatus.RUNNING.value,
        idempotency_key="terminal-heartbeat-race",
    )
    db_session.add(run)
    db_session.commit()

    lease = TaskLease(
        task_id=task_id,
        runner_id="terminal-runner",
        run_id="terminal-run",
    )
    terminal_committed = asyncio.Event()
    allow_execution_return = asyncio.Event()

    async def execute_then_wait(**_kwargs) -> None:
        def commit_terminal_status() -> None:
            SessionLocal = database_module.get_session_local()
            with SessionLocal() as terminal_db:
                terminal_task = terminal_db.query(Task).filter(Task.id == task_id).one()
                terminal_task.status = TaskStatus.COMPLETED
                terminal_task.control_state = "completed"
                terminal_db.commit()

        await asyncio.to_thread(commit_terminal_status)
        terminal_committed.set()
        await allow_execution_return.wait()

    monkeypatch.setattr(
        "xagent.web.services.task_lease_service.get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            return_value=MagicMock(),
        ),
        patch.object(
            task_orchestrator_module,
            "resolve_execution_scope",
            return_value=None,
            create=True,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=execute_then_wait,
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=task_id,
            task_owner_user_id=int(user.id),
            task_source=task.source,
            run_id="terminal-run",
            payload=TaskTurnPayload("hello"),
            force_fresh=False,
            context=None,
        )
        await asyncio.wait_for(terminal_committed.wait(), timeout=1)
        await asyncio.sleep(0.02)
        allow_execution_return.set()
        await asyncio.wait_for(bg_task, timeout=1)

    db_session.expire_all()
    persisted_task = db_session.query(Task).filter(Task.id == task_id).one()
    persisted_run = db_session.query(TriggerRun).filter(TriggerRun.id == run.id).one()
    assert persisted_task.status == TaskStatus.COMPLETED
    assert persisted_task.runner_id is None
    assert persisted_task.lease_expires_at is None
    assert persisted_run.status == TriggerRunStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_schedule_bg_releases_lease_without_blocking_pool_checkout(
    queue_pool_runtime_db_factory,
    monkeypatch,
) -> None:
    """Final status read and workforce-aware release share one worker session."""
    from xagent.web.api.websocket import background_task_manager

    # The release path must wait for the slot, never give up on it.
    engine, SessionLocal = queue_pool_runtime_db_factory(
        pool_timeout=CONTENTION_POOL_TIMEOUT
    )
    runner_id = get_runner_id()
    with SessionLocal() as seed_db:
        user = _create_user(seed_db)
        task = _create_task(seed_db, user.id, status=TaskStatus.RUNNING)
        task_id = int(task.id)
        user_id = int(user.id)
        task_source = task.source
        task.runner_id = runner_id
        task.run_id = "run-a"
        seed_db.commit()

    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    held_connection = engine.connect()
    ticker_stop = asyncio.Event()
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while not ticker_stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    with (
        gated_pool_checkout(engine) as gate,
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=TaskLease(
                task_id=task_id,
                runner_id=runner_id,
                run_id="run-a",
            ),
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            return_value=MagicMock(),
        ),
        patch.object(
            task_orchestrator_module,
            "resolve_execution_scope",
            return_value=None,
            create=True,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(),
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=task_id,
            task_owner_user_id=user_id,
            task_source=task_source,
            run_id="run-a",
            payload=TaskTurnPayload("hello"),
            force_fresh=False,
            context=None,
        )
        ticker_task = asyncio.create_task(ticker())
        try:
            await gate.wait_until_contending()
            observed = await wait_for_ticks(lambda: ticks)
            assert observed >= LOOP_LIVENESS_TICKS, (
                "lease release QueuePool checkout blocked the loop"
            )
            assert not bg_task.done(), "lease release skipped the contended checkout"
        finally:
            held_connection.close()
            gate.let_through()
            await asyncio.wait_for(bg_task, timeout=GUARD_TIMEOUT)
            ticker_stop.set()
            await ticker_task

    with SessionLocal() as verify_db:
        released = verify_db.query(Task).filter(Task.id == task_id).one()
        assert released.status == TaskStatus.FAILED
        assert released.run_id == "run-a"
        assert released.runner_id is None


@pytest.mark.asyncio
async def test_schedule_bg_releases_lease_on_execute_task_background_exception(
    db_session,
) -> None:
    """Lease must not leak when execute_task_background raises — _runner.finally
    must still call the lease release + workforce sync helper."""
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.services.task_lease_service import TaskLease

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    fake_lease = TaskLease(task_id=int(task.id), runner_id="test-runner")
    payload = TaskTurnPayload("x")
    _store_runtime_secret_for_turn(payload.turn_id)
    assert get_ephemeral_runtime_values(payload.turn_id) is not None

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
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
            "xagent.web.services.task_orchestrator.settle_task_lease_isolated",
        ) as mock_settle,
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=payload,
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

    mock_settle.assert_called_once()
    assert get_ephemeral_runtime_values(payload.turn_id) is None
    assert pop_ephemeral_runtime_values(payload.turn_id) is None


@pytest.mark.parametrize(
    ("settled", "expected_broadcasts"),
    [(True, 1), (False, 0)],
)
@pytest.mark.asyncio
async def test_schedule_bg_broadcasts_failure_only_after_exact_settlement(
    db_session,
    settled: bool,
    expected_broadcasts: int,
) -> None:
    """A replacement owner must never inherit a stale runner's error event."""
    from xagent.web.api.websocket import background_task_manager

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task_id = int(task.id)
    lease = TaskLease(task_id=task_id, runner_id="runner-a", run_id="run-a")
    events: list[str] = []

    def settle(*_args, **_kwargs) -> bool:
        events.append("settle")
        return settled

    async def broadcast(*_args, **_kwargs) -> None:
        events.append("broadcast")

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            return_value=MagicMock(),
        ),
        patch.object(
            task_orchestrator_module,
            "resolve_execution_scope",
            return_value=None,
            create=True,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(side_effect=RuntimeError("owned run failed")),
        ),
        patch(
            "xagent.web.services.task_orchestrator.settle_task_lease_isolated",
            side_effect=settle,
        ),
        patch(
            "xagent.web.api.websocket.manager",
            MagicMock(broadcast_to_task=AsyncMock(side_effect=broadcast)),
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        await _schedule_bg(
            task_id=task_id,
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=TaskTurnPayload("hello"),
            force_fresh=False,
            context=None,
        )

    assert events == ["settle"] + (["broadcast"] * expected_broadcasts)


@pytest.mark.asyncio
async def test_marked_append_rejects_before_persisting_turn(db_session) -> None:
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.COMPLETED)
    task.agent_config = {MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: True}
    original_run_id = task.run_id
    original_state_version = task.state_version
    db_session.commit()

    with pytest.raises(TaskTurnError, match="actor_task_reuse_unsupported"):
        await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            payload=TaskTurnPayload("generic append"),
            kind=TurnKind.APPEND,
        )

    db_session.expire_all()
    persisted = db_session.get(Task, int(task.id))
    assert persisted is not None
    assert persisted.status == TaskStatus.COMPLETED
    assert persisted.run_id == original_run_id
    assert persisted.state_version == original_state_version
    assert (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == int(task.id))
        .count()
        == 0
    )


@pytest.mark.asyncio
async def test_marked_legacy_execution_rejects_before_scheduling(db_session) -> None:
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.COMPLETED)
    task.agent_config = {MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: True}
    db_session.commit()

    with (
        patch("xagent.web.services.task_orchestrator._schedule_bg") as schedule,
        pytest.raises(
            MCPBuiltinOAuthActorPolicyRequiredError,
            match="legacy execution is unsupported",
        ),
    ):
        await TaskTurnOrchestrator.schedule_existing_task_execution(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=TaskTurnPayload("generic execute"),
        )

    schedule.assert_not_called()
    db_session.expire_all()
    persisted = db_session.get(Task, int(task.id))
    assert persisted is not None
    assert persisted.status == TaskStatus.COMPLETED
    assert persisted.runner_id is None


@pytest.mark.asyncio
async def test_trusted_marked_create_schedule_forwards_actor_policy() -> None:
    policy = MCPBuiltinOAuthActorPolicy(resource_owner_key="actor:alice")
    lease = TaskLease(task_id=42, runner_id="trusted-direct", run_id="run-42")
    claimed = _ClaimedTurn(
        task_lease=lease,
        status=TaskStatus.RUNNING,
        updated_at=None,
        before_message_id=None,
        task_source="external",
        run_id="run-42",
        agent_config={MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: True},
    )

    async def completed() -> None: ...

    background = asyncio.create_task(completed())
    with (
        patch(
            "xagent.web.services.task_orchestrator.invalidate_task_cache_best_effort"
        ),
        patch(
            "xagent.web.services.task_orchestrator._schedule_bg",
            return_value=background,
        ) as schedule,
    ):
        started = await TaskTurnOrchestrator.schedule_claimed_create_turn(
            task_id=42,
            task_owner_user_id=7,
            actor_user_id=7,
            payload=TaskTurnPayload("fresh actor message"),
            claimed=claimed,
            mcp_runtime_authorization_policy=policy,
        )

    assert started.background_task is background
    assert schedule.call_args.kwargs["mcp_runtime_authorization_policy"] is policy
    await background


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
    payload = TaskTurnPayload(
        transcript_message="summarize this",
        execution_message="summarize this\n\n[uploaded file: secret.txt]",
    )
    _store_runtime_secret_for_turn(payload.turn_id)
    assert get_ephemeral_runtime_values(payload.turn_id) is not None
    actor_policy = MCPBuiltinOAuthActorPolicy(resource_owner_key="actor:alice")
    agent_manager = MagicMock()

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
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
            "xagent.web.services.task_orchestrator.settle_task_lease_isolated",
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=agent_manager,
        ),
    ):
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=payload,
            force_fresh=False,
            context={"turn_id": "caller-turn", "existing": "value"},
            mcp_runtime_authorization_policy=actor_policy,
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
    assert kwargs["mcp_runtime_authorization_policy"] is actor_policy
    agent_manager.remove_agent.assert_called_once_with(
        int(task.id),
        int(user.id),
        expected_run_id=fake_lease.run_id,
    )
    assert get_ephemeral_runtime_values(payload.turn_id) is None
    assert pop_ephemeral_runtime_values(payload.turn_id) is None


@pytest.mark.asyncio
async def test_schedule_bg_acquires_expired_lease_on_first_try(db_session) -> None:
    """Expired lease columns are granted by acquire_task_lease's atomic WHERE."""
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.services.task_lease_service import utc_now

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task.runner_id = "dead-runner"
    task.lease_expires_at = utc_now() - timedelta(seconds=5)
    db_session.commit()

    fake_snapshot = MagicMock()

    with (
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            return_value=fake_snapshot,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(),
        ) as mock_exec,
        patch(
            "xagent.web.services.task_orchestrator.settle_task_lease_isolated",
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=TaskTurnPayload("hello"),
            force_fresh=False,
            context=None,
        )
        await bg_task

    mock_exec.assert_awaited_once()


# ---------------------------------------------------------------------------
# _runner setup-error → FAILED safety net: prevents the
# acquire_lease-sets-RUNNING-then-no-one-clears-it zombie state when
# snapshot load or execute_task_background raises.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_bg_marks_task_failed_when_snapshot_load_raises(
    db_session,
) -> None:
    """Snapshot-load exception must not leave the row visible-as-running.

    ``acquire_task_lease_isolated`` writes ``status=RUNNING`` as part
    of taking the lease. Without the outer ``except`` in ``_runner``,
    an exception out of ``load_task_setup_snapshot_sync`` propagates
    through ``_runner``'s inner ``try`` block; ``finish_turn`` and
    ``execute_task_background`` never run, and the outer release
    block reads the still-RUNNING status and writes it back --
    leaving the task displayed as running but with no worker
    executing it.

    The outer ``except`` records the error and the one fenced settlement
    transaction pushes the exact run to ``FAILED`` while releasing it.
    """
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.api.websocket import manager as ws_manager
    from xagent.web.services.task_lease_service import TaskLease

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task.runner_id = "test-runner"
    task.run_id = "run-a"
    db_session.commit()
    fake_lease = TaskLease(
        task_id=int(task.id), runner_id="test-runner", run_id="run-a"
    )
    payload = TaskTurnPayload("x")
    _store_runtime_secret_for_turn(payload.turn_id)
    assert get_ephemeral_runtime_values(payload.turn_id) is not None

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            side_effect=RuntimeError("simulated snapshot load failure"),
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(),
        ) as mock_exec,
        patch.object(background_task_manager, "register_task"),
        patch.object(ws_manager, "broadcast_to_task", new=AsyncMock()) as broadcast,
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=payload,
            force_fresh=False,
            context=None,
        )
        try:
            await bg_task
        except RuntimeError:
            pass

    # execute_task_background must not run when snapshot load raised.
    mock_exec.assert_not_called()
    # The row should now be FAILED, not the zombie RUNNING.
    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED, (
        f"Expected task.status == FAILED after snapshot raise, got {task.status}. "
        "If this fails, the fenced settlement did not close the zombie-RUNNING "
        "window."
    )
    assert task.runner_id is None
    assert task.error_message is not None
    assert "simulated snapshot load failure" in str(task.error_message)
    task_errors = [
        call.args[0]
        for call in broadcast.call_args_list
        if call.args[0].get("type") == "task_error"
    ]
    assert len(task_errors) == 1
    assert task_errors[0]["message"] == "Task execution failed."
    assert task_errors[0]["error"] == "Task execution failed."
    assert "simulated snapshot load failure" not in repr(task_errors[0])
    assert get_ephemeral_runtime_values(payload.turn_id) is None
    assert pop_ephemeral_runtime_values(payload.turn_id) is None


@pytest.mark.asyncio
async def test_schedule_bg_cleanup_handles_missing_payload_turn_id(db_session) -> None:
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.services.task_lease_service import TaskLease

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    fake_lease = TaskLease(task_id=int(task.id), runner_id="test-runner")

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            return_value=MagicMock(),
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(),
        ) as mock_exec,
        patch(
            "xagent.web.services.task_orchestrator.settle_task_lease_isolated",
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=None,  # type: ignore[arg-type]
            force_fresh=False,
            context=None,
        )
        await bg_task

    mock_exec.assert_not_called()


@pytest.mark.asyncio
async def test_schedule_bg_preserves_public_safe_required_mcp_failure(
    db_session,
) -> None:
    """A typed, curated setup failure remains actionable to the client."""
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.api.websocket import manager as ws_manager

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task.runner_id = "test-runner"
    task.run_id = "run-a"
    db_session.commit()
    lease = TaskLease(task_id=int(task.id), runner_id="test-runner", run_id="run-a")
    public_message = "Required MCP servers are unavailable."

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            side_effect=RequiredMCPUnavailableError([]),
        ),
        patch.object(background_task_manager, "register_task"),
        patch.object(ws_manager, "broadcast_to_task", new=AsyncMock()) as broadcast,
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        await _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=TaskTurnPayload("hello"),
            force_fresh=False,
            context=None,
        )

    db_session.expire_all()
    persisted = db_session.get(Task, int(task.id))
    assert persisted is not None
    assert persisted.error_message == public_message
    assistant = (
        db_session.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == task.id,
            TaskChatMessage.role == "assistant",
        )
        .one()
    )
    assert assistant.content == public_message
    assert assistant.message_type == CLIENT_SAFE_FAILURE_MESSAGE_TYPE
    broadcast.assert_awaited_once()
    event = broadcast.await_args.args[0]
    assert event["message"] == public_message
    assert event["error"] == public_message


@pytest.mark.asyncio
async def test_schedule_bg_marks_task_failed_when_execute_raises(
    db_session,
) -> None:
    """An execution exception is persisted by the one fenced settlement."""
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.api.websocket import manager as ws_manager
    from xagent.web.services.task_lease_service import TaskLease

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task.runner_id = "test-runner"
    task.run_id = "run-a"
    db_session.commit()
    fake_lease = TaskLease(
        task_id=int(task.id), runner_id="test-runner", run_id="run-a"
    )

    # Snapshot loader returns a minimal sentinel snapshot so the test
    # proceeds past the snapshot-None branch.
    fake_snapshot = MagicMock()

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            return_value=fake_snapshot,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(side_effect=RuntimeError("simulated agent boom")),
        ),
        patch.object(background_task_manager, "register_task"),
        patch.object(ws_manager, "broadcast_to_task", new=AsyncMock()) as broadcast,
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=TaskTurnPayload("x"),
            force_fresh=False,
            context=None,
        )
        try:
            await bg_task
        except RuntimeError:
            pass

    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    assert task.error_message is not None
    assert "simulated agent boom" in str(task.error_message)
    broadcast.assert_awaited_once()
    event = broadcast.await_args.args[0]
    assert event["message"] == "Task execution failed."
    assert event["error"] == "Task execution failed."
    assert "simulated agent boom" not in repr(event)


@pytest.mark.asyncio
async def test_schedule_bg_does_not_overwrite_terminal_status_from_execute(
    db_session,
) -> None:
    """Fenced settlement never overwrites a committed control status.

    Simulate the inner handler setting PAUSED before raising. After
    ``_runner`` returns, the row must remain PAUSED, not be flipped
    to FAILED by the outer safety net.
    """
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.services.task_lease_service import TaskLease

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task.runner_id = "test-runner"
    task.run_id = "run-a"
    db_session.commit()
    fake_lease = TaskLease(
        task_id=int(task.id), runner_id="test-runner", run_id="run-a"
    )
    fake_snapshot = MagicMock()

    async def fake_execute(*args, **kwargs):
        # Inner handler decides the turn is paused, commits, then a
        # later step raises. Outer except must not undo the PAUSED.
        from xagent.web.models.task import Task as TaskModel

        with sessionmaker(bind=get_engine())() as inner:
            row = inner.query(TaskModel).filter(TaskModel.id == task.id).first()
            row.status = TaskStatus.PAUSED
            inner.commit()
        raise RuntimeError("simulated late-stage error after PAUSED")

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            return_value=fake_snapshot,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=fake_execute,
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=TaskTurnPayload("x"),
            force_fresh=False,
            context=None,
        )
        try:
            await bg_task
        except RuntimeError:
            pass

    db_session.refresh(task)
    assert task.status == TaskStatus.PAUSED, (
        f"Expected PAUSED (set by execute), got {task.status}. If this fails, "
        "the concrete-lease settlement overwrote a control status."
    )
    assert task.runner_id is None


# ---------------------------------------------------------------------------
# Finalize-time delivery reconciliation (#332)
# ---------------------------------------------------------------------------


def _claim_pending_delivery(db, *, task_id: int, user_id: int, turn_id: str) -> None:
    """Create a committed ``pending`` delivery row, mirroring a turn claim."""

    claim = claim_user_message_delivery(
        db,
        task_id,
        user_id,
        "hello there",
        turn_id=turn_id,
    )
    assert claim.claimed is True
    assert claim.pending is True


def _delivery_status(db, turn_id: str) -> str | None:
    db.expire_all()
    row = (
        db.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == turn_id)
        .one_or_none()
    )
    return None if row is None else str(row.delivery_status)


def test_reconcile_finalized_delivery_completes_orphaned_pending_on_success(
    db_session,
) -> None:
    """A clean settlement advances an orphaned ``pending`` row to ``completed``."""
    user = _create_user(db_session)
    task = _create_task(db_session, int(user.id), status=TaskStatus.RUNNING)
    _claim_pending_delivery(
        db_session, task_id=int(task.id), user_id=int(user.id), turn_id="turn-ok"
    )

    _reconcile_finalized_turn_delivery(
        task_id=int(task.id),
        turn_id="turn-ok",
        settlement_error=None,
        execution_started=True,
    )

    assert _delivery_status(db_session, "turn-ok") == DELIVERY_COMPLETED


def test_reconcile_finalized_delivery_fails_orphaned_pending_when_never_ran(
    db_session,
) -> None:
    """A turn that provably never executed advances ``pending`` to ``failed``.

    ``failed`` invites a fresh-id retry downstream, so it is only written with
    positive evidence the message was never consumed
    (``execution_started=False``). That is the row a same-``client_message_id``
    retry needs: a usable terminal ack instead of an in-flight ``PENDING``
    that is rejected forever.
    """
    user = _create_user(db_session)
    task = _create_task(db_session, int(user.id), status=TaskStatus.RUNNING)
    _claim_pending_delivery(
        db_session, task_id=int(task.id), user_id=int(user.id), turn_id="turn-boom"
    )

    _reconcile_finalized_turn_delivery(
        task_id=int(task.id),
        turn_id="turn-boom",
        settlement_error="setup/run error: RuntimeError: boom",
        execution_started=False,
    )

    assert _delivery_status(db_session, "turn-boom") == DELIVERY_FAILED
    # The probe's ``pending`` predicate — what rejected the retry forever — is
    # now False, so the client can converge.
    inspected = inspect_user_message_delivery(
        db_session,
        int(task.id),
        "hello there",
        attachments=None,
        turn_id="turn-boom",
    )
    assert inspected is not None
    assert inspected.pending is False
    assert inspected.failed is True


def test_reconcile_finalized_delivery_marks_dispatched_when_run_failed_after_start(
    db_session,
) -> None:
    """A failure after execution began closes ``pending`` as ``dispatched``.

    The run may have consumed the message (and produced side effects) before
    failing, so writing ``failed`` would invite a fresh-id retry and a double
    execution. ``dispatched`` converges the client without a resend; the
    turn's failure is surfaced through task status.
    """
    user = _create_user(db_session)
    task = _create_task(db_session, int(user.id), status=TaskStatus.RUNNING)
    _claim_pending_delivery(
        db_session, task_id=int(task.id), user_id=int(user.id), turn_id="turn-late"
    )

    _reconcile_finalized_turn_delivery(
        task_id=int(task.id),
        turn_id="turn-late",
        settlement_error="setup/run error: RuntimeError: late boom",
        execution_started=True,
    )

    assert _delivery_status(db_session, "turn-late") == DELIVERY_DISPATCHED


def test_reconcile_finalized_delivery_noop_without_turn_id(db_session) -> None:
    """A turn with no durable delivery row (no turn id) is a safe no-op."""
    user = _create_user(db_session)
    task = _create_task(db_session, int(user.id), status=TaskStatus.RUNNING)

    # Must not raise even though there is nothing to reconcile.
    _reconcile_finalized_turn_delivery(
        task_id=int(task.id),
        turn_id=None,
        settlement_error=None,
        execution_started=True,
    )


@pytest.mark.parametrize("execution_started", [True, False])
def test_reconcile_finalized_delivery_does_not_regress_dispatched_on_error(
    db_session,
    execution_started,
) -> None:
    """``dispatched`` is terminal for dedup; a later failure must not regress it.

    A normally dispatched turn already gave the client a usable ack. With
    ``execution_started=True`` the ``dispatched`` target is idempotent; with
    ``False`` the monotonic state machine rejects ``dispatched -> failed``.
    Either way the row stays ``dispatched`` and the turn's failure is surfaced
    through task status, not by reopening delivery.
    """
    user = _create_user(db_session)
    task = _create_task(db_session, int(user.id), status=TaskStatus.RUNNING)
    _claim_pending_delivery(
        db_session, task_id=int(task.id), user_id=int(user.id), turn_id="turn-disp"
    )
    transition = mark_user_message_delivery(
        db_session,
        task_id=int(task.id),
        turn_id="turn-disp",
        status=DELIVERY_DISPATCHED,
    )
    assert transition.outcome == "updated"
    db_session.commit()
    db_session.expire_all()

    _reconcile_finalized_turn_delivery(
        task_id=int(task.id),
        turn_id="turn-disp",
        settlement_error="boom",
        execution_started=execution_started,
    )

    assert _delivery_status(db_session, "turn-disp") == DELIVERY_DISPATCHED


def test_reconcile_finalized_delivery_completes_dispatched_on_success(
    db_session,
) -> None:
    """A clean settlement closes a ``dispatched`` row to ``completed``.

    The orchestrator path never marked ``completed`` before; finalize now
    closes the lifecycle to match the resume path.
    """
    user = _create_user(db_session)
    task = _create_task(db_session, int(user.id), status=TaskStatus.RUNNING)
    _claim_pending_delivery(
        db_session, task_id=int(task.id), user_id=int(user.id), turn_id="turn-done"
    )
    mark_user_message_delivery(
        db_session,
        task_id=int(task.id),
        turn_id="turn-done",
        status=DELIVERY_DISPATCHED,
    )
    db_session.commit()

    _reconcile_finalized_turn_delivery(
        task_id=int(task.id),
        turn_id="turn-done",
        settlement_error=None,
        execution_started=True,
    )

    assert _delivery_status(db_session, "turn-done") == DELIVERY_COMPLETED


def test_reconcile_finalized_delivery_swallows_pool_timeout_leaving_pending(
    db_session,
    caplog,
) -> None:
    """A pool timeout leaves the row ``pending`` for durable recovery, no raise."""
    user = _create_user(db_session)
    task = _create_task(db_session, int(user.id), status=TaskStatus.RUNNING)
    _claim_pending_delivery(
        db_session, task_id=int(task.id), user_id=int(user.id), turn_id="turn-pool"
    )

    caplog.set_level(logging.ERROR, logger="xagent.web.services.task_orchestrator")
    with patch(
        "xagent.web.services.task_orchestrator.mark_user_message_delivery_sync",
        side_effect=SQLAlchemyTimeoutError("finalize delivery pool exhausted"),
    ):
        _reconcile_finalized_turn_delivery(
            task_id=int(task.id),
            turn_id="turn-pool",
            settlement_error=None,
            execution_started=True,
        )

    assert _delivery_status(db_session, "turn-pool") == DELIVERY_PENDING
    assert "component=turn-finalize-delivery" in caplog.text


def _finalize_turn_fixture(db_session, *, turn_id: str):
    """One RUNNING task with a committed lease and an orphaned ``pending`` row.

    Mirrors the #332 failure state: the turn claim committed (task RUNNING,
    lease owned, delivery ``pending``) but the ``dispatched`` projection never
    landed.
    """
    user = _create_user(db_session)
    task = _create_task(db_session, int(user.id), status=TaskStatus.RUNNING)
    payload = TaskTurnPayload("hello there", turn_id=turn_id)
    _claim_pending_delivery(
        db_session, task_id=int(task.id), user_id=int(user.id), turn_id=turn_id
    )
    task.runner_id = "test-runner"
    task.run_id = "run-a"
    db_session.commit()
    lease = TaskLease(task_id=int(task.id), runner_id="test-runner", run_id="run-a")
    return user, task, payload, lease


@contextmanager
def _finalize_runner_patches(
    fake_lease: TaskLease,
    *,
    execute=None,
    settle=None,
    mark_delivery=None,
    validate=None,
    snapshot=None,
):
    """Patch scaffolding for driving ``_schedule_bg``'s ``_runner`` end-to-end.

    Only the agent runtime is faked by default; lease settlement and delivery
    marking run for real against SQLite unless a stub is supplied.
    """
    from xagent.web.api.websocket import background_task_manager

    patches = [
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            new=snapshot
            if snapshot is not None
            else MagicMock(return_value=MagicMock()),
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=execute if execute is not None else AsyncMock(),
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ]
    if settle is not None:
        patches.append(
            patch(
                "xagent.web.services.task_orchestrator.settle_task_lease_isolated",
                settle,
            )
        )
    if mark_delivery is not None:
        patches.append(
            patch(
                "xagent.web.services.task_orchestrator.mark_user_message_delivery_sync",
                mark_delivery,
            )
        )
    if validate is not None:
        patches.append(
            patch(
                "xagent.web.services.task_orchestrator."
                "validate_preacquired_task_lease_isolated",
                validate,
            )
        )
    with ExitStack() as stack:
        for entry in patches:
            stack.enter_context(entry)
        yield


def _spawn_finalize_runner(task, user, payload, **schedule_kwargs):
    return _schedule_bg(
        task_id=int(task.id),
        task_owner_user_id=int(user.id),
        task_source=task.source,
        payload=payload,
        force_fresh=False,
        context=None,
        **schedule_kwargs,
    )


@pytest.mark.asyncio
async def test_runner_finalize_reconciles_orphaned_pending_delivery(
    db_session,
) -> None:
    """End-to-end: a turn whose ``dispatched`` projection never landed still
    finalizes its delivery row to ``completed`` when the run settles cleanly."""
    user, task, payload, lease = _finalize_turn_fixture(
        db_session, turn_id="turn-e2e-ok"
    )

    with _finalize_runner_patches(lease):
        await _spawn_finalize_runner(task, user, payload)

    assert _delivery_status(db_session, "turn-e2e-ok") == DELIVERY_COMPLETED


@pytest.mark.asyncio
async def test_runner_finalize_reconciles_alongside_a_genuinely_completed_turn(
    db_session,
) -> None:
    """A real success settles through ``finish_turn``'s COMPLETED branch and
    closes the orphaned delivery row in the same finalize.

    The sibling success test reaches ``completed`` delivery via
    ``settlement_error is None`` while the task itself lands FAILED (its mocked
    runtime never signals success), so it cannot catch a regression in
    ``finish_turn``'s COMPLETED branch. Here the fake runtime commits
    COMPLETED plus an assistant message — what that branch reads to write
    ``output`` — so task terminalization and delivery reconciliation are
    asserted together.
    """
    user, task, payload, lease = _finalize_turn_fixture(
        db_session, turn_id="turn-e2e-real-success"
    )

    async def succeeding_execute(*args, **kwargs):
        with sessionmaker(bind=get_engine())() as inner:
            row = inner.query(Task).filter(Task.id == task.id).one()
            row.status = TaskStatus.COMPLETED
            inner.add(
                TaskChatMessage(
                    task_id=int(task.id),
                    user_id=int(user.id),
                    role="assistant",
                    content="here is your answer",
                    message_type="chat_response",
                )
            )
            inner.commit()

    with _finalize_runner_patches(lease, execute=succeeding_execute):
        await _spawn_finalize_runner(task, user, payload)

    assert _delivery_status(db_session, "turn-e2e-real-success") == DELIVERY_COMPLETED
    db_session.expire_all()
    stored = db_session.query(Task).filter(Task.id == task.id).one()
    assert stored.status == TaskStatus.COMPLETED
    assert stored.output == "here is your answer"
    assert stored.runner_id is None


@pytest.mark.asyncio
async def test_runner_finalize_leaves_pending_when_deferred_to_ttl_recovery(
    db_session,
) -> None:
    """When settlement is deferred to TTL recovery (pool timeout), the delivery
    row must stay ``pending`` — the turn may be re-run by the recovery path."""
    user, task, payload, lease = _finalize_turn_fixture(
        db_session, turn_id="turn-e2e-defer"
    )

    async def pool_timeout_execute(*args, **kwargs):
        raise SQLAlchemyTimeoutError("setup/run pool exhausted")

    with _finalize_runner_patches(lease, execute=pool_timeout_execute):
        await _spawn_finalize_runner(task, user, payload)

    assert _delivery_status(db_session, "turn-e2e-defer") == DELIVERY_PENDING


@pytest.mark.asyncio
async def test_runner_finalize_leaves_pending_when_settlement_raises(
    db_session,
) -> None:
    """If lease settlement itself raises, the lease is retained for TTL recovery
    and the turn may re-run — so the delivery row must stay ``pending`` even
    though the run completed. Guards the settle-failure branch, which does not
    set ``defer_settlement_to_ttl_recovery``."""
    user, task, payload, lease = _finalize_turn_fixture(
        db_session, turn_id="turn-e2e-settle-raise"
    )

    with _finalize_runner_patches(
        lease, settle=MagicMock(side_effect=RuntimeError("settle boom"))
    ):
        await _spawn_finalize_runner(task, user, payload)

    assert _delivery_status(db_session, "turn-e2e-settle-raise") == DELIVERY_PENDING


@pytest.mark.asyncio
async def test_runner_finalize_leaves_pending_when_settle_is_fenced_out(
    db_session,
) -> None:
    """A fenced/superseded settlement must not close the delivery row.

    ``settle_task_lease_isolated`` returns ``False`` *without raising* when the
    lease row is gone or now owned by a different run. This stale coroutine is
    no longer authoritative for the turn, so reconciliation must be skipped and
    the ``pending`` row left to the authoritative owner."""
    user, task, payload, lease = _finalize_turn_fixture(
        db_session, turn_id="turn-e2e-fenced"
    )

    with _finalize_runner_patches(lease, settle=MagicMock(return_value=False)):
        await _spawn_finalize_runner(task, user, payload)

    assert _delivery_status(db_session, "turn-e2e-fenced") == DELIVERY_PENDING


@pytest.mark.asyncio
async def test_runner_finalize_skips_reconcile_on_settlement_ready_short_circuit(
    db_session,
) -> None:
    """Pre-execution SETTLEMENT_READY must not close the delivery row.

    A delayed run whose pre-acquired lease validates as SETTLEMENT_READY
    returns before the turn body executes. Settlement still runs (and here
    genuinely settles the lease), but the turn never ran, so the delivery row
    must not be reconciled to ``completed``."""
    user, task, payload, lease = _finalize_turn_fixture(
        db_session, turn_id="turn-e2e-settle-ready"
    )

    with _finalize_runner_patches(
        lease,
        validate=MagicMock(return_value=TaskLeaseRefreshState.SETTLEMENT_READY),
    ):
        await _spawn_finalize_runner(
            task, user, payload, task_lease=lease, run_id="run-a"
        )

    assert _delivery_status(db_session, "turn-e2e-settle-ready") == DELIVERY_PENDING


@pytest.mark.asyncio
async def test_runner_finalize_marks_dispatched_for_cancelled_turn(
    db_session,
) -> None:
    """A turn cancelled during execution closes its delivery row to
    ``dispatched``, not ``failed`` — execution had begun, so the message may
    already have been consumed and a fresh-id retry could double-execute it."""
    user, task, payload, lease = _finalize_turn_fixture(
        db_session, turn_id="turn-e2e-cancelled"
    )

    with _finalize_runner_patches(
        lease, execute=AsyncMock(side_effect=asyncio.CancelledError())
    ):
        bg_task = _spawn_finalize_runner(task, user, payload)
        with pytest.raises(asyncio.CancelledError):
            await bg_task

    assert _delivery_status(db_session, "turn-e2e-cancelled") == DELIVERY_DISPATCHED
    db_session.expire_all()
    stored = db_session.query(Task).filter(Task.id == task.id).one()
    assert stored.status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_runner_finalize_marks_dispatched_when_run_fails_after_start(
    db_session,
) -> None:
    """The double-execution hazard: a run that failed *after* execution began
    must close the orphaned ``pending`` row as ``dispatched`` (probe answers
    MATCHES, no retry), never ``failed`` (probe would invite a fresh-id resend
    of a message that already ran). This also pins the symmetry with the
    post-schedule projection: whichever best-effort write lands, the row
    converges on ``dispatched``."""
    user, task, payload, lease = _finalize_turn_fixture(
        db_session, turn_id="turn-e2e-late-fail"
    )

    with _finalize_runner_patches(
        lease,
        execute=AsyncMock(side_effect=RuntimeError("late boom after side effects")),
    ):
        await _spawn_finalize_runner(task, user, payload)

    assert _delivery_status(db_session, "turn-e2e-late-fail") == DELIVERY_DISPATCHED
    db_session.expire_all()
    stored = db_session.query(Task).filter(Task.id == task.id).one()
    assert stored.status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_runner_finalize_marks_failed_when_setup_fails_before_execute(
    db_session,
) -> None:
    """A failure before ``execute_task_background`` is ever invoked is positive
    evidence the message was never consumed, so the row closes as ``failed``
    and a fresh-id retry is safe."""
    user, task, payload, lease = _finalize_turn_fixture(
        db_session, turn_id="turn-e2e-pre-exec"
    )

    with _finalize_runner_patches(
        lease,
        snapshot=MagicMock(side_effect=RuntimeError("snapshot load exploded")),
    ):
        await _spawn_finalize_runner(task, user, payload)

    assert _delivery_status(db_session, "turn-e2e-pre-exec") == DELIVERY_FAILED
    db_session.expire_all()
    stored = db_session.query(Task).filter(Task.id == task.id).one()
    assert stored.status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_runner_finalize_preserves_cancellation_from_reconcile(
    db_session,
) -> None:
    """A ``CancelledError`` surfacing from the reconcile call site is recorded
    as cleanup cancellation and re-raised after connector cleanup; the delivery
    row stays ``pending``."""
    user, task, payload, lease = _finalize_turn_fixture(
        db_session, turn_id="turn-e2e-cancel-reconcile"
    )

    with _finalize_runner_patches(
        lease,
        mark_delivery=MagicMock(side_effect=asyncio.CancelledError()),
    ):
        bg_task = _spawn_finalize_runner(task, user, payload)
        with pytest.raises(asyncio.CancelledError):
            await bg_task

    assert _delivery_status(db_session, "turn-e2e-cancel-reconcile") == DELIVERY_PENDING
    # The lease settlement itself committed before the reconcile cancellation.
    db_session.expire_all()
    stored = db_session.query(Task).filter(Task.id == task.id).one()
    assert stored.runner_id is None


@pytest.mark.parametrize(
    "seeded_status, settlement_error, execution_started",
    [
        (DELIVERY_COMPLETED, None, True),
        (DELIVERY_COMPLETED, "boom", True),
        (DELIVERY_COMPLETED, "boom", False),
        (DELIVERY_FAILED, None, True),
    ],
)
def test_reconcile_finalized_delivery_noop_on_already_terminal_row(
    db_session,
    seeded_status,
    settlement_error,
    execution_started,
) -> None:
    """An already-terminal delivery row is never rewritten by reconciliation.

    ``completed`` and ``failed`` are terminal in the monotonic state machine;
    a later reconcile — idempotent or conflicting, under any target — must
    leave them as-is."""
    user = _create_user(db_session)
    task = _create_task(db_session, int(user.id), status=TaskStatus.RUNNING)
    _claim_pending_delivery(
        db_session, task_id=int(task.id), user_id=int(user.id), turn_id="turn-term"
    )
    transition = mark_user_message_delivery(
        db_session, task_id=int(task.id), turn_id="turn-term", status=seeded_status
    )
    assert transition.outcome == "updated"
    db_session.commit()
    db_session.expire_all()

    _reconcile_finalized_turn_delivery(
        task_id=int(task.id),
        turn_id="turn-term",
        settlement_error=settlement_error,
        execution_started=execution_started,
    )

    assert _delivery_status(db_session, "turn-term") == seeded_status


# ---------------------------------------------------------------------------
# Connector-runtime failures reach the client as a structured, wire-safe frame
# ---------------------------------------------------------------------------


CONNECTOR_RUNTIME_CODES = [
    "missing_runtime_context",
    "runtime_secret_unavailable",
    "scheduled_secret_unavailable",
]

# The reason a missing declared context key produces, used to populate the
# exception's own ``details`` below. The key half is a name the connector's
# owner chose; nothing under ``details`` reaches the terminal frame at all,
# so these fixtures exist to exercise the operator log, which still reads it.
WITHHELD_KEY_REASON = "missing_context.auth_token"
# An arbitrary reason value with no significance of its own -- it lives only
# in the exception's ``details``, which the frame never carries, so which
# string this is does not affect any assertion below.
PUBLIC_REASON = "not_provided"


@contextmanager
def _captured_terminal_broadcast(setup_or_run_error: BaseException, db_session):
    """Drive one owned run to failure and hand back both halves it produced.

    The branch under test writes two things: the broadcast frame the live
    client renders, and the durable settlement the transcript replays after a
    reload. Capturing only the frame would let the durable half be deleted
    with every test still green, so the settlement kwargs come back too.
    """

    from xagent.web.api.websocket import background_task_manager

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task_id = int(task.id)
    lease = TaskLease(task_id=task_id, runner_id="runner-a", run_id="run-a")
    frames: list[dict] = []
    settlements: list[dict] = []

    async def broadcast(event, *_args, **_kwargs) -> None:
        frames.append(event)

    def settle(*_args, **kwargs) -> bool:
        settlements.append(kwargs)
        return True

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            return_value=MagicMock(),
        ),
        patch.object(
            task_orchestrator_module,
            "resolve_execution_scope",
            return_value=None,
            create=True,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(side_effect=setup_or_run_error),
        ),
        patch(
            "xagent.web.services.task_orchestrator.settle_task_lease_isolated",
            side_effect=settle,
        ),
        patch(
            "xagent.web.api.websocket.manager",
            MagicMock(broadcast_to_task=AsyncMock(side_effect=broadcast)),
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        yield task_id, frames, settlements


async def _run_failing_turn(task_id: int, user_id: int, source) -> None:
    await _schedule_bg(
        task_id=task_id,
        task_owner_user_id=user_id,
        task_source=source,
        payload=TaskTurnPayload("hello"),
        force_fresh=False,
        context=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("code", CONNECTOR_RUNTIME_CODES)
async def test_connector_runtime_failure_broadcasts_its_safe_message(
    db_session,
    code: str,
) -> None:
    """The curated sentence replaces the opaque task-failure fallback."""

    safe_message = f"Required connector runtime input is missing ({code})."
    error = ConnectorRuntimeError(
        code,
        safe_message,
        details={"reason": WITHHELD_KEY_REASON},
    )

    with _captured_terminal_broadcast(error, db_session) as (
        task_id,
        frames,
        settlements,
    ):
        task = db_session.query(Task).filter(Task.id == task_id).one()
        await _run_failing_turn(task_id, int(task.user_id), task.source)

    assert [frame["message"] for frame in frames] == [safe_message]
    assert frames[0]["error"] == safe_message
    assert frames[0]["code"] == code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ValueError("secret-token-xyz"),
        KeyError("secret-token-xyz"),
        RuntimeError("secret-token-xyz"),
    ],
    ids=["value-error", "key-error", "runtime-error"],
)
async def test_incidental_failure_still_redacts(
    db_session,
    error: BaseException,
) -> None:
    """Only the connector-runtime class earns the new branch."""

    with _captured_terminal_broadcast(error, db_session) as (
        task_id,
        frames,
        settlements,
    ):
        task = db_session.query(Task).filter(Task.id == task_id).one()
        await _run_failing_turn(task_id, int(task.user_id), task.source)

    assert len(frames) == 1
    assert frames[0]["message"] == CLIENT_SAFE_TASK_FAILURE
    assert frames[0]["error"] == CLIENT_SAFE_TASK_FAILURE
    assert "secret-token-xyz" not in json.dumps(frames[0])
    assert "code" not in frames[0]
    assert "details" not in frames[0]


@pytest.mark.asyncio
async def test_connector_runtime_frame_carries_code_only(db_session) -> None:
    """Whatever the raise site attached to ``details``, none of it reaches the wire."""

    error = ConnectorRuntimeError(
        "runtime_secret_unavailable",
        "Required runtime secret is unavailable.",
        details={
            "reason": PUBLIC_REASON,
            "internal_sql": "SELECT value FROM task_connector_runtime_contexts",
            "raw_value": "tenant-secret",
            "connector_ref": {"connector_type": "mcp", "connector_id": 7},
        },
    )

    with _captured_terminal_broadcast(error, db_session) as (
        task_id,
        frames,
        settlements,
    ):
        task = db_session.query(Task).filter(Task.id == task_id).one()
        await _run_failing_turn(task_id, int(task.user_id), task.source)

    assert set(frames[0]) == {
        "type",
        "message",
        "task_id",
        "task",
        "error",
        "timestamp",
        "code",
    }
    assert "details" not in frames[0]


@pytest.mark.asyncio
async def test_connector_runtime_frame_never_carries_connector_ref(
    db_session,
) -> None:
    """The frame's audience includes anonymous widget and share visitors.

    The three assertions are structural on purpose. An earlier form of this
    test also asserted the connector's numeric id was absent from the
    serialized frame, which goes red on any fixture where that id collides
    with the task id or a timestamp digit.
    """

    error = ConnectorRuntimeError(
        "runtime_secret_unavailable",
        "Required runtime secret is unavailable.",
        connector_ref=ConnectorRef(connector_type="mcp", connector_id=7),
        details={"reason": PUBLIC_REASON},
    )

    with _captured_terminal_broadcast(error, db_session) as (
        task_id,
        frames,
        settlements,
    ):
        task = db_session.query(Task).filter(Task.id == task_id).one()
        await _run_failing_turn(task_id, int(task.user_id), task.source)

    frame = frames[0]
    assert set(frame) == {
        "type",
        "message",
        "task_id",
        "task",
        "error",
        "timestamp",
        "code",
    }
    serialized = json.dumps(frame)
    assert "connector_ref" not in serialized
    assert "connector_id" not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize("code", CONNECTOR_RUNTIME_CODES)
async def test_connector_runtime_failure_logs_missing_key(
    db_session,
    caplog,
    code: str,
) -> None:
    """Operators read the raw details, connector identity included."""

    error = ConnectorRuntimeError(
        code,
        "Required connector runtime context is missing.",
        connector_ref=ConnectorRef(connector_type="mcp", connector_id=7),
        details={"reason": WITHHELD_KEY_REASON},
    )

    with caplog.at_level(logging.ERROR):
        with _captured_terminal_broadcast(error, db_session) as (
            task_id,
            frames,
            settlements,
        ):
            task = db_session.query(Task).filter(Task.id == task_id).one()
            await _run_failing_turn(task_id, int(task.user_id), task.source)

    structured = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR
        and "component=connector-runtime" in record.getMessage()
    ]
    assert len(structured) == 1
    assert f"code={code}" in structured[0]
    assert f"reason={WITHHELD_KEY_REASON}" in structured[0]
    assert "connector=" in structured[0]
    assert "'connector_id': 7" in structured[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", CONNECTOR_RUNTIME_CODES)
async def test_connector_runtime_failure_persists_client_safe_history(
    db_session,
    code: str,
) -> None:
    """The durable half: what the transcript replays after a reload.

    The new branch writes three things -- the frame, the settlement error and
    the history message type. Without this test the whole
    ``client_history_message_type`` line could be deleted and every other test
    in this file would stay green, while a reloading user dropped back to the
    generic failure text the frame no longer shows.
    """

    safe_message = f"Required connector runtime input is missing ({code})."
    error = ConnectorRuntimeError(
        code,
        safe_message,
        details={"reason": WITHHELD_KEY_REASON},
    )

    with _captured_terminal_broadcast(error, db_session) as (
        task_id,
        frames,
        settlements,
    ):
        task = db_session.query(Task).filter(Task.id == task_id).one()
        await _run_failing_turn(task_id, int(task.user_id), task.source)

    assert len(settlements) == 1
    settled = settlements[0]
    assert settled["client_message_type"] == CLIENT_SAFE_FAILURE_MESSAGE_TYPE
    # The durable row and the frame's own message field are the same server
    # sentence. The live bubble is not that sentence: for a missing-value code
    # the client replaces it with its own localized wording (see the
    # "terminal error frames" suite in app-context-chat.test.tsx). What the two
    # views owe each other is the facts they carry, and the key name is in
    # neither -- the frame never carries a details object at all, so there is
    # no key name for the client to render.
    assert settled["client_error_message"] == safe_message
    assert settled["client_error_message"] == frames[0]["message"]
    assert "details" not in frames[0]
    assert "auth_token" not in json.dumps(frames[0])
    # The durable error keeps the code prefix operators grep for, and never
    # the "setup/run error: <ExceptionType>" shape the else branch produces.
    assert settled["error_message"] == f"{code}: {safe_message}"
    assert "setup/run error" not in settled["error_message"]


@pytest.mark.asyncio
async def test_incidental_failure_persists_the_generic_history_type(
    db_session,
) -> None:
    """The counterpart: an incidental failure keeps the untrusted settlement."""

    error = RuntimeError("secret-token-xyz")

    with _captured_terminal_broadcast(error, db_session) as (
        task_id,
        frames,
        settlements,
    ):
        task = db_session.query(Task).filter(Task.id == task_id).one()
        await _run_failing_turn(task_id, int(task.user_id), task.source)

    assert len(settlements) == 1
    settled = settlements[0]
    assert settled["client_message_type"] == TASK_FAILURE_MESSAGE_TYPE
    assert settled["client_error_message"] == CLIENT_SAFE_TASK_FAILURE
    assert "secret-token-xyz" not in settled["client_error_message"]


@pytest.mark.asyncio
async def test_leased_auto_failure_preserves_client_classification(db_session) -> None:
    error = AutoModelUnavailableError("private model binding details")
    with _captured_terminal_broadcast(error, db_session) as (
        task_id,
        frames,
        settlements,
    ):
        task = db_session.query(Task).filter(Task.id == task_id).one()
        with patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(side_effect=error),
        ) as execute:
            await _run_failing_turn(task_id, int(task.user_id), task.source)
        execute.assert_awaited_once()
        assert execute.await_args.kwargs["task_lease"] == TaskLease(
            task_id=task_id, runner_id="runner-a", run_id="run-a"
        )

    assert len(frames) == 1
    assert frames[0]["code"] == "auto_model_unavailable"
    assert frames[0]["message"] == CLIENT_SAFE_AUTO_MODEL_UNAVAILABLE
    assert len(settlements) == 1
    assert settlements[0]["client_message_type"] == CLIENT_SAFE_FAILURE_MESSAGE_TYPE
    assert settlements[0]["client_error_message"] == CLIENT_SAFE_AUTO_MODEL_UNAVAILABLE
    assert settlements[0]["error_message"] == CLIENT_SAFE_AUTO_MODEL_UNAVAILABLE
    assert "private model binding details" not in json.dumps(frames)
