"""Outcome-unknown settlement and orphan reconciliation on real PostgreSQL.

The SQLite suites pin the decisions; these pin what only a real MVCC database
can show: the unique-index collision a recovered turn used to hit, row-level
locking between lease recovery and a concurrent delivery write, and a sweep
that must neither block on nor touch a concurrently accepted, uncommitted
turn.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from typing import Iterator
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import text, update
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from tests.web.api.test_durable_message_resume_contention import (
    _live_control_environment,
    _message_command,
    _user,
)
from tests.web.api.test_recovered_delivery_outcome_unknown import (
    TURN_ID,
    _assert_outcome_unknown_frames,
    _outcome_unknown_result,
    _pending_row,
    _RecordingReply,
    _recovered_command,
    _row_status,
    _settled_task,
    _user_rows,
)
from xagent.web.models import database
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_command_execution as command_execution_service
from xagent.web.services import task_execution_controller as controller_module
from xagent.web.services import task_lease_recovery
from xagent.web.services.chat_history_service import (
    DELIVERY_COMPLETED,
    DELIVERY_DISPATCHED,
    DELIVERY_PENDING,
    mark_user_message_delivery,
    withdraw_pending_user_message_delivery_sync,
)
from xagent.web.services.task_command_execution import execute_durable_task_command
from xagent.web.services.task_command_transport import TaskCommandKind
from xagent.web.services.task_execution import (
    ResumeReservationOutcome,
    _acquire_resume_task_lease,
)
from xagent.web.services.task_execution_controller import (
    TaskControlState,
    TaskStatusRefusedError,
    transition_task_control_state_sync,
)
from xagent.web.services.task_lease_service import (
    get_expired_task_lease_candidates,
    recover_expired_task_lease_no_commit,
    utc_now,
)
from xagent.web.services.task_orchestrator import (
    TaskTurnOrchestrator,
    TaskTurnPayload,
)

pytestmark = pytest.mark.postgresql

# Bounds a regression that blocks on a row lock; a passing run never waits.
_BLOCKED_SECONDS = 30


@pytest.fixture()
def pg_sessions(monkeypatch) -> Iterator[sessionmaker[Session]]:
    with disposable_database_factory("unknown_delivery") as make_database:
        engine = make_database("t")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        # Production code resolves its Session factory and engine globally.
        monkeypatch.setattr(database, "_engine", engine)
        monkeypatch.setattr(database, "_SessionLocal", sessions)
        yield sessions


@pytest.fixture()
def db_session(pg_sessions) -> Iterator[Session]:
    db = pg_sessions()
    try:
        yield db
    finally:
        db.close()


def _expired_task(db: Session, owner_id: int, *, suffix: str) -> Task:
    task = Task(
        user_id=owner_id,
        title=f"expired {suffix}",
        description="d",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
        runner_id=f"dead-{suffix}",
        run_id=f"run-{suffix}",
        lease_expires_at=utc_now() - timedelta(seconds=5),
        last_heartbeat_at=utc_now() - timedelta(seconds=10),
        state_version=3,
        control_state="running",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _add_row(
    db: Session,
    task: Task,
    turn_id: str,
    *,
    old: bool = False,
) -> int:
    row = TaskChatMessage(
        task_id=int(task.id),
        user_id=int(task.user_id),
        role="user",
        message_type="user_message",
        content=f"message {turn_id}",
        turn_id=turn_id,
        delivery_status=DELIVERY_PENDING,
    )
    if old:
        row.created_at = utc_now() - timedelta(hours=1)
    db.add(row)
    db.commit()
    return int(row.id)


def _status(sessions: sessionmaker[Session], row_id: int) -> str | None:
    with sessions() as db:
        row = db.get(TaskChatMessage, row_id)
        assert row is not None
        return row.delivery_status


def _recover_expired_leases() -> int:
    return task_lease_recovery.recover_expired_task_leases_batch_isolated(
        cutoff=utc_now(),
        batch_size=10,
        after=None,
    ).recovered


@pytest.mark.asyncio
async def test_recovered_delivery_on_changed_run_settles_outcome_unknown_pg(
    db_session: Session,
) -> None:
    owner = _user(db_session, "pg-changed-run")
    task = _settled_task(db_session, int(owner.id), status=TaskStatus.PAUSED)
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    reply = _RecordingReply()
    begin_turn = AsyncMock(wraps=TaskTurnOrchestrator.begin_turn)

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch.object(command_execution_service, "command_reply", return_value=reply),
        patch.object(TaskTurnOrchestrator, "begin_turn", begin_turn),
    ):
        result = await execute_durable_task_command(_recovered_command(task, owner))

    assert result == _outcome_unknown_result(task)
    begin_turn.assert_not_awaited()
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    assert len(_user_rows(db_session, task_id)) == 1
    stored = db_session.get(Task, task_id)
    assert stored.status == TaskStatus.PAUSED
    assert stored.run_id == "newly-started-run"
    _assert_outcome_unknown_frames(reply)
    assert _user_rows(db_session, task_id)[0].turn_id == TURN_ID


def test_lease_recovery_dispatches_orphaned_pending_rows_pg(
    pg_sessions: sessionmaker[Session],
    db_session: Session,
) -> None:
    owner = _user(db_session, "pg-orphan")
    task = _expired_task(db_session, int(owner.id), suffix="orphan")
    orphan = _add_row(db_session, task, "orphan-turn")

    assert _recover_expired_leases() == 1

    assert _status(pg_sessions, orphan) == DELIVERY_DISPATCHED
    db_session.expire_all()
    assert db_session.get(Task, int(task.id)).status == TaskStatus.FAILED


def test_sweep_neither_blocks_on_nor_touches_an_uncommitted_new_turn(
    pg_sessions: sessionmaker[Session],
    db_session: Session,
) -> None:
    owner = _user(db_session, "pg-interleave")
    task = _settled_task(db_session, int(owner.id), status=TaskStatus.PAUSED)
    task_id = int(task.id)
    orphan = _add_row(db_session, task, "old-orphan", old=True)

    accepting = pg_sessions()
    try:
        # B: a new turn claims the task (RUNNING) and stages its pending row,
        # holding the task row lock, without committing yet.
        TaskTurnOrchestrator.claim_append_turn_no_commit(
            accepting,
            task_id=task_id,
            task_owner_user_id=int(owner.id),
            payload=TaskTurnPayload(transcript_message="new turn", turn_id="new-turn"),
        )
        accepting.flush()

        # A: the sweep commits from its own connection while B is open.
        with ThreadPoolExecutor(max_workers=1) as pool:
            swept = pool.submit(
                task_lease_recovery.reconcile_orphaned_pending_deliveries_isolated,
                batch_size=10,
            ).result(timeout=_BLOCKED_SECONDS)
        assert swept == 1

        accepting.commit()
    finally:
        accepting.rollback()
        accepting.close()

    assert _status(pg_sessions, orphan) == DELIVERY_DISPATCHED
    with pg_sessions() as db:
        new_row = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.turn_id == "new-turn",
            )
            .one()
        )
        assert new_row.delivery_status == DELIVERY_PENDING
        assert db.get(Task, task_id).status == TaskStatus.RUNNING


def test_lease_recovery_concurrent_with_completion_never_regresses(
    pg_sessions: sessionmaker[Session],
    db_session: Session,
) -> None:
    owner = _user(db_session, "pg-monotonic")
    task = _expired_task(db_session, int(owner.id), suffix="monotonic")
    row_id = _add_row(db_session, task, "finishing-turn")

    finishing = pg_sessions()
    try:
        # B: the turn's own finalize writes completed and holds the row lock.
        transition = mark_user_message_delivery(
            finishing,
            task_id=int(task.id),
            turn_id="finishing-turn",
            status=DELIVERY_COMPLETED,
        )
        assert transition.outcome == "updated"

        # A: lease recovery either waits on that lock or runs after the
        # commit; in both orders its guarded UPDATE finds no pending row.
        with ThreadPoolExecutor(max_workers=1) as pool:
            recovering = pool.submit(_recover_expired_leases)
            finishing.commit()
            assert recovering.result(timeout=_BLOCKED_SECONDS) == 1
    finally:
        finishing.rollback()
        finishing.close()

    assert _status(pg_sessions, row_id) == DELIVERY_COMPLETED
    with pg_sessions() as db:
        assert db.get(Task, int(task.id)).status == TaskStatus.FAILED


def _recovery_holding_failed(sessions: sessionmaker[Session], task_id: int) -> Session:
    """Stage lease recovery's FAILED settlement and keep its row lock."""

    recovering = sessions()
    now = utc_now()
    candidate = next(
        candidate
        for candidate in get_expired_task_lease_candidates(
            recovering, cutoff=now, limit=10
        )
        if candidate.task_id == task_id
    )
    assert recover_expired_task_lease_no_commit(
        recovering,
        candidate,
        status=TaskStatus.FAILED,
        recovered_at=now,
        error_message="not recoverable",
    )
    return recovering


def _completion_holding(sessions: sessionmaker[Session], task_id: int) -> Session:
    """Stage the run finalizer's COMPLETED write and keep its row lock."""

    finishing = sessions()
    finishing.execute(
        update(Task)
        .where(Task.id == task_id)
        .values(
            status=TaskStatus.COMPLETED,
            control_state=TaskControlState.COMPLETED.value,
            runner_id=None,
            lease_expires_at=None,
            output="final answer",
            state_version=Task.state_version + 1,
        )
    )
    return finishing


def _ending_writer(
    sessions: sessionmaker[Session], task_id: int, ended_status: TaskStatus
) -> Session:
    if ended_status == TaskStatus.FAILED:
        return _recovery_holding_failed(sessions, task_id)
    return _completion_holding(sessions, task_id)


def _wait_until_blocked_on_a_lock(sessions: sessionmaker[Session]) -> None:
    deadline = time.monotonic() + _BLOCKED_SECONDS
    while time.monotonic() < deadline:
        with sessions() as db:
            waiting = db.execute(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE datname = current_database() "
                    "AND wait_event_type = 'Lock'"
                )
            ).scalar_one()
        if waiting:
            return
        time.sleep(0.02)
    raise AssertionError("the fenced write never waited on the recovery lock")


@pytest.mark.asyncio
async def test_recovered_delivery_on_failed_run_settles_outcome_unknown_pg(
    pg_sessions: sessionmaker[Session],
    db_session: Session,
) -> None:
    owner = _user(db_session, "pg-failed-run")
    task = _expired_task(db_session, int(owner.id), suffix="failed-run")
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    recovering = _recovery_holding_failed(pg_sessions, task_id)
    recovering.commit()
    recovering.close()
    reply = _RecordingReply()
    command = replace(_recovered_command(task, owner), target_run_id="run-failed-run")

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            _,
        ),
        patch.object(command_execution_service, "command_reply", return_value=reply),
    ):
        result = await execute_durable_task_command(command)

    assert result == _outcome_unknown_result(task)
    agent.post_user_message.assert_not_awaited()
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    db_session.expire_all()
    stored = db_session.get(Task, task_id)
    assert stored.status == TaskStatus.FAILED
    assert stored.control_state == TaskControlState.FAILED.value
    assert stored.error_message == "not recoverable"
    assert stored.run_id == "run-failed-run"
    _assert_outcome_unknown_frames(reply)


@pytest.mark.parametrize("ended_status", [TaskStatus.FAILED, TaskStatus.COMPLETED])
def test_resume_transition_waits_on_the_ending_write_and_refuses_it_pg(
    pg_sessions: sessionmaker[Session],
    db_session: Session,
    ended_status: TaskStatus,
) -> None:
    """READ COMMITTED re-evaluates the fenced UPDATE on the ended row."""

    owner = _user(db_session, f"pg-transition-fence-{ended_status.value}")
    task = _expired_task(db_session, int(owner.id), suffix="transition-fence")
    task_id = int(task.id)

    recovering = _ending_writer(pg_sessions, task_id, ended_status)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            transition = pool.submit(
                transition_task_control_state_sync,
                task_id,
                TaskControlState.RESUME_REQUESTED,
                expected_run_id="run-transition-fence",
                refuse_terminal_status=True,
            )
            _wait_until_blocked_on_a_lock(pg_sessions)
            recovering.commit()
            with pytest.raises(TaskStatusRefusedError):
                transition.result(timeout=_BLOCKED_SECONDS)
    finally:
        recovering.rollback()
        recovering.close()

    with pg_sessions() as db:
        stored = db.get(Task, task_id)
        assert stored.status == ended_status
        assert stored.control_state == ended_status.value.lower()
        assert stored.run_id == "run-transition-fence"


def test_resume_lease_claim_waits_on_recovery_and_refuses_failed_pg(
    pg_sessions: sessionmaker[Session],
    db_session: Session,
) -> None:
    owner = _user(db_session, "pg-claim-fence")
    task = _expired_task(db_session, int(owner.id), suffix="claim-fence")
    task_id = int(task.id)
    refused: list[bool] = []

    recovering = _recovery_holding_failed(pg_sessions, task_id)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            claim = pool.submit(
                _acquire_resume_task_lease,
                task_id,
                int(owner.id),
                "run-claim-fence",
                refuse_terminal_status=True,
                run_not_resumable_out=refused,
            )
            _wait_until_blocked_on_a_lock(pg_sessions)
            recovering.commit()
            assert claim.result(timeout=_BLOCKED_SECONDS) is None
    finally:
        recovering.rollback()
        recovering.close()

    assert refused == [True]
    with pg_sessions() as db:
        stored = db.get(Task, task_id)
        assert stored.status == TaskStatus.FAILED
        assert stored.runner_id is None
        assert stored.error_message == "not recoverable"


def test_resume_lease_claim_waits_on_completion_and_refuses_completed_pg(
    pg_sessions: sessionmaker[Session],
    db_session: Session,
) -> None:
    """The run's own finalizer commits COMPLETED while the claim waits."""

    owner = _user(db_session, "pg-claim-completed")
    task = _expired_task(db_session, int(owner.id), suffix="claim-completed")
    task_id = int(task.id)
    refused: list[bool] = []

    finishing = _completion_holding(pg_sessions, task_id)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            claim = pool.submit(
                _acquire_resume_task_lease,
                task_id,
                int(owner.id),
                "run-claim-completed",
                refuse_terminal_status=True,
                run_not_resumable_out=refused,
            )
            _wait_until_blocked_on_a_lock(pg_sessions)
            finishing.commit()
            assert claim.result(timeout=_BLOCKED_SECONDS) is None
    finally:
        finishing.rollback()
        finishing.close()

    assert refused == [True]
    with pg_sessions() as db:
        stored = db.get(Task, task_id)
        assert stored.status == TaskStatus.COMPLETED
        assert stored.control_state == TaskControlState.COMPLETED.value
        assert stored.output == "final answer"
        assert stored.runner_id is None


@pytest.mark.parametrize("ended_status", [TaskStatus.FAILED, TaskStatus.COMPLETED])
def test_fresh_claim_waits_on_the_ending_write_and_withdraws_its_row_pg(
    pg_sessions: sessionmaker[Session],
    db_session: Session,
    ended_status: TaskStatus,
) -> None:
    """The resume claim for a fresh message races the run's end.

    It waits on the ending write's row lock, is refused once that commits,
    and reports the ended status, which lets the never-injected row be
    withdrawn so the message can start a new turn.
    """

    owner = _user(db_session, f"pg-fresh-claim-{ended_status.value}")
    task = _expired_task(db_session, int(owner.id), suffix="fresh-claim")
    task_id = int(task.id)
    row_id = _add_row(db_session, task, TURN_ID)
    ended: list[TaskStatus] = []

    ending = _ending_writer(pg_sessions, task_id, ended_status)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            claim = pool.submit(
                _acquire_resume_task_lease,
                task_id,
                int(owner.id),
                "run-fresh-claim",
                refuse_terminal_status=True,
                ended_status_out=ended,
            )
            _wait_until_blocked_on_a_lock(pg_sessions)
            ending.commit()
            assert claim.result(timeout=_BLOCKED_SECONDS) is None
    finally:
        ending.rollback()
        ending.close()

    assert ended == [ended_status]
    assert withdraw_pending_user_message_delivery_sync(task_id, TURN_ID)
    with pg_sessions() as db:
        assert db.get(TaskChatMessage, row_id) is None
        stored = db.get(Task, task_id)
        assert stored.status == ended_status
        assert stored.runner_id is None
        assert stored.run_id == "run-fresh-claim"


@pytest.mark.asyncio
@pytest.mark.parametrize("ended_status", [TaskStatus.FAILED, TaskStatus.COMPLETED])
async def test_fresh_message_whose_run_ends_at_the_transition_appends_pg(
    pg_sessions: sessionmaker[Session],
    db_session: Session,
    ended_status: TaskStatus,
) -> None:
    """Withdrawal, then the new turn's insert of the same turn id, on PG."""

    owner = _user(db_session, f"pg-fresh-transition-{ended_status.value}")
    task = _expired_task(db_session, int(owner.id), suffix="fresh-transition")
    # The run the command targeted.
    task.run_id = "live-run"
    db_session.commit()
    task_id = int(task.id)
    reply = _RecordingReply()
    begin_turn = AsyncMock(wraps=TaskTurnOrchestrator.begin_turn)
    real_sync = controller_module.transition_task_control_state_sync

    def end_then_transition(*args, **kwargs):
        ending = _ending_writer(pg_sessions, task_id, ended_status)
        ending.commit()
        ending.close()
        return real_sync(*args, **kwargs)

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            _,
        ),
        patch.object(command_execution_service, "command_reply", return_value=reply),
        patch.object(TaskTurnOrchestrator, "begin_turn", begin_turn),
        patch.object(
            controller_module,
            "transition_task_control_state_sync",
            side_effect=end_then_transition,
        ),
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=1)
        )

    assert result == {
        "task_id": task_id,
        "command_id": TURN_ID,
        "kind": TaskCommandKind.MESSAGE.value,
    }
    agent.post_user_message.assert_not_awaited()
    begin_turn.assert_awaited_once()
    rows = [row for row in _user_rows(db_session, task_id) if row.turn_id == TURN_ID]
    assert len(rows) == 1
    db_session.expire_all()
    stored = db_session.get(Task, task_id)
    assert stored.status == TaskStatus.RUNNING
    assert stored.run_id not in {None, "live-run"}
    assert stored.error_message is None


def test_withdrawal_waits_on_a_concurrent_row_write_and_then_refuses_pg(
    pg_sessions: sessionmaker[Session],
    db_session: Session,
) -> None:
    """A withdrawal racing another writer of the same row never forces it.

    The DELETE waits on the uncommitted write's row lock; once it commits
    ``dispatched``, READ COMMITTED re-evaluates the ``pending`` predicate,
    nothing is deleted, and the caller settles conservatively.
    """

    owner = _user(db_session, "pg-withdraw-race")
    task = _expired_task(db_session, int(owner.id), suffix="withdraw-race")
    task_id = int(task.id)
    row_id = _add_row(db_session, task, TURN_ID)

    writing = pg_sessions()
    try:
        transition = mark_user_message_delivery(
            writing, task_id=task_id, turn_id=TURN_ID, status=DELIVERY_DISPATCHED
        )
        assert transition.status == DELIVERY_DISPATCHED
        with ThreadPoolExecutor(max_workers=1) as pool:
            withdrawal = pool.submit(
                withdraw_pending_user_message_delivery_sync, task_id, TURN_ID
            )
            _wait_until_blocked_on_a_lock(pg_sessions)
            writing.commit()
            assert withdrawal.result(timeout=_BLOCKED_SECONDS) is False
    finally:
        writing.rollback()
        writing.close()

    assert _status(pg_sessions, row_id) == DELIVERY_DISPATCHED
