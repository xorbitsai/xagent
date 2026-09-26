"""Outcome-unknown settlement and orphan reconciliation on real PostgreSQL.

The SQLite suites pin the decisions; these pin what only a real MVCC database
can show: the unique-index collision a recovered turn used to hit, row-level
locking between lease recovery and a concurrent delivery write, and a sweep
that must neither block on nor touch a concurrently accepted, uncommitted
turn.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Iterator
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from tests.web.api.test_durable_message_resume_contention import (
    _live_control_environment,
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
from xagent.web.services import task_lease_recovery
from xagent.web.services.chat_history_service import (
    DELIVERY_COMPLETED,
    DELIVERY_DISPATCHED,
    DELIVERY_PENDING,
    mark_user_message_delivery,
)
from xagent.web.services.task_command_execution import execute_durable_task_command
from xagent.web.services.task_execution import ResumeReservationOutcome
from xagent.web.services.task_lease_service import utc_now
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
