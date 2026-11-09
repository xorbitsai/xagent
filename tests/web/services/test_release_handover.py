"""Hand-over of unfinished runs to a successor (``release_handover``).

Runs on SQLite, and on PostgreSQL when ``XAGENT_TEST_POSTGRES_URL`` is set,
through the shared ``engine`` fixture.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models import database
from xagent.web.models.database import Base as DatabaseBase
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import release_handover, task_auto_resume
from xagent.web.services.release_handover import (
    begin_handover_sync,
    handover_command_id,
    settle_handover_sync,
)
from xagent.web.services.task_auto_resume import (
    cancel_pending_auto_resume_no_commit,
    run_auto_resume_tick,
)
from xagent.web.services.task_command_transport import COMMAND_COMPLETED
from xagent.web.services.task_lease_service import utc_now

engine = engine_fixture

RELEASE = "rel-1"
RUNNER = "runner-old"
RUN = "run-handover"


@pytest.fixture
def factory(engine, monkeypatch) -> Iterator[sessionmaker]:
    DatabaseBase.metadata.create_all(engine)
    result = sessionmaker(engine, autoflush=False)
    monkeypatch.setattr(database, "_SessionLocal", result)
    monkeypatch.setattr(database, "_engine", engine)
    for name in (
        "XAGENT_TASK_AUTO_RESUME_ENABLED",
        "XAGENT_SHARED_TASK_EXECUTION_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "true")
    notified: list[int] = []
    for module in (release_handover, task_auto_resume):
        monkeypatch.setattr(
            module, "notify_task_command_dispatcher", lambda: notified.append(1)
        )
    yield result


def _task(db: Session, **fields: Any) -> int:
    user = User(username=f"handover-{uuid.uuid4().hex[:8]}", password_hash="x")
    db.add(user)
    db.flush()
    values: dict[str, Any] = {
        "status": TaskStatus.RUNNING,
        "control_state": "running",
        "run_id": RUN,
        "runner_id": RUNNER,
        "state_version": 3,
    }
    values.update(fields)
    task = Task(
        user_id=user.id,
        title="Handover",
        description="handover test",
        execution_mode="balanced",
        **values,
    )
    db.add(task)
    db.flush()
    return int(task.id)


def _commands(factory: sessionmaker, task_id: int) -> list[TaskExecutionCommand]:
    with factory() as db:
        rows = list(
            db.scalars(
                sa.select(TaskExecutionCommand)
                .where(TaskExecutionCommand.task_id == task_id)
                .order_by(TaskExecutionCommand.id)
            )
        )
        db.expunge_all()
        return rows


def _row(factory: sessionmaker, task_id: int) -> TaskAutoRecovery | None:
    with factory() as db:
        row = db.get(TaskAutoRecovery, task_id)
        db.expunge_all()
        return row


def _pause_applied(factory: sessionmaker, task_id: int) -> None:
    """What the lease owner does with the staged PAUSE: apply and complete."""
    with factory() as db:
        task = db.get(Task, task_id)
        task.status = TaskStatus.PAUSED
        task.control_state = "paused"
        task.state_version = int(task.state_version) + 1
        for command in db.scalars(
            sa.select(TaskExecutionCommand).where(
                TaskExecutionCommand.task_id == task_id
            )
        ):
            command.status = COMMAND_COMPLETED
        db.commit()


def test_begin_stages_one_pause_for_an_eligible_run_of_the_given_runner(factory):
    with factory() as db:
        mine = _task(db)
        other_runner = _task(db, runner_id="runner-elsewhere")
        db.commit()

    report = begin_handover_sync(RELEASE, {RUNNER})

    assert (report.staged, report.kept) == (1, 0)
    commands = _commands(factory, mine)
    assert [(c.kind, c.command_id, c.target_run_id) for c in commands] == [
        ("pause", handover_command_id(RELEASE, mine), RUN)
    ]
    assert _commands(factory, other_runner) == []

    # The host repeats the call while it waits: nothing is staged twice.
    again = begin_handover_sync(RELEASE, {RUNNER})
    assert again.staged == 0
    assert len(_commands(factory, mine)) == 1


def test_begin_leaves_runs_automatic_recovery_would_not_resume(factory):
    with factory() as db:
        a2a = _task(db, source="a2a")
        user_paused = _task(db, control_state="pause_requested")
        db.commit()

    report = begin_handover_sync(RELEASE, {RUNNER})

    assert (report.staged, report.kept) == (0, 2)
    assert _commands(factory, a2a) == []
    assert _commands(factory, user_paused) == []


def test_begin_stages_nothing_when_automatic_resume_is_off(factory, monkeypatch):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")
    with factory() as db:
        task_id = _task(db)
        db.commit()

    assert begin_handover_sync(RELEASE, {RUNNER}).staged == 0
    assert _commands(factory, task_id) == []


def test_settle_waits_for_the_pause_then_records_a_shutdown_resume(factory):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    begin_handover_sync(RELEASE, {RUNNER})

    # The run has not reached its safe point yet: still unsettled, nothing recorded.
    waiting = settle_handover_sync(RELEASE)
    assert (waiting.recorded, waiting.unsettled) == (0, 1)
    assert _row(factory, task_id) is None

    _pause_applied(factory, task_id)
    settled = settle_handover_sync(RELEASE)

    assert (settled.recorded, settled.unsettled) == (1, 0)
    row = _row(factory, task_id)
    assert row is not None
    assert (row.reason, row.state, row.run_id) == ("shutdown", "scheduled", RUN)
    assert row.next_attempt_at is not None
    # Idempotent: the same pause is not recorded twice.
    assert settle_handover_sync(RELEASE).recorded == 0


def test_a_run_that_moved_on_is_not_recorded(factory):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    begin_handover_sync(RELEASE, {RUNNER})
    _pause_applied(factory, task_id)
    with factory() as db:
        task = db.get(Task, task_id)
        task.run_id = "run-next-turn"
        db.commit()

    assert settle_handover_sync(RELEASE).recorded == 0
    assert _row(factory, task_id) is None


def test_the_successor_resumes_the_handed_over_run_exactly_once(factory):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    begin_handover_sync(RELEASE, {RUNNER})
    _pause_applied(factory, task_id)
    settle_handover_sync(RELEASE)

    now = utc_now() + timedelta(seconds=30)
    assert run_auto_resume_tick(now=now).staged_commands == 1
    assert run_auto_resume_tick(now=now).staged_commands == 0

    resumes = [c for c in _commands(factory, task_id) if c.kind == "resume"]
    assert len(resumes) == 1
    assert resumes[0].command_id.startswith("auto-resume:")


def test_a_user_cancel_during_the_handover_keeps_the_run_paused(factory):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    begin_handover_sync(RELEASE, {RUNNER})
    _pause_applied(factory, task_id)
    settle_handover_sync(RELEASE)

    with factory() as db:
        task = db.execute(
            sa.select(Task).where(Task.id == task_id).with_for_update()
        ).scalar_one()
        cancel = cancel_pending_auto_resume_no_commit(db, task, at="web_cancel")
        assert cancel.cancelled
        db.commit()

    now = utc_now() + timedelta(seconds=30)
    assert run_auto_resume_tick(now=now).staged_commands == 0
    assert [c.kind for c in _commands(factory, task_id)] == ["pause"]
    with factory() as db:
        task = db.get(Task, task_id)
        assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")
