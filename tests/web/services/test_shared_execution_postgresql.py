"""Shared handoff and deployment schema checks against disposable PostgreSQL."""

from unittest.mock import Mock

import pytest
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from tests.web.services.coordinator_command_shared import claim_for_owner
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_admission import TaskAdmissionBucket, TaskAdmissionTicket
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_command_terminal_event import TaskCommandTerminalEvent
from xagent.web.models.user import User
from xagent.web.services import task_coordinator_service as ownership
from xagent.web.services import task_event_bridge, task_execution_admission
from xagent.web.services import task_start_consumer as consumer
from xagent.web.services.task_command_transport import TaskCommandRejected
from xagent.web.services.task_orchestrator import TaskTurnOrchestrator, TaskTurnPayload

pytestmark = pytest.mark.postgresql


@pytest.mark.parametrize("fail_completion", [False, True])
def test_postgres_handoff_completion_and_lease_are_atomic(monkeypatch, fail_completion):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(task_event_bridge, "get_task_event_bridge", lambda: Mock())
    with disposable_database_factory("shared_worker") as make_database:
        engine = make_database("handoff")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        monkeypatch.setattr(consumer, "get_session_local", lambda: sessions)
        with sessions() as db:
            owner = User(username="owner", password_hash="unused")
            db.add(owner)
            db.flush()
            task = Task(
                user_id=owner.id,
                title="Shared",
                source="sdk",
                status=TaskStatus.PENDING,
            )
            db.add(task)
            db.flush()
            accepted = TaskTurnOrchestrator.claim_created_turn_no_commit(
                db,
                task_id=task.id,
                task_owner_user_id=owner.id,
                payload=TaskTurnPayload("hello"),
            )
            db.commit()
            task_id = task.id
            owner_lease = ownership.acquire_task_lease_no_commit(
                db, task_id, runner_id="worker"
            )
            db.commit()
            command = claim_for_owner(db, owner_lease, accepted.command_db_id)
        assert owner_lease is not None
        if fail_completion:
            monkeypatch.setattr(
                consumer, "finish_task_command_no_commit", lambda *args, **kwargs: False
            )
            with pytest.raises(TaskCommandRejected):
                consumer._commit_handoff(command, owner_lease)
        else:
            handoff = consumer._commit_handoff(command, owner_lease)
            with pytest.raises(TaskCommandRejected):
                consumer._commit_handoff(command, owner_lease)
        with sessions() as db:
            task = db.get(Task, task_id)
            row = db.get(TaskExecutionCommand, command.id)
            if fail_completion:
                assert task.lease_attempt_id == owner_lease.attempt_id
                assert task.status == TaskStatus.PENDING
                assert task.run_id is None
                assert row.status == "processing"
            else:
                assert task.lease_attempt_id == handoff.claimed.task_lease.attempt_id
                assert task.run_id == accepted.run_id
                assert task.status == TaskStatus.RUNNING
                assert row.status == "completed"
                assert row.result["lease_attempt_id"] == task.lease_attempt_id


def _settle_control(db: Session, control_id: int) -> None:
    task_execution_admission.settle_cap_held_commands_no_commit(db, control_id)
    db.commit()


def _snap(row: TaskExecutionCommand) -> tuple[object, ...]:
    return (row.status, row.attempt_count, row.error, row.result, row.completed_at)


@pytest.mark.parametrize(
    "control,case",
    [
        ("cancel", "claimed_once"),
        ("cancel", "already_settled"),
        ("pause", "claimed_once"),
        ("pause", "already_settled"),
        ("pause", "pause_ticket_row"),
    ],
)
def test_control_settle_skips_claimed_or_settled_rows(monkeypatch, control, case):
    """The settlement writes a row only while it is still pending, never
    claimed and not waiting for an admission ticket; a row claimed once, one an
    earlier control already settled, or one holding a ticket is left exactly as
    it is."""

    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    with disposable_database_factory("control_settle") as make_database:
        engine = make_database("cas")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        with sessions() as db:
            owner = User(username="owner", password_hash="unused")
            db.add(owner)
            db.flush()
            task = Task(
                user_id=owner.id,
                title="Shared",
                source="a2a",
                status=TaskStatus.FAILED,
                state_version=2,
            )
            db.add(task)
            db.flush()
            task_id = task.id

            def add(command_id: str, kind: str, **fields) -> int:
                row = TaskExecutionCommand(
                    task_id=task_id,
                    command_id=command_id,
                    kind=kind,
                    payload={},
                    actor_user_id=owner.id,
                    target_state_version=fields.pop("target_state_version", 1),
                    **fields,
                )
                db.add(row)
                db.flush()
                return int(row.id)

            if case == "claimed_once":
                held_id = add("held", "start", attempt_count=1)
            elif case == "pause_ticket_row":
                held_id = add("held", "resume", target_state_version=2)
                db.add(TaskAdmissionBucket(key="bucket", capacity=1, max_pending=10))
                db.flush()
                db.add(
                    TaskAdmissionTicket(
                        command_id=held_id,
                        bucket_key="bucket",
                        task_id=task_id,
                        owner_attempt_id=None,
                    )
                )
                db.flush()
            else:
                held_id = add("held", "start")
                first_control = add(f"{control}-1", control, status="completed")
                _settle_control(db, first_control)
                assert db.get(TaskExecutionCommand, held_id).status == "failed"
            second_control = add(f"{control}-2", control, status="completed")
            db.commit()
            snapshot = _snap(db.get(TaskExecutionCommand, held_id))
            events_before = db.query(TaskCommandTerminalEvent).count()

            _settle_control(db, second_control)

            db.expire_all()
            assert _snap(db.get(TaskExecutionCommand, held_id)) == snapshot
            assert db.query(TaskCommandTerminalEvent).count() == events_before
