"""Shared handoff and deployment schema checks against disposable PostgreSQL."""

from unittest.mock import Mock

import pytest
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import (
    disposable_database_factory,
)
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import task_event_bridge
from xagent.web.services import task_start_consumer as consumer
from xagent.web.services.task_command_transport import (
    TaskCommandRejected,
    claim_task_command,
)
from xagent.web.services.task_orchestrator import TaskTurnOrchestrator, TaskTurnPayload

pytestmark = pytest.mark.postgresql


@pytest.mark.parametrize("fail_completion", [False, True])
def test_postgres_handoff_completion_and_lease_are_atomic(monkeypatch, fail_completion):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(task_event_bridge, "get_task_event_bridge", lambda: Mock())
    monkeypatch.setattr(consumer, "get_runner_id", lambda: "worker")
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
            command = claim_task_command(
                db, runner_id="worker", command_db_id=accepted.command_db_id
            )
            task_id = task.id
        if fail_completion:
            monkeypatch.setattr(
                consumer, "finish_task_command_no_commit", lambda *args, **kwargs: False
            )
            with pytest.raises(TaskCommandRejected):
                consumer._commit_handoff(command)
        else:
            handoff = consumer._commit_handoff(command)
            with pytest.raises(TaskCommandRejected):
                consumer._commit_handoff(command)
        with sessions() as db:
            task = db.get(Task, task_id)
            row = db.get(TaskExecutionCommand, command.id)
            if fail_completion:
                assert task.lease_attempt_id is None
                assert row.status == "processing"
            else:
                assert task.lease_attempt_id == handoff.claimed.task_lease.attempt_id
                assert row.status == "completed"
                assert row.result["lease_attempt_id"] == task.lease_attempt_id
