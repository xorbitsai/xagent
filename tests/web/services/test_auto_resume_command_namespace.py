"""The ``auto-resume:`` command id namespace is reserved at the transport.

Episode bookkeeping treats a RESUME with this prefix as automatic, so no
ingress may stage one; only a caller passing ``reserved=True`` (the
auto-resume sweeper) can.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services.task_auto_recovery import is_auto_resume_command_id
from xagent.web.services.task_command_transport import (
    TaskCommandKind,
    enqueue_task_command,
    stage_task_command,
)

engine = engine_fixture


@pytest.fixture
def factory(engine) -> sessionmaker:
    Base.metadata.create_all(engine)
    return sessionmaker(engine, autoflush=False)


@pytest.fixture
def task(factory) -> tuple[int, int]:
    with factory() as db:
        user = User(username="namespace-owner", password_hash="x")
        db.add(user)
        db.flush()
        task = Task(
            user_id=user.id,
            title="t",
            description="d",
            status=TaskStatus.PAUSED,
            control_state="paused",
            run_id="run-1",
            state_version=5,
        )
        db.add(task)
        db.commit()
        return int(task.id), int(user.id)


def _commands(factory, task_id: int) -> list[str]:
    with factory() as db:
        return list(
            db.scalars(
                sa.select(TaskExecutionCommand.command_id).where(
                    TaskExecutionCommand.task_id == task_id
                )
            )
        )


@pytest.mark.parametrize("kind", [TaskCommandKind.RESUME, TaskCommandKind.MESSAGE])
@pytest.mark.parametrize("command_id", ["auto-resume:5:1:run-1", " auto-resume:x"])
def test_transport_refuses_the_reserved_prefix(factory, task, kind, command_id):
    task_id, user_id = task
    with factory() as db:
        with pytest.raises(ValueError, match="reserved"):
            stage_task_command(
                db,
                task_id=task_id,
                actor_user_id=user_id,
                command_id=command_id,
                kind=kind,
                payload={},
            )
        with pytest.raises(ValueError, match="reserved"):
            enqueue_task_command(
                db,
                task_id=task_id,
                actor_user_id=user_id,
                command_id=command_id,
                kind=kind,
                payload={},
            )
    assert _commands(factory, task_id) == []


def test_reserved_callers_may_stage_the_namespace(factory, task):
    task_id, user_id = task
    with factory() as db:
        staged = stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=user_id,
            command_id="auto-resume:5:1:run-1",
            kind=TaskCommandKind.RESUME,
            payload={},
            reserved=True,
        )
        db.commit()
    assert staged.created
    assert _commands(factory, task_id) == ["auto-resume:5:1:run-1"]


def test_other_ids_are_unaffected(factory, task):
    task_id, user_id = task
    with factory() as db:
        staged = stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=user_id,
            command_id="resume:auto-resume:x",
            kind=TaskCommandKind.RESUME,
            payload={},
        )
        db.commit()
    assert staged.created


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto-resume:1:1:r", True),
        ("resume:1", False),
        (None, False),
        (7, False),
    ],
)
def test_is_auto_resume_command_id(value, expected):
    assert is_auto_resume_command_id(value) is expected
