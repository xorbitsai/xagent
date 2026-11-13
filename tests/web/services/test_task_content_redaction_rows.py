"""Row-level redaction and purge on both SQLite and PostgreSQL.

``redact_task_rows`` and ``purge_task_rows`` share their content-row delete,
whose statement order exists for foreign keys and the active-interaction
CHECK. Both are run here against both dialects with foreign keys enforced,
starting from a task whose checkpoint pointer and active interaction both
name its trace row -- the shape that order exists for.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_interaction_schema_shared import make_row
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import (
    DAGExecution,
    Task,
    TaskStatus,
    TraceCheckpointBlob,
    TraceEvent,
    TraceMessageBlob,
)
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.task_deletion import purge_task_rows, redact_task_rows

engine = engine_fixture

NOW = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
UPDATED_AT = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

CONTENT_TABLES = (
    TraceEvent,
    TraceMessageBlob,
    TraceCheckpointBlob,
    DAGExecution,
    TaskChatMessage,
    TaskExecutionEvent,
    TaskInteractionRequest,
)


@pytest.fixture
def sessions(engine) -> sessionmaker[Session]:
    """Sessions shaped like production's: ``autoflush=False``."""
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False)


def _seed(db: Session) -> tuple[int, int]:
    """A completed task with content, an anchored active interaction and a file.

    Returns ``(task_id, owner_id)``.
    """
    user = User(username="owner", password_hash="unused")
    db.add(user)
    db.flush()
    task = Task(
        user_id=user.id,
        title="secret",
        input="secret",
        output="secret",
        status=TaskStatus.COMPLETED,
        input_tokens=7,
        output_tokens=3,
    )
    db.add(task)
    db.flush()
    trace = TraceEvent(
        task_id=task.id,
        event_id="evt-1",
        event_type="agent_execution_checkpoint",
        timestamp=NOW,
        data={"content": "secret"},
    )
    db.add(trace)
    db.flush()
    db.add_all(
        [
            TraceMessageBlob(
                task_id=task.id,
                execution_id="exec",
                message_hash="m",
                message_data={},
                message_bytes=2,
            ),
            TraceCheckpointBlob(
                task_id=task.id,
                execution_id="exec",
                blob_kind="messages",
                blob_hash="c",
                blob_data={},
                blob_bytes=2,
            ),
            DAGExecution(task_id=task.id),
            TaskChatMessage(
                task_id=task.id,
                user_id=user.id,
                role="user",
                content="secret",
                message_type="user_message",
            ),
            TaskExecutionEvent(
                event_id="exec-event-1",
                task_id=task.id,
                sequence=1,
                scope_id="root",
                idempotency_key="k1",
                kind="assistant_message",
                payload_version=1,
                payload={"content": "secret"},
                occurred_at=NOW,
            ),
            TaskExecutionCommand(
                task_id=task.id,
                command_id="turn-1",
                kind="message",
                payload={"content": "secret"},
                status="completed",
            ),
            TaskInteractionRequest(
                **make_row(task_id=task.id, resume_trace_event_id=trace.id, db=db)
            ),
            # No durable key and a path outside the uploads root: the
            # redaction deletes the row and has no bytes to touch.
            UploadedFile(
                user_id=user.id,
                task_id=task.id,
                filename="a.txt",
                storage_path="/nonexistent/a.txt",
                file_size=1,
            ),
        ]
    )
    db.execute(
        sa.update(Task)
        .where(Task.id == task.id)
        .values(
            last_checkpoint_event_id="evt-1",
            last_checkpoint_trace_event_id=trace.id,
            interaction_protocol_version=1,
            updated_at=UPDATED_AT,
        )
    )
    db.commit()
    return int(task.id), int(user.id)


def _count(db: Session, model: type, task_id: int) -> int:
    return int(
        db.execute(
            sa.select(sa.func.count())
            .select_from(model)
            .where(model.task_id == task_id)  # type: ignore[attr-defined]
        ).scalar_one()
    )


def test_redact_task_rows_keeps_the_task_and_removes_its_content(sessions) -> None:
    with sessions() as db:
        task_id, owner_id = _seed(db)

    for _ in range(2):  # the second run is the idempotency check
        with sessions() as db:
            assert redact_task_rows(db, task_id=task_id, now=NOW) is True
            db.commit()

    with sessions() as db:
        assert {
            model.__tablename__: _count(db, model, task_id) for model in CONTENT_TABLES
        } == {model.__tablename__: 0 for model in CONTENT_TABLES}
        assert (
            db.query(UploadedFile).filter(UploadedFile.user_id == owner_id).count() == 0
        )
        task = db.get(Task, task_id)
        assert task is not None
        assert (task.title, task.input, task.output) == ("", None, None)
        assert (task.input_tokens, task.output_tokens) == (7, 3)
        assert task.updated_at.replace(tzinfo=timezone.utc) == UPDATED_AT
        assert task.traces_expired_at.replace(tzinfo=timezone.utc) == NOW
        [command] = db.query(TaskExecutionCommand).filter_by(task_id=task_id).all()
        assert command.payload == {}


def test_redact_task_rows_returns_false_for_a_missing_task(sessions) -> None:
    with sessions() as db:
        assert redact_task_rows(db, task_id=987654321, now=NOW) is False


def test_purge_task_rows_still_deletes_the_task_and_detaches_files(sessions) -> None:
    with sessions() as db:
        task_id, owner_id = _seed(db)

    with sessions() as db:
        assert purge_task_rows(db, task_id=task_id, detached_reason="task_deleted")
        db.commit()

    with sessions() as db:
        assert db.get(Task, task_id) is None
        assert all(_count(db, model, task_id) == 0 for model in CONTENT_TABLES)
        [file] = db.query(UploadedFile).filter(UploadedFile.user_id == owner_id).all()
        assert (file.task_id, file.detached_reason) == (None, "task_deleted")
