"""Live trace frames carry the identity of the persisted trace row."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from xagent.core.agent.trace import (
    TraceAction,
    TraceCategory,
    TraceEvent,
    TraceEventType,
    TraceScope,
)
from xagent.web.services import task_event_trace_handler as handler_module
from xagent.web.services.task_event_trace_handler import TaskEventTraceHandler

TASK_ID = 4242


def _ai_message_event(task_id: int = TASK_ID) -> TraceEvent:
    return TraceEvent(
        event_type=TraceEventType(
            TraceScope.TASK, TraceAction.UPDATE, TraceCategory.MESSAGE
        ),
        task_id=str(task_id),
        data={"content": "hello"},
    )


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []

    async def fake_publish(frame: dict[str, Any], task_id: int) -> None:
        frames.append(frame)

    async def no_description(self: TaskEventTraceHandler) -> None:
        return None

    monkeypatch.setattr(handler_module, "publish_task_event", fake_publish)
    monkeypatch.setattr(handler_module, "task_has_audience", lambda task_id: True)
    monkeypatch.setattr(TaskEventTraceHandler, "_load_task_description", no_description)
    monkeypatch.setattr(
        TaskEventTraceHandler,
        "_has_prior_user_message_turn",
        lambda self, *args, **kwargs: False,
    )
    return frames


@pytest.mark.asyncio
async def test_non_authoritative_live_frame_uses_persisted_event_id(
    published: list[dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from xagent.web.models.database import Base
    from xagent.web.models.task import Task
    from xagent.web.models.task import TraceEvent as TraceEventRow
    from xagent.web.models.user import User
    from xagent.web.services import trace_handlers
    from xagent.web.services.trace_database import TraceDatabaseRuntime

    source = create_engine(f"sqlite:///{tmp_path / 'trace.db'}")
    Base.metadata.create_all(source)
    with Session(source) as db:
        user = User(username="trace-event-id", password_hash="unused")
        db.add(user)
        db.flush()
        task = Task(user_id=user.id, title="Event id")
        db.add(task)
        db.flush()
        task_id = task.id
        db.commit()
    runtime = TraceDatabaseRuntime(source, use_async=True, limit=4)
    monkeypatch.setattr(trace_handlers, "get_trace_database_runtime", lambda: runtime)
    event = _ai_message_event(task_id)
    try:
        # The same event object goes to both handlers, as the tracer does.
        await trace_handlers.DatabaseTraceHandler(task_id).handle_event(event)
        await TaskEventTraceHandler(task_id).handle_event(event)
        with Session(source) as db:
            persisted = list(
                db.scalars(
                    select(TraceEventRow.event_id).where(
                        TraceEventRow.task_id == task_id
                    )
                )
            )
    finally:
        await runtime.close()
        source.dispose()

    assert persisted == [str(event.id)]
    assert len(published) == 1
    assert published[0]["event_id"] == persisted[0]


@pytest.mark.asyncio
async def test_authoritative_branch_frame_is_not_rewritten(
    published: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    authoritative_frame = {
        "type": "trace_event",
        "event_id": str(uuid.uuid4()),
        "message_id": "message-from-display-store",
        "event_type": "ai_message",
        "task_id": TASK_ID,
        "data": {"content": "hello"},
    }
    monkeypatch.setattr(
        TaskEventTraceHandler,
        "_load_event_display",
        lambda self, protocol_event_id: dict(authoritative_frame),
    )

    await TaskEventTraceHandler(TASK_ID, authoritative=True).handle_event(
        _ai_message_event()
    )

    assert published == [authoritative_frame]
