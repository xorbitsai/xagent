"""Regression pin for xorbitsai/xagent#1328.

``execute_resume_background`` resumes an agent that may have been rebuilt
from history (process restart, cache miss, other worker) with no outbound
message handler installed. Without one, ``PatternRuntime.send_message``
only appends the payload to an in-memory list and drops it -- a follow-up
``ask_user_question`` is then neither persisted to the chat transcript nor
broadcast to any listener. This pins that the resume path installs the
same task-scoped handler the initial run installs, so a question raised
mid-resume is persisted and broadcast like any other agent message.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.core.agent.runtime import PatternRuntime
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services import task_events
from xagent.web.services.task_execution import (
    background_task_manager,
    execute_resume_background,
)

FOLLOW_UP_QUESTION = "What is the second value?"


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'resume_outbound_handler.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _user(db, username, *, is_admin=False) -> User:
    u = User(username=username, password_hash="x", is_admin=is_admin)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _task(db, owner_id: int, status: TaskStatus = TaskStatus.RUNNING) -> Task:
    t = Task(
        user_id=owner_id,
        title="t",
        description="d",
        status=status,
        execution_mode="balanced",
        source="sdk",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


def _register_current_resume(task_id: int) -> None:
    current = asyncio.current_task()
    assert current is not None
    background_task_manager.resume_tasks[task_id] = current


@pytest.mark.asyncio
async def test_resume_installs_outbound_handler_so_follow_up_question_persists(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resumed agent that was rebuilt from history has no handler until
    ``execute_resume_background`` installs one. The fake resume drives a
    follow-up question through a real ``PatternRuntime`` wired to whatever
    handler the resume installed, so the payload is the shape production
    emits, and checks it lands in the chat transcript, the trace log, and
    the live event sink -- exactly like a question raised during the
    initial run does. Without the install the runtime has no handler, only
    warns, and none of the three assertions below can hold."""

    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    task_id = int(task.id)

    agent = MagicMock()
    agent.set_outbound_message_handler = MagicMock()

    async def _resume(_task_id_str: str) -> dict[str, Any]:
        call = agent.set_outbound_message_handler.call_args
        handler = call.args[0] if call is not None else None
        runtime = PatternRuntime(
            execution_id=str(task_id), outbound_message_handler=handler
        )
        await runtime.send_message(
            message=FOLLOW_UP_QUESTION,
            message_type="question",
            expect_response=True,
        )
        return {
            "status": "waiting_for_user",
            "success": True,
            "output": "",
            "agent_result": {"context": SimpleNamespace(messages=[])},
        }

    agent.resume_execution_by_id = AsyncMock(side_effect=_resume)

    sink = AsyncMock()
    monkeypatch.setattr(task_events, "_task_event_sink", sink)

    try:
        _register_current_resume(task_id)
        await execute_resume_background(
            task_id=task_id,
            agent_service=agent,
            task_owner_user_id=int(owner.id),
        )
    finally:
        background_task_manager.resume_tasks.pop(task_id, None)

    agent.set_outbound_message_handler.assert_called_once()

    # Persisted to the transcript.
    db_session.expire_all()
    stored = (
        db_session.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == task_id,
            TaskChatMessage.role == "assistant",
        )
        .one()
    )
    assert stored.content == FOLLOW_UP_QUESTION
    assert stored.message_type == "question"

    # Recorded as the agent_message trace event the projections read.
    trace_event = (
        db_session.query(TraceEvent)
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.event_type == "agent_message",
        )
        .one()
    )
    assert trace_event.data["message"] == FOLLOW_UP_QUESTION
    assert trace_event.data["expect_response"] is True

    # Delivered to the live subscriber.
    published = [
        call.args[0] for call in sink.call_args_list if isinstance(call.args[0], dict)
    ]
    question_events = [e for e in published if e.get("event_type") == "agent_message"]
    assert len(question_events) == 1
    assert question_events[0]["data"]["message"] == FOLLOW_UP_QUESTION
    assert question_events[0]["data"]["expect_response"] is True
