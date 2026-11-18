"""A user message flagged ``metadata.hidden`` is stored and replayed unchanged.

The flag is plumbing for clients (for example a resume nudge they do not want
to draw); derived surfaces must not surface the message, while the text still
reaches the model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.core.agent.attachments import SERVER_METADATA_KEY
from xagent.core.agent.context import ContextManager
from xagent.core.agent.runner import AgentRunner
from xagent.core.agent.tracing import TraceEventCallback
from xagent.web.api.chat import get_tasks
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.models.user import User
from xagent.web.services.chat_history_service import (
    claim_user_message_delivery_no_commit,
    persist_user_message_no_commit,
)
from xagent.web.services.message_metadata import (
    is_hidden_message_metadata,
    validate_server_message_metadata,
)
from xagent.web.services.task_event_display import (
    convert_display_event,
    project_event_message,
)
from xagent.web.services.task_start_protocol import TaskStartPayload

HIDDEN = {"hidden": True, "kind": "interaction_settled", "interaction_id": "i-1"}


def test_validate_accepts_well_formed_server_metadata() -> None:
    assert validate_server_message_metadata(HIDDEN) == HIDDEN
    assert validate_server_message_metadata({"kind": "k"}) == {"kind": "k"}
    assert validate_server_message_metadata(None) is None
    assert validate_server_message_metadata({}) is None


@pytest.mark.parametrize(
    "raw",
    [
        [],
        "hidden",
        {"extra": 1},
        {**HIDDEN, "extra": {"a": 1}},
        {"hidden": True},
        {"hidden": "true", "kind": "k"},
        {"hidden": 1, "kind": "k"},
        {"kind": "Bad Kind"},
        {"kind": "k" * 65},
        {"kind": 3},
        {"kind": "ok", "interaction_id": "a b"},
        {"kind": "ok", "interaction_id": "x" * 65},
    ],
)
def test_validate_rejects_malformed_server_metadata(raw) -> None:
    with pytest.raises(ValueError):
        validate_server_message_metadata(raw)


def test_hidden_requires_literal_true() -> None:
    assert is_hidden_message_metadata({"hidden": True})
    assert not is_hidden_message_metadata({"hidden": "true"})
    assert not is_hidden_message_metadata({"hidden": 1})
    assert not is_hidden_message_metadata(None)


def _sqlite_session():
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    db = sessionmaker(bind=eng)()
    user = User(username="u", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    task = Task(
        user_id=int(user.id),
        title="t",
        description="d",
        status=TaskStatus.PENDING,
    )
    db.add(task)
    db.commit()
    return db, user, task


def test_legacy_row_round_trips_metadata() -> None:
    db, user, task = _sqlite_session()
    persist_user_message_no_commit(
        db,
        task.id,
        user.id,
        "[connector settled]",
        turn_id="t1",
        message_metadata=HIDDEN,
    )
    persist_user_message_no_commit(db, task.id, user.id, "plain", turn_id="t2")
    claim = claim_user_message_delivery_no_commit(
        db,
        task.id,
        user.id,
        "[connector settled]",
        turn_id="t3",
        message_metadata=HIDDEN,
    )
    db.commit()
    rows = {
        r.turn_id: r.message_metadata for r in db.scalars(select(TaskChatMessage)).all()
    }
    assert rows["t1"] == HIDDEN
    assert rows["t3"] == HIDDEN
    assert not rows["t2"]
    assert claim.message.message_metadata == HIDDEN


@pytest.mark.asyncio
async def test_task_search_skips_hidden_messages() -> None:
    db, user, task = _sqlite_session()
    visible = Task(
        user_id=int(user.id), title="v", description="d", status=TaskStatus.PENDING
    )
    db.add(visible)
    db.commit()
    persist_user_message_no_commit(
        db,
        task.id,
        user.id,
        "[connector settled]",
        turn_id="t1",
        message_metadata=HIDDEN,
    )
    persist_user_message_no_commit(
        db, visible.id, user.id, "[connector settled] typed by a human", turn_id="t2"
    )
    db.commit()
    result = await get_tasks(search="connector settled", db=db, user=user)
    assert [t["task_id"] for t in result["tasks"]] == [visible.id]


def _input_event(metadata):
    payload = {
        "user_id": 1,
        "role": "user",
        "content": "[connector settled]",
        "message_type": "user_message",
        "attachments": None,
        "interactions": None,
        "turn_id": "h1",
    }
    if metadata:
        payload["message_metadata"] = metadata
    return TaskExecutionEvent(
        event_id="e1",
        task_id=1,
        sequence=1,
        scope_id="root",
        kind="input_accepted",
        turn_id="h1",
        payload_version=1,
        payload=payload,
        occurred_at=datetime(2026, 11, 17, tzinfo=timezone.utc),
    )


def test_event_backed_message_keeps_metadata_on_replay_and_live() -> None:
    hidden_event = _input_event(HIDDEN)
    assert project_event_message(hidden_event)["message_metadata"] == HIDDEN
    wire = convert_display_event(hidden_event, replay=True)
    assert wire["data"]["metadata"] == HIDDEN
    assert wire["data"]["content"] == "[connector settled]"

    plain_event = _input_event(None)
    assert project_event_message(plain_event)["message_metadata"] is None
    assert "metadata" not in convert_display_event(plain_event, replay=True)["data"]


class _Recorder:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def trace_event(self, event_type, *, task_id=None, data=None, **kw):
        self.events.append({"event_type": event_type, "data": data})


@pytest.mark.asyncio
async def test_runner_keeps_metadata_on_message_and_trace_event() -> None:
    manager = ContextManager()
    context = manager.create_context("exec-hidden")
    context.add_user_message("original")
    tracer = SimpleNamespace(
        load_latest_checkpoint=AsyncMock(return_value=None), checkpoint=AsyncMock()
    )
    runner = AgentRunner(
        SimpleNamespace(llm=None), tracer=tracer, context_manager=manager
    )
    await runner.inject_user_message(
        "exec-hidden",
        "[connector settled]",
        turn_id="turn-h",
        message_metadata=HIDDEN,
    )
    message = context.messages[-1]
    assert message.content == "[connector settled]"  # still model context
    assert message.metadata[SERVER_METADATA_KEY] == HIDDEN

    recorder = _Recorder()
    await TraceEventCallback().on_user_message_posted(
        runner=SimpleNamespace(tracer=recorder), context=context, message=message
    )
    assert recorder.events[0]["data"]["metadata"] == HIDDEN
    assert recorder.events[0]["data"]["turn_id"] == "turn-h"


def test_start_payload_wire_shape_unchanged_without_metadata() -> None:
    base = dict(version=1, run_id="r", state_version=0, turn_id="t", kind="append")
    plain = TaskStartPayload(**base, message="m").model_dump(mode="json")
    assert "message_metadata" not in plain
    hidden = TaskStartPayload(**base, message="m", message_metadata=HIDDEN)
    assert hidden.model_dump(mode="json")["message_metadata"] == HIDDEN


@pytest.mark.parametrize(
    ("metadata", "expected_input"),
    [(HIDDEN, "original request"), (None, "[connector settled]")],
    ids=["hidden", "visible"],
)
def test_a_hidden_turn_leaves_the_task_input_alone(
    monkeypatch, metadata, expected_input
) -> None:
    from xagent.web.services.task_orchestrator import (
        TaskTurnPayload,
        TurnKind,
        _accept_turn_no_commit,
    )

    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    db, user, task = _sqlite_session()
    task.status = TaskStatus.COMPLETED
    task.input = "original request"
    db.commit()

    _accept_turn_no_commit(
        db,
        int(task.id),
        int(user.id),
        payload=TaskTurnPayload(
            "[connector settled]", turn_id="t1", message_metadata=metadata
        ),
        kind=TurnKind.APPEND,
    )
    db.commit()
    db.expire_all()
    assert db.get(Task, task.id).input == expected_input
