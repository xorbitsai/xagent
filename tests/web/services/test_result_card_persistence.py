"""A result card is persisted as an assistant chat message and replayed intact."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.core.agent import PatternRuntime
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.models.user import User
from xagent.web.services.assistant_history_safety import (
    KNOWN_SAFE_ASSISTANT_MESSAGE_TYPES,
)
from xagent.web.services.task_event_display import (
    convert_display_event,
    project_event_message,
)
from xagent.web.services.task_execution import make_agent_outbound_handler

CARD = {
    "title": "Query · top customers",
    "source": {"name": "Azure Cosmos DB", "app_id": "cosmos", "logo": None},
    "items": ["Contoso Ltd · $182,400"],
    "footnote": "1 of 1,284 customers",
    "created_at": "2026-11-17T06:00:00Z",
}
CARD_METADATA = {"display": "result_card", "result_card": CARD}


@pytest.fixture
def db_factory(monkeypatch: pytest.MonkeyPatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        user = User(username="u", password_hash="x", is_admin=False)
        db.add(user)
        db.commit()
        db.add(
            Task(
                id=7,
                user_id=int(user.id),
                title="t",
                description="d",
                status=TaskStatus.RUNNING,
            )
        )
        db.commit()
    monkeypatch.setattr(
        "xagent.web.services.task_execution.get_db", lambda: iter([factory()])
    )
    return factory


@pytest.mark.asyncio
async def test_card_is_broadcast_as_a_chat_message_and_persisted(
    db_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    broadcast: list[dict[str, Any]] = []

    async def fake_publish(event: dict[str, Any], _task_id: int) -> None:
        broadcast.append(event)

    monkeypatch.setattr(
        "xagent.web.services.task_execution.publish_task_event", fake_publish
    )
    runtime = PatternRuntime(
        execution_id="exec-1", outbound_message_handler=make_agent_outbound_handler(7)
    )
    await runtime.send_message(
        message=CARD["title"],
        message_type="result_card",
        expect_response=False,
        metadata=CARD_METADATA,
    )

    (event,) = broadcast
    assert event["event_type"] == "agent_message"
    assert event["data"]["display"] == "chat"
    assert event["data"]["expect_response"] is False
    assert event["data"]["message_type"] == "result_card"
    assert event["data"]["metadata"]["display"] == "result_card"
    assert event["data"]["metadata"]["result_card"] == CARD

    with db_factory() as db:
        (row,) = db.scalars(select(TaskChatMessage)).all()
    assert row.role == "assistant"
    assert row.message_type == "result_card"
    assert row.content == CARD["title"]
    assert row.message_metadata == CARD_METADATA


@pytest.mark.asyncio
async def test_plain_progress_message_is_still_timeline_only(
    db_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    broadcast: list[dict[str, Any]] = []

    async def fake_publish(event: dict[str, Any], _task_id: int) -> None:
        broadcast.append(event)

    monkeypatch.setattr(
        "xagent.web.services.task_execution.publish_task_event", fake_publish
    )
    runtime = PatternRuntime(
        execution_id="exec-1", outbound_message_handler=make_agent_outbound_handler(7)
    )
    await runtime.send_message(message="working", message_type="info")

    assert broadcast[0]["event_type"] == "agent_progress"
    with db_factory() as db:
        assert db.scalars(select(TaskChatMessage)).all() == []


def _card_event(metadata: dict[str, Any]) -> TaskExecutionEvent:
    return TaskExecutionEvent(
        event_id="e1",
        task_id=7,
        sequence=3,
        scope_id="root",
        kind="agent_message",
        payload_version=1,
        payload={
            "protocol_event_id": "evt-1",
            "data": {
                "message": CARD["title"],
                "message_type": "result_card",
                "expect_response": False,
                "visible": True,
                "metadata": metadata,
            },
        },
        occurred_at=datetime(2026, 11, 17, tzinfo=timezone.utc),
    )


def test_event_backed_card_projects_as_an_assistant_message() -> None:
    projected = project_event_message(_card_event(CARD_METADATA))
    assert projected is not None
    assert projected["role"] == "assistant"
    assert projected["message_type"] == "result_card"
    assert projected["message_metadata"] == CARD_METADATA
    wire = convert_display_event(_card_event(CARD_METADATA), replay=True)
    assert wire["data"]["metadata"]["result_card"] == CARD


def test_other_agent_messages_are_not_projected_as_cards() -> None:
    assert project_event_message(_card_event({"display": "result_card"})) is None
    assert project_event_message(_card_event({"interactions": []})) is None


def test_result_card_is_a_known_safe_assistant_message_type() -> None:
    assert "result_card" in KNOWN_SAFE_ASSISTANT_MESSAGE_TYPES
