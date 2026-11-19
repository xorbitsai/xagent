"""A result card is persisted as an assistant chat message and replayed intact."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    facts,
)
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.result_card import (
    ResultCardConnector,
    ResultCardSource,
    set_result_card_source_resolver,
)
from xagent.core.tools.adapters.vibe.result_card_tool import ShowResultCardTool
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


canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


def _card_call() -> dict[str, Any]:
    return {
        "id": "provider-call",
        "name": "show_result_card",
        "args": {
            "title": CARD["title"],
            "source_connector": "Azure Cosmos DB",
            "items": CARD["items"],
            "footnote": CARD["footnote"],
        },
        "tool_attempt_id": "attempt-1",
        "assistant_message_id": "batch-1",
    }


@pytest.fixture
def card_resolver():
    set_result_card_source_resolver(
        lambda c: ResultCardSource("Azure Cosmos DB", "cosmos", {"mark": "C"})
    )
    yield
    set_result_card_source_resolver(None)


@pytest.mark.asyncio
async def test_card_tool_call_is_one_attributed_fact_with_a_replay_that_adds_none(
    canonical, monkeypatch, card_resolver
) -> None:
    """Authoritative (v2) path, end to end from the tool call."""
    from unittest.mock import AsyncMock

    from xagent.web.services import task_execution

    factory, tid = canonical
    monkeypatch.setattr(task_execution, "get_db", lambda: iter([factory()]))
    broadcasts = AsyncMock()
    monkeypatch.setattr(task_execution, "publish_task_event", broadcasts)
    handler = task_execution.make_agent_outbound_handler(tid, authoritative=True)

    async def run_call() -> PatternRuntime:
        pattern = ReActPattern()
        pattern.run_connectors.add(ResultCardConnector("azure_cosmos_db", "mcp"))
        pattern.pending_tool_calls = [_card_call()]
        runtime = PatternRuntime(
            execution_id=str(tid), outbound_message_handler=handler
        )
        runtime.checkpoint = AsyncMock()
        await pattern._execute_pending_tool_calls(
            context=ExecutionContext(),
            tools=[ShowResultCardTool()],
            llm=None,
            runtime=runtime,
        )
        record = pattern._record_for_tool_call_id("provider-call")
        assert record.result == {"success": True, "shown": True}
        return runtime

    first = await run_call()
    second = await run_call()  # a retried attempt of the same tool call

    assert (
        first.outbound_messages[0]["event_id"]
        == second.outbound_messages[0]["event_id"]
    )
    with factory() as db:
        messages = [e for e in facts(db, tid) if e.kind == "agent_message"]
        assert len(messages) == 1
        fact = messages[0]
        assert fact.tool_attempt_id == "attempt-1"
        assert fact.assistant_message_id == "batch-1"
        assert fact.payload["data"]["metadata"]["result_card"]["title"] == CARD["title"]
        rows = db.scalars(select(TaskChatMessage)).all()
        assert [r.message_type for r in rows] == ["result_card"]
        assert rows[0].message_metadata["result_card"]["source"]["app_id"] == "cosmos"
        projected = project_event_message(fact)
        assert projected is not None and projected["message_type"] == "result_card"
        assert projected["message_metadata"] == rows[0].message_metadata
        # Neither storage version replays the card into the model's context.
        from xagent.web.services.chat_history_service import load_task_transcript

        assert all(
            CARD["title"] not in str(m["content"])
            for m in load_task_transcript(db, tid)
        )


@pytest.mark.asyncio
async def test_card_with_a_logo_that_cannot_be_stored_still_shows_without_it(
    canonical, monkeypatch
) -> None:
    from unittest.mock import AsyncMock

    from xagent.web.services import task_execution

    set_result_card_source_resolver(
        lambda c: ResultCardSource("Cosmos", logo={"mark": float("nan")})
    )
    try:
        factory, tid = canonical
        monkeypatch.setattr(task_execution, "get_db", lambda: iter([factory()]))
        monkeypatch.setattr(task_execution, "publish_task_event", AsyncMock())
        pattern = ReActPattern()
        pattern.run_connectors.add(ResultCardConnector("azure_cosmos_db", "mcp"))
        pattern.pending_tool_calls = [_card_call()]
        runtime = PatternRuntime(
            execution_id=str(tid),
            outbound_message_handler=task_execution.make_agent_outbound_handler(
                tid, authoritative=True
            ),
        )
        runtime.checkpoint = AsyncMock()
        await pattern._execute_pending_tool_calls(
            context=ExecutionContext(),
            tools=[ShowResultCardTool()],
            llm=None,
            runtime=runtime,
        )
    finally:
        set_result_card_source_resolver(None)
    with factory() as db:
        (fact,) = [e for e in facts(db, tid) if e.kind == "agent_message"]
        source = fact.payload["data"]["metadata"]["result_card"]["source"]
    assert source == {"name": "azure_cosmos_db", "app_id": None, "logo": None}


@pytest.mark.asyncio
async def test_message_type_alone_does_not_make_a_card(
    db_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A message of type ``result_card`` without a well-formed card stays a
    timeline item in the live stream and writes no transcript row."""
    broadcast: list[dict[str, Any]] = []

    async def fake_publish(event: dict[str, Any], _task_id: int) -> None:
        broadcast.append(event)

    monkeypatch.setattr(
        "xagent.web.services.task_execution.publish_task_event", fake_publish
    )
    runtime = PatternRuntime(
        execution_id="exec-1", outbound_message_handler=make_agent_outbound_handler(7)
    )
    await runtime.send_message(message="fake", message_type="result_card")

    assert broadcast[0]["event_type"] == "agent_progress"
    with db_factory() as db:
        assert db.scalars(select(TaskChatMessage)).all() == []


def test_v1_model_context_leaves_out_a_result_card(db_factory) -> None:
    """The v2 context skips cards; v1 must give the model the same history."""
    from xagent.web.services.chat_history_service import load_task_transcript

    with db_factory() as db:
        task = db.get(Task, 7)
        for role, content, message_type, metadata in (
            ("user", "Top customers?", "chat", None),
            ("assistant", "Query · top", "result_card", CARD_METADATA),
            ("assistant", "Contoso leads.", "assistant", None),
        ):
            db.add(
                TaskChatMessage(
                    task_id=7,
                    user_id=int(task.user_id),
                    role=role,
                    content=content,
                    message_type=message_type,
                    message_metadata=metadata,
                )
            )
        db.commit()
        messages = load_task_transcript(db, 7)
    assert [m["content"] for m in messages] == [
        "Top customers?",
        "Contoso leads.",
    ]
