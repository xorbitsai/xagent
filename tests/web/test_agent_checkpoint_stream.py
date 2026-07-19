from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE, CHECKPOINT_TYPE
from xagent.core.agent.trace import (
    TraceAction,
    TraceCategory,
    TraceEvent,
    TraceEventType,
    TraceScope,
)
from xagent.web.api.trace_handlers import DatabaseTraceHandler
from xagent.web.api.websocket import (
    _agent_outbound_event_type,
    _is_agent_checkpoint_data,
    _is_duplicate_user_message_turn,
    _persist_agent_outbound_event,
    create_final_answer_stream_event,
    create_stream_event,
    make_agent_outbound_handler,
    send_historical_data_as_stream,
)
from xagent.web.api.ws_trace_handlers import (
    WebSocketTraceHandler,
    get_event_type_mapping,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task import TraceEvent as DatabaseTraceEvent
from xagent.web.models.user import User


def test_agent_checkpoint_is_not_converted_to_websocket_stream_event() -> None:
    event = TraceEvent(
        CHECKPOINT_EVENT_TYPE,
        task_id="365",
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": "365",
            "snapshot": {"label": "dag_before_llm"},
        },
    )

    stream_event = WebSocketTraceHandler(365)._convert_trace_event_to_stream_event(
        event
    )

    assert stream_event is None


def test_action_tool_error_maps_to_tool_execution_failed() -> None:
    event = TraceEvent(
        TraceEventType(TraceScope.ACTION, TraceAction.ERROR, TraceCategory.TOOL),
        task_id="365",
        step_id="default",
        data={"tool_name": "execute_python_code", "error_message": "failed"},
    )

    assert get_event_type_mapping(event) == "tool_execution_failed"


def test_action_llm_error_maps_to_llm_call_failed() -> None:
    event = TraceEvent(
        TraceEventType(TraceScope.ACTION, TraceAction.ERROR, TraceCategory.LLM),
        task_id="365",
        step_id="365",
        data={"error_message": "read timed out"},
    )

    assert get_event_type_mapping(event) == "llm_call_failed"


def test_workforce_delegation_summary_maps_to_public_stream_event() -> None:
    event = TraceEvent(
        TraceEventType(TraceScope.TASK, TraceAction.UPDATE, TraceCategory.GENERAL),
        task_id="365",
        data={
            "event_type": "workforce_delegation_start",
            "status": "start",
            "agent_id": 12,
            "agent_name": "Researcher",
            "worker_alias": "Research",
            "worker_task_id": "agent_12_abcd1234",
            "messages": [{"role": "user", "content": "raw prompt"}],
        },
    )

    stream_event = WebSocketTraceHandler(365)._convert_trace_event_to_stream_event(
        event
    )

    assert stream_event is not None
    assert stream_event["event_type"] == "workforce_delegation_start"
    assert stream_event["data"]["worker_alias"] == "Research"
    assert stream_event["data"]["worker_task_id"] == "agent_12_abcd1234"
    assert "messages" not in stream_event["data"]
    assert "event_type" not in stream_event["data"]


def test_non_task_update_event_with_delegation_payload_is_not_promoted() -> None:
    event = TraceEvent(
        TraceEventType(TraceScope.ACTION, TraceAction.END, TraceCategory.TOOL),
        task_id="365",
        step_id="step-1",
        data={
            "event_type": "workforce_delegation_end",
            "tool_name": "agent_12",
            "output": "worker response",
        },
    )

    stream_event = WebSocketTraceHandler(365)._convert_trace_event_to_stream_event(
        event
    )

    assert stream_event is not None
    assert stream_event["event_type"] == "tool_execution_end"
    assert stream_event["data"]["event_type"] == "workforce_delegation_end"
    assert stream_event["data"]["output"] == "worker response"


def test_historical_stream_identifies_agent_checkpoint_payload() -> None:
    assert _is_agent_checkpoint_data(
        {
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": "365",
            "snapshot": {"label": "dag_before_llm"},
        }
    )
    assert _is_agent_checkpoint_data(
        {
            "type": "checkpoint",
            "execution_id": "365",
            "pattern_state": {"status": "running"},
            "context": {"messages": []},
        }
    )
    assert not _is_agent_checkpoint_data({"event": "ai_message"})


def test_final_answer_stream_event_is_not_trace_event() -> None:
    event = create_final_answer_stream_event(
        "final_answer_delta",
        365,
        {
            "type": "final_answer_delta",
            "message_id": "final_answer_1",
            "delta": "hello",
        },
    )

    assert event["type"] == "final_answer_delta"
    assert event["task_id"] == 365
    assert event["message_id"] == "final_answer_1"
    assert event["delta"] == "hello"
    assert "event_type" not in event
    assert "data" not in event


def test_agent_outbound_event_type_separates_progress_from_questions() -> None:
    assert (
        _agent_outbound_event_type(
            {
                "message": "Still working",
                "message_type": "progress",
                "expect_response": False,
            }
        )
        == "agent_progress"
    )
    assert (
        _agent_outbound_event_type(
            {
                "message": "Need input",
                "message_type": "question",
                "expect_response": False,
            }
        )
        == "agent_message"
    )
    assert (
        _agent_outbound_event_type(
            {
                "message": "Need input",
                "message_type": "info",
                "expect_response": True,
            }
        )
        == "agent_message"
    )


@pytest.mark.asyncio
async def test_agent_outbound_handler_skips_hidden_messages(monkeypatch) -> None:
    persisted_calls: list[tuple[int, dict[str, object]]] = []
    broadcast_calls: list[tuple[dict[str, object], int]] = []
    to_thread_calls: list[tuple[object, tuple[object, ...]]] = []

    def fake_persist(task_id: int, event: dict[str, object]) -> None:
        persisted_calls.append((task_id, event))

    async def fake_to_thread(func: object, /, *args: object) -> None:
        to_thread_calls.append((func, args))

    async def fake_broadcast(event: dict[str, object], task_id: int) -> None:
        broadcast_calls.append((event, task_id))

    monkeypatch.setattr(
        "xagent.web.api.websocket._persist_agent_outbound_event", fake_persist
    )
    monkeypatch.setattr("xagent.web.api.websocket.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr(
        "xagent.web.api.websocket.manager.broadcast_to_task", fake_broadcast
    )

    handler = make_agent_outbound_handler(365)
    await handler(
        {
            "execution_id": "exec-1",
            "message": "Hidden progress",
            "message_type": "progress",
            "expect_response": False,
            "visible": False,
        }
    )

    assert persisted_calls == []
    assert to_thread_calls == []
    assert broadcast_calls == []


@pytest.mark.asyncio
async def test_agent_outbound_handler_repairs_completed_final_answer(
    monkeypatch,
) -> None:
    broadcast_calls: list[tuple[dict[str, object], int]] = []

    def fake_reconcile(task_id: int, content: str) -> str:
        assert task_id == 365
        assert content == "[video.mp4](file:invented-id)"
        return "[video.mp4](file:real-id)"

    async def fake_to_thread(func: object, /, *args: object):
        return func(*args)

    async def fake_broadcast(event: dict[str, object], task_id: int) -> None:
        broadcast_calls.append((event, task_id))

    monkeypatch.setattr(
        "xagent.web.api.websocket._reconcile_streamed_final_answer",
        fake_reconcile,
    )
    monkeypatch.setattr("xagent.web.api.websocket.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr(
        "xagent.web.api.websocket.manager.broadcast_to_task", fake_broadcast
    )

    handler = make_agent_outbound_handler(365)
    await handler(
        {
            "type": "final_answer_end",
            "message_id": "final-answer-1",
            "content": "[video.mp4](file:invented-id)",
        }
    )

    assert len(broadcast_calls) == 1
    event, task_id = broadcast_calls[0]
    assert task_id == 365
    assert event["type"] == "final_answer_end"
    assert event["content"] == "[video.mp4](file:real-id)"


def test_persist_agent_outbound_event_uses_payload_ids(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="tester", password_hash="hashed_password", is_admin=False)
        db.add(user)
        db.commit()
        db.refresh(user)
        task = Task(
            user_id=int(user.id),
            title="Chat task",
            description="Task chat",
            status=TaskStatus.PENDING,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
    finally:
        db.close()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.websocket.get_db", get_test_db)

    event = create_stream_event(
        "agent_progress",
        int(task.id),
        {
            "event_id": "agent-event-1",
            "step_id": "react-step-1",
            "message": "Still working",
            "expect_response": False,
        },
    )

    _persist_agent_outbound_event(int(task.id), event)

    db = SessionLocal()
    try:
        trace_event = db.query(DatabaseTraceEvent).filter_by(task_id=int(task.id)).one()
        assert trace_event.event_id == "agent-event-1"
        assert trace_event.event_type == "agent_progress"
        assert trace_event.step_id == "react-step-1"
    finally:
        db.close()


def _create_trace_handler_test_task(
    username: str,
    *,
    title: str = "Chat task",
    description: str = "Task chat",
):
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    user = User(username=username, password_hash="hashed_password", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    task = Task(
        user_id=int(user.id),
        title=title,
        description=description,
        status=TaskStatus.PENDING,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return SessionLocal, db, task


@pytest.mark.asyncio
async def test_paused_replay_event_embeds_known_control_state(monkeypatch) -> None:
    SessionLocal, db, task = _create_trace_handler_test_task("paused-replay")
    task.status = TaskStatus.PAUSED
    task.run_id = "run-paused"
    task.state_version = 9
    task.control_state = "paused"
    db.commit()
    task_id = int(task.id)
    user_id = int(task.user_id)
    db.close()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    sent_events: list[dict] = []

    async def send_personal_message(event: dict, websocket: object) -> None:
        sent_events.append(event)

    monkeypatch.setattr("xagent.web.models.database.get_db", get_test_db)
    monkeypatch.setattr("xagent.web.api.websocket.cache_get", lambda *args: None)
    monkeypatch.setattr(
        "xagent.web.api.websocket.cache_set", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "xagent.web.api.websocket.manager.send_personal_message",
        send_personal_message,
    )

    await send_historical_data_as_stream(
        websocket=object(),
        task_id=task_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )

    paused_event = next(
        event for event in sent_events if event.get("type") == "task_paused"
    )
    assert paused_event["run_id"] == "run-paused"
    assert paused_event["state_version"] == 9
    assert paused_event["control_state"] == "paused"
    assert paused_event["status"] == "paused"


def test_database_trace_handler_dedupes_user_message_turn_id() -> None:
    _, db, task = _create_trace_handler_test_task("tester")
    try:
        handler = DatabaseTraceHandler(int(task.id))
        event_type = TraceEventType(
            TraceScope.TASK,
            TraceAction.START,
            TraceCategory.MESSAGE,
        )
        first = TraceEvent(
            event_type,
            task_id=str(task.id),
            data={"message": "Repeat", "turn_id": "turn-1"},
        )
        duplicate = TraceEvent(
            event_type,
            task_id=str(task.id),
            data={"message": "Repeat", "turn_id": "turn-1"},
        )
        different_turn = TraceEvent(
            event_type,
            task_id=str(task.id),
            data={"message": "Repeat", "turn_id": "turn-2"},
        )

        handler._save_trace_event(db, first)
        handler._save_trace_event(db, duplicate)
        handler._save_trace_event(db, different_turn)

        rows = (
            db.query(DatabaseTraceEvent)
            .filter_by(task_id=int(task.id), event_type="user_message")
            .order_by(DatabaseTraceEvent.id)
            .all()
        )
        assert [row.data["turn_id"] for row in rows] == ["turn-1", "turn-2"]
    finally:
        db.close()


def test_database_trace_handler_dedupes_user_message_turn_id_per_build() -> None:
    _, db, task = _create_trace_handler_test_task("build-tester")
    try:
        event_type = TraceEventType(
            TraceScope.TASK,
            TraceAction.START,
            TraceCategory.MESSAGE,
        )
        parent_handler = DatabaseTraceHandler(int(task.id))
        worker_handler = DatabaseTraceHandler(
            int(task.id),
            build_id="agent_123_abcd1234",
        )

        parent_handler._save_trace_event(
            db,
            TraceEvent(
                event_type,
                task_id=str(task.id),
                data={"message": "Repeat", "turn_id": "turn-1"},
            ),
        )
        worker_handler._save_trace_event(
            db,
            TraceEvent(
                event_type,
                task_id="agent_123_abcd1234",
                data={"message": "Repeat", "turn_id": "turn-1"},
            ),
        )
        worker_handler._save_trace_event(
            db,
            TraceEvent(
                event_type,
                task_id="agent_123_abcd1234",
                data={"message": "Repeat", "turn_id": "turn-1"},
            ),
        )

        rows = (
            db.query(DatabaseTraceEvent)
            .filter_by(task_id=int(task.id), event_type="user_message")
            .order_by(DatabaseTraceEvent.id)
            .all()
        )
        assert [(row.build_id, row.data["turn_id"]) for row in rows] == [
            (None, "turn-1"),
            ("agent_123_abcd1234", "turn-1"),
        ]
    finally:
        db.close()


def test_database_trace_handler_build_checkpoint_does_not_update_task_pointer() -> None:
    _, db, task = _create_trace_handler_test_task(
        "checkpoint-user",
        title="Checkpoint task",
        description="Task with worker checkpoint",
    )
    try:
        handler = DatabaseTraceHandler(
            int(task.id),
            build_id="agent_123_abcd1234",
        )
        event = TraceEvent(
            CHECKPOINT_EVENT_TYPE,
            task_id="agent_123_abcd1234",
            data={
                "checkpoint_type": CHECKPOINT_TYPE,
                "execution_id": "agent_123_abcd1234",
                "snapshot": {"label": "worker_checkpoint"},
            },
        )

        handler._save_trace_event(db, event)
        db.refresh(task)

        row = db.query(DatabaseTraceEvent).filter_by(task_id=int(task.id)).one()
        assert row.build_id == "agent_123_abcd1234"
        assert task.last_checkpoint_event_id is None
    finally:
        db.close()


def test_database_trace_handler_load_latest_checkpoint_is_build_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SessionLocal, db, task = _create_trace_handler_test_task(
        "load-user",
        title="Checkpoint task",
        description="Task with scoped checkpoints",
    )

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.trace_handlers.get_db", get_test_db)

    try:
        parent_handler = DatabaseTraceHandler(int(task.id))
        worker_handler = DatabaseTraceHandler(
            int(task.id),
            build_id="agent_123_abcd1234",
        )
        parent_handler._save_trace_event(
            db,
            TraceEvent(
                CHECKPOINT_EVENT_TYPE,
                task_id=str(task.id),
                data={
                    "checkpoint_type": CHECKPOINT_TYPE,
                    "execution_id": "shared-execution",
                    "snapshot": {"label": "parent_checkpoint"},
                },
            ),
        )
        worker_handler._save_trace_event(
            db,
            TraceEvent(
                CHECKPOINT_EVENT_TYPE,
                task_id="agent_123_abcd1234",
                data={
                    "checkpoint_type": CHECKPOINT_TYPE,
                    "execution_id": "shared-execution",
                    "snapshot": {"label": "worker_checkpoint"},
                },
            ),
        )

        assert parent_handler._sync_load_latest_checkpoint("shared-execution") == {
            "label": "parent_checkpoint"
        }
        assert worker_handler._sync_load_latest_checkpoint("shared-execution") == {
            "label": "worker_checkpoint"
        }
    finally:
        db.close()


def test_websocket_trace_handler_dedupes_prior_user_message_turn_id(
    monkeypatch,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="tester", password_hash="hashed_password", is_admin=False)
        db.add(user)
        db.commit()
        db.refresh(user)
        task = Task(
            user_id=int(user.id),
            title="Chat task",
            description="Task chat",
            status=TaskStatus.PENDING,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="first-event",
                event_type="user_message",
                timestamp=task.created_at,
                data={"message": "Repeat", "turn_id": "turn-1"},
            )
        )
        db.commit()
    finally:
        db.close()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.models.database.get_db", get_test_db)

    handler = WebSocketTraceHandler(task_id)
    assert not handler._has_prior_user_message_turn(
        "user_message", {"turn_id": "turn-1"}, "first-event"
    )
    assert handler._has_prior_user_message_turn(
        "user_message", {"turn_id": "turn-1"}, "second-event"
    )
    assert not handler._has_prior_user_message_turn(
        "user_message", {"turn_id": "turn-2"}, "second-event"
    )


def test_historical_replay_duplicate_turn_helper_allows_distinct_turns() -> None:
    seen: set[str] = set()

    assert not _is_duplicate_user_message_turn(
        "user_message", {"message": "Repeat", "turn_id": "turn-1"}, seen
    )
    assert _is_duplicate_user_message_turn(
        "user_message", {"message": "Repeat", "turn_id": "turn-1"}, seen
    )
    assert not _is_duplicate_user_message_turn(
        "user_message", {"message": "Repeat", "turn_id": "turn-2"}, seen
    )


@pytest.mark.asyncio
async def test_historical_replay_skips_audit_only_trace_events(monkeypatch) -> None:
    SessionLocal, db, task = _create_trace_handler_test_task("audit-history")
    try:
        task_id = int(task.id)
        user_id = int(task.user_id)
        base_time = datetime(2026, 5, 22, tzinfo=timezone.utc)
        db.add_all(
            [
                DatabaseTraceEvent(
                    task_id=task_id,
                    event_id="audit-workforce",
                    event_type="task_update_general",
                    timestamp=base_time + timedelta(seconds=1),
                    data={
                        "__audit_only__": True,
                        "event_type": "workforce_delegation_start",
                        "worker_task_id": "agent_123_abcd1234",
                    },
                ),
                DatabaseTraceEvent(
                    task_id=task_id,
                    event_id="visible-tool",
                    event_type="tool_execution_start",
                    timestamp=base_time + timedelta(seconds=2),
                    data={"tool_name": "call_agent_worker"},
                ),
            ]
        )
        db.commit()
    finally:
        db.close()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    sent_events: list[dict] = []

    async def send_personal_message(event: dict, websocket: object) -> None:
        sent_events.append(event)

    monkeypatch.setattr("xagent.web.models.database.get_db", get_test_db)
    monkeypatch.setattr("xagent.web.api.websocket.cache_get", lambda *args: None)
    monkeypatch.setattr(
        "xagent.web.api.websocket.cache_set", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "xagent.web.api.websocket.manager.send_personal_message",
        send_personal_message,
    )

    await send_historical_data_as_stream(
        websocket=object(),
        task_id=task_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )

    trace_event_ids = [
        event.get("event_id")
        for event in sent_events
        if event.get("type") == "trace_event"
    ]

    assert "audit-workforce" not in trace_event_ids
    assert "visible-tool" in trace_event_ids


@pytest.mark.asyncio
async def test_historical_replay_promotes_workforce_delegation_summary(
    monkeypatch,
) -> None:
    SessionLocal, db, task = _create_trace_handler_test_task("delegation-history")
    try:
        task_id = int(task.id)
        user_id = int(task.user_id)
        base_time = datetime(2026, 5, 22, tzinfo=timezone.utc)
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="delegation-end",
                event_type="task_update_general",
                timestamp=base_time + timedelta(seconds=1),
                data={
                    "event_type": "workforce_delegation_end",
                    "status": "end",
                    "worker_task_id": "agent_123_abcd1234",
                    "worker_alias": "Writer",
                    "output": "draft complete",
                    "messages": [{"role": "user", "content": "raw prompt"}],
                },
            )
        )
        db.commit()
    finally:
        db.close()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    sent_events: list[dict] = []

    async def send_personal_message(event: dict, websocket: object) -> None:
        sent_events.append(event)

    monkeypatch.setattr("xagent.web.models.database.get_db", get_test_db)
    monkeypatch.setattr("xagent.web.api.websocket.cache_get", lambda *args: None)
    monkeypatch.setattr(
        "xagent.web.api.websocket.cache_set", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "xagent.web.api.websocket.manager.send_personal_message",
        send_personal_message,
    )

    await send_historical_data_as_stream(
        websocket=object(),
        task_id=task_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )

    delegation_event = next(
        event for event in sent_events if event.get("event_id") == "delegation-end"
    )
    assert delegation_event["event_type"] == "workforce_delegation_end"
    assert delegation_event["data"]["worker_alias"] == "Writer"
    assert delegation_event["data"]["worker_task_id"] == "agent_123_abcd1234"
    assert delegation_event["data"]["output"] == "draft complete"
    assert "messages" not in delegation_event["data"]
    assert "event_type" not in delegation_event["data"]


@pytest.mark.asyncio
async def test_historical_replay_skips_checkpoint_rows_before_streaming(
    monkeypatch,
) -> None:
    SessionLocal, db, task = _create_trace_handler_test_task("checkpoint-history")
    try:
        task_id = int(task.id)
        user_id = int(task.user_id)
        base_time = datetime(2026, 5, 22, tzinfo=timezone.utc)
        db.add_all(
            [
                DatabaseTraceEvent(
                    task_id=task_id,
                    event_id="checkpoint-row",
                    event_type=str(CHECKPOINT_EVENT_TYPE),
                    timestamp=base_time + timedelta(seconds=1),
                    data={
                        "checkpoint_type": CHECKPOINT_TYPE,
                        "execution_id": str(task_id),
                        "snapshot": {"context": {"messages": ["large"]}},
                    },
                ),
                DatabaseTraceEvent(
                    task_id=task_id,
                    event_id="llm-row",
                    event_type="llm_call_start",
                    timestamp=base_time + timedelta(seconds=2),
                    data={"model_name": "test-model"},
                ),
            ]
        )
        db.commit()
    finally:
        db.close()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    sent_events: list[dict] = []

    async def send_personal_message(event: dict, websocket: object) -> None:
        sent_events.append(event)

    monkeypatch.setattr("xagent.web.models.database.get_db", get_test_db)
    monkeypatch.setattr("xagent.web.api.websocket.cache_get", lambda *args: None)
    monkeypatch.setattr(
        "xagent.web.api.websocket.cache_set", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "xagent.web.api.websocket.manager.send_personal_message",
        send_personal_message,
    )

    await send_historical_data_as_stream(
        websocket=object(),
        task_id=task_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )

    streamed_event_ids = {
        event.get("event_id")
        for event in sent_events
        if event.get("type") == "trace_event"
    }
    assert "checkpoint-row" not in streamed_event_ids
    assert "llm-row" in streamed_event_ids


@pytest.mark.asyncio
async def test_historical_replay_marks_assistant_chat_history_for_chat_display(
    monkeypatch,
) -> None:
    SessionLocal, db, task = _create_trace_handler_test_task("chat-history-display")
    try:
        task_id = int(task.id)
        user_id = int(task.user_id)
        db.add(
            TaskChatMessage(
                task_id=task_id,
                user_id=user_id,
                role="assistant",
                content="Final answer",
                message_type="assistant",
                created_at=datetime(2026, 5, 27, tzinfo=timezone.utc),
            )
        )
        db.commit()
    finally:
        db.close()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    sent_events: list[dict] = []

    async def send_personal_message(event: dict, websocket: object) -> None:
        sent_events.append(event)

    monkeypatch.setattr("xagent.web.models.database.get_db", get_test_db)
    monkeypatch.setattr("xagent.web.api.websocket.cache_get", lambda *args: None)
    monkeypatch.setattr(
        "xagent.web.api.websocket.cache_set", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "xagent.web.api.websocket.manager.send_personal_message",
        send_personal_message,
    )

    await send_historical_data_as_stream(
        websocket=object(),
        task_id=task_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )

    assistant_events = [
        event
        for event in sent_events
        if event.get("type") == "trace_event"
        and event.get("event_type") == "agent_message"
        and event.get("data", {}).get("message") == "Final answer"
    ]
    assert len(assistant_events) == 1
    assistant_data = assistant_events[0]["data"]
    assert assistant_data["role"] == "assistant"
    assert assistant_data["expect_response"] is False
    assert assistant_data["source"] == "chat_history"
    assert assistant_data["display"] == "chat"


@pytest.mark.asyncio
async def test_historical_replay_orders_equal_timestamps_by_id(
    monkeypatch,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="tester", password_hash="hashed_password", is_admin=False)
        db.add(user)
        db.commit()
        db.refresh(user)
        task = Task(
            user_id=int(user.id),
            title="Chat task",
            description="Chat task",
            status=TaskStatus.COMPLETED,
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        task_id = int(task.id)
        user_id = int(user.id)
        timestamp = datetime(2026, 5, 22, tzinfo=timezone.utc)
        db.add_all(
            [
                DatabaseTraceEvent(
                    task_id=task_id,
                    event_id="first-row",
                    event_type="llm_call_start",
                    timestamp=timestamp,
                    data={"model_name": "first-model"},
                ),
                DatabaseTraceEvent(
                    task_id=task_id,
                    event_id="second-row",
                    event_type="llm_call_end",
                    timestamp=timestamp,
                    data={"response": "done"},
                ),
            ]
        )
        db.commit()
    finally:
        db.close()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    sent_events: list[dict] = []

    async def send_personal_message(event: dict, websocket: object) -> None:
        sent_events.append(event)

    monkeypatch.setattr("xagent.web.models.database.get_db", get_test_db)
    monkeypatch.setattr("xagent.web.api.websocket.cache_get", lambda *args: None)
    monkeypatch.setattr(
        "xagent.web.api.websocket.cache_set", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "xagent.web.api.websocket.manager.send_personal_message",
        send_personal_message,
    )

    await send_historical_data_as_stream(
        websocket=object(),
        task_id=task_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )

    streamed_event_ids = [
        event.get("event_id")
        for event in sent_events
        if event.get("type") == "trace_event"
        and event.get("event_id") in {"first-row", "second-row"}
    ]
    assert streamed_event_ids == ["first-row", "second-row"]


@pytest.mark.asyncio
async def test_historical_replay_uses_turn_id_before_legacy_content_dedupe(
    monkeypatch,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="tester", password_hash="hashed_password", is_admin=False)
        db.add(user)
        db.commit()
        db.refresh(user)
        task = Task(
            user_id=int(user.id),
            title="Chat task",
            description="Chat task",
            status=TaskStatus.COMPLETED,
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        task_id = int(task.id)
        user_id = int(user.id)
        base_time = datetime(2026, 5, 22, tzinfo=timezone.utc)
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="trace-turn-a",
                event_type="user_message",
                timestamp=base_time + timedelta(seconds=1),
                data={"message": "Repeat", "turn_id": "turn-A"},
            )
        )
        db.add_all(
            [
                TaskChatMessage(
                    task_id=task_id,
                    user_id=user_id,
                    role="user",
                    content="Repeat",
                    message_type="user_message",
                    turn_id="turn-A",
                    created_at=base_time + timedelta(seconds=2),
                ),
                TaskChatMessage(
                    task_id=task_id,
                    user_id=user_id,
                    role="user",
                    content="Repeat",
                    message_type="user_message",
                    turn_id="turn-B",
                    created_at=base_time + timedelta(seconds=3),
                ),
            ]
        )
        db.commit()
    finally:
        db.close()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    sent_events: list[dict] = []

    async def send_personal_message(event: dict, websocket: object) -> None:
        sent_events.append(event)

    monkeypatch.setattr("xagent.web.models.database.get_db", get_test_db)
    monkeypatch.setattr("xagent.web.api.websocket.cache_get", lambda *args: None)
    monkeypatch.setattr(
        "xagent.web.api.websocket.cache_set", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "xagent.web.api.websocket.manager.send_personal_message",
        send_personal_message,
    )

    await send_historical_data_as_stream(
        websocket=object(),
        task_id=task_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )

    user_message_events = [
        event
        for event in sent_events
        if event.get("type") == "trace_event"
        and event.get("event_type") == "user_message"
    ]

    assert [
        (event["data"].get("message"), event["data"].get("turn_id"))
        for event in user_message_events
    ] == [("Repeat", "turn-A"), ("Repeat", "turn-B")]


@pytest.mark.asyncio
async def test_historical_replay_dedupes_file_only_turns_by_turn_id(
    monkeypatch,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="tester", password_hash="hashed_password", is_admin=False)
        db.add(user)
        db.commit()
        db.refresh(user)
        task = Task(
            user_id=int(user.id),
            title="Chat task",
            description="Chat task",
            status=TaskStatus.COMPLETED,
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        task_id = int(task.id)
        user_id = int(user.id)
        base_time = datetime(2026, 5, 22, tzinfo=timezone.utc)
        attachments = [{"file_id": "fid-only", "name": "only.pdf"}]
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="trace-file-only",
                event_type="user_message",
                timestamp=base_time + timedelta(seconds=1),
                data={"message": "", "turn_id": "turn-file", "files": attachments},
            )
        )
        db.add(
            TaskChatMessage(
                task_id=task_id,
                user_id=user_id,
                role="user",
                content="",
                message_type="user_message",
                turn_id="turn-file",
                attachments=attachments,
                created_at=base_time + timedelta(seconds=2),
            )
        )
        db.commit()
    finally:
        db.close()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    sent_events: list[dict] = []

    async def send_personal_message(event: dict, websocket: object) -> None:
        sent_events.append(event)

    monkeypatch.setattr("xagent.web.models.database.get_db", get_test_db)
    monkeypatch.setattr("xagent.web.api.websocket.cache_get", lambda *args: None)
    monkeypatch.setattr(
        "xagent.web.api.websocket.cache_set", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "xagent.web.api.websocket.manager.send_personal_message",
        send_personal_message,
    )

    await send_historical_data_as_stream(
        websocket=object(),
        task_id=task_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )

    user_message_events = [
        event
        for event in sent_events
        if event.get("type") == "trace_event"
        and event.get("event_type") == "user_message"
    ]

    assert [
        (event["data"].get("turn_id"), event["data"].get("files"))
        for event in user_message_events
    ] == [("turn-file", attachments)]
