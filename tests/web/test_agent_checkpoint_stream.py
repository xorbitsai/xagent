from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Query, Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from tests.web.services.task_interaction_schema_shared import (
    make_row as make_interaction_row,
)
from xagent.core.agent.checkpoint import (
    CHECKPOINT_EVENT_TYPE,
    CHECKPOINT_TYPE,
    CheckpointAccessRefusedError,
    CheckpointCorruptError,
    CheckpointUnavailableError,
)
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
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.models.user import User
from xagent.web.models.workforce import WorkforceRun
from xagent.web.services import interaction_rollout as ir
from xagent.web.services.ops_signals import (
    CHECKPOINT_PRUNE_FAILED,
    active_degradations,
    clear_degradation,
)
from xagent.web.services.task_interaction_schema import (
    interaction_requests_table_exists,
)
from xagent.web.services.task_lease_service import (
    TASK_RUN_ID_TRACE_FIELD,
    TaskLease,
    bind_task_lease_context,
    current_task_lease,
)


def _shared_memory_sqlite_engine():
    """Keep the in-memory test database visible to off-loop worker Sessions."""

    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


@pytest.fixture(autouse=True)
def _reset_interaction_counters() -> Iterator[None]:
    """The counter registry is process-global; clear it on both sides of
    every test so one test's counts can never be read by another."""
    ir._counters.clear()
    try:
        yield
    finally:
        ir._counters.clear()


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
    engine = _shared_memory_sqlite_engine()
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


def test_persist_agent_outbound_event_sanitizes_payload(monkeypatch) -> None:
    """This function builds its own TraceEvent row without going through
    stage_trace_event_row (the staging module's "known bypass"), so it must
    sanitize for itself: PostgreSQL's jsonb rejects NUL and lone-surrogate
    code points at INSERT (#1248)."""
    engine = _shared_memory_sqlite_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(
            username="sanitizer", password_hash="hashed_password", is_admin=False
        )
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

    # chr, not source escapes: a lone surrogate is not encodable as UTF-8.
    nul, lone_high, replacement = chr(0x0000), chr(0xD800), chr(0xFFFD)
    event = create_stream_event(
        "agent_progress",
        int(task.id),
        {
            "event_id": "agent-event-dirty",
            "message": f"head{nul}mid{lone_high}tail",
            "expect_response": False,
        },
    )

    _persist_agent_outbound_event(int(task.id), event)

    db = SessionLocal()
    try:
        trace_event = (
            db.query(DatabaseTraceEvent).filter_by(event_id="agent-event-dirty").one()
        )
        assert trace_event.data["message"] == f"head{replacement}mid{replacement}tail"
    finally:
        db.close()


def _create_trace_handler_test_task(
    username: str,
    *,
    title: str = "Chat task",
    description: str = "Task chat",
):
    engine = _shared_memory_sqlite_engine()
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


def _checkpoint_trace_row(
    *,
    task_id: int,
    event_id: str,
    execution_id: str,
    label: str,
    timestamp: datetime,
    run_id: str | None = None,
    build_id: str | None = None,
) -> DatabaseTraceEvent:
    data = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "execution_id": execution_id,
        "snapshot": {"label": label},
    }
    if run_id is not None:
        data[TASK_RUN_ID_TRACE_FIELD] = run_id
    return DatabaseTraceEvent(
        task_id=task_id,
        build_id=build_id,
        event_id=event_id,
        event_type="system_update_general",
        timestamp=timestamp,
        data=data,
    )


def _create_trace_handler_test_task_without_interaction_table(
    username: str,
    *,
    title: str = "Chat task",
    description: str = "Task chat",
):
    """Same shape as _create_trace_handler_test_task, built with a filtered
    create_all that excludes task_interaction_requests -- reproducing a
    deployment upgraded to a revision before that table exists (the
    migration that creates it is not merged). The table has no inbound
    foreign keys, so excluding it cannot break create order (verified in
    the audit: 52 of 53 tables created, tasks and trace_events both
    present, has_table(target) False).
    """
    engine = _shared_memory_sqlite_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    tables = [
        table
        for table in Base.metadata.sorted_tables
        if table.name != TaskInteractionRequest.__tablename__
    ]
    Base.metadata.create_all(bind=engine, tables=tables)
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


def _seed_interaction_row(
    db: Session,
    *,
    task_id: int,
    resume_trace_event_id: int | None,
    status: str = "active",
    **overrides: object,
) -> TaskInteractionRequest:
    row = TaskInteractionRequest(
        **make_interaction_row(
            task_id=task_id,
            resume_trace_event_id=resume_trace_event_id,
            status=status,
            **overrides,
        )
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


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


@pytest.mark.asyncio
async def test_historical_stream_format_error_redacts_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.api import websocket as websocket_api

    secret = "history-storage-secret"
    sent_events: list[dict] = []

    async def fail_load(_operation):
        raise ValueError(f"malformed history: {secret}")

    async def send_personal_message(event: dict, _websocket: object) -> None:
        sent_events.append(event)

    monkeypatch.setattr(websocket_api, "run_db_io_cancellation_safe", fail_load)
    monkeypatch.setattr(
        websocket_api.manager,
        "send_personal_message",
        send_personal_message,
    )

    with pytest.raises(ValueError, match=secret):
        await send_historical_data_as_stream(
            websocket=object(),
            task_id=42,
            user=SimpleNamespace(id=1, is_admin=False),
        )

    assert len(sent_events) == 1
    assert sent_events[0]["event_type"] == "error"
    assert (
        sent_events[0]["data"]["message"]
        == "Task history could not be loaded. Please try again."
    )
    assert secret not in repr(sent_events[0])


@pytest.mark.asyncio
async def test_historical_replay_detaches_before_slow_network_send(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History DB/cache work finishes off-loop before the first network wait."""

    from xagent.web.api import websocket as websocket_api
    from xagent.web.models import database as database_module

    engine = create_engine(
        f"sqlite:///{tmp_path / 'history-one-slot.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.25,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with SessionLocal() as db:
        owner = User(
            username="history-one-slot-owner",
            password_hash="x",
            is_admin=False,
        )
        db.add(owner)
        db.flush()
        task = Task(
            user_id=int(owner.id),
            title="history one slot",
            description="history",
            status=TaskStatus.COMPLETED,
        )
        db.add(task)
        db.commit()
        owner_id = int(owner.id)
        task_id = int(task.id)

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    event_loop_thread = threading.get_ident()
    cache_threads: list[int] = []
    sent_events: list[dict] = []
    ticker_stop = asyncio.Event()
    ticks = 0

    def slow_cache_get(_key: str):
        assert engine.pool.checkedout() == 0
        cache_threads.append(threading.get_ident())
        time.sleep(0.05)
        return None

    def record_cache_set(*_args, **_kwargs) -> None:
        assert engine.pool.checkedout() == 0
        cache_threads.append(threading.get_ident())

    async def slow_send(event: dict, _websocket: object) -> None:
        assert engine.pool.checkedout() == 0

        def read_during_send() -> None:
            with SessionLocal() as db:
                assert db.query(Task.id).filter(Task.id == task_id).scalar() == task_id

        await asyncio.to_thread(read_during_send)
        sent_events.append(event)
        await asyncio.sleep(0.02)

    async def ticker() -> None:
        nonlocal ticks
        while not ticker_stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    monkeypatch.setattr(database_module, "get_db", get_test_db)
    monkeypatch.setattr(websocket_api, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(websocket_api, "cache_get", slow_cache_get)
    monkeypatch.setattr(websocket_api, "cache_set", record_cache_set)
    monkeypatch.setattr(
        websocket_api.manager,
        "send_personal_message",
        slow_send,
    )

    ticker_task = asyncio.create_task(ticker())
    try:
        await send_historical_data_as_stream(
            websocket=object(),
            task_id=task_id,
            user=SimpleNamespace(id=owner_id, is_admin=False),
        )
    finally:
        ticker_stop.set()
        await ticker_task
        engine.dispose()

    assert ticks >= 5
    assert cache_threads
    assert all(thread_id != event_loop_thread for thread_id in cache_threads)
    assert [event["event_type"] for event in sent_events] == [
        "task_info",
        "historical_data_complete",
    ]


def test_historical_replay_does_not_backfill_legacy_file_outputs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A history cache miss is a read path, never a durable-file writer."""

    from xagent.core.file_storage.factory import get_unscoped_file_storage
    from xagent.core.file_storage.storage import FsspecFileStorage
    from xagent.web.api import websocket as websocket_api
    from xagent.web.models import database as database_module
    from xagent.web.models.uploaded_file import UploadedFile

    engine = create_engine(
        f"sqlite:///{tmp_path / 'history-file-output.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.25,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()

    with SessionLocal() as db:
        owner = User(
            username="history-file-output-owner",
            password_hash="x",
            is_admin=False,
        )
        db.add(owner)
        db.flush()
        task = Task(
            user_id=int(owner.id),
            title="legacy output history",
            description="history",
            status=TaskStatus.COMPLETED,
        )
        db.add(task)
        db.flush()
        owner_id = int(owner.id)
        task_id = int(task.id)
        output_path = (
            uploads_dir
            / f"user_{owner_id}"
            / f"web_task_{task_id}"
            / "output"
            / "legacy.txt"
        )
        output_path.parent.mkdir(parents=True)
        output_path.write_text("legacy bytes", encoding="utf-8")
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="legacy-file-output",
                event_type="tool_execution_start",
                timestamp=datetime.now(timezone.utc),
                data={
                    "file_outputs": [
                        {
                            "path": str(output_path),
                            "filename": "legacy.txt",
                        }
                    ]
                },
            )
        )
        db.commit()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    put_observations: list[int] = []
    original_put_file = FsspecFileStorage.put_file

    def observe_put_file(
        self: FsspecFileStorage,
        source,
        key: str,
        content_type: str | None = None,
    ):
        put_observations.append(engine.pool.checkedout())
        return original_put_file(self, source, key, content_type)

    monkeypatch.setattr(database_module, "get_db", get_test_db)
    monkeypatch.setattr(websocket_api, "cache_get", lambda *_args: None)
    monkeypatch.setattr(
        websocket_api,
        "cache_set",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(FsspecFileStorage, "put_file", observe_put_file)
    try:
        snapshot = websocket_api._load_historical_stream_snapshot_sync(
            task_id,
            actor_user_id=owner_id,
            actor_is_admin=False,
        )
        assert snapshot is not None
        with SessionLocal() as db:
            assert db.query(UploadedFile).count() == 0
        assert put_observations == []
    finally:
        engine.dispose()
        get_unscoped_file_storage.cache_clear()


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

        with bind_task_lease_context(TaskLease(int(task.id), "runner-a", "run-a")):
            handler._save_trace_event(db, event)
        db.refresh(task)

        row = db.query(DatabaseTraceEvent).filter_by(task_id=int(task.id)).one()
        assert row.build_id == "agent_123_abcd1234"
        assert TASK_RUN_ID_TRACE_FIELD not in row.data
        assert task.last_checkpoint_event_id is None
    finally:
        db.close()


def test_database_trace_handler_tags_and_points_to_current_run_checkpoint() -> None:
    _, db, task = _create_trace_handler_test_task("current-run-checkpoint")
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    lease = TaskLease(
        task_id=int(task.id),
        runner_id="runner-a",
        run_id="run-a",
    )
    event = TraceEvent(
        CHECKPOINT_EVENT_TYPE,
        task_id=str(task.id),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": str(task.id),
            "snapshot": {"label": "current"},
        },
    )

    try:
        with bind_task_lease_context(lease):
            DatabaseTraceHandler(int(task.id))._save_trace_event(db, event)

        db.refresh(task)
        row = db.query(DatabaseTraceEvent).filter_by(task_id=int(task.id)).one()
        assert row.data[TASK_RUN_ID_TRACE_FIELD] == "run-a"
        assert task.last_checkpoint_event_id == str(event.id)
        # Dual write: the exact-row anchor points at the same row the
        # legacy string column names.
        assert task.last_checkpoint_trace_event_id == row.id
        # Mixed-version invariant: exactly one row in the task's root
        # partition carries the legacy event_id the pointer names, so an
        # old reader still resolving through the string column stays
        # unambiguous.
        matching = (
            db.query(DatabaseTraceEvent)
            .filter(
                DatabaseTraceEvent.task_id == int(task.id),
                DatabaseTraceEvent.build_id.is_(None),
                DatabaseTraceEvent.event_type == "system_update_general",
                DatabaseTraceEvent.event_id == task.last_checkpoint_event_id,
            )
            .all()
        )
        assert len(matching) == 1
        assert matching[0].id == row.id
    finally:
        db.close()


@pytest.mark.parametrize("require_persisted", [False, True])
def test_database_trace_handler_rejects_stale_checkpoint(
    require_persisted: bool,
) -> None:
    _, db, task = _create_trace_handler_test_task("stale-run-checkpoint")
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-b"
    task.run_id = "run-b"
    db.commit()
    stale_lease = TaskLease(
        task_id=int(task.id),
        runner_id="runner-a",
        run_id="run-a",
    )
    event = TraceEvent(
        CHECKPOINT_EVENT_TYPE,
        task_id=str(task.id),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": str(task.id),
            "snapshot": {"label": "stale"},
        },
        require_persisted=require_persisted,
    )

    try:
        with (
            bind_task_lease_context(stale_lease),
            pytest.raises(RuntimeError, match="lease changed"),
        ):
            DatabaseTraceHandler(int(task.id))._save_trace_event(db, event)

        db.expire_all()
        persisted = db.get(Task, int(task.id))
        assert persisted.last_checkpoint_event_id is None
        assert persisted.last_checkpoint_trace_event_id is None
        assert db.query(DatabaseTraceEvent).filter_by(task_id=int(task.id)).count() == 0
    finally:
        db.close()


@pytest.mark.parametrize("require_persisted", [False, True])
def test_database_trace_handler_pointer_update_skips_silently_when_task_is_gone(
    require_persisted: bool,
) -> None:
    """A 0-row pointer UPDATE has two distinct causes: the lease moved (an
    error worth raising), or the task row itself is gone. This test's
    in-memory engine has no FK pragma enabled, so deleting the task row
    does not fail the trace_event insert -- it reaches the pointer UPDATE
    against a task_id that matches nothing, the same state a deployment
    without FK enforcement on this legacy column would reach.

    The task-gone branch is classified exactly as the trace_events.task_id
    FK violation is classified: always zero residue, quiet for a
    best-effort event, loud for a require_persisted one. Reporting success
    for an event that was dropped is what the FK path has never done.
    """
    _, db, task = _create_trace_handler_test_task(
        f"task-gone-checkpoint-{require_persisted}"
    )
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)
    lease = TaskLease(task_id=task_id, runner_id="runner-a", run_id="run-a")
    event = TraceEvent(
        CHECKPOINT_EVENT_TYPE,
        task_id=str(task_id),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": str(task_id),
            "snapshot": {"label": "orphaned"},
        },
        require_persisted=require_persisted,
    )

    db.query(Task).filter(Task.id == task_id).delete(synchronize_session=False)
    db.commit()

    try:
        with bind_task_lease_context(lease):
            if require_persisted:
                with pytest.raises(RuntimeError, match="no longer exists"):
                    DatabaseTraceHandler(task_id)._save_trace_event(db, event)
            else:
                # Must not raise "lease changed" -- the task is gone, not
                # leased elsewhere.
                DatabaseTraceHandler(task_id)._save_trace_event(db, event)

        # Either way the staged row is discarded, not committed as an orphan.
        assert db.query(DatabaseTraceEvent).filter_by(task_id=task_id).count() == 0
    finally:
        db.close()


def test_cached_database_trace_handler_reads_run_context_per_event() -> None:
    _, db, task = _create_trace_handler_test_task("cached-run-checkpoint")
    handler = DatabaseTraceHandler(int(task.id))
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    first = TraceEvent(
        CHECKPOINT_EVENT_TYPE,
        task_id=str(task.id),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": str(task.id),
            "snapshot": {"label": "first"},
        },
    )
    second = TraceEvent(
        CHECKPOINT_EVENT_TYPE,
        task_id=str(task.id),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": str(task.id),
            "snapshot": {"label": "second"},
        },
    )

    try:
        with bind_task_lease_context(TaskLease(int(task.id), "runner-a", "run-a")):
            handler._save_trace_event(db, first)

        task.runner_id = "runner-b"
        task.run_id = "run-b"
        task.last_checkpoint_event_id = None
        db.commit()
        with bind_task_lease_context(TaskLease(int(task.id), "runner-b", "run-b")):
            handler._save_trace_event(db, second)

        rows = (
            db.query(DatabaseTraceEvent)
            .filter_by(task_id=int(task.id))
            .order_by(DatabaseTraceEvent.id)
            .all()
        )
        assert [row.data[TASK_RUN_ID_TRACE_FIELD] for row in rows] == [
            "run-a",
            "run-b",
        ]
        db.refresh(task)
        assert task.last_checkpoint_event_id == str(second.id)
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

        with bind_task_lease_context(TaskLease(int(task.id), "runner-a", "run-a")):
            assert (
                parent_handler._sync_load_latest_checkpoint("shared-execution") is None
            )
            assert worker_handler._sync_load_latest_checkpoint("shared-execution") == {
                "label": "worker_checkpoint"
            }

        assert parent_handler._sync_load_latest_checkpoint("shared-execution") == {
            "label": "parent_checkpoint"
        }
    finally:
        db.close()


def test_database_trace_handler_loads_checkpoint_only_from_bound_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SessionLocal, db, task = _create_trace_handler_test_task("run-scoped-load")
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-b"
    task.run_id = "run-b"
    db.commit()
    task_id = int(task.id)
    now = datetime.now(timezone.utc)
    db.add_all(
        [
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="run-b-checkpoint",
                execution_id="shared-execution",
                label="run-b",
                timestamp=now,
                run_id="run-b",
            ),
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="newer-run-a-checkpoint",
                execution_id="shared-execution",
                label="run-a",
                timestamp=now + timedelta(seconds=1),
                run_id="run-a",
            ),
        ]
    )
    db.commit()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.trace_handlers.get_db", get_test_db)

    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-b", "run-b")):
            assert DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                "shared-execution"
            ) == {"label": "run-b"}
    finally:
        db.close()


@pytest.mark.asyncio
async def test_database_trace_handler_load_worker_inherits_run_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = DatabaseTraceHandler(7)

    def observe_bound_run(_execution_id: str) -> dict[str, str | None]:
        lease = current_task_lease()
        return {"run_id": lease.run_id if lease is not None else None}

    monkeypatch.setattr(handler, "_sync_load_latest_checkpoint", observe_bound_run)

    with bind_task_lease_context(TaskLease(7, "runner-b", "run-b")):
        assert await handler.load_latest_checkpoint("shared-execution") == {
            "run_id": "run-b"
        }

    assert current_task_lease() is None


def test_database_trace_handler_bound_run_does_not_fallback_to_legacy_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SessionLocal, db, task = _create_trace_handler_test_task("run-legacy-load")
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-b"
    task.run_id = "run-b"
    db.commit()
    task_id = int(task.id)
    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="legacy-checkpoint",
            execution_id="shared-execution",
            label="legacy",
            timestamp=datetime.now(timezone.utc),
        )
    )
    db.commit()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.trace_handlers.get_db", get_test_db)

    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-b", "run-b")):
            assert (
                DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                    "shared-execution"
                )
                is None
            )
    finally:
        db.close()


def test_database_trace_handler_load_without_pk_anchor_uses_legacy_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contract for the mixed-data window: with the pointer unset, the read
    path is exactly the pre-existing legacy scan."""
    SessionLocal, db, task = _create_trace_handler_test_task("no-pk-anchor-load")
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)
    assert task.last_checkpoint_trace_event_id is None
    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="legacy-checkpoint",
            execution_id="shared-execution",
            label="legacy-only",
            timestamp=datetime.now(timezone.utc),
            run_id="run-a",
        )
    )
    db.commit()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.trace_handlers.get_db", get_test_db)

    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            assert DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                "shared-execution"
            ) == {"label": "legacy-only"}
    finally:
        db.close()


def test_database_trace_handler_load_uses_pk_anchor_over_newer_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once set, the pointer is authoritative -- even over a row the legacy
    scan (newest timestamp first) would otherwise pick instead."""
    SessionLocal, db, task = _create_trace_handler_test_task("pk-anchor-load")
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)
    now = datetime.now(timezone.utc)
    anchored_row = _checkpoint_trace_row(
        task_id=task_id,
        event_id="anchored-checkpoint",
        execution_id="shared-execution",
        label="anchored",
        timestamp=now,
        run_id="run-a",
    )
    db.add(anchored_row)
    db.commit()
    db.refresh(anchored_row)
    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="newer-checkpoint",
            execution_id="shared-execution",
            label="newer",
            timestamp=now + timedelta(seconds=1),
            run_id="run-a",
        )
    )
    task.last_checkpoint_trace_event_id = anchored_row.id
    db.commit()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.trace_handlers.get_db", get_test_db)

    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            assert DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                "shared-execution"
            ) == {"label": "anchored"}
    finally:
        db.close()


_PK_ANCHOR_SINGLE_FAULT_FIELDS = [
    "task_id",
    "event_type",
    "build_id",
    "checkpoint_type",
    "run_partition",
    "execution_id",
]


def _mutate_pk_anchor_field(
    field: str,
    row_kwargs: Dict[str, Any],
    data: Dict[str, Any],
    other_task_id: int,
) -> None:
    """Corrupt exactly one field the PK-anchored validation checks, leaving
    every other field -- including the run field -- valid, so a passing
    case can only be explained by that one conjunct doing nothing."""
    if field == "task_id":
        row_kwargs["task_id"] = other_task_id
    elif field == "event_type":
        row_kwargs["event_type"] = "agent_progress"
    elif field == "build_id":
        row_kwargs["build_id"] = "agent_123_wrongscope"
    elif field == "checkpoint_type":
        data["checkpoint_type"] = "not_a_readable_checkpoint_type"
    elif field == "run_partition":
        data[TASK_RUN_ID_TRACE_FIELD] = "run-b"
    elif field == "execution_id":
        data["execution_id"] = "other-execution"
    else:
        raise AssertionError(f"unknown single-fault field {field}")


@pytest.mark.parametrize("field", _PK_ANCHOR_SINGLE_FAULT_FIELDS)
def test_database_trace_handler_load_pk_anchor_single_fault_raises_corrupt(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Each conjunct of the PK-anchored validation must independently be
    load-bearing: a row wrong in exactly one field (every other field,
    including the run partition, valid) must still raise
    CheckpointCorruptError. A row that violates several conjuncts at once
    proves nothing about any one of them -- deleting one check still leaves
    the others to raise -- so every conjunct gets its own case here.
    """
    SessionLocal, db, task = _create_trace_handler_test_task(
        f"pk-anchor-single-fault-{field}"
    )
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)

    other_task = Task(
        user_id=int(task.user_id),
        title="Other task",
        description="Other task",
        status=TaskStatus.PENDING,
    )
    db.add(other_task)
    db.commit()
    db.refresh(other_task)
    other_task_id = int(other_task.id)

    row_kwargs: Dict[str, Any] = dict(
        task_id=task_id,
        build_id=None,
        event_id="pk-anchor-single-fault",
        event_type="system_update_general",
        timestamp=datetime.now(timezone.utc),
    )
    data: Dict[str, Any] = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "execution_id": "shared-execution",
        TASK_RUN_ID_TRACE_FIELD: "run-a",
        "snapshot": {"label": "single-fault"},
    }
    _mutate_pk_anchor_field(field, row_kwargs, data, other_task_id)

    row = DatabaseTraceEvent(data=data, **row_kwargs)
    db.add(row)
    db.commit()
    db.refresh(row)
    task.last_checkpoint_trace_event_id = row.id
    db.commit()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.trace_handlers.get_db", get_test_db)

    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            with pytest.raises(CheckpointCorruptError):
                DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                    "shared-execution"
                )
    finally:
        db.close()


def test_database_trace_handler_load_pk_anchor_absent_run_field_defers_to_scan(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pointer row missing the run-partition field entirely -- the shape
    the 20260804 backfill produces from a trace_events row written before
    that field existed -- is a pre-existing row, not a corrupt one. It
    defers to the legacy scan, which filters candidates by run partition and
    therefore finds none. Unlike a genuine "no checkpoint" query, this
    reader still cannot verify it is authoritative for the anchored row, so
    the read raises CheckpointAccessRefusedError (reason
    "run_provenance_unavailable") instead of resolving to None -- a caller
    that treated this as "no checkpoint" would silently rebuild an empty
    context instead of reporting that the row could not be verified. Pairs
    with the "run_partition" case of
    test_database_trace_handler_load_pk_anchor_single_fault_raises_corrupt,
    which covers the opposite: a run-partition field that is present and
    wrong.
    """
    SessionLocal, db, task = _create_trace_handler_test_task(
        "pk-anchor-absent-run-field"
    )
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)

    row = DatabaseTraceEvent(
        task_id=task_id,
        build_id=None,
        event_id="pk-anchor-absent-run-field",
        event_type="system_update_general",
        timestamp=datetime.now(timezone.utc),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": "shared-execution",
            "snapshot": {"label": "absent-run-field"},
        },
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    row_id = row.id
    task.last_checkpoint_trace_event_id = row_id
    db.commit()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.trace_handlers.get_db", get_test_db)

    caplog.set_level(logging.WARNING, logger="xagent.web.api.trace_handlers")
    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            with pytest.raises(CheckpointAccessRefusedError) as excinfo:
                DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                    "shared-execution"
                )
            assert excinfo.value.reason == "run_provenance_unavailable"
    finally:
        db.close()

    # The refusal is the contract now, but the warning and counter this
    # condition has always set remain the record of how often it fires,
    # independent of how each caller reacts to the refusal it triggers.
    # Assert both, at the same strength the write-direction resolver's own
    # test asserts them (tests/web/services/test_task_interaction_anchor.py).
    assert ir.counters_snapshot() == {
        ir.COUNTER_CHECKPOINT_ABSENT_MISSING_RUN_PARTITION: 1
    }
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, caplog.records
    assert warnings[0].msg == (
        "Task %s's checkpoint pointer %s is missing its "
        "run-partition field; deferring to the legacy scan "
        "rather than reporting the row as corrupt"
    )
    assert warnings[0].args == (task_id, row_id)


def test_database_trace_handler_load_pk_anchor_without_execution_identity_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The execution-identity conjunct is verification, not filtering: a
    row that carries no execution identity at all -- a legacy row written
    before root_execution_id/execution_id/snapshot.execution_id existed --
    must still load rather than being treated as a mismatch. Pairs with
    the "execution_id" case of
    test_database_trace_handler_load_pk_anchor_single_fault_raises_corrupt,
    which covers the opposite: an execution identity that is present and
    wrong.
    """
    SessionLocal, db, task = _create_trace_handler_test_task(
        "pk-anchor-no-execution-identity"
    )
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)

    row = DatabaseTraceEvent(
        task_id=task_id,
        build_id=None,
        event_id="pk-anchor-no-execution-identity",
        event_type="system_update_general",
        timestamp=datetime.now(timezone.utc),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            TASK_RUN_ID_TRACE_FIELD: "run-a",
            "snapshot": {"label": "no-execution-identity"},
        },
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    task.last_checkpoint_trace_event_id = row.id
    db.commit()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.trace_handlers.get_db", get_test_db)

    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            assert DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                "shared-execution"
            ) == {"label": "no-execution-identity"}
    finally:
        db.close()


def test_database_trace_handler_load_dangling_pk_anchor_falls_back_with_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pointer whose row no longer exists is only reachable on a database
    upgraded without the DB-level FK for this column (see the migration).
    The compatibility-window response is a legacy fallback plus a
    self-clearing degradation signal, not a corruption verdict."""
    from xagent.web.services.ops_signals import (
        CHECKPOINT_PK_ANCHOR_DANGLING,
        active_degradations,
        clear_degradation,
    )

    SessionLocal, db, task = _create_trace_handler_test_task("pk-anchor-dangling")
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)
    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="legacy-checkpoint",
            execution_id="shared-execution",
            label="legacy-fallback",
            timestamp=datetime.now(timezone.utc),
            run_id="run-a",
        )
    )
    task.last_checkpoint_trace_event_id = 999999999
    db.commit()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.trace_handlers.get_db", get_test_db)

    clear_degradation(CHECKPOINT_PK_ANCHOR_DANGLING)
    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            assert DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                "shared-execution"
            ) == {"label": "legacy-fallback"}
        assert CHECKPOINT_PK_ANCHOR_DANGLING in active_degradations()
    finally:
        clear_degradation(CHECKPOINT_PK_ANCHOR_DANGLING)
        db.close()


def test_database_trace_handler_unbound_legacy_load_fails_closed_after_tagged_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SessionLocal, db, task = _create_trace_handler_test_task("legacy-only-load")
    task_id = int(task.id)
    now = datetime.now(timezone.utc)
    db.add_all(
        [
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="legacy-checkpoint",
                execution_id="shared-execution",
                label="legacy",
                timestamp=now,
            ),
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="newer-tagged-checkpoint",
                execution_id="shared-execution",
                label="tagged",
                timestamp=now + timedelta(seconds=1),
                run_id="run-a",
            ),
        ]
    )
    db.commit()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.trace_handlers.get_db", get_test_db)

    try:
        # A tagged run has positive proof a checkpoint exists in a partition
        # this unbound legacy reader is not authoritative for. That is a
        # refusal, not the "queried successfully, found nothing" fact that
        # ``None`` reserves.
        with pytest.raises(CheckpointAccessRefusedError) as excinfo:
            DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                "shared-execution"
            )
        assert excinfo.value.reason == "superseded_legacy"
    finally:
        db.close()


def test_database_trace_handler_unbound_legacy_load_stays_legacy_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SessionLocal, db, task = _create_trace_handler_test_task("legacy-only-load")
    task_id = int(task.id)
    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="legacy-checkpoint",
            execution_id="shared-execution",
            label="legacy",
            timestamp=datetime.now(timezone.utc),
        )
    )
    db.commit()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.trace_handlers.get_db", get_test_db)

    try:
        assert DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
            "shared-execution"
        ) == {"label": "legacy"}
    finally:
        db.close()


def test_database_trace_handler_unbound_root_load_fails_closed_for_active_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SessionLocal, db, task = _create_trace_handler_test_task("unbound-active-load")
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)
    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="legacy-checkpoint",
            execution_id="shared-execution",
            label="legacy",
            timestamp=datetime.now(timezone.utc),
        )
    )
    db.commit()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.trace_handlers.get_db", get_test_db)

    try:
        # An active run is in progress under a different lease; this unbound
        # legacy reader is refused, not told the checkpoint is absent.
        with pytest.raises(CheckpointAccessRefusedError) as excinfo:
            DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                "shared-execution"
            )
        assert excinfo.value.reason == "active_run"
    finally:
        db.close()


def test_database_trace_handler_load_refuses_lease_bound_to_another_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound lease for a different task id is not this reader's partition
    -- distinct from ``active_run``/``superseded_legacy``: the read itself
    is contaminated by a stray context, not a policy decision about this
    task's own rows."""
    SessionLocal, db, task = _create_trace_handler_test_task("lease-mismatch-load")
    task_id = int(task.id)

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.trace_handlers.get_db", get_test_db)

    try:
        with (
            bind_task_lease_context(TaskLease(task_id + 1, "runner-a", "run-a")),
            pytest.raises(CheckpointAccessRefusedError) as excinfo,
        ):
            DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                "shared-execution"
            )
        assert excinfo.value.reason == "lease_mismatch"
    finally:
        db.close()


def test_partition_refusal_is_distinct_from_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused partition read must never be reported the same way as a
    query that succeeded and genuinely found nothing."""
    SessionLocal, db, task = _create_trace_handler_test_task("refusal-vs-absence")
    task_id = int(task.id)
    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="tagged-checkpoint",
            execution_id="shared-execution",
            label="tagged",
            timestamp=datetime.now(timezone.utc),
            run_id="run-a",
        )
    )
    db.commit()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.trace_handlers.get_db", get_test_db)

    try:
        # No lease bound, and a tagged run has positive proof a checkpoint
        # exists: this legacy reader is not authoritative for that
        # partition and must be refused, not told "absent".
        with pytest.raises(CheckpointAccessRefusedError):
            DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                "shared-execution"
            )
        # A reader that IS bound to the tagged run reads its own partition
        # and, for an execution id with no matching row there, gets the
        # authoritative "queried successfully, found nothing" result.
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            assert (
                DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                    "other-execution"
                )
                is None
            )
    finally:
        db.close()


def test_database_trace_handler_prunes_only_bound_run_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, db, task = _create_trace_handler_test_task("run-scoped-prune")
    task_id = int(task.id)
    now = datetime.now(timezone.utc)
    db.add_all(
        [
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="run-a-old",
                execution_id="shared-execution",
                label="run-a-old",
                timestamp=now,
                run_id="run-a",
            ),
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="run-a-new",
                execution_id="shared-execution",
                label="run-a-new",
                timestamp=now + timedelta(seconds=1),
                run_id="run-a",
            ),
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="run-b-old",
                execution_id="shared-execution",
                label="run-b-old",
                timestamp=now + timedelta(seconds=2),
                run_id="run-b",
            ),
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="run-b-new",
                execution_id="shared-execution",
                label="run-b-new",
                timestamp=now + timedelta(seconds=3),
                run_id="run-b",
            ),
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="legacy",
                execution_id="shared-execution",
                label="legacy",
                timestamp=now + timedelta(seconds=4),
            ),
        ]
    )
    db.commit()
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.get_checkpoint_history_limit",
        lambda: 1,
    )

    try:
        DatabaseTraceHandler(task_id)._prune_checkpoint_history(
            db,
            {
                "checkpoint_type": CHECKPOINT_TYPE,
                "execution_id": "shared-execution",
                TASK_RUN_ID_TRACE_FIELD: "run-b",
            },
        )

        assert {
            row.event_id
            for row in db.query(DatabaseTraceEvent).filter_by(task_id=task_id).all()
        } == {"run-a-old", "run-a-new", "run-b-new", "legacy"}
    finally:
        db.close()


def test_database_trace_handler_prunes_legacy_partition_without_tagged_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, db, task = _create_trace_handler_test_task("legacy-scoped-prune")
    task_id = int(task.id)
    now = datetime.now(timezone.utc)
    db.add_all(
        [
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="legacy-old",
                execution_id="shared-execution",
                label="legacy-old",
                timestamp=now,
            ),
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="legacy-new",
                execution_id="shared-execution",
                label="legacy-new",
                timestamp=now + timedelta(seconds=1),
            ),
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="tagged-run",
                execution_id="shared-execution",
                label="tagged-run",
                timestamp=now + timedelta(seconds=2),
                run_id="run-a",
            ),
        ]
    )
    db.commit()
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.get_checkpoint_history_limit",
        lambda: 1,
    )

    try:
        DatabaseTraceHandler(task_id)._prune_checkpoint_history(
            db,
            {
                "checkpoint_type": CHECKPOINT_TYPE,
                "execution_id": "shared-execution",
            },
        )

        assert {
            row.event_id
            for row in db.query(DatabaseTraceEvent).filter_by(task_id=task_id).all()
        } == {"legacy-new", "tagged-run"}
    finally:
        db.close()


def test_database_trace_handler_prune_excludes_the_anchored_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic: the task's pointer is wired to an older row that the
    retention ranking would otherwise prune. Nothing in today's
    steady-state writer produces this -- the pointer always names the row
    just written, which ranks newest -- but a future back-pointing anchor
    (or a backfilled pointer) could, so prune must never delete the row a
    pointer references regardless of its rank."""
    _, db, task = _create_trace_handler_test_task("anchor-protected-prune")
    task_id = int(task.id)
    now = datetime.now(timezone.utc)
    old_row = _checkpoint_trace_row(
        task_id=task_id,
        event_id="anchored-old",
        execution_id="shared-execution",
        label="anchored-old",
        timestamp=now,
        run_id="run-a",
    )
    db.add(old_row)
    db.commit()
    db.refresh(old_row)
    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="run-a-new",
            execution_id="shared-execution",
            label="run-a-new",
            timestamp=now + timedelta(seconds=1),
            run_id="run-a",
        )
    )
    # Synthetic: wire the pointer to the older row, which the retention
    # ranking (newest first, limit=1) would otherwise prune.
    task.last_checkpoint_trace_event_id = old_row.id
    db.commit()
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.get_checkpoint_history_limit",
        lambda: 1,
    )

    try:
        DatabaseTraceHandler(task_id)._prune_checkpoint_history(
            db,
            {
                "checkpoint_type": CHECKPOINT_TYPE,
                "execution_id": "shared-execution",
                TASK_RUN_ID_TRACE_FIELD: "run-a",
            },
        )

        assert {
            row.event_id
            for row in db.query(DatabaseTraceEvent).filter_by(task_id=task_id).all()
        } == {"anchored-old", "run-a-new"}
    finally:
        db.close()


def test_prune_protects_a_trace_row_anchored_by_an_active_interaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second protection source (PR-C2b): an active interaction row's
    resume_trace_event_id keeps its anchor alive even when the anchor ranks
    outside the retention window, the same way the task's exact-row pointer
    does in the test above -- but through task_interaction_requests instead
    of tasks.last_checkpoint_trace_event_id."""
    _, db, task = _create_trace_handler_test_task("interaction-anchor-protected")
    task_id = int(task.id)
    now = datetime.now(timezone.utc)
    old_row = _checkpoint_trace_row(
        task_id=task_id,
        event_id="interaction-anchored-old",
        execution_id="shared-execution",
        label="interaction-anchored-old",
        timestamp=now,
        run_id="run-a",
    )
    db.add(old_row)
    db.commit()
    db.refresh(old_row)
    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="run-a-new",
            execution_id="shared-execution",
            label="run-a-new",
            timestamp=now + timedelta(seconds=1),
            run_id="run-a",
        )
    )
    db.commit()
    _seed_interaction_row(
        db, task_id=task_id, resume_trace_event_id=old_row.id, status="active"
    )
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.get_checkpoint_history_limit",
        lambda: 1,
    )

    try:
        DatabaseTraceHandler(task_id)._prune_checkpoint_history(
            db,
            {
                "checkpoint_type": CHECKPOINT_TYPE,
                "execution_id": "shared-execution",
                TASK_RUN_ID_TRACE_FIELD: "run-a",
            },
        )

        assert {
            row.event_id
            for row in db.query(DatabaseTraceEvent).filter_by(task_id=task_id).all()
        } == {"interaction-anchored-old", "run-a-new"}
    finally:
        db.close()


def test_prune_does_not_protect_a_terminal_interaction_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reverse control for the test above: a terminated interaction row's
    resume_trace_event_id must not enter the protection set -- only
    active_slot IS NOT NULL rows do -- or the protection set would keep
    growing forever as interactions terminate instead of shrinking back to
    just the exact-row pointer."""
    _, db, task = _create_trace_handler_test_task("interaction-anchor-terminal")
    task_id = int(task.id)
    now = datetime.now(timezone.utc)
    old_row = _checkpoint_trace_row(
        task_id=task_id,
        event_id="terminal-anchored-old",
        execution_id="shared-execution",
        label="terminal-anchored-old",
        timestamp=now,
        run_id="run-a",
    )
    db.add(old_row)
    db.commit()
    db.refresh(old_row)
    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="run-a-new",
            execution_id="shared-execution",
            label="run-a-new",
            timestamp=now + timedelta(seconds=1),
            run_id="run-a",
        )
    )
    db.commit()
    _seed_interaction_row(
        db, task_id=task_id, resume_trace_event_id=old_row.id, status="terminated"
    )
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.get_checkpoint_history_limit",
        lambda: 1,
    )

    try:
        DatabaseTraceHandler(task_id)._prune_checkpoint_history(
            db,
            {
                "checkpoint_type": CHECKPOINT_TYPE,
                "execution_id": "shared-execution",
                TASK_RUN_ID_TRACE_FIELD: "run-a",
            },
        )

        assert {
            row.event_id
            for row in db.query(DatabaseTraceEvent).filter_by(task_id=task_id).all()
        } == {"run-a-new"}
    finally:
        db.close()


def test_prune_protects_an_expired_but_active_interaction_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliberately no expires_at filter on the protection-set query: an
    expired-but-still-active request (nothing has terminated it yet) is
    still answerable, so its anchor must stay protected."""
    _, db, task = _create_trace_handler_test_task("interaction-anchor-expired")
    task_id = int(task.id)
    now = datetime.now(timezone.utc)
    old_row = _checkpoint_trace_row(
        task_id=task_id,
        event_id="expired-anchored-old",
        execution_id="shared-execution",
        label="expired-anchored-old",
        timestamp=now,
        run_id="run-a",
    )
    db.add(old_row)
    db.commit()
    db.refresh(old_row)
    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="run-a-new",
            execution_id="shared-execution",
            label="run-a-new",
            timestamp=now + timedelta(seconds=1),
            run_id="run-a",
        )
    )
    db.commit()
    _seed_interaction_row(
        db,
        task_id=task_id,
        resume_trace_event_id=old_row.id,
        status="active",
        created_at=now - timedelta(hours=2),
        expires_at=now - timedelta(hours=1),
    )
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.get_checkpoint_history_limit",
        lambda: 1,
    )

    try:
        DatabaseTraceHandler(task_id)._prune_checkpoint_history(
            db,
            {
                "checkpoint_type": CHECKPOINT_TYPE,
                "execution_id": "shared-execution",
                TASK_RUN_ID_TRACE_FIELD: "run-a",
            },
        )

        assert {
            row.event_id
            for row in db.query(DatabaseTraceEvent).filter_by(task_id=task_id).all()
        } == {"expired-anchored-old", "run-a-new"}
    finally:
        db.close()


def test_prune_retains_at_most_limit_plus_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both protection sources at once, anchoring two distinct rows: the
    exact-row pointer and the active interaction anchor are the two
    exceptions the docstring's limit + 2 bound accounts for."""
    _, db, task = _create_trace_handler_test_task("limit-plus-two")
    task_id = int(task.id)
    now = datetime.now(timezone.utc)
    pointer_row = _checkpoint_trace_row(
        task_id=task_id,
        event_id="pointer-anchored-oldest",
        execution_id="shared-execution",
        label="pointer-anchored-oldest",
        timestamp=now,
        run_id="run-a",
    )
    db.add(pointer_row)
    db.commit()
    db.refresh(pointer_row)
    interaction_anchor_row = _checkpoint_trace_row(
        task_id=task_id,
        event_id="interaction-anchored-middle",
        execution_id="shared-execution",
        label="interaction-anchored-middle",
        timestamp=now + timedelta(seconds=1),
        run_id="run-a",
    )
    db.add(interaction_anchor_row)
    db.commit()
    db.refresh(interaction_anchor_row)
    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="run-a-newest",
            execution_id="shared-execution",
            label="run-a-newest",
            timestamp=now + timedelta(seconds=2),
            run_id="run-a",
        )
    )
    task.last_checkpoint_trace_event_id = pointer_row.id
    db.commit()
    _seed_interaction_row(
        db,
        task_id=task_id,
        resume_trace_event_id=interaction_anchor_row.id,
        status="active",
    )
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.get_checkpoint_history_limit",
        lambda: 1,
    )

    try:
        DatabaseTraceHandler(task_id)._prune_checkpoint_history(
            db,
            {
                "checkpoint_type": CHECKPOINT_TYPE,
                "execution_id": "shared-execution",
                TASK_RUN_ID_TRACE_FIELD: "run-a",
            },
        )

        remaining = {
            row.event_id
            for row in db.query(DatabaseTraceEvent).filter_by(task_id=task_id).all()
        }
        assert remaining == {
            "pointer-anchored-oldest",
            "interaction-anchored-middle",
            "run-a-newest",
        }
        assert len(remaining) == 3  # limit (1) + 2
    finally:
        db.close()


def test_prune_retains_exactly_the_limit_when_the_interaction_anchor_is_in_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interaction-anchor analog of
    test_database_trace_handler_prune_retains_exactly_the_limit_when_the_anchor_is_in_range:
    when the protected row already ranks inside the retention window, its
    protection is redundant with natural retention, and the retained count
    must stay exactly ``limit``.

    This is also the test that actually distinguishes "exclude protected
    rows from the ranking query, then OFFSET" (M9's bug) from "OFFSET
    first, then exclude protected rows from the stale set" (the correct
    order): when every protected row ranks outside the window (as in
    test_prune_retains_at_most_limit_plus_two above), both orderings
    produce the identical final set -- removing a row that already sits
    past the OFFSET boundary cannot change which rows are within it. The
    divergence only appears when a protected row ranks inside the window,
    which is exactly this scenario.
    """
    _, db, task = _create_trace_handler_test_task("interaction-anchor-in-range")
    task_id = int(task.id)
    now = datetime.now(timezone.utc)
    rows = [
        _checkpoint_trace_row(
            task_id=task_id,
            event_id=event_id,
            execution_id="shared-execution",
            label=event_id,
            timestamp=now + timedelta(seconds=offset),
            run_id="run-a",
        )
        for offset, event_id in enumerate(["oldest", "middle", "newest"])
    ]
    db.add_all(rows)
    db.commit()
    db.refresh(rows[-1])
    _seed_interaction_row(
        db, task_id=task_id, resume_trace_event_id=rows[-1].id, status="active"
    )
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.get_checkpoint_history_limit",
        lambda: 1,
    )

    try:
        DatabaseTraceHandler(task_id)._prune_checkpoint_history(
            db,
            {
                "checkpoint_type": CHECKPOINT_TYPE,
                "execution_id": "shared-execution",
                TASK_RUN_ID_TRACE_FIELD: "run-a",
            },
        )

        assert {
            row.event_id
            for row in db.query(DatabaseTraceEvent).filter_by(task_id=task_id).all()
        } == {"newest"}
    finally:
        db.close()


def test_prune_runs_without_the_interaction_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The has_table gate's reason to exist: on a deployment upgraded to a
    revision before task_interaction_requests exists, prune must still
    complete, still reclaim stale rows, and must not register
    CHECKPOINT_PRUNE_FAILED -- registering it here would be exactly the
    failure mode measured in the audit (SQLite's OperationalError landing
    in the (IntegrityError, OperationalError) handler on every checkpoint
    write, with no way for an operator to ever clear it)."""
    _, db, task = _create_trace_handler_test_task_without_interaction_table(
        "prune-without-interaction-table"
    )
    task_id = int(task.id)
    assert interaction_requests_table_exists(db) is False
    now = datetime.now(timezone.utc)
    db.add_all(
        [
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="no-table-old",
                execution_id="shared-execution",
                label="no-table-old",
                timestamp=now,
                run_id="run-a",
            ),
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="no-table-new",
                execution_id="shared-execution",
                label="no-table-new",
                timestamp=now + timedelta(seconds=1),
                run_id="run-a",
            ),
        ]
    )
    db.commit()
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.get_checkpoint_history_limit",
        lambda: 1,
    )

    clear_degradation(CHECKPOINT_PRUNE_FAILED)
    try:
        DatabaseTraceHandler(task_id)._prune_checkpoint_history(
            db,
            {
                "checkpoint_type": CHECKPOINT_TYPE,
                "execution_id": "shared-execution",
                TASK_RUN_ID_TRACE_FIELD: "run-a",
            },
        )

        assert CHECKPOINT_PRUNE_FAILED not in active_degradations()
        assert {
            row.event_id
            for row in db.query(DatabaseTraceEvent).filter_by(task_id=task_id).all()
        } == {"no-table-new"}
    finally:
        clear_degradation(CHECKPOINT_PRUNE_FAILED)
        db.close()


@pytest.mark.parametrize(
    "injected_exception",
    [
        pytest.param(
            IntegrityError("DELETE", {}, Exception("restrict violation")),
            id="integrity_error",
        ),
        pytest.param(
            OperationalError("DELETE", {}, Exception("serialization failure")),
            id="operational_error",
        ),
    ],
)
def test_database_trace_handler_prune_registers_degradation_on_delete_error(
    monkeypatch: pytest.MonkeyPatch,
    injected_exception: Exception,
) -> None:
    """Synthetic delete-failure injection: the delete step is replaced to
    raise, isolating the except-(IntegrityError, OperationalError) branch
    from needing genuine FK enforcement (which only PostgreSQL, or a
    freshly create_all'd SQLite database, provides for this column) or a
    genuine serialization conflict. Both exception classes are covered
    because PostgreSQL's restrict-violation and its SerializationFailure /
    DeadlockDetected surface through different SQLAlchemy wrapper classes
    (IntegrityError vs. OperationalError) for the same retention race."""
    from sqlalchemy.orm import Query

    from xagent.web.services.ops_signals import (
        CHECKPOINT_PRUNE_FAILED,
        active_degradations,
        clear_degradation,
    )

    _, db, task = _create_trace_handler_test_task("prune-failure")
    task_id = int(task.id)
    now = datetime.now(timezone.utc)
    db.add_all(
        [
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="old",
                execution_id="shared-execution",
                label="old",
                timestamp=now,
            ),
            _checkpoint_trace_row(
                task_id=task_id,
                event_id="new",
                execution_id="shared-execution",
                label="new",
                timestamp=now + timedelta(seconds=1),
            ),
        ]
    )
    db.commit()
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.get_checkpoint_history_limit",
        lambda: 1,
    )

    def failing_delete(self: Query, *args: object, **kwargs: object) -> int:
        raise injected_exception

    monkeypatch.setattr(Query, "delete", failing_delete)

    clear_degradation(CHECKPOINT_PRUNE_FAILED)
    try:
        # Must not raise -- a prune failure can never break the checkpoint
        # write it follows.
        DatabaseTraceHandler(task_id)._prune_checkpoint_history(
            db,
            {"checkpoint_type": CHECKPOINT_TYPE, "execution_id": "shared-execution"},
        )
        assert CHECKPOINT_PRUNE_FAILED in active_degradations()
        assert db.query(DatabaseTraceEvent).filter_by(task_id=task_id).count() == 2
    finally:
        clear_degradation(CHECKPOINT_PRUNE_FAILED)
        db.close()


def test_websocket_trace_handler_dedupes_prior_user_message_turn_id(
    monkeypatch,
) -> None:
    engine = _shared_memory_sqlite_engine()
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
        db.add_all(
            [
                WorkforceRun(
                    workforce_id=999,
                    task_id=task_id,
                    user_id=user_id,
                    status="completed",
                    snapshot={},
                ),
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
                ),
                DatabaseTraceEvent(
                    task_id=task_id,
                    build_id="agent_123_abcd1234",
                    event_id="delegated-internal",
                    event_type="llm_call_start",
                    timestamp=base_time,
                    data={
                        "source": "xagent-agent-tool-child",
                        "worker_task_id": "agent_123_abcd1234",
                    },
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

    delegation_event = next(
        event for event in sent_events if event.get("event_id") == "delegation-end"
    )
    assert delegation_event["event_type"] == "workforce_delegation_end"
    assert delegation_event["data"]["worker_alias"] == "Writer"
    assert delegation_event["data"]["worker_task_id"] == "agent_123_abcd1234"
    assert delegation_event["data"]["output"] == "draft complete"
    assert "messages" not in delegation_event["data"]
    assert "event_type" not in delegation_event["data"]
    assert not any(
        event.get("event_id") == "delegated-internal" for event in sent_events
    )


@pytest.mark.asyncio
async def test_historical_replay_includes_child_trace_for_non_workforce_task(
    monkeypatch,
) -> None:
    SessionLocal, db, task = _create_trace_handler_test_task(
        "non-workforce-child-history"
    )
    try:
        task_id = int(task.id)
        user_id = int(task.user_id)
        base_time = datetime(2026, 5, 22, tzinfo=timezone.utc)
        db.add_all(
            [
                DatabaseTraceEvent(
                    task_id=task_id,
                    build_id="agent_123_abcd1234",
                    event_id="delegated-progress",
                    event_type="agent_progress",
                    timestamp=base_time,
                    data={
                        "source": "xagent-agent-tool-child",
                        "worker_task_id": "agent_123_abcd1234",
                        "message": "Child is working",
                    },
                ),
                DatabaseTraceEvent(
                    task_id=task_id,
                    build_id="builder-session",
                    event_id="unrelated-build",
                    event_type="agent_progress",
                    timestamp=base_time + timedelta(seconds=1),
                    data={"source": "builder", "message": "Internal build trace"},
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

    delegated_event = next(
        event for event in sent_events if event.get("event_id") == "delegated-progress"
    )
    assert delegated_event["event_type"] == "agent_progress"
    assert delegated_event["data"]["source"] == "xagent-agent-tool-child"
    assert not any(event.get("event_id") == "unrelated-build" for event in sent_events)


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
    engine = _shared_memory_sqlite_engine()
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
    engine = _shared_memory_sqlite_engine()
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
    engine = _shared_memory_sqlite_engine()
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


_BROKEN_ANCHOR_LABEL = "anchored-broken"


def _unidentified_checkpoint_row(
    *,
    task_id: int,
    label: str,
    timestamp: datetime,
    run_id: str,
) -> DatabaseTraceEvent:
    """A checkpoint row carrying no execution identity at all.

    The pointer anchors it (checkpoint_execution_id() returns "" and the
    anchor's identity conjunct short-circuits), but the legacy scan excludes
    it: its coalesce(nullif(...)) predicate is NULL and NULL = :execution_id
    is never true. This is the shape that makes the anchor's payload verdict
    load-bearing -- without it seeding the scan's accumulators, an unreadable
    checkpoint would come back as "no checkpoint".
    """
    return DatabaseTraceEvent(
        task_id=task_id,
        build_id=None,
        event_id=f"unidentified-{label}",
        event_type="system_update_general",
        timestamp=timestamp,
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            TASK_RUN_ID_TRACE_FIELD: run_id,
            "snapshot": {"label": label},
        },
    )


def _decode_failing_on(label: str, exception: Exception):
    """Decode stub that fails for exactly one row, identified by its label.

    Injected rather than built from a genuinely broken blob because these
    tests pin the anchor's *classification* of a decode failure, and the
    generic (non-CheckpointMessageDecodeError) branch has no data-level
    trigger at all.
    """

    def fake_decode(db, *, task_id, data, strict=False, verify_blob_hashes=True):
        snapshot = data.get("snapshot") if isinstance(data, dict) else None
        if isinstance(snapshot, dict) and snapshot.get("label") == label:
            raise exception
        return data

    return fake_decode


def _bind_checkpoint_read_session(
    monkeypatch: pytest.MonkeyPatch,
    SessionLocal,
) -> None:
    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.trace_handlers.get_db", get_test_db)


def test_database_trace_handler_load_pk_anchor_undecodable_payload_falls_back_to_older_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retention guarantee _prune_checkpoint_history documents: older
    rows are kept so an unreadable latest can fall back. The anchor's
    identity stays authoritative, but its payload does not."""
    from xagent.web.services.trace_message_storage import CheckpointMessageDecodeError

    SessionLocal, db, task = _create_trace_handler_test_task("anchor-undecodable-older")
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)
    now = datetime.now(timezone.utc)
    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="older",
            execution_id="shared-execution",
            label="older-readable",
            timestamp=now,
            run_id="run-a",
        )
    )
    broken = _checkpoint_trace_row(
        task_id=task_id,
        event_id="broken",
        execution_id="shared-execution",
        label=_BROKEN_ANCHOR_LABEL,
        timestamp=now + timedelta(seconds=1),
        run_id="run-a",
    )
    db.add(broken)
    db.commit()
    db.refresh(broken)
    task.last_checkpoint_trace_event_id = broken.id
    db.commit()

    _bind_checkpoint_read_session(monkeypatch, SessionLocal)
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.decode_trace_event_data",
        _decode_failing_on(
            _BROKEN_ANCHOR_LABEL, CheckpointMessageDecodeError("blob is gone")
        ),
    )

    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            assert DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                "shared-execution"
            ) == {"label": "older-readable"}
    finally:
        db.close()


def test_database_trace_handler_load_pk_anchor_undecodable_payload_without_older_row_is_corrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The anchored row is the only checkpoint and the scan cannot even see
    it (no execution identity). The deferral must still resolve to corrupt,
    never to ``None`` -- an unreadable checkpoint is not an absent one."""
    from xagent.web.services.trace_message_storage import CheckpointMessageDecodeError

    SessionLocal, db, task = _create_trace_handler_test_task("anchor-undecodable-only")
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)
    broken = _unidentified_checkpoint_row(
        task_id=task_id,
        label=_BROKEN_ANCHOR_LABEL,
        timestamp=datetime.now(timezone.utc),
        run_id="run-a",
    )
    db.add(broken)
    db.commit()
    db.refresh(broken)
    task.last_checkpoint_trace_event_id = broken.id
    db.commit()

    _bind_checkpoint_read_session(monkeypatch, SessionLocal)
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.decode_trace_event_data",
        _decode_failing_on(
            _BROKEN_ANCHOR_LABEL, CheckpointMessageDecodeError("blob is gone")
        ),
    )

    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            with pytest.raises(CheckpointCorruptError):
                DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                    "shared-execution"
                )
    finally:
        db.close()


def test_database_trace_handler_load_pk_anchor_generic_decode_failure_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient (non-decode-class) failure on the anchored row is
    retryable, not terminal: unavailable, so a2a maps it to 503 rather than
    the terminal 400 a corrupt verdict earns."""
    from xagent.web.services.ops_signals import (
        CHECKPOINT_DECODE_FALLBACK,
        CHECKPOINT_LOAD_UNAVAILABLE,
        active_degradations,
        clear_degradation,
    )

    SessionLocal, db, task = _create_trace_handler_test_task("anchor-generic-only")
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)
    broken = _unidentified_checkpoint_row(
        task_id=task_id,
        label=_BROKEN_ANCHOR_LABEL,
        timestamp=datetime.now(timezone.utc),
        run_id="run-a",
    )
    db.add(broken)
    db.commit()
    db.refresh(broken)
    task.last_checkpoint_trace_event_id = broken.id
    db.commit()

    _bind_checkpoint_read_session(monkeypatch, SessionLocal)
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.decode_trace_event_data",
        _decode_failing_on(_BROKEN_ANCHOR_LABEL, RuntimeError("blob prefetch failed")),
    )

    clear_degradation(CHECKPOINT_DECODE_FALLBACK)
    clear_degradation(CHECKPOINT_LOAD_UNAVAILABLE)
    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            with pytest.raises(CheckpointUnavailableError):
                DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                    "shared-execution"
                )
        active = active_degradations()
        assert CHECKPOINT_DECODE_FALLBACK in active
        assert CHECKPOINT_LOAD_UNAVAILABLE in active
    finally:
        clear_degradation(CHECKPOINT_DECODE_FALLBACK)
        clear_degradation(CHECKPOINT_LOAD_UNAVAILABLE)
        db.close()


def test_database_trace_handler_absent_run_field_with_scan_generic_failure_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pointer row missing its run-partition field defers to the legacy
    scan (see test_database_trace_handler_load_pk_anchor_absent_run_field_defers_to_scan),
    which the scan's own partition filter excludes it from. If the scan then
    finds a *different*, correctly-partitioned row that fails to decode for
    a transient (non-decode-class) reason, that is a stronger, more specific
    verdict than the anchor's own run-provenance refusal -- the scan
    actually examined a row and hit a real failure. CheckpointUnavailableError
    (retryable) must win, not CheckpointAccessRefusedError."""
    SessionLocal, db, task = _create_trace_handler_test_task(
        "pk-anchor-absent-run-field-scan-generic"
    )
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)

    pointer_row = DatabaseTraceEvent(
        task_id=task_id,
        build_id=None,
        event_id="pointer-absent-run-field-generic",
        event_type="system_update_general",
        timestamp=datetime.now(timezone.utc),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": "shared-execution",
            "snapshot": {"label": "absent-run-field-pointer"},
        },
    )
    db.add(pointer_row)
    db.commit()
    db.refresh(pointer_row)
    task.last_checkpoint_trace_event_id = pointer_row.id
    db.commit()

    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="scan-row-generic",
            execution_id="shared-execution",
            label=_BROKEN_ANCHOR_LABEL,
            timestamp=datetime.now(timezone.utc),
            run_id="run-a",
        )
    )
    db.commit()

    _bind_checkpoint_read_session(monkeypatch, SessionLocal)
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.decode_trace_event_data",
        _decode_failing_on(_BROKEN_ANCHOR_LABEL, RuntimeError("blob prefetch failed")),
    )

    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            with pytest.raises(CheckpointUnavailableError):
                DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                    "shared-execution"
                )
    finally:
        db.close()


def test_database_trace_handler_absent_run_field_with_scan_undecodable_row_is_corrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same setup as the generic-failure sibling above, but the scan's
    matching row is permanently undecodable instead of transiently failed.
    A fully scanned, exclusively-undecodable candidate set outranks the
    anchor's run-provenance refusal the same way: CheckpointCorruptError
    (terminal, not retryable) must win, not CheckpointAccessRefusedError."""
    from xagent.web.services.trace_message_storage import CheckpointMessageDecodeError

    SessionLocal, db, task = _create_trace_handler_test_task(
        "pk-anchor-absent-run-field-scan-undecodable"
    )
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)

    pointer_row = DatabaseTraceEvent(
        task_id=task_id,
        build_id=None,
        event_id="pointer-absent-run-field-undecodable",
        event_type="system_update_general",
        timestamp=datetime.now(timezone.utc),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": "shared-execution",
            "snapshot": {"label": "absent-run-field-pointer"},
        },
    )
    db.add(pointer_row)
    db.commit()
    db.refresh(pointer_row)
    task.last_checkpoint_trace_event_id = pointer_row.id
    db.commit()

    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="scan-row-undecodable",
            execution_id="shared-execution",
            label=_BROKEN_ANCHOR_LABEL,
            timestamp=datetime.now(timezone.utc),
            run_id="run-a",
        )
    )
    db.commit()

    _bind_checkpoint_read_session(monkeypatch, SessionLocal)
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.decode_trace_event_data",
        _decode_failing_on(
            _BROKEN_ANCHOR_LABEL, CheckpointMessageDecodeError("blob is gone")
        ),
    )

    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            with pytest.raises(CheckpointCorruptError):
                DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                    "shared-execution"
                )
    finally:
        db.close()


def _decode_dropping_snapshot(label: str):
    """Decode stub that succeeds but strips the snapshot for one row,
    identified by its label -- the "readable checkpoint_type but no
    snapshot" shape, which has no data-level trigger of its own."""

    def fake_decode(db, *, task_id, data, strict=False, verify_blob_hashes=True):
        snapshot = data.get("snapshot") if isinstance(data, dict) else None
        if isinstance(snapshot, dict) and snapshot.get("label") == label:
            return {"checkpoint_type": CHECKPOINT_TYPE}
        return data

    return fake_decode


def _absent_run_field_pointer_row(
    *, task_id: int, label: str, timestamp: datetime
) -> DatabaseTraceEvent:
    """A pointer row identified by execution id but missing the run-partition
    field entirely -- the shape ``_classify_anchor_identity`` defers on
    rather than raising. No other row is added alongside it in the tests
    below, so the legacy scan the anchor defers to always comes back empty:
    each test pins what the anchor's own payload verdict resolves to on its
    own, before that empty scan is even reached."""
    return _checkpoint_trace_row(
        task_id=task_id,
        event_id=f"absent-run-field-{label}",
        execution_id="shared-execution",
        label=label,
        timestamp=timestamp,
        run_id=None,
    )


def test_database_trace_handler_absent_run_field_undecodable_payload_is_corrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F1: a pointer row missing the run-partition field whose payload is
    also undecodable (a broken blob reference) must not be masked by the
    run-provenance refusal -- the payload verdict is a stronger, more
    specific fact and has to win. With nothing else for the legacy scan to
    find, the read raises CheckpointCorruptError, not
    CheckpointAccessRefusedError."""
    from xagent.web.services.trace_message_storage import CheckpointMessageDecodeError

    SessionLocal, db, task = _create_trace_handler_test_task(
        "pk-anchor-absent-run-field-undecodable"
    )
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)

    label = "absent-run-field-undecodable"
    row = _absent_run_field_pointer_row(
        task_id=task_id, label=label, timestamp=datetime.now(timezone.utc)
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    task.last_checkpoint_trace_event_id = row.id
    db.commit()

    _bind_checkpoint_read_session(monkeypatch, SessionLocal)
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.decode_trace_event_data",
        _decode_failing_on(label, CheckpointMessageDecodeError("blob is gone")),
    )

    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            with pytest.raises(CheckpointCorruptError):
                DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                    "shared-execution"
                )
    finally:
        db.close()

    # The condition _classify_anchor_identity establishes still fires
    # independently of how the payload stage then resolves.
    assert ir.counters_snapshot() == {
        ir.COUNTER_CHECKPOINT_ABSENT_MISSING_RUN_PARTITION: 1
    }


def test_database_trace_handler_absent_run_field_transient_decode_failure_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F1: same setup, but the payload fetch fails transiently (e.g. a blob
    prefetch hitting a momentary database error) rather than being
    permanently undecodable. That is retryable, so the read must raise
    CheckpointUnavailableError, not the non-retryable
    CheckpointAccessRefusedError the run-provenance condition alone would
    produce. This is the case identified as the one that actually cost
    something under the old, pre-refactor ordering: a transient failure
    could previously be swallowed into a refusal a retry would never clear."""
    SessionLocal, db, task = _create_trace_handler_test_task(
        "pk-anchor-absent-run-field-transient"
    )
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)

    label = "absent-run-field-transient"
    row = _absent_run_field_pointer_row(
        task_id=task_id, label=label, timestamp=datetime.now(timezone.utc)
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    task.last_checkpoint_trace_event_id = row.id
    db.commit()

    _bind_checkpoint_read_session(monkeypatch, SessionLocal)
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.decode_trace_event_data",
        _decode_failing_on(label, RuntimeError("blob prefetch failed")),
    )

    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            with pytest.raises(CheckpointUnavailableError):
                DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                    "shared-execution"
                )
    finally:
        db.close()


def test_database_trace_handler_absent_run_field_no_snapshot_is_corrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F1: same setup, but the row's decode succeeds while carrying a
    readable checkpoint_type and no snapshot at all. Same reasoning as the
    undecodable case: the payload verdict is more specific than an
    unverifiable provenance and must win, so the read raises
    CheckpointCorruptError."""
    SessionLocal, db, task = _create_trace_handler_test_task(
        "pk-anchor-absent-run-field-no-snapshot"
    )
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)

    label = "absent-run-field-no-snapshot"
    row = _absent_run_field_pointer_row(
        task_id=task_id, label=label, timestamp=datetime.now(timezone.utc)
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    task.last_checkpoint_trace_event_id = row.id
    db.commit()

    _bind_checkpoint_read_session(monkeypatch, SessionLocal)
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.decode_trace_event_data",
        _decode_dropping_snapshot(label),
    )

    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            with pytest.raises(CheckpointCorruptError):
                DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                    "shared-execution"
                )
    finally:
        db.close()

    assert ir.counters_snapshot() == {
        ir.COUNTER_CHECKPOINT_ABSENT_MISSING_RUN_PARTITION: 1
    }


def test_database_trace_handler_absent_run_field_clean_payload_clears_decode_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F1's one deliberate behavior change: a pointer row missing the
    run-partition field whose payload decodes cleanly still reaches
    CheckpointAccessRefusedError (reason "run_provenance_unavailable"),
    unchanged from before the refactor -- but it now also clears
    CHECKPOINT_DECODE_FALLBACK on the way, because _read_anchor_payload
    really did decode this row's payload before the refusal is raised. Under
    the old single-pass ordering this row returned before ever reaching that
    clear. The scan that follows this refusal would clear the same signal a
    few lines later anyway (the first row it decodes, if any); this pins
    that the clear now also happens here, rather than leaving it to be
    noticed later."""
    from xagent.web.services.ops_signals import (
        CHECKPOINT_DECODE_FALLBACK,
        active_degradations,
        register_degradation,
    )

    SessionLocal, db, task = _create_trace_handler_test_task(
        "pk-anchor-absent-run-field-clears-fallback"
    )
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)

    label = "absent-run-field-clears-fallback"
    row = _absent_run_field_pointer_row(
        task_id=task_id, label=label, timestamp=datetime.now(timezone.utc)
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    task.last_checkpoint_trace_event_id = row.id
    db.commit()

    _bind_checkpoint_read_session(monkeypatch, SessionLocal)
    register_degradation(CHECKPOINT_DECODE_FALLBACK, "pre-existing from an earlier row")

    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            with pytest.raises(CheckpointAccessRefusedError) as excinfo:
                DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                    "shared-execution"
                )
            assert excinfo.value.reason == "run_provenance_unavailable"
        assert CHECKPOINT_DECODE_FALLBACK not in active_degradations()
    finally:
        clear_degradation(CHECKPOINT_DECODE_FALLBACK)
        db.close()


@pytest.mark.parametrize("decodes_a_row", [True, False])
def test_database_trace_handler_decode_fallback_clears_after_a_successful_decode(
    monkeypatch: pytest.MonkeyPatch,
    decodes_a_row: bool,
) -> None:
    """The decode-fallback signal is process-wide and its clear is evidence
    that a decode worked, so an anchored read that resolves its pointer and
    decodes the row has to retire it: that read returns before the legacy
    scan, and the scan holds the only other clear. Without this the signal
    stays on /health for the life of the process as soon as the anchor
    starts succeeding, which is the normal steady state.

    The second case pins the clear to the decode rather than to the attempt.
    A read with nothing to decode -- no pointer, no rows -- proves nothing
    about the decode layer and must leave a set signal alone; a clear placed
    at the top of the anchored read next to the dangling clear would retire
    it there and report a false all-clear.
    """
    from xagent.web.services.ops_signals import (
        CHECKPOINT_DECODE_FALLBACK,
        active_degradations,
        clear_degradation,
        register_degradation,
    )

    SessionLocal, db, task = _create_trace_handler_test_task(
        f"decode-fallback-clear-{decodes_a_row}"
    )
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)
    expected: Dict[str, Any] | None = None
    if decodes_a_row:
        row = _checkpoint_trace_row(
            task_id=task_id,
            event_id="anchored",
            execution_id="shared-execution",
            label="anchored",
            timestamp=datetime.now(timezone.utc),
            run_id="run-a",
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        task.last_checkpoint_trace_event_id = row.id
        db.commit()
        expected = {"label": "anchored"}

    _bind_checkpoint_read_session(monkeypatch, SessionLocal)

    register_degradation(
        CHECKPOINT_DECODE_FALLBACK, "left over from an earlier fallback"
    )
    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            assert (
                DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                    "shared-execution"
                )
                == expected
            )
        if decodes_a_row:
            assert CHECKPOINT_DECODE_FALLBACK not in active_degradations()
        else:
            assert CHECKPOINT_DECODE_FALLBACK in active_degradations()
    finally:
        clear_degradation(CHECKPOINT_DECODE_FALLBACK)
        db.close()


@pytest.mark.parametrize("failing_call", ["pointer_query", "row_fetch"])
def test_database_trace_handler_load_pk_anchor_database_failure_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    failing_call: str,
) -> None:
    """_sync_load_latest_checkpoint promises every database touch is
    translated into a CheckpointReadError subclass. Both of the anchor's own
    database calls have to honour that, or a transient connection failure
    escapes as a raw driver exception no caller catches."""
    from xagent.web.services.ops_signals import (
        CHECKPOINT_LOAD_UNAVAILABLE,
        active_degradations,
        clear_degradation,
    )

    SessionLocal, db, task = _create_trace_handler_test_task(
        f"anchor-db-{failing_call}"
    )
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)
    row = _checkpoint_trace_row(
        task_id=task_id,
        event_id="anchored",
        execution_id="shared-execution",
        label="anchored",
        timestamp=datetime.now(timezone.utc),
        run_id="run-a",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    task.last_checkpoint_trace_event_id = row.id
    db.commit()

    _bind_checkpoint_read_session(monkeypatch, SessionLocal)

    boom = OperationalError("SELECT", {}, Exception("connection reset"))
    if failing_call == "pointer_query":
        original_one_or_none = Query.one_or_none

        def failing_one_or_none(self: Query):  # type: ignore[no-untyped-def]
            descriptions = self.column_descriptions
            if (
                descriptions
                and descriptions[0]["name"] == "last_checkpoint_trace_event_id"
            ):
                raise boom
            return original_one_or_none(self)

        monkeypatch.setattr(Query, "one_or_none", failing_one_or_none)
    else:

        def failing_get(self: Session, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            raise boom

        monkeypatch.setattr(Session, "get", failing_get)

    clear_degradation(CHECKPOINT_LOAD_UNAVAILABLE)
    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            with pytest.raises(CheckpointUnavailableError):
                DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                    "shared-execution"
                )
        assert CHECKPOINT_LOAD_UNAVAILABLE in active_degradations()
    finally:
        clear_degradation(CHECKPOINT_LOAD_UNAVAILABLE)
        db.close()


@pytest.mark.parametrize("anchor_set", [True, False])
def test_database_trace_handler_anchored_read_clears_the_dangling_signal(
    monkeypatch: pytest.MonkeyPatch,
    anchor_set: bool,
) -> None:
    """The clear must sit before the pointer lookup, not after a pointer
    resolves to a row.

    ops_signals state is process-wide, so a signal that only cleared once
    some pointer resolved would stay set forever in a process where no task
    carries an anchor. The anchorless case is the one that pins the
    placement: move the clear below the pointer lookup and the anchored case
    still passes while that one fails.
    """
    from xagent.web.services.ops_signals import (
        CHECKPOINT_PK_ANCHOR_DANGLING,
        active_degradations,
        clear_degradation,
        register_degradation,
    )

    SessionLocal, db, task = _create_trace_handler_test_task(
        f"anchor-clears-dangling-{anchor_set}"
    )
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)
    expected: Dict[str, Any] | None = None
    if anchor_set:
        row = _checkpoint_trace_row(
            task_id=task_id,
            event_id="anchored",
            execution_id="shared-execution",
            label="anchored",
            timestamp=datetime.now(timezone.utc),
            run_id="run-a",
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        task.last_checkpoint_trace_event_id = row.id
        db.commit()
        expected = {"label": "anchored"}
    assert (task.last_checkpoint_trace_event_id is not None) is anchor_set

    _bind_checkpoint_read_session(monkeypatch, SessionLocal)

    register_degradation(
        CHECKPOINT_PK_ANCHOR_DANGLING, "left over from another task's read"
    )
    try:
        with bind_task_lease_context(TaskLease(task_id, "runner-a", "run-a")):
            assert (
                DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                    "shared-execution"
                )
                == expected
            )
        assert CHECKPOINT_PK_ANCHOR_DANGLING not in active_degradations()
    finally:
        clear_degradation(CHECKPOINT_PK_ANCHOR_DANGLING)
        db.close()


def test_database_trace_handler_prune_retains_exactly_the_limit_when_the_anchor_is_in_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The steady state: the pointer names the row just written, which ranks
    inside the retention window. Ranking before protecting makes the
    documented retention count exact -- excluding the anchor from the
    candidate set instead shifts every remaining row's OFFSET rank by one
    and keeps limit + 1."""
    _, db, task = _create_trace_handler_test_task("anchor-in-range-prune")
    task_id = int(task.id)
    now = datetime.now(timezone.utc)
    rows = [
        _checkpoint_trace_row(
            task_id=task_id,
            event_id=event_id,
            execution_id="shared-execution",
            label=event_id,
            timestamp=now + timedelta(seconds=offset),
            run_id="run-a",
        )
        for offset, event_id in enumerate(["oldest", "middle", "newest"])
    ]
    db.add_all(rows)
    db.commit()
    db.refresh(rows[-1])
    task.last_checkpoint_trace_event_id = rows[-1].id
    db.commit()
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.get_checkpoint_history_limit",
        lambda: 1,
    )

    try:
        DatabaseTraceHandler(task_id)._prune_checkpoint_history(
            db,
            {
                "checkpoint_type": CHECKPOINT_TYPE,
                "execution_id": "shared-execution",
                TASK_RUN_ID_TRACE_FIELD: "run-a",
            },
        )

        assert {
            row.event_id
            for row in db.query(DatabaseTraceEvent).filter_by(task_id=task_id).all()
        } == {"newest"}
    finally:
        db.close()


def test_database_trace_handler_prune_with_nothing_stale_clears_the_failure_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy steady state prunes nothing. If the clear sat after the
    delete, that state could never retire a previously-latched signal, and
    it would show up as permanent /health noise."""
    from xagent.web.services.ops_signals import (
        CHECKPOINT_PRUNE_FAILED,
        active_degradations,
        clear_degradation,
        register_degradation,
    )

    _, db, task = _create_trace_handler_test_task("prune-nothing-stale")
    task_id = int(task.id)
    db.add(
        _checkpoint_trace_row(
            task_id=task_id,
            event_id="only",
            execution_id="shared-execution",
            label="only",
            timestamp=datetime.now(timezone.utc),
        )
    )
    db.commit()
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.get_checkpoint_history_limit",
        lambda: 5,
    )

    register_degradation(CHECKPOINT_PRUNE_FAILED, "left over from an earlier race")
    try:
        DatabaseTraceHandler(task_id)._prune_checkpoint_history(
            db,
            {"checkpoint_type": CHECKPOINT_TYPE, "execution_id": "shared-execution"},
        )
        assert CHECKPOINT_PRUNE_FAILED not in active_degradations()
    finally:
        clear_degradation(CHECKPOINT_PRUNE_FAILED)
        db.close()


def test_database_trace_handler_flush_without_primary_key_refuses_to_write_the_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard that replaced a bare assert. A no-op flush (rather than a
    raising one) is what exercises it: a raising flush would land in the
    generic handler instead and prove nothing about the guard."""
    _, db, task = _create_trace_handler_test_task("flush-without-pk")
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db.commit()
    task_id = int(task.id)
    lease = TaskLease(task_id=task_id, runner_id="runner-a", run_id="run-a")
    event = TraceEvent(
        CHECKPOINT_EVENT_TYPE,
        task_id=str(task_id),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": str(task_id),
            "snapshot": {"label": "unflushed"},
        },
    )

    monkeypatch.setattr(Session, "flush", lambda self, *a, **k: None)

    try:
        with (
            bind_task_lease_context(lease),
            pytest.raises(
                RuntimeError, match="refusing to write a NULL checkpoint anchor"
            ),
        ):
            DatabaseTraceHandler(task_id)._save_trace_event(db, event)

        db.expire_all()
        persisted = db.get(Task, task_id)
        assert persisted.last_checkpoint_trace_event_id is None
        assert db.query(DatabaseTraceEvent).filter_by(task_id=task_id).count() == 0
    finally:
        db.close()
