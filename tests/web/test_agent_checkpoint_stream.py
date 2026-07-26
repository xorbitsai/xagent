from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

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
from xagent.web.models.workforce import WorkforceRun
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
        assert db.query(DatabaseTraceEvent).filter_by(task_id=int(task.id)).count() == 0
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
        assert (
            DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                "shared-execution"
            )
            is None
        )
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
        assert (
            DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                "shared-execution"
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
