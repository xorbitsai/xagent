"""Identical durable-write contracts for sync and native async PostgreSQL."""

import asyncio
import os
import threading
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE, CHECKPOINT_TYPE
from xagent.core.agent.trace import TASK_START_GENERAL
from xagent.core.agent.trace import TraceEvent as CoreTraceEvent
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent, TraceMessageBlob
from xagent.web.models.user import User
from xagent.web.services import trace_handlers
from xagent.web.services.task_lease_service import TaskLease, bind_task_lease_context
from xagent.web.services.trace_database import TraceDatabaseRuntime
from xagent.web.services.trace_message_storage import decode_trace_event_data

pytestmark = [pytest.mark.postgresql, pytest.mark.asyncio]


@pytest_asyncio.fixture(params=[False, True], ids=["sync", "async"])
async def storage(request, monkeypatch):
    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    if request.param:
        pytest.importorskip("psycopg")
    schema = "trace_async_test_" + uuid.uuid4().hex
    admin = create_engine(url)
    with admin.begin() as db:
        db.execute(text(f'CREATE SCHEMA "{schema}"'))
    source = create_engine(
        make_url(url).update_query_dict({"options": f"-csearch_path={schema}"})
    )
    runtime = None
    try:
        Base.metadata.create_all(source)
        with Session(source) as db:
            user = User(username="trace-test", password_hash="unused")
            db.add(user)
            db.flush()
            task = Task(
                user_id=user.id,
                title="Trace test",
                description="Trace test",
                status=TaskStatus.RUNNING,
                runner_id="runner",
                run_id="run",
                lease_attempt_id="attempt",
            )
            db.add(task)
            db.flush()
            task_id = int(task.id)
            db.commit()
        runtime = TraceDatabaseRuntime(source, use_async=request.param, limit=1)
        monkeypatch.setattr(
            trace_handlers, "get_trace_database_runtime", lambda: runtime
        )
        monkeypatch.setenv("XAGENT_CHECKPOINT_HISTORY_LIMIT", "8")

        def sessions():
            with Session(source, autoflush=False) as db:
                yield db

        monkeypatch.setattr(trace_handlers, "get_db", sessions)
        yield source, runtime, trace_handlers.DatabaseTraceHandler(task_id), task_id
    finally:
        if runtime is not None:
            await runtime.close()
        source.dispose()
        with admin.begin() as db:
            db.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def checkpoint(task_id, step=0):
    execution_id = f"exec-{task_id}"
    return CoreTraceEvent(
        CHECKPOINT_EVENT_TYPE,
        task_id=str(task_id),
        timestamp=1800000000 + step,
        require_persisted=True,
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": execution_id,
            "root_execution_id": execution_id,
            "label": "after_llm",
            "snapshot": {
                "type": "checkpoint",
                "execution_id": execution_id,
                "label": "after_llm",
                "pattern": "ReActPattern",
                "pattern_state": {"current_iteration": step},
                "context": {"messages": [{"role": "user", "content": "same message"}]},
            },
        },
    )


async def test_checkpoint_retention_dedup_pointer_and_required_data(
    storage, monkeypatch
):
    source, runtime, handler, task_id = storage
    if runtime.engine is not None:

        def no_sync_worker(*args):
            pytest.fail("Async persistence must not submit a synchronous worker")

        monkeypatch.setattr(handler, "_sync_save_to_database", no_sync_worker)
        assert runtime.engine.pool.size() == 1
        assert runtime.engine.sync_engine.hide_parameters is True
    last = None
    with bind_task_lease_context(TaskLease(task_id, "runner", "run", "attempt")):
        for step in range(12):
            last = checkpoint(task_id, step)
            await handler._save_to_database(last)
    with Session(source) as db:
        rows = (
            db.query(TraceEvent)
            .filter_by(task_id=task_id)
            .order_by(TraceEvent.timestamp)
            .all()
        )
        assert len(rows) == 8
        assert db.query(TraceMessageBlob).filter_by(task_id=task_id).count() == 1
        task = db.get(Task, task_id)
        assert task.last_checkpoint_event_id == last.id
        assert task.last_checkpoint_trace_event_id == rows[-1].id
        decoded = decode_trace_event_data(
            db,
            task_id=task_id,
            data=rows[-1].data,
            strict=True,
            verify_blob_hashes=True,
        )
        assert decoded["snapshot"] == last.data["snapshot"]


async def test_stale_lease_rolls_back_staged_checkpoint_and_blobs(storage):
    source, runtime, handler, task_id = storage
    with bind_task_lease_context(TaskLease(task_id, "runner", "run", "superseded")):
        with pytest.raises(RuntimeError, match="lease changed"):
            await handler._save_to_database(checkpoint(task_id))
    with Session(source) as db:
        assert db.query(TraceEvent).filter_by(task_id=task_id).count() == 0
        assert db.query(TraceMessageBlob).filter_by(task_id=task_id).count() == 0
        assert db.get(Task, task_id).last_checkpoint_event_id is None


async def test_commit_failure_rolls_back_and_subsequent_write_works(storage):
    source, runtime, handler, task_id = storage

    def fail_commit(db):
        raise RuntimeError("injected commit failure")

    event.listen(Session, "before_commit", fail_commit)
    try:
        with bind_task_lease_context(TaskLease(task_id, "runner", "run", "attempt")):
            with pytest.raises(RuntimeError, match="injected commit failure"):
                await handler._save_to_database(checkpoint(task_id))
    finally:
        event.remove(Session, "before_commit", fail_commit)
    with Session(source) as db:
        assert db.query(TraceEvent).count() == 0
        assert db.query(TraceMessageBlob).count() == 0
        assert db.get(Task, task_id).last_checkpoint_event_id is None
    with bind_task_lease_context(TaskLease(task_id, "runner", "run", "attempt")):
        await handler._save_to_database(checkpoint(task_id, 1))


async def test_trace_non_checkpoint_uses_same_backend(storage):
    source, runtime, handler, task_id = storage
    trace = CoreTraceEvent(
        TASK_START_GENERAL,
        task_id=str(task_id),
        data={"message": "started"},
        require_persisted=True,
    )
    with bind_task_lease_context(TaskLease(task_id, "runner", "run", "attempt")):
        await handler._save_to_database(trace)
    with Session(source) as db:
        assert (
            db.query(TraceEvent).filter_by(event_id=trace.id).one().data["message"]
            == "started"
        )


async def test_cancel_during_database_wait_drains_transaction_before_next_write(
    storage, monkeypatch
):
    source, runtime, handler, task_id = storage
    blocker = source.connect()
    lock_key = uuid.uuid4().int % (2**62)
    blocker.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_key})
    entered = threading.Event()
    original = handler._save_trace_event
    saved = []

    def blocked(db, trace):
        entered.set()
        db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_key})
        original(db, trace)
        saved.append(trace.id)

    monkeypatch.setattr(handler, "_save_trace_event", blocked)
    events = [checkpoint(task_id, step) for step in range(2)]
    callers = []
    try:
        with bind_task_lease_context(TaskLease(task_id, "runner", "run", "attempt")):
            callers = [
                asyncio.create_task(handler._save_to_database(trace))
                for trace in events
            ]

        async def wait_entered():
            while not entered.is_set():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait_entered(), 3)
        callers[0].cancel()
        await asyncio.sleep(0.01)
        callers[0].cancel()
        await asyncio.sleep(0.01)
        assert not callers[0].done()
        assert not saved
    finally:
        blocker.rollback()
        blocker.close()
        outcomes = await asyncio.wait_for(
            asyncio.gather(*callers, return_exceptions=True), 5
        )
    assert isinstance(outcomes[0], asyncio.CancelledError)
    assert outcomes[1] is None
    assert saved == [trace.id for trace in events]
    if runtime.engine is not None:
        assert runtime.engine.pool.checkedout() == 0
    with Session(source) as db:
        assert db.get(Task, task_id).last_checkpoint_event_id == events[-1].id
