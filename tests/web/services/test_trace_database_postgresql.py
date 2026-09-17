"""Identical durable-write contracts for sync and native async PostgreSQL."""

import asyncio
import copy
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
from xagent.web.models.task import (
    Task,
    TaskStatus,
    TraceCheckpointBlob,
    TraceEvent,
    TraceMessageBlob,
)
from xagent.web.models.user import User
from xagent.web.services import trace_event_staging, trace_handlers
from xagent.web.services import trace_message_storage as codec
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


async def test_large_checkpoint_preparation_offloads_payload_work(storage):
    source, runtime, handler, task_id = storage
    if runtime.engine is None:
        return  # This contract targets the default async implementation.
    # Initialize the database path before checking the large checkpoint.
    with bind_task_lease_context(TaskLease(task_id, "runner", "run", "attempt")):
        await handler._save_to_database(checkpoint(task_id))
    event = checkpoint(task_id, 1)
    event.data["snapshot"]["context"]["messages"] = [
        {"role": "user", "content": "large payload " * 160, "index": index}
        for index in range(3000)
    ]
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    prepare = handler._prepare_async_trace_transaction
    prepared = asyncio.Event()
    release = threading.Event()
    encoded_messages = []
    bound_messages = []
    canonical = codec.canonical_json_bytes
    materialize = codec._materialize_blob_data

    def checked_canonical(value):
        assert threading.get_ident() != loop_thread
        if isinstance(value, dict) and "index" in value:
            encoded_messages.append(value["index"])
        return canonical(value)

    def checked_materialize(value):
        result = materialize(value)
        # The async transaction must bind the worker's encoded string rather
        # than decode/copy/re-encode the large message bodies on the loop.
        assert isinstance(value, codec.CanonicalJSON)
        assert isinstance(result, codec.PreparedJSON)
        assert result.encoded is value.encoded
        bound_messages.append(result)
        return result

    def checked_prepare(event):
        assert threading.get_ident() != loop_thread
        transaction = prepare(event)
        loop.call_soon_threadsafe(prepared.set)
        # A deadlock watchdog, not a machine-speed/runner-scheduling SLA.
        assert release.wait(30), "event loop did not release preparation worker"
        return transaction

    # Wall-clock heartbeat gaps include OS descheduling under parallel CI.
    # Check actual payload work and a loop/worker rendezvous instead.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(handler, "_prepare_async_trace_transaction", checked_prepare)
        patch.setattr(codec, "canonical_json_bytes", checked_canonical)
        patch.setattr(codec, "_materialize_blob_data", checked_materialize)
        with bind_task_lease_context(TaskLease(task_id, "runner", "run", "attempt")):
            write = asyncio.create_task(handler._save_to_database(event))
        try:
            await asyncio.wait_for(prepared.wait(), 30)
            assert not write.done()
        finally:
            release.set()
            await write
    assert encoded_messages == list(range(3000))
    assert len(bound_messages) == 3000
    with Session(source) as db:
        task = db.get(Task, task_id)
        row = db.get(TraceEvent, task.last_checkpoint_trace_event_id)
        restored = decode_trace_event_data(
            db, task_id=task_id, data=row.data, strict=True, verify_blob_hashes=True
        )
        assert restored["snapshot"] == event.data["snapshot"]


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

    def blocked(db, trace, **kwargs):
        entered.set()
        db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_key})
        original(db, trace, **kwargs)
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


@pytest.mark.parametrize("storage", [True], indirect=True, ids=["async"])
@pytest.mark.parametrize("use_v2", [False, True], ids=["v1", "v2"])
async def test_async_candidates_materialize_only_after_metadata_filtering(
    storage, monkeypatch, use_v2
):
    source, runtime, handler, task_id = storage
    monkeypatch.setenv("XAGENT_CHECKPOINT_ENCODING_V2", str(use_v2).lower())
    first = checkpoint(task_id)
    first.data["snapshot"]["context"]["metadata"] = {"value": "元数据" * 1000}
    first.data["snapshot"]["context"]["system_prompt"] = "系统提示" * 1000
    first.data["snapshot"]["pattern_state"]["tool_ledger"] = {
        "old": {"result": "old result" * 1000}
    }
    second = checkpoint(task_id, 1)
    second.data["snapshot"] = copy.deepcopy(first.data["snapshot"])
    second.data["snapshot"]["context"]["messages"].append(
        {"role": "assistant", "content": "新消息" * 1000}
    )
    second.data["snapshot"]["pattern_state"]["tool_ledger"]["new"] = {
        "result": "new result" * 1000
    }
    candidates = []
    prepare = codec.prepare_checkpoint_data

    def checked_prepare(*args):
        prepared = prepare(*args)
        values = list(prepared.messages.values()) + list(prepared.blobs.values())
        # Preparation owns canonical strings, with no decoded object trees or
        # eagerly prepared binds, for both reused and missing candidates.
        assert all(isinstance(value.data, codec.CanonicalJSON) for value in values)
        candidates.append({value.data.encoded for value in values})
        return prepared

    materialized = []
    materialize = codec._materialize_blob_data

    def checked_materialize(data):
        assert isinstance(data, codec.CanonicalJSON)
        materialized.append(data.encoded)
        return materialize(data)

    monkeypatch.setattr(trace_event_staging, "prepare_checkpoint_data", checked_prepare)
    monkeypatch.setattr(codec, "_materialize_blob_data", checked_materialize)
    with bind_task_lease_context(TaskLease(task_id, "runner", "run", "attempt")):
        await handler._save_to_database(first)
        assert set(materialized) == candidates[0]
        materialized.clear()
        await handler._save_to_database(second)
        assert candidates[0] & candidates[1]
        misses = candidates[1] - candidates[0]
        assert misses
        assert set(materialized) == misses
        assert len(materialized) == len(misses)
    with Session(source) as db:
        task = db.get(Task, task_id)
        row = db.get(TraceEvent, task.last_checkpoint_trace_event_id)
        restored = decode_trace_event_data(
            db, task_id=task_id, data=row.data, strict=True, verify_blob_hashes=True
        )
        assert restored["snapshot"] == second.data["snapshot"]


@pytest.mark.parametrize("storage", [True], indirect=True, ids=["async"])
@pytest.mark.parametrize("kind", ["message", "checkpoint"])
async def test_async_existing_size_mismatch_precedes_materialization(
    storage, monkeypatch, kind
):
    source, runtime, handler, task_id = storage
    trace = checkpoint(task_id)
    trace.data["snapshot"]["context"]["metadata"] = {"value": "metadata"}
    with bind_task_lease_context(TaskLease(task_id, "runner", "run", "attempt")):
        await handler._save_to_database(trace)
    with Session(source) as db:
        if kind == "message":
            db.query(TraceMessageBlob).filter_by(task_id=task_id).update(
                {TraceMessageBlob.message_bytes: 0}
            )
        else:
            db.query(TraceCheckpointBlob).filter_by(task_id=task_id).update(
                {TraceCheckpointBlob.blob_bytes: 0}
            )
        db.commit()

    def unexpected_materialization(data):
        pytest.fail("Existing blobs must be size-checked before materialization")

    monkeypatch.setattr(codec, "_materialize_blob_data", unexpected_materialization)
    trace.id = "size-mismatch-checkpoint"
    with bind_task_lease_context(TaskLease(task_id, "runner", "run", "attempt")):
        with pytest.raises(ValueError, match="hash collision"):
            await handler._save_to_database(trace)
    with Session(source) as db:
        assert db.query(TraceEvent).filter_by(task_id=task_id).count() == 1
