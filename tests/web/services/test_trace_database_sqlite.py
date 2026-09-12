"""File SQLite async writes preserve transactions without occupying API workers."""

import asyncio
import threading

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from tests.web.services import test_trace_database_postgresql as contracts
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.services.db_runtime import drain_async_task_cancellation_safe
from xagent.web.services.trace_database import TraceDatabaseRuntime


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contract",
    [
        contracts.test_checkpoint_retention_dedup_pointer_and_required_data,
        contracts.test_stale_lease_rolls_back_staged_checkpoint_and_blobs,
        contracts.test_commit_failure_rolls_back_and_subsequent_write_works,
    ],
)
async def test_checkpoint_contracts(tmp_path, monkeypatch, contract):
    source = create_engine(f"sqlite:///{tmp_path / 'checkpoints.db'}")
    apply_sqlite_concurrency_pragmas(source)
    contracts.Base.metadata.create_all(source)
    with Session(source) as db:
        user = contracts.User(username="trace-test", password_hash="unused")
        db.add(user)
        db.flush()
        task = contracts.Task(
            user_id=user.id,
            title="Trace test",
            description="Trace test",
            status=contracts.TaskStatus.RUNNING,
            runner_id="runner",
            run_id="run",
            lease_attempt_id="attempt",
        )
        db.add(task)
        db.flush()
        task_id = task.id
        db.commit()
    runtime = TraceDatabaseRuntime(source, use_async=True, limit=4)
    monkeypatch.setattr(
        contracts.trace_handlers, "get_trace_database_runtime", lambda: runtime
    )
    monkeypatch.setenv("XAGENT_CHECKPOINT_HISTORY_LIMIT", "8")
    handler = contracts.trace_handlers.DatabaseTraceHandler(task_id)
    storage = source, runtime, handler, task_id
    try:
        if (
            contract
            is contracts.test_checkpoint_retention_dedup_pointer_and_required_data
        ):
            await contract(storage, monkeypatch)
        else:
            await contract(storage)
    finally:
        await runtime.close()
        source.dispose()


@pytest.mark.asyncio
async def test_file_database_transactions_and_pragmas(tmp_path):
    source = create_engine(f"sqlite:///{tmp_path / 'trace.db'}")
    apply_sqlite_concurrency_pragmas(source)
    with source.begin() as db:
        db.execute(text("CREATE TABLE items (id INTEGER PRIMARY KEY)"))
    runtime = TraceDatabaseRuntime(source, use_async=True, limit=8)
    assert runtime.limit == 1
    assert runtime.engine is not None
    try:
        async with runtime.engine.connect() as db:
            assert await db.scalar(text("PRAGMA foreign_keys")) == 1
            assert await db.scalar(text("PRAGMA busy_timeout")) == 5000
            assert await db.scalar(text("PRAGMA journal_mode")) == "wal"

        def transaction(db):
            db.execute(text("INSERT INTO items VALUES (1)"))
            db.commit()

        def unexpected_sync():
            pytest.fail("async path submitted synchronous worker")

        await runtime.run(unexpected_sync, transaction)

        def fail(db):
            db.execute(text("INSERT INTO items VALUES (2)"))
            raise ValueError("rollback")

        with pytest.raises(ValueError, match="rollback"):
            await runtime.run(unexpected_sync, fail)
        with source.connect() as db:
            assert db.execute(text("SELECT id FROM items")).scalars().all() == [1]
    finally:
        await runtime.close()
        source.dispose()


@pytest.mark.asyncio
async def test_lock_wait_cancellation_and_loop_responsiveness(tmp_path):
    source = create_engine(f"sqlite:///{tmp_path / 'locked.db'}")
    apply_sqlite_concurrency_pragmas(source)
    with source.begin() as db:
        db.execute(text("CREATE TABLE items (id INTEGER PRIMARY KEY)"))
    runtime = TraceDatabaseRuntime(source, use_async=True, limit=4)
    # Establish the async connection before taking the external write lock.
    async with runtime.engine.connect() as db:
        await db.execute(text("SELECT 1"))
    blocker = source.connect()
    blocker.execute(text("BEGIN IMMEDIATE"))
    started = asyncio.Event()

    def transaction(db):
        started.set()
        db.execute(text("INSERT INTO items VALUES (1)"))
        db.commit()

    async def owned_write():
        await drain_async_task_cancellation_safe(
            asyncio.create_task(runtime.run(lambda: None, transaction))
        )

    caller = asyncio.create_task(owned_write())
    try:
        await asyncio.wait_for(started.wait(), 2)
        caller.cancel()
        await asyncio.sleep(0.03)
        caller.cancel()
        # Both the loop and default executor remain available during DB lock wait.
        assert await asyncio.wait_for(asyncio.to_thread(lambda: 42), 1) == 42
        assert not caller.done()
        close = asyncio.create_task(runtime.close())
        await asyncio.sleep(0.03)
        assert not close.done()
    finally:
        blocker.rollback()
        blocker.close()
        with pytest.raises(asyncio.CancelledError):
            await caller
        await runtime.close()
        with source.connect() as db:
            assert db.scalar(text("SELECT count(*) FROM items")) == 1
        source.dispose()


@pytest.mark.asyncio
async def test_memory_fallback_preserves_existing_database():
    from sqlalchemy.pool import StaticPool

    source = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    with source.begin() as db:
        db.execute(text("CREATE TABLE items (id INTEGER PRIMARY KEY)"))
    runtime = TraceDatabaseRuntime(source, use_async=True, limit=4)
    assert runtime.engine is None
    thread_ids = []

    def write():
        thread_ids.append(threading.get_ident())
        with Session(source) as db:
            db.execute(text("INSERT INTO items VALUES (1)"))
            db.commit()

    try:
        await runtime.run(write, lambda db: pytest.fail("separate memory database"))
        assert thread_ids != [threading.get_ident()]
        with source.connect() as db:
            assert db.scalar(text("SELECT count(*) FROM items")) == 1
    finally:
        await runtime.close()
        source.dispose()
