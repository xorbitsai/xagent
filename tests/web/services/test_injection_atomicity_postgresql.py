"""Atomic user-message injection against a real PostgreSQL checkpoint store.

The unit tests in ``tests/core/agent/test_atomic_injection.py`` fake the
tracer. These drive the production stack (``Tracer`` ->
``DatabaseTraceHandler`` -> PostgreSQL) so the read-back that classifies an
uncertain write sees what the database actually committed.
"""

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.core.agent.checkpoint import TraceCheckpointStore
from xagent.core.agent.context import ContextManager
from xagent.core.agent.runner import (
    AgentRunner,
    InjectionDisposition,
    UserMessageInjectionOutcome,
    UserMessageInjectionRejectedError,
    classify_injection,
    track_user_message_injection,
)
from xagent.core.agent.runtime import ExecutionInterrupted, PatternRuntime
from xagent.core.agent.trace import Tracer
from xagent.web.models import database
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services import task_events, trace_handlers
from xagent.web.services.task_lease_service import TaskLease, bind_task_lease_context
from xagent.web.services.task_orchestrator import pause_unknown_task_lease
from xagent.web.services.trace_database import TraceDatabaseRuntime

pytestmark = [pytest.mark.postgresql, pytest.mark.asyncio]

EXECUTION_ID = "atomic-pg"


@pytest_asyncio.fixture(params=[False, True], ids=["sync", "async"])
async def stack(request, monkeypatch):
    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    if request.param:
        pytest.importorskip("psycopg")
    schema = "injection_atomicity_" + uuid.uuid4().hex
    admin = create_engine(url)
    with admin.begin() as db:
        db.execute(text(f'CREATE SCHEMA "{schema}"'))
    source = create_engine(
        make_url(url).update_query_dict({"options": f"-csearch_path={schema}"})
    )
    runtime = None
    manager = ContextManager()
    try:
        Base.metadata.create_all(source)
        with Session(source) as db:
            user = User(username="injection-test", password_hash="unused")
            db.add(user)
            db.flush()
            task = Task(
                user_id=user.id,
                title="Injection test",
                description="Injection test",
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

        def sessions():
            with Session(source, autoflush=False) as db:
                yield db

        monkeypatch.setattr(trace_handlers, "get_db", sessions)
        tracer = Tracer()
        tracer.add_handler(trace_handlers.DatabaseTraceHandler(task_id))
        store = TraceCheckpointStore(tracer, require_persisted=True)
        context = manager.create_context(EXECUTION_ID)
        context.add_user_message("original")
        runner = AgentRunner(
            SimpleNamespace(llm=None), tracer=store, context_manager=manager
        )
        with bind_task_lease_context(TaskLease(task_id, "runner", "run", "attempt")):
            yield SimpleNamespace(
                source=source,
                store=store,
                runner=runner,
                context=context,
                task_id=task_id,
            )
    finally:
        if runtime is not None:
            await runtime.close()
        source.dispose()
        with admin.begin() as db:
            db.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def checkpoint_count(stack) -> int:
    with Session(stack.source) as db:
        return db.query(TraceEvent).filter_by(task_id=stack.task_id).count()


async def persisted_turns(stack) -> list[str]:
    payload = await stack.store.load_latest_checkpoint(EXECUTION_ID)
    if payload is None:
        return []
    return [
        (message.get("metadata") or {}).get("turn_id")
        for message in payload["context"]["messages"]
        if message.get("role") == "user"
    ]


async def seed(stack) -> None:
    result = await stack.runner.inject_user_message(
        EXECUTION_ID, "seed", turn_id="seed"
    )
    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH


async def test_committed_injection_is_visible_and_replays(stack):
    result = await stack.runner.inject_user_message(EXECUTION_ID, "new", turn_id="t1")

    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert [m.content for m in stack.context.messages] == ["original", "new"]
    assert "t1" in await persisted_turns(stack)
    replay = await stack.runner.inject_user_message(EXECUTION_ID, "new", turn_id="t1")
    assert replay.outcome is UserMessageInjectionOutcome.POSTED_REPLAY
    assert checkpoint_count(stack) == 1


async def test_lost_commit_acknowledgement_is_confirmed_by_readback(stack):
    await seed(stack)

    def lose_ack(db):
        raise RuntimeError("injected lost acknowledgement")

    # after_commit runs once the transaction is durable: the caller sees an
    # error for a write PostgreSQL did commit.
    event.listen(Session, "after_commit", lose_ack)
    try:
        with track_user_message_injection() as attempt:
            result = await stack.runner.inject_user_message(
                EXECUTION_ID, "new", turn_id="t1"
            )
    finally:
        event.remove(Session, "after_commit", lose_ack)

    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert classify_injection(attempt.outcome, posted=result.outcome) is (
        InjectionDisposition.ACCEPTED
    )
    assert [m.content for m in stack.context.messages][-1] == "new"
    assert "t1" in await persisted_turns(stack)


async def test_rolled_back_commit_is_confirmed_absent_and_retryable(stack):
    await seed(stack)
    before = checkpoint_count(stack)

    def fail_commit(db):
        raise RuntimeError("injected commit failure")

    event.listen(Session, "before_commit", fail_commit)
    try:
        with track_user_message_injection() as attempt:
            with pytest.raises(UserMessageInjectionRejectedError) as rejected:
                await stack.runner.inject_user_message(
                    EXECUTION_ID, "new", turn_id="t1"
                )
    finally:
        event.remove(Session, "before_commit", fail_commit)

    assert classify_injection(attempt.outcome, error=rejected.value) is (
        InjectionDisposition.NOT_ACCEPTED_RETRYABLE
    )
    assert checkpoint_count(stack) == before
    assert "t1" not in await persisted_turns(stack)
    assert [m.content for m in stack.context.messages][-1] == "seed"

    retry = await stack.runner.inject_user_message(EXECUTION_ID, "new", turn_id="t1")
    assert retry.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert "t1" in await persisted_turns(stack)


async def test_unreadable_store_after_failed_commit_fences_the_context(
    stack, monkeypatch
):
    await seed(stack)
    before = checkpoint_count(stack)
    healthy_sessions = trace_handlers.get_db

    def fail_commit(db):
        # The read-back that follows cannot open a session either.
        def unavailable():
            raise RuntimeError("injected session checkout failure")
            yield  # pragma: no cover

        monkeypatch.setattr(trace_handlers, "get_db", unavailable)
        raise RuntimeError("injected commit failure")

    event.listen(Session, "before_commit", fail_commit)
    try:
        with track_user_message_injection() as attempt:
            result = await stack.runner.inject_user_message(
                EXECUTION_ID, "new", turn_id="t1"
            )
    finally:
        event.remove(Session, "before_commit", fail_commit)
        monkeypatch.setattr(trace_handlers, "get_db", healthy_sessions)

    assert result.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    assert classify_injection(attempt.outcome, posted=result.outcome) is (
        InjectionDisposition.UNKNOWN
    )
    assert [m.content for m in stack.context.messages][-1] == "seed"

    # The fence refuses later checkpoints even though storage is healthy again.
    runtime = PatternRuntime(execution_id=EXECUTION_ID, tracer=stack.store)
    with pytest.raises(ExecutionInterrupted):
        await runtime.checkpoint(
            "late", context=stack.context, pattern=SimpleNamespace()
        )
    assert checkpoint_count(stack) == before
    assert "t1" not in await persisted_turns(stack)


@pytest.fixture
def leased_task(monkeypatch):
    """A RUNNING task under one exact lease, on a disposable database."""
    with disposable_database_factory("injection_pause") as make_database:
        engine = make_database("pause")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        monkeypatch.setattr(database, "get_session_local", lambda: sessions)
        publish = AsyncMock()
        monkeypatch.setattr(task_events, "publish_task_event", publish)
        with sessions() as db:
            user = User(username="pause-test", password_hash="unused")
            db.add(user)
            db.flush()
            task = Task(
                user_id=user.id,
                title="Pause test",
                description="Pause test",
                status=TaskStatus.RUNNING,
                runner_id="runner",
                run_id="run",
                lease_attempt_id="attempt",
                lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            )
            db.add(task)
            db.commit()
            task_id = int(task.id)
        yield SimpleNamespace(
            engine=engine,
            sessions=sessions,
            publish=publish,
            lease=TaskLease(task_id, "runner", "run", "attempt"),
        )


def load_task(leased_task) -> Task:
    with leased_task.sessions() as db:
        task = db.get(Task, leased_task.lease.task_id)
        db.expunge(task)
        return task


async def test_unknown_input_pause_commits_under_the_exact_lease(leased_task):
    before = load_task(leased_task).state_version

    assert await pause_unknown_task_lease(leased_task.lease) is True

    task = load_task(leased_task)
    assert task.status == TaskStatus.PAUSED
    assert task.state_version == before + 1
    assert task.lease_attempt_id is None
    leased_task.publish.assert_awaited_once()


async def test_unknown_input_pause_ignores_a_superseded_lease(leased_task):
    stale = TaskLease(leased_task.lease.task_id, "runner", "run", "superseded")

    assert await pause_unknown_task_lease(stale) is False

    task = load_task(leased_task)
    assert task.status == TaskStatus.RUNNING
    assert task.lease_attempt_id == "attempt"
    leased_task.publish.assert_not_awaited()


async def test_unknown_input_pause_never_revives_an_explicitly_failed_task(
    leased_task,
):
    with leased_task.sessions() as db:
        db.get(Task, leased_task.lease.task_id).status = TaskStatus.FAILED
        db.commit()

    # The exact lease is still released, but nothing was paused or announced.
    assert await pause_unknown_task_lease(leased_task.lease) is True

    task = load_task(leased_task)
    assert task.status == TaskStatus.FAILED
    assert task.lease_attempt_id is None
    leased_task.publish.assert_not_awaited()


async def _lock_waiter_appears(engine) -> None:
    query = text(
        "SELECT count(*) FROM pg_stat_activity "
        "WHERE datname = current_database() AND wait_event_type = 'Lock'"
    )
    while True:
        with engine.connect() as probe:
            if probe.execute(query).scalar_one():
                return
        await asyncio.sleep(0.02)


async def test_unknown_input_pause_waits_for_a_concurrent_row_lock(leased_task):
    holder = leased_task.engine.connect()
    holder.begin()
    holder.execute(
        text("SELECT id FROM tasks WHERE id = :id FOR UPDATE"),
        {"id": leased_task.lease.task_id},
    )
    try:
        pause = asyncio.create_task(pause_unknown_task_lease(leased_task.lease))
        # Observe the settlement actually waiting on the row lock rather than
        # inferring it from elapsed time.
        await asyncio.wait_for(_lock_waiter_appears(leased_task.engine), 10)
        assert not pause.done()
        assert load_task(leased_task).status == TaskStatus.RUNNING
    finally:
        holder.rollback()
        holder.close()
    assert await asyncio.wait_for(pause, 10) is True
    assert load_task(leased_task).status == TaskStatus.PAUSED
