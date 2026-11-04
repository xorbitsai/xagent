"""Same-identity retries of V2 trace-fact commits.

Faults are raised at ``Session.commit`` so a "lost acknowledgement" really
commits first: the retry must then replay the committed fact, never append a
second one or run a tool twice.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from collections.abc import Callable
from copy import deepcopy

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

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
from xagent.core.agent.checkpoint import (
    ExecutionEventPersistenceError,
    TraceCheckpointStore,
)
from xagent.core.agent.runtime import PatternRuntime
from xagent.core.agent.trace import (
    TraceAction,
    TraceCategory,
    TraceEvent,
    TraceEventType,
    TraceScope,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task
from xagent.web.models.task import TraceEvent as StoredTraceEvent
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.services.chat_history_service import persist_user_message_no_commit
from xagent.web.services.db_write_retry import DbWriteRetryPolicy
from xagent.web.services.task_execution_event_store import ExecutionEventConflict
from xagent.web.services.task_lease_service import (
    acquire_task_lease,
    bind_task_lease_context,
)
from xagent.web.services.trace_handlers import CommitState, DatabaseTraceHandler
from xagent.web.tracing import ExecutionEventTraceAdapter, create_task_tracer

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


@pytest.fixture(autouse=True)
def immediate_retries(monkeypatch):
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_BASE_DELAY_MS", "0")
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_MAX_DELAY_MS", "0")


@pytest.fixture
def attempts(monkeypatch):
    """Count transactions the trace writer runs, across all handlers."""
    calls: list[str] = []
    original = DatabaseTraceHandler._save_trace_event

    def save(self, db, event, **kwargs):
        calls.append(str(event.id))
        return original(self, db, event, **kwargs)

    monkeypatch.setattr(DatabaseTraceHandler, "_save_trace_event", save)
    return calls


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def trace_runtime(request, canonical, engine, monkeypatch):
    """Run writes through the bounded sync worker or the async prepared path."""
    from xagent.web.services import trace_handlers
    from xagent.web.services.trace_database import TraceDatabaseRuntime

    native_async = request.param
    source = engine
    if native_async and engine.dialect.name == "postgresql":
        # This fixture uses creator= for disposable databases; recover its URL
        # so the async driver connects to the same isolated database.
        connection = engine.raw_connection()
        try:
            parameters = connection.driver_connection.info.dsn_parameters
            source = sa.create_engine(
                sa.URL.create(
                    "postgresql",
                    username=parameters["user"],
                    host=parameters["host"],
                    port=int(parameters["port"]),
                    database=parameters["dbname"],
                )
            )
        finally:
            connection.close()
    database_runtime = TraceDatabaseRuntime(source, use_async=native_async, limit=1)
    monkeypatch.setattr(
        trace_handlers, "get_trace_database_runtime", lambda: database_runtime
    )
    yield database_runtime
    if source is not engine:
        source.dispose()


def invalidated() -> OperationalError:
    # psycopg2 reports a lost COMMIT acknowledgement exactly like this.
    return OperationalError(
        "COMMIT",
        {},
        Exception("server closed the connection unexpectedly"),
        connection_invalidated=True,
    )


def holds_fact(kind: str | None = None) -> Callable[[Session], bool]:
    """Match a commit whose transaction staged or read a fact of ``kind``."""

    def matches(db: Session) -> bool:
        return any(
            isinstance(row, TaskExecutionEvent) and kind in (None, row.kind)
            for row in db.identity_map.values()
        )

    return matches


def fail_commits(
    monkeypatch,
    target: Callable[[Session], bool],
    *,
    committed: bool | list[bool],
    times: int = 1,
    error: Callable[[], Exception] = invalidated,
    between: Callable[[], None] | None = None,
) -> list[Exception]:
    """Fail the first ``times`` matching commits, after or before COMMIT.

    ``committed`` may give each failure its own outcome, in order.
    ``between`` runs after each failed transaction has committed or rolled
    back, i.e. between two attempts, outside the writer's locks.
    """
    raised: list[Exception] = []
    original = Session.commit
    outcomes = committed if isinstance(committed, list) else [committed] * times

    def commit(self):
        if len(raised) >= len(outcomes) or not target(self):
            return original(self)
        if outcomes[len(raised)]:
            original(self)
        else:
            self.rollback()
        if between is not None:
            between()
        raised.append(error())
        raise raised[-1]

    monkeypatch.setattr(Session, "commit", commit)
    return raised


def task_start(task_id: int, **data) -> TraceEvent:
    return TraceEvent(
        TraceEventType(TraceScope.TASK, TraceAction.START, TraceCategory.GENERAL),
        task_id=str(task_id),
        data=data,
    )


def projections(db, event_id: str) -> int:
    return db.query(StoredTraceEvent).filter_by(event_id=event_id).count()


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True], ids=["rolled-back", "lost-ack"])
async def test_retryable_commit_failure_reruns_into_one_fact(
    canonical, trace_runtime, attempts, monkeypatch, committed
):
    factory, task_id = canonical
    raised = fail_commits(monkeypatch, holds_fact(), committed=committed)
    event = task_start(task_id, input="hello")
    original_data = deepcopy(event.data)
    try:
        await ExecutionEventTraceAdapter(task_id).commit_event(event)
    finally:
        await trace_runtime.close()
    assert len(raised) == 1
    assert len(attempts) == 2
    assert event.data == original_data
    with factory() as db:
        rows = facts(db, task_id)
        assert [row.idempotency_key for row in rows] == [f"runtime:{event.id}"]
        assert projections(db, str(event.id)) == 1
        assert db.get(Task, task_id).conversation_event_sequence == 1


def react_call(task_id: int, tool):
    """Run one real ReAct tool call through the V2 fact writer."""
    from tests.core.agent.test_react import FakeLLM
    from xagent.core.agent import ExecutionContext, ReActPattern

    tracer = create_task_tracer(task_id)
    tracer.handlers = [
        h for h in tracer.handlers if isinstance(h, ExecutionEventTraceAdapter)
    ]
    context = ExecutionContext(system_prompt="Calculate")
    context.add_user_message("2+2", metadata={"turn_id": "retry-turn"})
    return ReActPattern(max_iterations=3).run(
        context=context,
        tools=[tool],
        runtime=PatternRuntime(tracer=TraceCheckpointStore(tracer)),
        llm=FakeLLM(
            responses=[
                {
                    "content": "calculate",
                    "tool_calls": [
                        {
                            "id": "call1",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression":"2+2"}',
                            },
                        }
                    ],
                },
                {"content": "4", "done": True},
            ]
        ),
    )


def tool_facts(db, task_id):
    return [row for row in facts(db, task_id) if row.kind.startswith("tool_")]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tool_execution_start", "tool_execution_end"])
async def test_lost_tool_ack_runs_the_tool_once(
    canonical, trace_runtime, monkeypatch, kind
):
    from tests.core.agent.test_react import FakeTool

    factory, task_id = canonical
    raised = fail_commits(monkeypatch, holds_fact(kind), committed=True)
    tool = FakeTool()
    try:
        result = await react_call(task_id, tool)
    finally:
        await trace_runtime.close()
    assert result["success"]
    assert len(raised) == 1
    assert len(tool.calls) == 1
    with factory() as db:
        rows = tool_facts(db, task_id)
        assert [row.kind for row in rows] == [
            "tool_execution_start",
            "tool_execution_end",
        ]
        assert len({row.tool_attempt_id for row in rows}) == 1
        for row in rows:
            assert projections(db, row.payload["protocol_event_id"]) == 1


@pytest.mark.asyncio
async def test_start_replaced_between_attempts_is_still_refused(
    canonical, monkeypatch, attempts
):
    from tests.core.agent.test_react import FakeTool

    factory, task_id = canonical

    def replace_start():
        # Another writer's start for the same attempt, not this call's own.
        with factory() as db:
            db.execute(
                sa.update(TaskExecutionEvent)
                .where(TaskExecutionEvent.kind == "tool_execution_start")
                .values(event_id=str(uuid.uuid4()))
            )
            db.commit()

    fail_commits(
        monkeypatch,
        holds_fact("tool_execution_start"),
        committed=True,
        between=replace_start,
    )
    tool = FakeTool()
    with pytest.raises(ExecutionEventPersistenceError) as caught:
        await react_call(task_id, tool)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "already started" in str(caught.value.__cause__)
    assert tool.calls == []
    with factory() as db:
        assert [row.kind for row in tool_facts(db, task_id)] == ["tool_execution_start"]


def tool_start(task_id: int, attempt: str) -> TraceEvent:
    return TraceEvent(
        TraceEventType(TraceScope.ACTION, TraceAction.START, TraceCategory.TOOL),
        task_id=str(task_id),
        step_id="tool-step",
        data={
            "tool_name": "write",
            "tool_call_id": "call1",
            "tool_attempt_id": attempt,
            "assistant_message_id": "batch1",
        },
    )


def tool_end(task_id: int, attempt: str) -> TraceEvent:
    return TraceEvent(
        TraceEventType(TraceScope.ACTION, TraceAction.END, TraceCategory.TOOL),
        task_id=str(task_id),
        step_id="tool-step",
        data={
            "tool_name": "write",
            "tool_call_id": "call1",
            "tool_attempt_id": attempt,
            "assistant_message_id": "batch1",
            "result": {"success": True},
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("claim", ["none", "other", "own-with-end"])
async def test_start_guard_accepts_only_this_calls_own_start(
    canonical, trace_runtime, claim
):
    try:
        await _check_start_guard(canonical, claim)
    finally:
        await trace_runtime.close()


async def _check_start_guard(canonical, claim):
    factory, task_id = canonical
    adapter = ExecutionEventTraceAdapter(task_id)
    first = tool_start(task_id, "attempt1")
    first.require_persisted = True
    state = CommitState()
    await adapter._save_to_database(first, commit_state=state)
    assert state.start_event_id is not None
    if claim == "none":
        retry = CommitState()
    elif claim == "other":
        retry = CommitState(start_event_id=str(uuid.uuid4()))
    else:
        retry = CommitState(start_event_id=state.start_event_id)
        await adapter.commit_event(tool_end(task_id, "attempt1"))
    second = tool_start(task_id, "attempt1")
    second.require_persisted = True
    with pytest.raises(RuntimeError, match="already started"):
        await adapter._save_to_database(second, commit_state=retry)
    # The own start alone is the idempotent rerun this guard admits.
    if claim != "own-with-end":
        rerun = tool_start(task_id, "attempt1")
        rerun.require_persisted = True
        await adapter._save_to_database(
            rerun, commit_state=CommitState(start_event_id=state.start_event_id)
        )
        assert rerun.id == first.id
    with factory() as db:
        starts = [r for r in facts(db, task_id) if r.kind == "tool_execution_start"]
        assert [r.event_id for r in starts] == [state.start_event_id]


@pytest.mark.asyncio
async def test_lease_replaced_between_attempts_is_not_retried(
    canonical, monkeypatch, attempts
):
    factory, task_id = canonical
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)

    def replace_lease():
        with factory() as db:
            db.get(Task, task_id).run_id = "replacement-run"
            db.commit()

    raised = fail_commits(
        monkeypatch,
        holds_fact(),
        committed=False,
        between=replace_lease,
    )
    with bind_task_lease_context(lease):
        with pytest.raises(ExecutionEventPersistenceError) as caught:
            await ExecutionEventTraceAdapter(task_id).commit_event(task_start(task_id))
    assert len(raised) == 1
    assert len(attempts) == 2
    assert "lost its task lease" in str(caught.value.__cause__)
    with factory() as db:
        assert facts(db, task_id) == []
        assert db.query(StoredTraceEvent).count() == 0


def integrity_error() -> IntegrityError:
    return IntegrityError("INSERT", {}, Exception("duplicate key value"))


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [integrity_error, lambda: ValueError("bad")])
async def test_non_retryable_error_is_attempted_once(
    canonical, monkeypatch, attempts, error
):
    factory, task_id = canonical
    raised = fail_commits(monkeypatch, holds_fact(), committed=False, error=error)
    with pytest.raises(ExecutionEventPersistenceError) as caught:
        await ExecutionEventTraceAdapter(task_id).commit_event(task_start(task_id))
    assert caught.value.__cause__ is raised[0]
    assert len(attempts) == 1
    with factory() as db:
        assert facts(db, task_id) == []


@pytest.mark.asyncio
async def test_payload_drift_under_same_identity_conflicts_without_retry(
    canonical, attempts
):
    factory, task_id = canonical
    adapter = ExecutionEventTraceAdapter(task_id)
    event = task_start(task_id, input="hello")
    await adapter.commit_event(event)
    event.data = {"input": "changed"}
    with pytest.raises(ExecutionEventPersistenceError) as caught:
        await adapter.commit_event(event)
    assert isinstance(caught.value.__cause__, ExecutionEventConflict)
    assert len(attempts) == 2
    with factory() as db:
        assert [row.payload["data"] for row in facts(db, task_id)] == [
            {"input": "hello"}
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("orig", "expected"),
    [
        (Exception("server closed the connection unexpectedly"), 4),
        # SQLite reports BUSY only after its busy timeout: one retry at most.
        (sqlite3.OperationalError("database is locked"), 2),
    ],
    ids=["connection", "sqlite-busy"],
)
async def test_exhausted_retries_keep_the_database_error_as_cause(
    canonical, monkeypatch, attempts, orig, expected
):
    factory, task_id = canonical
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_ATTEMPTS", "4")
    raised = fail_commits(
        monkeypatch,
        holds_fact(),
        committed=False,
        times=10,
        error=lambda: OperationalError("COMMIT", {}, orig),
    )
    with pytest.raises(ExecutionEventPersistenceError) as caught:
        await ExecutionEventTraceAdapter(task_id).commit_event(task_start(task_id))
    assert len(attempts) == expected
    assert len(raised) == expected
    assert caught.value.__cause__ is raised[-1]
    with factory() as db:
        assert facts(db, task_id) == []


@pytest.mark.asyncio
async def test_cancellation_during_backoff_stops_retrying(
    canonical, monkeypatch, attempts
):
    factory, task_id = canonical
    backing_off = asyncio.Event()

    def delay(self, retry, *, rng=None):
        backing_off.set()
        return 60.0

    monkeypatch.setattr(DbWriteRetryPolicy, "delay", delay)
    fail_commits(monkeypatch, holds_fact(), committed=False)
    write = asyncio.create_task(
        ExecutionEventTraceAdapter(task_id).commit_event(task_start(task_id))
    )
    await asyncio.wait_for(backing_off.wait(), timeout=10)
    write.cancel()
    with pytest.raises(asyncio.CancelledError):
        await write
    assert len(attempts) == 1
    with factory() as db:
        assert facts(db, task_id) == []


@pytest.mark.asyncio
async def test_v1_task_writes_are_not_retried(canonical, monkeypatch, attempts):
    factory, task_id = canonical
    with factory() as db:
        db.get(Task, task_id).conversation_storage_version = 1
        db.commit()
    tracer = create_task_tracer(task_id)
    # No fact writer: the V2 commit path, and its retries, never runs.
    assert tracer.event_writer is None
    raised = fail_commits(
        monkeypatch,
        lambda db: any(
            isinstance(row, StoredTraceEvent)
            for row in [*db.new, *db.identity_map.values()]
        ),
        committed=False,
    )
    with pytest.raises(RuntimeError):
        await tracer.trace_event(
            TraceEventType(TraceScope.TASK, TraceAction.START, TraceCategory.GENERAL),
            task_id=str(task_id),
            require_persisted=True,
        )
    assert len(raised) == 1
    assert len(attempts) == 1
    with factory() as db:
        assert db.query(StoredTraceEvent).count() == 0


@pytest.mark.asyncio
async def test_compaction_lost_ack_keeps_the_first_watermark(canonical, monkeypatch):
    factory, task_id = canonical
    with factory() as db:
        message = persist_user_message_no_commit(
            db, task_id, db.get(Task, task_id).user_id, "earlier", turn_id="turn-1"
        )
        db.commit()
        anchor = facts(db, task_id)[0]
        expected = {
            "scope_id": "root",
            "event_id": anchor.event_id,
            "sequence": anchor.sequence,
        }
        data = {"summary": "Earlier conversation", "watermark_message_id": message.id}

    def drop_transcript():
        # A recomputed watermark would now fail; the rerun must reuse the first.
        with factory() as db:
            db.execute(
                sa.delete(TaskChatMessage).where(TaskChatMessage.task_id == task_id)
            )
            db.commit()

    raised = fail_commits(
        monkeypatch,
        holds_fact("action_end_compact"),
        committed=True,
        between=drop_transcript,
    )
    event = TraceEvent(
        TraceEventType(TraceScope.ACTION, TraceAction.END, TraceCategory.COMPACT),
        task_id=str(task_id),
        step_id="compact-step",
        data=deepcopy(data),
    )
    await ExecutionEventTraceAdapter(task_id).commit_event(event)
    assert len(raised) == 1
    assert event.data == data
    with factory() as db:
        summaries = [r for r in facts(db, task_id) if r.kind == "action_end_compact"]
        assert len(summaries) == 1
        assert summaries[0].payload["transcript_watermark"] == expected
        assert summaries[0].payload["data"] == data
        assert projections(db, str(event.id)) == 1


@pytest.mark.asyncio
async def test_start_rolled_back_then_lost_ack_admits_only_the_second_insert(
    canonical, monkeypatch, attempts
):
    factory, task_id = canonical
    # Attempt 1 rolls back, attempt 2 commits but loses its ack, attempt 3
    # must recognise attempt 2's row, not attempt 1's, as this call's own.
    raised = fail_commits(
        monkeypatch,
        holds_fact("tool_execution_start"),
        committed=[False, True],
    )
    await ExecutionEventTraceAdapter(task_id).commit_event(
        tool_start(task_id, "attempt1")
    )
    assert len(raised) == 2
    assert len(attempts) == 3
    with factory() as db:
        assert [row.kind for row in tool_facts(db, task_id)] == ["tool_execution_start"]


def settlement_start(task_id: int, attempt: str) -> TraceEvent:
    event = tool_start(task_id, attempt)
    event.data = {**event.data, "settlement_delivery": True}
    return event


@pytest.mark.asyncio
async def test_lost_settlement_ack_replays_the_settlement_fact(
    canonical, monkeypatch, attempts
):
    factory, task_id = canonical
    adapter = ExecutionEventTraceAdapter(task_id)
    await adapter.commit_event(tool_start(task_id, "attempt1"))
    await adapter.commit_event(tool_end(task_id, "attempt1"))
    raised = fail_commits(monkeypatch, holds_fact(), committed=True)
    event = settlement_start(task_id, "attempt1")
    await adapter.commit_event(event)
    assert len(raised) == 1
    with factory() as db:
        rows = tool_facts(db, task_id)
        assert [row.idempotency_key for row in rows] == [
            "tool:attempt1:tool_execution_start",
            "tool:attempt1:tool_execution_end",
            "tool:attempt1:settlement:tool_execution_start",
        ]
        assert projections(db, str(event.id)) == 1


@pytest.mark.asyncio
async def test_rerun_waits_a_bounded_time_for_the_task_lock(
    canonical, engine, monkeypatch, attempts
):
    if engine.dialect.name != "postgresql":
        pytest.skip("lock_timeout is a PostgreSQL setting")
    from xagent.web.services import db_write_retry
    from xagent.web.services.task_execution_event_store import (
        lock_task_execution_events_no_commit,
    )

    factory, task_id = canonical
    monkeypatch.setattr(db_write_retry, "_RERUN_LOCK_TIMEOUT", "200ms")
    holder = factory()

    def abandon_lock():
        # A dropped connection's transaction can keep the task row locked.
        lock_task_execution_events_no_commit(holder, task_id)

    fail_commits(monkeypatch, holds_fact(), committed=False, between=abandon_lock)
    try:
        with pytest.raises(ExecutionEventPersistenceError) as caught:
            await asyncio.wait_for(
                ExecutionEventTraceAdapter(task_id).commit_event(task_start(task_id)),
                timeout=10,
            )
    finally:
        holder.rollback()
        holder.close()
    # Both reruns gave up on the lock instead of waiting for it indefinitely.
    assert len(attempts) == 3
    assert getattr(caught.value.__cause__.orig, "pgcode", None) == "55P03"


@pytest.mark.asyncio
@pytest.mark.parametrize("times", [1, 10], ids=["recovered", "exhausted"])
async def test_retried_attempts_warn_and_only_a_failed_write_errors(
    canonical, monkeypatch, caplog, times
):
    factory, task_id = canonical
    fail_commits(monkeypatch, holds_fact(), committed=False, times=times)
    with caplog.at_level(logging.WARNING):
        try:
            await ExecutionEventTraceAdapter(task_id).commit_event(task_start(task_id))
        except ExecutionEventPersistenceError:
            assert times == 10
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings
    if times == 1:
        assert errors == []
    else:
        assert len(errors) == 1
        assert errors[0].startswith("execution_event_commit_failed ")
        assert "site=trace_fact" in errors[0]
