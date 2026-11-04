"""Same-identity retries of authoritative outbound and existing-input writes.

Faults are raised at ``Session.commit`` so a "lost acknowledgement" really
commits first: the rerun must replay the committed fact, chat row and
envelope, never append a second message or publish twice.
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError, OperationalError

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
from tests.web.services.test_trace_fact_commit_retry import (
    fail_commits,
    holds_fact,
)
from tests.web.services.test_trace_fact_commit_retry import (
    immediate_retries as immediate_retries_fixture,
)
from xagent.core.agent.checkpoint import ExecutionEventPersistenceError
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task
from xagent.web.models.task import TraceEvent as StoredTraceEvent
from xagent.web.services import db_write_retry, task_execution, task_orchestrator
from xagent.web.services.task_lease_service import (
    TaskLeaseLostError,
    acquire_task_lease,
    acquire_task_lease_isolated,
    bind_task_lease_context,
)
from xagent.web.services.task_orchestrator import TaskTurnPayload

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture
immediate_retries = immediate_retries_fixture


@pytest.fixture
def outbound(canonical, monkeypatch):
    """Route outbound writes to the test database and record each attempt."""
    factory, task_id = canonical
    monkeypatch.setattr(task_execution, "get_db", lambda: iter([factory()]))
    published = AsyncMock()
    monkeypatch.setattr(task_execution, "publish_task_event", published)
    attempts: list[dict] = []
    original = task_execution._persist_agent_outbound_event

    def persist(task_id, event, **kwargs):
        attempts.append(deepcopy(event))
        return original(task_id, event, **kwargs)

    monkeypatch.setattr(task_execution, "_persist_agent_outbound_event", persist)
    return factory, task_id, published, attempts


def question() -> dict:
    return {
        "event_id": "question-event",
        "message": "Which file?",
        "expect_response": True,
    }


def tool_message() -> dict:
    return {
        "event_id": "tool-message",
        "message": "Uploading",
        "metadata": {"tool_attempt_id": "attempt-1", "assistant_message_id": "m-1"},
    }


def final_answer() -> dict:
    return {
        "type": "final_answer_end",
        "content": "hello",
        "stream_id": "stream1",
        "message_id": "final_answer_1",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [question, tool_message, final_answer])
@pytest.mark.parametrize("committed", [False, True], ids=["rolled-back", "lost-ack"])
async def test_outbound_rerun_keeps_one_message_and_publishes_once(
    outbound, monkeypatch, payload, committed
):
    factory, task_id, published, attempts = outbound
    raised = fail_commits(monkeypatch, holds_fact(), committed=committed)
    handler = task_execution.make_agent_outbound_handler(task_id, authoritative=True)
    await handler(payload())
    assert len(raised) == 1
    assert len(attempts) == 2
    # Every attempt writes the event as the handler built it.
    assert attempts[0] == attempts[1]
    with factory() as db:
        (fact,) = facts(db, task_id)
        protocol_id = fact.payload["protocol_event_id"]
        # A final answer has no legacy trace projection.
        projections = 0 if payload is final_answer else 1
        assert (
            db.query(StoredTraceEvent).filter_by(event_id=protocol_id).count()
            == projections
        )
        chat_rows = db.query(TaskChatMessage).all()
        if payload is question:
            assert [row.execution_event_id for row in chat_rows] == [fact.event_id]
        else:
            assert chat_rows == []
    published.assert_awaited_once()
    (event, published_task_id), _ = published.await_args
    assert published_task_id == task_id
    assert event["event_id"] == protocol_id


@pytest.mark.asyncio
async def test_replayed_tool_message_rerun_starts_from_the_original_event(
    outbound, monkeypatch
):
    """The replay branch rewrites the event before its COMMIT fails."""
    factory, task_id, published, attempts = outbound
    raised = fail_commits(monkeypatch, holds_fact(), committed=[True, False])
    await task_execution.make_agent_outbound_handler(task_id, authoritative=True)(
        tool_message()
    )
    assert len(raised) == 2
    assert len(attempts) == 3
    assert attempts[0] == attempts[1] == attempts[2]
    with factory() as db:
        assert len(facts(db, task_id)) == 1
    published.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [question, tool_message])
async def test_leased_outbound_rerun_after_lost_ack_relocks_its_lease(
    outbound, monkeypatch, payload
):
    factory, task_id, published, attempts = outbound
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)
    raised = fail_commits(monkeypatch, holds_fact(), committed=True)
    with bind_task_lease_context(lease):
        await task_execution.make_agent_outbound_handler(task_id, authoritative=True)(
            payload()
        )
    assert len(raised) == 1
    assert len(attempts) == 2
    with factory() as db:
        (fact,) = facts(db, task_id)
        assert fact.run_id == lease.run_id
    published.assert_awaited_once()


@pytest.mark.asyncio
async def test_outbound_lease_replaced_between_attempts_is_not_retried(
    outbound, monkeypatch
):
    factory, task_id, published, attempts = outbound
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)

    def replace_lease():
        with factory() as db:
            db.get(Task, task_id).run_id = "replacement-run"
            db.commit()

    raised = fail_commits(
        monkeypatch, holds_fact(), committed=False, between=replace_lease
    )
    with bind_task_lease_context(lease):
        with pytest.raises(ExecutionEventPersistenceError) as caught:
            await task_execution.make_agent_outbound_handler(
                task_id, authoritative=True
            )(question())
    assert len(raised) == 1
    assert len(attempts) == 2
    assert isinstance(caught.value.__cause__, TaskLeaseLostError)
    with factory() as db:
        assert facts(db, task_id) == []
        assert db.query(TaskChatMessage).count() == 0
    published.assert_not_awaited()


@pytest.mark.asyncio
async def test_outbound_non_retryable_error_is_attempted_once(outbound, monkeypatch):
    factory, task_id, published, attempts = outbound
    fail_commits(
        monkeypatch,
        holds_fact(),
        committed=False,
        error=lambda: IntegrityError("INSERT", {}, Exception("duplicate key value")),
    )
    with pytest.raises(ExecutionEventPersistenceError):
        await task_execution.make_agent_outbound_handler(task_id, authoritative=True)(
            question()
        )
    assert len(attempts) == 1
    published.assert_not_awaited()


@pytest.mark.asyncio
async def test_outbound_exhausted_retries_keep_the_database_error_as_cause(
    outbound, monkeypatch
):
    factory, task_id, published, attempts = outbound
    raised = fail_commits(monkeypatch, holds_fact(), committed=False, times=3)
    with pytest.raises(ExecutionEventPersistenceError) as caught:
        await task_execution.make_agent_outbound_handler(task_id, authoritative=True)(
            question()
        )
    assert len(attempts) == 3
    assert caught.value.__cause__ is raised[-1]
    with factory() as db:
        assert facts(db, task_id) == []
    published.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_authoritative_outbound_failure_is_logged_once_and_not_retried(
    outbound, monkeypatch, caplog
):
    factory, task_id, published, attempts = outbound
    raised = fail_commits(
        monkeypatch, lambda db: db.new or db.dirty, committed=False, times=3
    )
    with caplog.at_level(logging.ERROR, logger=task_execution.logger.name):
        await task_execution.make_agent_outbound_handler(task_id)(question())
    assert len(raised) == 1
    assert len(attempts) == 1
    assert "Failed to persist agent outbound message" in caplog.text
    published.assert_awaited_once()


@pytest.fixture
def accept_input(canonical, monkeypatch):
    """Accept a local execute input under an exact lease, recording attempts."""
    factory, task_id = canonical
    with factory() as db:
        owner = db.get(Task, task_id).user_id
    lease = acquire_task_lease_isolated(task_id)
    attempts: list[None] = []
    original = task_orchestrator._accept_existing_event_input_sync

    def accept(*args, **kwargs):
        attempts.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(task_orchestrator, "_accept_existing_event_input_sync", accept)

    async def run() -> None:
        await task_orchestrator._accept_existing_event_input(
            lease, owner, TaskTurnPayload("existing input")
        )

    return factory, task_id, run, attempts


def any_commit(db) -> bool:
    # The acceptance transaction is the only commit these tests make.
    return True


def accepted_inputs(db, task_id):
    return [fact for fact in facts(db, task_id) if fact.kind == "input_accepted"]


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True], ids=["rolled-back", "lost-ack"])
async def test_existing_input_rerun_keeps_one_accepted_fact(
    accept_input, monkeypatch, committed
):
    factory, task_id, run, attempts = accept_input
    raised = fail_commits(monkeypatch, any_commit, committed=committed)
    await run()
    assert len(raised) == 1
    assert len(attempts) == 2
    with factory() as db:
        (fact,) = accepted_inputs(db, task_id)
        assert fact.payload["content"] == "existing input"


@pytest.mark.asyncio
async def test_existing_input_lease_replaced_between_attempts_is_not_retried(
    accept_input, monkeypatch
):
    factory, task_id, run, attempts = accept_input

    def replace_lease():
        # A Core transaction, so the patched Session.commit does not fail it.
        with factory.kw["bind"].begin() as connection:
            connection.execute(
                sa.update(Task)
                .where(Task.id == task_id)
                .values(lease_attempt_id="replacement")
            )

    fail_commits(monkeypatch, any_commit, committed=False, between=replace_lease)
    with pytest.raises(TaskLeaseLostError):
        await run()
    assert len(attempts) == 2
    with factory() as db:
        assert accepted_inputs(db, task_id) == []


@pytest.mark.asyncio
async def test_existing_input_exhausted_retries_raise_the_database_error(
    accept_input, monkeypatch
):
    factory, task_id, run, attempts = accept_input
    raised = fail_commits(monkeypatch, any_commit, committed=False, times=3)
    with pytest.raises(OperationalError) as caught:
        await run()
    assert len(attempts) == 3
    assert caught.value is raised[-1]
    with factory() as db:
        assert accepted_inputs(db, task_id) == []


@pytest.fixture
def abandoned_task_lock(canonical, engine, monkeypatch):
    """Hold the task row lock from between two attempts, as a dropped
    connection's abandoned transaction can."""
    if engine.dialect.name != "postgresql":
        pytest.skip("lock_timeout is a PostgreSQL setting")
    from xagent.web.services.task_execution_event_store import (
        lock_task_execution_events_no_commit,
    )

    factory, task_id = canonical
    monkeypatch.setattr(db_write_retry, "_RERUN_LOCK_TIMEOUT", "200ms")
    holder = factory()
    yield lambda: lock_task_execution_events_no_commit(holder, task_id)
    holder.rollback()
    holder.close()


def lock_timed_out(error: BaseException) -> bool:
    return getattr(getattr(error, "orig", None), "pgcode", None) == "55P03"


@pytest.mark.asyncio
async def test_outbound_rerun_waits_a_bounded_time_for_the_task_lock(
    outbound, abandoned_task_lock, monkeypatch
):
    factory, task_id, published, attempts = outbound
    fail_commits(
        monkeypatch, holds_fact(), committed=False, between=abandoned_task_lock
    )
    with pytest.raises(ExecutionEventPersistenceError) as caught:
        await asyncio.wait_for(
            task_execution.make_agent_outbound_handler(task_id, authoritative=True)(
                question()
            ),
            timeout=10,
        )
    # Both reruns gave up on the lock instead of waiting for it indefinitely.
    assert len(attempts) == 3
    assert lock_timed_out(caught.value.__cause__)
    published.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_input_rerun_waits_a_bounded_time_for_the_task_lock(
    accept_input, abandoned_task_lock, monkeypatch
):
    factory, task_id, run, attempts = accept_input
    fail_commits(monkeypatch, any_commit, committed=False, between=abandoned_task_lock)
    with pytest.raises(OperationalError) as caught:
        await asyncio.wait_for(run(), timeout=10)
    assert len(attempts) == 3
    assert lock_timed_out(caught.value)
