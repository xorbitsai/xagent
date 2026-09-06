from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone
from threading import Event
from unittest.mock import Mock
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.models.user import User
from xagent.web.services.task_execution_event_store import (
    ExecutionEventConflict,
    append_task_execution_event_no_commit,
    load_task_execution_events,
)


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgresql)]
)
def engine(request, tmp_path):
    if request.param == "postgresql":
        with disposable_database_factory("execution_events") as make:
            yield make("store")
    else:
        result = sa.create_engine(f"sqlite:///{tmp_path / 'events.db'}")

        @sa.event.listens_for(result, "connect")
        def enable_foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

        try:
            yield result
        finally:
            result.dispose()


@pytest.fixture
def task_id(engine):
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = User(username="event-owner", password_hash="unused")
        db.add(user)
        db.flush()
        task = Task(user_id=user.id, title="Existing task", description="unchanged")
        db.add(task)
        db.commit()
        return task.id


def append(db, task_id, key="turn-1", **overrides):
    values = dict(
        task_id=task_id,
        scope_id="root",
        idempotency_key=key,
        kind="user_message_accepted",
        payload={"content": "hello", "attachments": []},
        occurred_at=datetime(2026, 9, 5, tzinfo=timezone.utc),
        turn_id=key,
    )
    values.update(overrides)
    return append_task_execution_event_no_commit(db, **values)


def test_defaults_pin_legacy_without_events(engine, task_id):
    with Session(engine) as db:
        task = db.get(Task, task_id)
        assert task.conversation_storage_version == 1
        assert task.conversation_event_sequence == 0
        assert load_task_execution_events(db, task_id=task_id, scope_id="root") == []
        task.conversation_storage_version = 3
        with pytest.raises(IntegrityError):
            db.flush()


def test_append_is_atomic_with_business_state_and_does_not_touch_task_time(
    engine, task_id
):
    with Session(engine) as writer, Session(engine) as reader:
        original_time = writer.get(Task, task_id).updated_at
        writer.get(Task, task_id).description = "uncommitted business change"
        row = append(writer, task_id)
        assert row.sequence == 1
        assert (
            load_task_execution_events(reader, task_id=task_id, scope_id="root") == []
        )
        assert reader.get(Task, task_id).conversation_event_sequence == 0
        assert reader.get(Task, task_id).description == "unchanged"
        writer.rollback()
        assert writer.get(Task, task_id).conversation_event_sequence == 0
        assert writer.get(Task, task_id).description == "unchanged"
        assert (
            load_task_execution_events(writer, task_id=task_id, scope_id="root") == []
        )
        row = append(writer, task_id)
        assert row.sequence == 1
        writer.commit()
        reader.rollback()
        assert reader.get(Task, task_id).updated_at == original_time
        assert [
            r.sequence
            for r in load_task_execution_events(
                reader, task_id=task_id, scope_id="root"
            )
        ] == [1]


def test_replay_preserves_identity_and_rejects_conflicting_fact(engine, task_id):
    with Session(engine) as db:
        first = append(db, task_id)
        identity = first.event_id
        db.commit()
        replay = append(db, task_id, occurred_at=datetime.now(timezone.utc))
        assert replay.event_id == identity
        assert replay.sequence == 1
        assert db.scalar(sa.select(Task.conversation_event_sequence)) == 1
        db.commit()
        with pytest.raises(ExecutionEventConflict):
            append(db, task_id, payload={"content": "different"})
        db.rollback()
        assert (
            db.scalar(sa.select(sa.func.count()).select_from(TaskExecutionEvent)) == 1
        )


def test_scope_cursor_and_tool_batch_metadata(engine, task_id):
    with Session(engine) as db:
        append(db, task_id, scope_id="child-1")
        append(
            db,
            task_id,
            kind="assistant_tools_requested",
            run_id="run-1",
            assistant_message_id="assistant-1",
        )
        append(
            db,
            task_id,
            key="tool-1",
            kind="tool_attempt_finished",
            assistant_message_id="assistant-1",
            tool_attempt_id="attempt-1",
        )
        db.commit()
        rows = load_task_execution_events(db, task_id=task_id, scope_id="root", limit=1)
        assert [row.sequence for row in rows] == [2]
        assert rows[0].run_id == "run-1"
        rows = load_task_execution_events(
            db, task_id=task_id, scope_id="root", after_sequence=2
        )
        assert [
            (r.sequence, r.assistant_message_id, r.tool_attempt_id) for r in rows
        ] == [(3, "assistant-1", "attempt-1")]
        assert (
            load_task_execution_events(db, task_id=task_id + 1, scope_id="root") == []
        )


def test_idempotency_distinguishes_json_booleans_and_snapshots_input(engine, task_id):
    payload = {"result": {"value": True}}
    with Session(engine) as db:
        original = append(db, task_id, payload=payload)
        payload["result"]["value"] = False
        assert original.payload == {"result": {"value": True}}
        db.commit()
        with pytest.raises(ExecutionEventConflict):
            append(db, task_id, payload={"result": {"value": 1}})


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_json_does_not_allocate_sequence(engine, task_id, value):
    with Session(engine) as db:
        with pytest.raises(ValueError):
            append(db, task_id, payload={"value": value})
        assert db.scalar(sa.select(Task.conversation_event_sequence)) == 0


@pytest.mark.parametrize("rollback_first", [False, True])
@pytest.mark.parametrize("same_key", [False, True])
def test_concurrent_append_cannot_commit_past_pending_event(
    engine, task_id, rollback_first, same_key
):
    reached_update = Event()

    def second_writer():
        with Session(engine) as db:
            connection = db.connection()

            @sa.event.listens_for(connection, "before_cursor_execute")
            def reached_lock(_conn, _cursor, statement, _params, _context, _many):
                if statement.startswith("UPDATE tasks"):
                    reached_update.set()

            row = append(db, task_id, key="turn-1" if same_key else "turn-2")
            result = (row.sequence, row.event_id)
            db.commit()
            return result

    with Session(engine) as first, ThreadPoolExecutor(max_workers=1) as pool:
        row = append(first, task_id)
        first_id = row.event_id
        future = pool.submit(second_writer)
        try:
            assert reached_update.wait(5)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.1)
        finally:
            if rollback_first:
                first.rollback()
            else:
                first.commit()
        sequence, event_id = future.result(timeout=10)
        assert sequence == (1 if rollback_first or same_key else 2)
        if same_key and not rollback_first:
            assert event_id == first_id


def test_database_constraints_and_task_cascade(engine, task_id):
    with Session(engine) as db:
        row = append(db, task_id)
        db.commit()
        for replacement in ({"sequence": 0}, {"payload_version": 0}, {"scope_id": ""}):
            with pytest.raises(IntegrityError):
                with db.begin_nested():
                    db.execute(
                        sa.update(TaskExecutionEvent)
                        .where(TaskExecutionEvent.id == row.id)
                        .values(**replacement)
                    )
        stored = dict(
            db.execute(sa.select(TaskExecutionEvent.__table__)).mappings().one()
        )
        stored.pop("id")
        for duplicate in (
            {"event_id": str(uuid4()), "idempotency_key": "other"},
            {"sequence": 2, "idempotency_key": "other"},
            {"sequence": 2, "event_id": str(uuid4())},
        ):
            with pytest.raises(IntegrityError):
                with db.begin_nested():
                    db.execute(sa.insert(TaskExecutionEvent).values(stored | duplicate))
        db.execute(sa.delete(Task).where(Task.id == task_id))
        db.commit()
        assert (
            db.scalar(sa.select(sa.func.count()).select_from(TaskExecutionEvent)) == 0
        )


def test_payload_keeps_complete_tool_results_and_normalizes_jsonb_hazards(
    engine, task_id
):
    result = {
        "handle": {"branch": "调查完整性", "node_id": "node-1"},
        "output": "结果" * 5000,
    }
    with Session(engine) as db:
        append(db, task_id, kind="tool_attempt_finished", payload=result)
        append(db, task_id, key="unsafe", payload={"value": "a\x00b\ud800c"})
        db.commit()
    with Session(engine) as db:
        rows = load_task_execution_events(db, task_id=task_id, scope_id="root")
        assert rows[0].payload == result
        assert rows[1].payload == {"value": "a\ufffdb\ufffdc"}
        assert (
            append(
                db, task_id, key="unsafe", payload={"value": "a\x00b\ud800c"}
            ).event_id
            == rows[1].event_id
        )


@pytest.mark.parametrize("limit", [-1, 0, 101, 10**9])
def test_page_size_rejected_before_database_access(limit):
    db = Mock(spec=Session)
    with pytest.raises(ValueError, match="limit must be between 1 and 100"):
        load_task_execution_events(db, task_id=1, scope_id="root", limit=limit)
    db.scalars.assert_not_called()


def test_page_size_bounds_and_default_preserve_cursor_pagination(engine, task_id):
    with Session(engine) as db:
        for index in range(101):
            append(db, task_id, key=f"event-{index}")
        db.commit()
        for options in ({}, {"limit": 100}):
            rows = load_task_execution_events(
                db, task_id=task_id, scope_id="root", **options
            )
            assert [row.sequence for row in rows] == list(range(1, 101))
        first = load_task_execution_events(
            db, task_id=task_id, scope_id="root", limit=1
        )
        assert [row.sequence for row in first] == [1]
        last = load_task_execution_events(
            db, task_id=task_id, scope_id="root", after_sequence=100
        )
        assert [row.sequence for row in last] == [101]
