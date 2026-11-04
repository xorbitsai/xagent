"""A native summary bounds model-context reads by its suffix, not by history."""

import gc
import json
import tracemalloc
from datetime import datetime, timezone

import pytest
import sqlalchemy as sa

from tests.web.services.task_event_context_reference import (
    reference_load_task_event_context,
)
from tests.web.services.test_task_event_context_equivalence import (
    _live_projection,
    call,
    done,
    last_root,
    say,
    summarize,
)
from tests.web.services.test_task_event_context_service import (
    accept,
    apply,
    fact,
    purge_legacy,
)
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.web.models.task import Task
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.services.task_event_context_service import load_task_event_context

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


class _CountFetchedRows:
    """Observe fetched values without buffering or changing the result shape."""

    def __init__(self, cursor, metrics):
        self.cursor, self.metrics = cursor, metrics

    def __getattr__(self, name):
        return getattr(self.cursor, name)

    def _record(self, rows):
        self.metrics["rows"] += len(rows)
        self.metrics["bytes"] += sum(
            len(json.dumps(tuple(row), ensure_ascii=False, default=str).encode())
            for row in rows
        )
        return rows

    def fetchone(self):
        row = self.cursor.fetchone()
        if row is not None:
            self._record([row])
        return row

    def fetchmany(self, size=None):
        rows = self.cursor.fetchmany() if size is None else self.cursor.fetchmany(size)
        return self._record(rows)

    def fetchall(self):
        return self._record(self.cursor.fetchall())


def _measure(factory, task_id):
    engine = factory.kw["bind"]
    metrics = {"queries": 0, "rows": 0, "bytes": 0}

    def counted(_conn, cursor, _sql, _params, context, _many):
        metrics["queries"] += 1
        context.cursor = _CountFetchedRows(cursor, metrics)

    sa.event.listen(engine, "after_cursor_execute", counted)
    try:
        with factory() as db:
            loaded = load_task_event_context(db, task_id)
    finally:
        sa.event.remove(engine, "after_cursor_execute", counted)
    gc.collect()
    tracemalloc.start()
    try:
        with factory() as db:
            assert load_task_event_context(db, task_id) == loaded
        metrics["peak"] = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    with factory() as db:
        assert reference_load_task_event_context(db, task_id) == loaded
    return loaded, metrics


def _seed(factory, owner_task_id, size):
    """A V2 task with ``size`` covered 8 KiB messages, 150 older summaries and a small suffix."""
    now = datetime.now(timezone.utc)
    with factory() as db:
        task = Task(
            user_id=db.get(Task, owner_task_id).user_id,
            title="budget",
            description="budget",
            conversation_storage_version=2,
        )
        db.add(task)
        db.flush()
        task_id = int(task.id)
        db.execute(
            sa.insert(TaskExecutionEvent),
            [
                {
                    "task_id": task_id,
                    "scope_id": "root",
                    "sequence": i,
                    "event_id": f"{task_id}-history-{i}",
                    "idempotency_key": f"history-{i}",
                    "kind": "assistant_message",
                    "payload_version": 1,
                    "occurred_at": now,
                    "payload": {
                        "content": "x" * 8192,
                        "message_type": "assistant_response",
                    },
                }
                for i in range(1, size + 1)
            ],
        )
        task.conversation_event_sequence = size
        db.commit()
        # Many earlier usable summaries with large bodies: reading more than
        # the newest one would grow rows and bytes.
        for i in range(150):
            summarize(db, task_id, f"old-{i}", last_root(db, task_id), text="y" * 8192)
        accept(db, task_id, "late", "accepted before summary")
        call(db, task_id, "cross", "a")
        call(db, task_id, "cross", "b")
        done(db, task_id, "cross", "a")
        summarize(db, task_id, "s", last_root(db, task_id))
        apply(db, task_id, "late")
        done(db, task_id, "cross", "b")
        say(db, task_id, "tail")
        purge_legacy(db, task_id)
    return task_id


def _assert_bounded(factory, owner_task_id, sizes):
    results = [_measure(factory, _seed(factory, owner_task_id, n)) for n in sizes]
    (first, small), *rest = results
    assert [m["role"] for m in first.messages] == [
        "system",
        "assistant",
        "tool",
        "tool",
        "user",
        "assistant",
    ]
    assert small["rows"] > 0 and small["bytes"] > 0
    for loaded, metrics in rest:
        assert metrics["rows"] > 0 and metrics["bytes"] > 0
        assert loaded.messages == first.messages
        assert metrics["queries"] == small["queries"]
        assert metrics["rows"] == small["rows"]
        # Only identities and sequence numbers grow with the covered prefix.
        assert abs(metrics["bytes"] - small["bytes"]) < 1024
        assert metrics["bytes"] < 64 * 1024
        assert abs(metrics["peak"] - small["peak"]) < 1024 * 1024


def test_reads_do_not_grow_with_covered_history(canonical):
    factory, task_id = canonical
    _assert_bounded(factory, task_id, [100, 2000])


@pytest.mark.slow
def test_reads_do_not_grow_with_large_covered_history(canonical):
    factory, task_id = canonical
    _assert_bounded(factory, task_id, [100, 10000])


def _settle_with_agent_result(db, task_id, key, status, size):
    return fact(
        db,
        task_id,
        "execution_settled",
        key,
        {
            "status": status,
            "result": {
                "success": status != "failed",
                "status": status,
                "failure_reason": f"{key} reason",
                "agent_result": {
                    "context": {"messages": [{"role": "user", "content": "c" * size}]}
                },
            },
        },
    )


def _seed_settlements(factory, owner_task_id, mode, size):
    with factory() as db:
        task = Task(
            user_id=db.get(Task, owner_task_id).user_id,
            title="settlements",
            description="settlements",
            conversation_storage_version=2,
        )
        db.add(task)
        db.flush()
        task_id = int(task.id)
        _settle_with_agent_result(db, task_id, "before", "failed", size)
        say(db, task_id, "covered")
        if mode != "no_summary":
            summarize(db, task_id, "s", last_root(db, task_id), legacy=mode == "legacy")
        _settle_with_agent_result(db, task_id, "after", "failed", size)
        _settle_with_agent_result(db, task_id, "after-ok", "completed", size)
        purge_legacy(db, task_id)
        db.commit()
    return task_id


@pytest.mark.parametrize("mode", ["no_summary", "native", "legacy"])
def test_settlement_agent_result_is_not_fetched(canonical, mode):
    factory, owner_task_id = canonical
    size = 256 * 1024
    task_id = _seed_settlements(factory, owner_task_id, mode, size)
    loaded, metrics = _measure(factory, task_id)
    with factory() as db:
        assert _live_projection(db, task_id) == loaded
    reasons = [m["content"] for m in loaded.messages if m["role"] == "system"]
    assert any("after reason" in text for text in reasons)
    assert any("before reason" in text for text in reasons) == (mode != "native")
    assert metrics["rows"] > 0
    # Each of the three settlements embeds ``size`` bytes of agent context.
    assert metrics["bytes"] < size // 4


def test_anchor_lookup_filters_by_event_id_only(canonical):
    """event_id is unique; a task/scope filter lets the planner scan the task."""
    factory, task_id = canonical
    with factory() as db:
        say(db, task_id, "covered")
        summarize(db, task_id, "s", last_root(db, task_id))
        say(db, task_id, "tail")
        purge_legacy(db, task_id)
    statements = []

    def capture(_conn, _cursor, statement, _params, _context, _many):
        statements.append(" ".join(statement.split()))

    engine = factory.kw["bind"]
    sa.event.listen(engine, "before_cursor_execute", capture)
    try:
        with factory() as db:
            assert load_task_event_context(db, task_id).messages
    finally:
        sa.event.remove(engine, "before_cursor_execute", capture)
    anchor_lookups = [
        s
        for s in statements
        if "task_execution_events.event_id IN" in s
        and "task_execution_events.scope_id," in s
    ]
    assert anchor_lookups
    for lookup in anchor_lookups:
        where = lookup.split(" WHERE ", 1)[1]
        assert not any(c in where for c in ("task_id", "scope_id", "sequence"))
