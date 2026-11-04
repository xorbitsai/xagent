"""Reproducible stage-E measurements; thresholds belong to activation (3.4)."""

import gc
import json
import time
import tracemalloc
from datetime import datetime, timezone

import pytest
import sqlalchemy as sa

from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.core.agent.context.execution import MODEL_CONTEXT_WATERMARK_METADATA_KEY
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TraceEvent
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.services.chat_history_service import load_task_transcript_window
from xagent.web.services.task_event_context_service import load_task_event_context
from xagent.web.services.task_event_display import load_event_display_snapshot

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
        self.metrics["serialized_value_bytes"] += sum(
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


def _measure(engine, read):
    metrics = {"queries": 0, "rows": 0, "serialized_value_bytes": 0}

    def counted(_conn, cursor, _sql, _params, context, _many):
        metrics["queries"] += 1
        context.cursor = _CountFetchedRows(cursor, metrics)

    sa.event.listen(engine, "after_cursor_execute", counted)
    try:
        expected = read()
    finally:
        sa.event.remove(engine, "after_cursor_execute", counted)
    # Count, memory, and timing passes are separate: byte serialization must
    # not inflate either peak memory or elapsed time. Each uses a new Session.
    gc.collect()
    tracemalloc.start()
    try:
        assert read() == expected
        metrics["python_peak_bytes"] = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    started = time.perf_counter()
    assert read() == expected
    metrics["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return expected, metrics


def _seed(factory, task_id, size):
    now = datetime.now(timezone.utc)
    text = "x" * 8192
    with factory() as db:
        task = db.get(Task, task_id)
        legacy = Task(user_id=task.user_id, title="baseline", description="baseline")
        db.add(legacy)
        db.flush()
        legacy_id = legacy.id
        db.execute(
            sa.insert(TaskChatMessage),
            [
                {
                    "task_id": legacy_id,
                    "user_id": task.user_id,
                    "role": "assistant",
                    "message_type": "assistant_response",
                    "content": text,
                }
                for _ in range(size)
            ],
        )
        watermark = db.scalar(
            sa.select(sa.func.max(TaskChatMessage.id)).where(
                TaskChatMessage.task_id == legacy_id
            )
        )
        db.add(
            TraceEvent(
                task_id=legacy_id,
                event_id="summary",
                event_type="action_end_compact",
                timestamp=now,
                data={"summary": "saved summary", "watermark_message_id": watermark},
            )
        )
        db.add(
            TaskChatMessage(
                task_id=legacy_id,
                user_id=task.user_id,
                role="assistant",
                message_type="assistant_response",
                content="retained answer",
            )
        )
        events = [
            {
                "task_id": task_id,
                "scope_id": "root",
                "sequence": i,
                "event_id": f"history-{i}",
                "idempotency_key": f"history-{i}",
                "kind": "assistant_message",
                "payload_version": 1,
                "occurred_at": now,
                "payload": {"content": text, "message_type": "assistant_response"},
            }
            for i in range(1, size + 1)
        ]
        events.extend(
            [
                {
                    "task_id": task_id,
                    "scope_id": "root",
                    "sequence": size + 1,
                    "event_id": "summary",
                    "idempotency_key": "summary",
                    "kind": "action_end_compact",
                    "payload_version": 1,
                    "occurred_at": now,
                    "payload": {
                        "protocol_event_id": "summary",
                        "data": {
                            "summary": "saved summary",
                            MODEL_CONTEXT_WATERMARK_METADATA_KEY: {
                                "scope_id": "root",
                                "sequence": size,
                                "event_id": f"history-{size}",
                            },
                        },
                    },
                },
                {
                    "task_id": task_id,
                    "scope_id": "root",
                    "sequence": size + 2,
                    "event_id": "tail",
                    "idempotency_key": "tail",
                    "kind": "assistant_message",
                    "payload_version": 1,
                    "occurred_at": now,
                    "payload": {
                        "content": "retained answer",
                        "message_type": "assistant_response",
                    },
                },
            ]
        )
        db.execute(sa.insert(TaskExecutionEvent), events)
        task.conversation_event_sequence = size + 2
        db.commit()
        return legacy_id


@pytest.mark.slow
@pytest.mark.parametrize("size", [100, 1000, 10000])
def test_compacted_history_read_baseline(canonical, size, record_property):
    factory, task_id = canonical
    legacy_id = _seed(factory, task_id, size)
    engine = factory.kw["bind"]

    def legacy_context():
        with factory() as db:
            return load_task_transcript_window(db, legacy_id).messages

    def event_context():
        with factory() as db:
            return load_task_event_context(db, task_id).messages

    def event_display():
        with factory() as db:
            view = load_event_display_snapshot(db, task_id)
            return (view.horizon, len(view.messages), len(view.events))

    old, v1 = _measure(engine, legacy_context)
    new, v2 = _measure(engine, event_context)
    assert (
        new
        == old
        == [
            {"role": "system", "content": "saved summary"},
            {"role": "assistant", "content": "retained answer"},
        ]
    )
    display, display_metrics = _measure(engine, event_display)
    assert display == (size + 2, size + 1, size + 2)
    result = {
        "dialect": engine.dialect.name,
        "covered_messages": size,
        "message_bytes": 8192,
        "horizon": size + 2,
        "v1_model": v1,
        "v2_model": v2,
        "v2_display": display_metrics,
    }
    record_property("event_reader_baseline", json.dumps(result))
    print("EVENT_READER_BASELINE " + json.dumps(result))
