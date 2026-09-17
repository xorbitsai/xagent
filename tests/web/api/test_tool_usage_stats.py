"""Tool statistics come from completion events rather than shared counters."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from tests.web.services.test_task_execution_event_store import engine as engine_fixture
from tests.web.services.test_task_execution_event_store import (
    task_id as task_id_fixture,
)
from xagent.core.agent.trace import ACTION_END_TOOL
from xagent.core.agent.trace import TraceEvent as CoreTraceEvent
from xagent.web.api.tools import _tool_usage_query, get_tool_usage
from xagent.web.models.task import Task, TraceEvent
from xagent.web.models.tool_config import ToolUsage
from xagent.web.services import task_lease_service
from xagent.web.services.trace_handlers import DatabaseTraceHandler

engine = engine_fixture
task_id = task_id_fixture


@pytest.mark.asyncio
async def test_usage_aggregates_completed_events_across_tasks(engine, task_id):
    start = datetime(2026, 9, 8, tzinfo=timezone.utc)
    with Session(engine) as db:
        assert await get_tool_usage(db) == []
        task = db.get(Task, task_id)
        other = Task(user_id=task.user_id, title="Other task")
        db.add(other)
        db.flush()
        payloads = [
            (task_id, "tool_execution_start", {"tool_name": "search"}),
            (task_id, "tool_execution_end", {"tool_name": "search", "success": True}),
            (other.id, "tool_execution_end", {"tool_name": "search", "success": False}),
            (other.id, "tool_execution_end", {"tool_name": "search"}),
            (other.id, "tool_execution_failed", {"tool_name": "search"}),
            (
                task_id,
                "tool_execution_end",
                {"tool_name": "计算器🔎", "success": False},
            ),
            (task_id, "tool_execution_end", {}),
            (task_id, "tool_execution_end", {"tool_name": ""}),
            (task_id, "tool_execution_end", {"tool_name": None}),
        ]
        for index, (tid, kind, data) in enumerate(payloads):
            db.add(
                TraceEvent(
                    task_id=tid,
                    event_id=str(index),
                    event_type=kind,
                    timestamp=start + timedelta(seconds=index),
                    data=data,
                )
            )
        # Legacy totals must not be added to the same events a second time.
        db.add(ToolUsage(tool_name="search", usage_count=999))
        db.commit()
        counts = {row.tool_name: row.usage_count for row in _tool_usage_query(db).all()}
        assert counts == {"search": 3, "计算器🔎": 1}
        response = {row["tool_name"]: row for row in await get_tool_usage(db)}
        assert {name: row["usage_count"] for name, row in response.items()} == counts
        assert response["search"]["success_count"] == 2
        assert response["search"]["error_count"] == 1
        assert response["search"]["success_rate"] == pytest.approx(200 / 3)
        last_used = datetime.fromisoformat(response["search"]["last_used_at"])
        if last_used.tzinfo is None:  # SQLite stores UTC without an offset.
            last_used = last_used.replace(tzinfo=timezone.utc)
        assert last_used == start + timedelta(seconds=3)
        assert response["计算器🔎"]["success_count"] == 0
        assert response["计算器🔎"]["error_count"] == 1


def test_tool_trace_persists_without_accessing_shared_counters(engine, task_id):
    with Session(engine) as db:
        lease = task_lease_service.acquire_task_lease_no_commit(
            db, task_id, runner_id="tool-stats-test", new_run=True
        )
        db.commit()
    assert lease is not None
    statements = []

    def record(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement.lower())

    event.listen(engine, "before_cursor_execute", record)
    try:
        with Session(engine) as db:
            trace = CoreTraceEvent(
                ACTION_END_TOOL,
                task_id=str(task_id),
                step_id="tool-step",
                data={"tool_name": "search", "success": True},
                require_persisted=True,
            )
            with task_lease_service.bind_task_lease_context(lease):
                DatabaseTraceHandler(task_id)._save_trace_event(db, trace)
        with Session(engine) as db:
            assert (
                db.query(TraceEvent)
                .filter(TraceEvent.event_id == str(trace.id))
                .count()
                == 1
            )
            assert _tool_usage_query(db).one().usage_count == 1
        assert not any("tool_usage" in statement for statement in statements)
    finally:
        event.remove(engine, "before_cursor_execute", record)
