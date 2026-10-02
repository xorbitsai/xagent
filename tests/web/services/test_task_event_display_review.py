"""Display regressions at persistence, publication and replay boundaries."""

from unittest.mock import AsyncMock, Mock

import pytest

from tests.web.services.test_task_event_display import fact
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services.task_event_display import load_event_display_snapshot

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


@pytest.mark.parametrize(
    "broken", ["missing_completion", "missing_run", "missing_link", "invalid_result"]
)
def test_completed_stream_requires_settlement_provenance(canonical, broken):
    factory, task_id = canonical
    with factory() as db:
        fact(
            db,
            task_id,
            "final_answer_end",
            "end",
            {"message_id": "final_answer_1", "content": "answer"},
            run_id="run",
        )
        if broken != "missing_completion":
            result = (
                []
                if broken == "invalid_result"
                else {}
                if broken == "missing_link"
                else {"stream_message_id": "final_answer_1"}
            )
            fact(
                db,
                task_id,
                "task_completion",
                "completion",
                {"result": result},
                run_id=None if broken == "missing_run" else "run",
            )
        fact(
            db,
            task_id,
            "assistant_message",
            "answer",
            {"content": "answer", "message_type": "assistant_response"},
            flat=True,
            run_id="run",
        )
        settlement = fact(
            db,
            task_id,
            "execution_settled",
            "settlement",
            {"status": "completed"},
            flat=True,
            run_id="run",
        )
        db.commit()
        with pytest.raises(
            ValueError, match=f"task_id={task_id}, event_id={settlement.event_id}"
        ):
            load_event_display_snapshot(db, task_id)


def test_snapshot_integrity_failure_does_not_block_other_tasks(canonical, monkeypatch):
    from xagent.web.models.user import User
    from xagent.web.services import task_stream_snapshot

    factory, task_id = canonical
    monkeypatch.setattr(task_stream_snapshot, "get_session_local", lambda: factory)
    with factory() as db:
        task = db.get(Task, task_id)
        task.status = TaskStatus.COMPLETED
        task.run_id = "run"
        task.state_version = 1
        fact(
            db,
            task_id,
            "final_answer_end",
            "end",
            {"message_id": "final_answer_1", "content": "answer"},
            run_id="run",
        )
        fact(
            db,
            task_id,
            "task_completion",
            "completion",
            {"result": {"stream_message_id": "final_answer_1"}},
            run_id="run",
        )
        for key in ("answer", "duplicate"):
            fact(
                db,
                task_id,
                "assistant_message",
                key,
                {"content": "answer", "message_type": "assistant_response"},
                flat=True,
                run_id="run",
            )
        fact(
            db,
            task_id,
            "execution_settled",
            "result:run:1:completed",
            {"status": "completed"},
            flat=True,
            run_id="run",
        )
        user = User(username="other-owner", password_hash="unused")
        db.add(user)
        db.flush()
        healthy = Task(
            user_id=user.id,
            title="healthy",
            description="healthy",
            status=TaskStatus.COMPLETED,
            output="healthy answer",
        )
        db.add(healthy)
        db.commit()
        healthy_id = healthy.id
    snapshots = task_stream_snapshot.load_task_stream_snapshots([task_id, healthy_id])
    assert any(
        row["task_id"] == healthy_id and row.get("output") == "healthy answer"
        for row in snapshots
    )
    assert any(
        row["task_id"] == task_id and row["type"] == "error" for row in snapshots
    )


def test_history_reasserts_control_state_after_releasing_snapshot_transaction(
    canonical, monkeypatch
):
    from xagent.web.api import websocket

    factory, task_id = canonical
    with factory() as db:
        task = db.get(Task, task_id)
        task.status = TaskStatus.RUNNING
        task.control_state = "running"
        task.state_version = 1
        db.commit()
    original = websocket.release_db_connection_if_clean
    releases = 0

    def transition_after_release(db):
        nonlocal releases
        released = original(db)
        releases += 1
        if releases == 1:
            with factory() as writer:
                task = writer.get(Task, task_id)
                task.status = TaskStatus.PAUSED
                task.control_state = "paused"
                task.state_version = 2
                writer.commit()
        return released

    monkeypatch.setattr(
        websocket, "release_db_connection_if_clean", transition_after_release
    )
    monkeypatch.setattr(websocket, "cache_get", lambda _: None)
    cache_write = Mock()
    monkeypatch.setattr(websocket, "cache_set", cache_write)
    with factory() as db:
        snapshot = websocket._load_event_historical_stream_snapshot(
            db, db.get(Task, task_id)
        )
    paused = next(row for row in snapshot.events if row["type"] == "task_paused")
    assert paused["status"] == "paused"
    assert paused["state_version"] == 2
    cache_write.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["broadcast", "dropped", "integrity_gap"])
async def test_authoritative_live_display_records_outcomes_and_safe_errors(
    canonical, monkeypatch, outcome
):
    from xagent.core.agent.trace import (
        TraceAction,
        TraceCategory,
        TraceEvent,
        TraceEventType,
        TraceScope,
    )
    from xagent.web.services import task_event_trace_handler
    from xagent.web.services.client_error_messages import CLIENT_SAFE_TASK_FAILURE

    factory, task_id = canonical
    event = TraceEvent(
        TraceEventType(TraceScope.TASK, TraceAction.START, TraceCategory.REACT),
        task_id=str(task_id),
        data={},
    )
    if outcome != "integrity_gap":
        with factory() as db:
            fact(
                db,
                task_id,
                "react_task_start",
                str(event.id),
                {"__audit_only__": True}
                if outcome == "dropped"
                else {"execution_id": "execution"},
            )
            db.commit()
    sink, counter = AsyncMock(), Mock()
    monkeypatch.setattr(task_event_trace_handler, "publish_task_event", sink)
    monkeypatch.setattr(task_event_trace_handler, "increment_counter", counter)
    monkeypatch.setattr(task_event_trace_handler, "task_has_audience", lambda _: True)
    await task_event_trace_handler.TaskEventTraceHandler(
        task_id, authoritative=True
    ).handle_event(event)
    counter.assert_any_call(
        "xagent.websocket.trace.events", attributes={"outcome": outcome}
    )
    if outcome == "dropped":
        sink.assert_not_awaited()
    elif outcome == "integrity_gap":
        assert sink.await_args.args == (
            {"type": "error", "task_id": task_id, "message": CLIENT_SAFE_TASK_FAILURE},
            task_id,
        )
    else:
        assert sink.await_args.args[0]["event_id"] == event.id


def test_ordinary_task_child_route_authorizes_parent_and_filters_scope(canonical):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from xagent.web.api import chat
    from xagent.web.models.user import User
    from xagent.web.models.workforce import WorkforceRun

    factory, task_id = canonical
    with factory() as db:
        owner = db.get(User, db.get(Task, task_id).user_id)
        owner.is_admin = False
        stranger = User(username="unrelated", password_hash="unused", is_admin=False)
        db.add(stranger)
        db.flush()
        stranger_id = stranger.id
        owner_id = owner.id
        fact(
            db,
            task_id,
            "react_task_start",
            "child",
            {
                "source": "xagent-agent-tool-child",
                "worker_task_id": "child-scope",
                "agent_name": "Worker",
            },
            scope_id="child-scope",
        )
        fact(
            db,
            task_id,
            "react_task_start",
            "other-child",
            {"source": "xagent-agent-tool-child", "agent_name": "Other worker"},
            scope_id="other-scope",
        )
        fact(
            db, task_id, "react_task_start", "not-delegated", [], scope_id="child-scope"
        )
        fact(
            db,
            task_id,
            "recovery_state",
            "checkpoint",
            {"source": "xagent-agent-tool-child", "secret": "internal"},
            scope_id="child-scope",
        )
        db.commit()
        assert db.query(WorkforceRun).count() == 0

    def session():
        with factory() as db:
            yield db

    user_id = owner_id

    def current_user():
        with factory() as db:
            user = db.get(User, user_id)
            db.expunge(user)
            return user

    app = FastAPI()
    app.include_router(chat.chat_router)
    app.dependency_overrides[chat.get_db] = session
    app.dependency_overrides[chat.get_current_user] = current_user
    with TestClient(app) as client:
        response = client.get(f"/api/chat/task/{task_id}/agent-executions/child-scope")
        assert response.status_code == 200, response.text
        assert [row["event_id"] for row in response.json()["trace_events"]] == ["child"]
        assert response.json()["agent_name"] == "Worker"
        assert "internal" not in response.text
        assert (
            client.get(f"/api/chat/task/{task_id}/agent-executions/absent").status_code
            == 404
        )
        user_id = stranger_id
        assert (
            client.get(
                f"/api/chat/task/{task_id}/agent-executions/child-scope"
            ).status_code
            == 404
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_version", [1, 2])
async def test_ordinary_v2_delegation_emits_root_inspector_summary(
    canonical, storage_version
):
    from xagent.core.tools.adapters.vibe.agent_tool import AgentTool
    from xagent.web.tracing import create_task_tracer

    factory, task_id = canonical
    with factory() as db:
        task = db.get(Task, task_id)
        task.conversation_storage_version = storage_version
        owner_id = task.user_id
        db.commit()
    tracer = create_task_tracer(task_id)
    tool = AgentTool(
        agent_id=17,
        agent_name="Worker",
        agent_description="",
        session_factory=factory,
        user_id=owner_id,
        task_id=str(task_id),
        parent_tracer=tracer,
    )
    await tool._trace_delegation("start", execution_task_id="child-scope")
    if storage_version == 2:
        with factory() as db:
            events = load_event_display_snapshot(db, task_id).events
        assert events[0]["event_type"] == "workforce_delegation_start"
        assert events[0]["data"]["worker_task_id"] == "child-scope"
    else:
        from xagent.web.models.task import TraceEvent

        with factory() as db:
            assert (
                db.query(TraceEvent).filter(TraceEvent.task_id == task_id).count() == 0
            )


def test_activity_aggregation_is_limited_to_candidate_tasks(canonical):
    from datetime import datetime, timezone

    from sqlalchemy import select

    from xagent.web.api.conversation_logs import _latest_message_activity_subquery
    from xagent.web.models.chat_message import TaskChatMessage

    factory, task_id = canonical
    with factory() as db:
        owner = db.get(Task, task_id).user_id
        legacy = Task(user_id=owner, title="legacy", description="legacy")
        unrelated = Task(
            user_id=owner,
            title="unrelated",
            description="unrelated",
            conversation_storage_version=2,
        )
        db.add_all([legacy, unrelated])
        db.flush()
        fact(
            db, task_id, "assistant_message", "answer", {"content": "answer"}, flat=True
        )
        fact(
            db,
            unrelated.id,
            "assistant_message",
            "other",
            {"content": "other"},
            flat=True,
        )
        db.add(
            TaskChatMessage(
                task_id=legacy.id,
                user_id=owner,
                role="user",
                message_type="user",
                content="legacy",
                created_at=datetime.now(timezone.utc),
            )
        )
        db.flush()
        candidates = (
            select(Task.id).where(Task.id.in_([task_id, legacy.id])).correlate(None)
        )
        activity = _latest_message_activity_subquery(db, candidates)
        rows = db.execute(select(activity.c.task_id, activity.c.message_count)).all()
        assert dict(rows) == {task_id: 1, legacy.id: 1}


def test_monitor_source_filters_owner_before_union_and_excludes_recovery(canonical):
    from sqlalchemy import select

    from xagent.web.models.user import User
    from xagent.web.services.task_event_metrics import monitoring_trace_source

    factory, task_id = canonical
    with factory() as db:
        owner = db.get(Task, task_id).user_id
        other_user = User(username="metrics-other", password_hash="unused")
        db.add(other_user)
        db.flush()
        other = Task(
            user_id=other_user.id,
            title="other",
            description="other",
            conversation_storage_version=2,
        )
        db.add(other)
        db.flush()
        fact(db, task_id, "llm_call_end", "owned", {})
        fact(db, task_id, "llm_call_end", "child", {}, scope_id="child")
        fact(
            db,
            task_id,
            "recovery_state",
            "checkpoint",
            {"snapshot": {"private": "state"}},
        )
        fact(db, other.id, "llm_call_end", "foreign", {})
        db.flush()
        own_source = monitoring_trace_source(user_id=owner)
        assert db.scalars(select(own_source.event_id)).all() == ["owned"]
        all_source = monitoring_trace_source()
        assert set(db.scalars(select(all_source.event_id)).all()) == {
            "owned",
            "foreign",
        }


def test_settlement_reconciliation_reads_only_transcript_provenance(canonical):
    from xagent.web.services.task_event_display import settlement_display_events

    factory, task_id = canonical
    with factory() as db:
        fact(db, task_id, "llm_call_end", "nonessential", [])
        fact(
            db,
            task_id,
            "assistant_message",
            "answer",
            {"content": "answer", "message_type": "assistant_response"},
            flat=True,
            run_id="run",
        )
        settlement = fact(
            db,
            task_id,
            "execution_settled",
            "result:run:1:completed",
            {"status": "completed"},
            flat=True,
            run_id="run",
        )
        db.flush()
        frames = settlement_display_events(db, settlement)
        assert len(frames) == 1
        assert frames[0]["data"]["message"] == "answer"


@pytest.mark.parametrize("field", ["visible", "__audit_only__", "expect_response"])
@pytest.mark.parametrize(
    "value",
    [None, False, True, 0, 0.0, 1, "", "false", "invalid", [], [1], {}, {"flag": True}],
)
def test_log_activity_flags_match_transcript_json_semantics(canonical, field, value):
    from sqlalchemy import select

    from xagent.web.api.conversation_logs import _latest_message_activity_subquery

    factory, task_id = canonical
    with factory() as db:
        data = {
            "message": "Question?",
            "expect_response": True,
            "visible": True,
            field: value,
        }
        fact(db, task_id, "agent_message", "question", data)
        db.flush()
        snapshot = load_event_display_snapshot(db, task_id)
        activity = _latest_message_activity_subquery(
            db, select(Task.id).where(Task.id == task_id)
        )
        count = db.scalar(select(activity.c.message_count)) or 0
        assert count == len(snapshot.messages)
