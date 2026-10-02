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
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TraceEvent
from xagent.web.services import task_events
from xagent.web.services.task_event_display import load_event_display_snapshot
from xagent.web.services.task_execution_event_writer import append_fact_no_commit
from xagent.web.services.task_stream_snapshot import load_task_stream_snapshots

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


def fact(db, task_id, kind, key, data, *, flat=False, **kwargs):
    return append_fact_no_commit(
        db,
        task_id=task_id,
        kind=kind,
        key=key,
        payload=data if flat else {"data": data, "protocol_event_id": key},
        occurred_at=datetime(2026, 9, 30, tzinfo=timezone.utc),
        **kwargs,
    )


def test_display_uses_occurrences_and_excludes_internal_and_child_content(canonical):
    factory, task_id = canonical
    with factory() as db:
        for turn in ("first", "second"):
            fact(
                db,
                task_id,
                "input_accepted",
                turn,
                {"content": "same", "message_type": "user"},
                flat=True,
                turn_id=turn,
            )
        fact(
            db,
            task_id,
            "user_message",
            "runtime-user",
            {"message": "same", "context": {"secret": "secret"}},
            turn_id="first",
        )
        fact(db, task_id, "recovery_state", "state", {"snapshot": {"secret": "secret"}})
        fact(
            db,
            task_id,
            "llm_call_end",
            "audit",
            {"__audit_only__": True, "response": "secret"},
        )
        fact(
            db,
            task_id,
            "agent_message",
            "child",
            {"message": "secret"},
            scope_id="child",
        )
        question = fact(
            db,
            task_id,
            "agent_message",
            "question",
            {
                "message": "Which?",
                "expect_response": True,
                "visible": True,
                "metadata": {"interactions": [{"type": "text"}]},
            },
        )
        fact(
            db,
            task_id,
            "assistant_message",
            "failure",
            {
                "content": "secret",
                "message_type": "task_failure",
                "attachments": [{"secret": "secret"}],
            },
            flat=True,
        )
        db.commit()
        db.execute(sa.delete(TaskChatMessage).where(TaskChatMessage.task_id == task_id))
        db.execute(sa.delete(TraceEvent).where(TraceEvent.task_id == task_id))
        db.commit()
        view = load_event_display_snapshot(db, task_id)
        assert len(view.messages) == 4
        assert [m["turn_id"] for m in view.messages[:2]] == ["first", "second"]
        assert len({m["id"] for m in view.messages}) == 4
        assert view.messages[2]["id"] == question.sequence
        assert "secret" not in repr(view)
        assert len(view.events) == 4
        assert (
            view.events[0]["timestamp"]
            == datetime(2026, 9, 30, tzinfo=timezone.utc).timestamp()
        )
        assert view.events[2]["event_id"] == "question"
        assert (
            view.events[2]["data"]["message_id"]
            == f"execution_message_{question.event_id}"
        )
        assert view.events[2]["data"]["expect_response"] is False


def test_stream_boundaries_fixed_horizon_and_settlement_share_one_bubble(canonical):
    factory, task_id = canonical
    with factory() as db:
        start = fact(
            db,
            task_id,
            "final_answer_start",
            "start",
            {"message_id": "final_answer_1"},
            run_id="run",
        )
        db.commit()
        running = load_event_display_snapshot(
            db, task_id, through_sequence=start.sequence, active_run_id="run"
        )
        assert running.messages == ()
        assert running.events[0]["message_id"] == "final_answer_1"
        assert "status" not in running.events[0]
        interrupted = load_event_display_snapshot(
            db, task_id, through_sequence=start.sequence
        )
        assert interrupted.events[0]["status"] == "interrupted"
        end = fact(
            db,
            task_id,
            "final_answer_end",
            "end",
            {"message_id": "final_answer_1", "content": "answer"},
            run_id="run",
        )
        completion = fact(
            db,
            task_id,
            "task_completion",
            "completion",
            {"result": {"stream_message_id": "final_answer_1"}},
            run_id="run",
        )
        answer = fact(
            db,
            task_id,
            "assistant_message",
            "answer",
            {"content": "answer", "message_type": "assistant_response"},
            flat=True,
            run_id="run",
        )
        fact(
            db,
            task_id,
            "execution_settled",
            "settlement",
            {"status": "completed"},
            flat=True,
            run_id="run",
        )
        db.commit()
        before = load_event_display_snapshot(db, task_id, through_sequence=end.sequence)
        assert before.events[-1]["event_id"] == "end"
        assert before.messages == ()
        still_old = load_event_display_snapshot(
            db, task_id, through_sequence=start.sequence
        )
        assert len(still_old.events) == 1
        settled = load_event_display_snapshot(db, task_id)
        assert len(settled.messages) == 1
        assert settled.messages[0]["id"] == answer.sequence
        assert settled.events[-1]["data"]["message_id"] == "final_answer_1"
        assert settled.events[-2]["event_id"] == completion.payload["protocol_event_id"]
        assert settled.events[-2]["data"]["display"] == "timeline"


@pytest.mark.parametrize("bad", ["missing_end", "duplicate_completion", "null_run"])
def test_stream_settlement_rejects_insufficient_provenance(canonical, bad):
    factory, task_id = canonical
    run = None if bad == "null_run" else "run"
    with factory() as db:
        if bad != "missing_end":
            fact(
                db,
                task_id,
                "final_answer_end",
                "end",
                {"message_id": "final_answer_1", "content": "answer"},
                run_id=run,
            )
        for key in (
            ["completion", "another"]
            if bad == "duplicate_completion"
            else ["completion"]
        ):
            fact(
                db,
                task_id,
                "task_completion",
                key,
                {"result": {"stream_message_id": "final_answer_1"}},
                run_id=run,
            )
        fact(
            db,
            task_id,
            "assistant_message",
            "answer",
            {"content": "answer", "message_type": "assistant_response"},
            flat=True,
            run_id=run,
        )
        fact(
            db,
            task_id,
            "execution_settled",
            "settlement",
            {"status": "completed"},
            flat=True,
            run_id=run,
        )
        db.commit()
        with pytest.raises(ValueError, match=f"task_id={task_id}"):
            load_event_display_snapshot(db, task_id)


def test_websocket_reads_v2_without_legacy_queries_and_enforces_access(
    canonical, monkeypatch
):
    from xagent.web.api import websocket

    factory, task_id = canonical
    monkeypatch.setattr("xagent.web.models.database.get_db", lambda: iter([factory()]))
    monkeypatch.setattr(websocket, "cache_get", lambda *_: None)
    monkeypatch.setattr(websocket, "cache_set", lambda *_a, **_k: None)
    with factory() as db:
        owner = db.get(Task, task_id).user_id
        fact(
            db,
            task_id,
            "input_accepted",
            "first",
            {"content": "hello", "message_type": "user"},
            flat=True,
            turn_id="first",
        )
        db.commit()
    statements = []

    def record(_conn, _cursor, statement, _params, _context, _many):
        statements.append(statement.lower())

    sa.event.listen(factory.kw["bind"], "before_cursor_execute", record)
    try:
        snapshot = websocket._load_historical_stream_snapshot_sync(
            task_id, actor_user_id=owner, actor_is_admin=False
        )
        assert snapshot is not None
        assert any(e.get("event_type") == "user_message" for e in snapshot.events)
        assert not any(
            "trace_events" in s or "task_chat_messages" in s for s in statements
        )
        assert (
            websocket._load_historical_stream_snapshot_sync(
                task_id, actor_user_id=owner + 1, actor_is_admin=False
            )
            is None
        )
    finally:
        sa.event.remove(factory.kw["bind"], "before_cursor_execute", record)


@pytest.mark.asyncio
async def test_v2_log_detail_and_list_use_event_counts_activity_and_compaction(
    canonical,
):
    from xagent.web.api.conversation_logs import (
        get_conversation_log_detail,
        list_conversation_logs,
    )
    from xagent.web.models.user import User

    factory, task_id = canonical
    with factory() as db:
        task = db.get(Task, task_id)
        task.source = "sdk"
        task.is_visible = False
        user = db.get(User, task.user_id)
        first = fact(
            db,
            task_id,
            "input_accepted",
            "first",
            {"content": "hello", "message_type": "user"},
            flat=True,
            turn_id="first",
        )
        compact = fact(
            db,
            task_id,
            "action_end_compact",
            "compact",
            {"original_tokens": 100, "compacted_tokens": 10},
        )
        answer = fact(
            db,
            task_id,
            "assistant_message",
            "answer",
            {"content": "answer", "message_type": "assistant_response"},
            flat=True,
        )
        fact(
            db,
            task_id,
            "agent_message",
            "audit-question",
            {
                "__audit_only__": True,
                "expect_response": True,
                "visible": True,
                "message": "private audit",
            },
        )
        # A contradictory compatibility row must not contribute a log message.
        db.add(
            TaskChatMessage(
                task_id=task_id,
                user_id=user.id,
                role="user",
                content="legacy secret",
                message_type="user",
            )
        )
        db.commit()
        detail = await get_conversation_log_detail(task_id, db=db, user=user)
        assert [row["id"] for row in detail["transcript"]] == [
            first.sequence,
            f"compact-{compact.event_id}",
            answer.sequence,
        ]
        assert detail["log"]["message_count"] == 2
        assert "legacy secret" not in repr(detail)
        listed = await list_conversation_logs(
            source="all",
            agent_id=None,
            search=None,
            page=1,
            per_page=20,
            db=db,
            user=user,
        )
        assert listed["logs"][0]["message_count"] == 2
        assert (
            listed["logs"][0]["last_activity_at"] == detail["log"]["last_activity_at"]
        )


@pytest.mark.asyncio
async def test_monitor_counts_each_task_source_once_and_isolates_child_scope(canonical):
    from xagent.web.api.monitor import (
        get_model_stats,
        get_monitoring_stats,
        get_popular_tools,
    )
    from xagent.web.models.user import User

    factory, task_id = canonical
    with factory() as db:
        user = db.get(User, db.get(Task, task_id).user_id)
        for kind, data in [
            ("llm_call_start", {"model_name": "event-model"}),
            ("llm_call_end", {"usage": {"total_tokens": 123}}),
            ("tool_execution_start", {"tool_name": "event-tool"}),
            ("tool_execution_end", {"tool_name": "event-tool"}),
        ]:
            fact(db, task_id, kind, kind, data)
            # Keeping both representations must not double count a V2 task.
            db.add(
                TraceEvent(
                    task_id=task_id,
                    event_id=kind,
                    event_type=kind,
                    timestamp=datetime.now(timezone.utc),
                    data=data,
                )
            )
            fact(db, task_id, kind, f"child-{kind}", data, scope_id="child")
        legacy = Task(
            user_id=user.id,
            title="legacy",
            description="legacy",
            conversation_storage_version=1,
        )
        db.add(legacy)
        db.flush()
        db.add(
            TraceEvent(
                task_id=legacy.id,
                event_id="v1-call",
                event_type="llm_call_end",
                timestamp=datetime.now(timezone.utc),
                data={"usage": {"total_tokens": 7}},
            )
        )
        db.commit()
        stats = await get_monitoring_stats(db=db, current_user=user)
        assert stats["totalCalls"] == 3
        assert stats["totalTokens"] == 130
        assert stats["llmCalls"] == 1
        assert stats["toolExecutions"] == 1
        assert (await get_popular_tools(db=db, current_user=user))[0][
            "usage_count"
        ] == 1
        assert (await get_model_stats(db=db, current_user=user))[0]["total_tasks"] == 1


def test_v2_rest_steps_and_child_details_read_events_without_legacy_content(
    canonical, monkeypatch
):
    from xagent.web.api.v1 import tasks
    from xagent.web.models.agent import Agent
    from xagent.web.services.task_agent_execution import (
        _serialize_agent_execution_traces,
    )

    factory, task_id = canonical
    with factory() as db:
        task = db.get(Task, task_id)
        agent = Agent(name="display", user_id=task.user_id)
        db.add(agent)
        db.flush()
        task.agent_id = agent.id
        fact(
            db,
            task_id,
            "tool_execution_start",
            "tool",
            {"tool_name": "root-tool", "step_id": "step"},
        )
        fact(
            db,
            task_id,
            "react_task_start",
            "child",
            {"source": "xagent-agent-tool-child", "agent_name": "child-agent"},
            scope_id="worker",
        )
        failed = fact(
            db,
            task_id,
            "task_error_general",
            "child-failed",
            {"source": "xagent-agent-tool-child", "error": "private details"},
            scope_id="worker",
        )
        db.commit()
        monkeypatch.setattr(tasks, "get_session_local", lambda: factory)
        monkeypatch.setattr(
            tasks,
            "_resolve_task_or_404",
            lambda task_id, _principal, db: db.get(Task, task_id),
        )
        version = tasks._load_task_steps_version_snapshot(task_id, None)
        snapshot = tasks._load_task_steps_snapshot(task_id, None)
        assert version.storage_version == snapshot.storage_version == 2
        assert snapshot.max_event_id == failed.sequence
        assert [row.event_id for row in snapshot.events] == ["tool"]
        events, metadata = _serialize_agent_execution_traces(
            db, task_id=task_id, worker_task_id="worker", event_horizon=failed.sequence
        )
        assert [event["event_id"] for event in events] == ["child", "child-failed"]
        assert "private details" not in repr(events)
        assert metadata["agent_name"] == "child-agent"


@pytest.mark.asyncio
async def test_live_terminal_publication_uses_same_owner_as_replay(
    canonical, monkeypatch
):
    from unittest.mock import AsyncMock

    from xagent.web.models.task import TaskStatus
    from xagent.web.services.managed_task_lease import (
        finalize_managed_task_lease_result,
    )
    from xagent.web.services.task_event_display import publish_task_result
    from xagent.web.services.task_lease_service import acquire_task_lease

    factory, task_id = canonical
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)
        fact(
            db,
            task_id,
            "final_answer_end",
            "end",
            {"message_id": "final_answer_1", "content": "answer"},
            run_id=lease.run_id,
        )
        fact(
            db,
            task_id,
            "task_completion",
            "completion",
            {"result": {"stream_message_id": "final_answer_1"}},
            run_id=lease.run_id,
        )
        db.commit()
        assert finalize_managed_task_lease_result(
            db,
            lease,
            status=TaskStatus.COMPLETED,
            assistant_content="answer",
            execution_result={"success": True, "output": "answer"},
        )
        task = db.get(Task, task_id)
        notice = {
            "type": "task_completed",
            "task": {"id": task_id, "status": "completed"},
            "run_id": lease.run_id,
            "state_version": task.state_version,
        }
        historical = load_event_display_snapshot(db, task_id).events[-1]
    sink = AsyncMock()
    monkeypatch.setattr(task_events, "_task_event_sink", sink)
    await publish_task_result(notice, task_id)
    assert sink.await_args_list[0].args[0] == historical
    assert sink.await_args_list[1].args[0]["message_id"] == "final_answer_1"


@pytest.mark.asyncio
async def test_live_runtime_trace_uses_canonical_protocol_identity(
    canonical, monkeypatch
):
    from unittest.mock import AsyncMock

    from xagent.core.agent.trace import (
        TraceAction,
        TraceCategory,
    )
    from xagent.core.agent.trace import TraceEvent as RuntimeEvent
    from xagent.core.agent.trace import (
        TraceEventType,
        TraceScope,
    )
    from xagent.web.services import task_event_trace_handler
    from xagent.web.tracing import create_task_tracer

    factory, task_id = canonical
    sink = AsyncMock()
    monkeypatch.setattr(task_event_trace_handler, "publish_task_event", sink)
    monkeypatch.setattr(task_event_trace_handler, "task_has_audience", lambda _: True)
    event = RuntimeEvent(
        TraceEventType(TraceScope.TASK, TraceAction.START, TraceCategory.REACT),
        task_id=str(task_id),
        data={"execution_id": "execution"},
    )
    tracer = create_task_tracer(task_id)
    await tracer.event_writer(event)
    await next(
        handler
        for handler in tracer.handlers
        if isinstance(handler, task_event_trace_handler.TaskEventTraceHandler)
    ).handle_event(event)
    with factory() as db:
        historical = load_event_display_snapshot(db, task_id).events[0]
    assert sink.await_args.args[0] == historical
    assert historical["event_id"] == event.id


@pytest.mark.asyncio
async def test_failure_publication_and_shared_reconciliation_use_durable_messages(
    canonical,
    monkeypatch,
):
    from unittest.mock import AsyncMock

    from xagent.web.models.task import TaskStatus
    from xagent.web.services.managed_task_lease import (
        finalize_managed_task_lease_result,
    )
    from xagent.web.services.task_event_display import publish_task_result
    from xagent.web.services.task_lease_service import acquire_task_lease

    factory, task_id = canonical
    monkeypatch.setattr(
        "xagent.web.services.task_stream_snapshot.get_session_local", lambda: factory
    )
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)
        assert finalize_managed_task_lease_result(
            db,
            lease,
            status=TaskStatus.FAILED,
            assistant_content="private failure",
            message_type="task_failure",
            execution_result={"error": "private failure"},
        )
        task = db.get(Task, task_id)
        notice = {
            "type": "task_error",
            "task": {"id": task_id, "status": "failed"},
            "run_id": lease.run_id,
            "state_version": task.state_version,
        }
        historical = load_event_display_snapshot(db, task_id).events[-1]
        task.output = "legacy output must never become a bubble"
        db.commit()
    sink = AsyncMock()
    monkeypatch.setattr(task_events, "_task_event_sink", sink)
    await publish_task_result(notice, task_id)
    assert sink.await_args_list[0].args[0] == historical
    assert (
        sink.await_args_list[1].args[0]["message_id"]
        == historical["data"]["message_id"]
    )
    snapshots = load_task_stream_snapshots([task_id])
    assert snapshots[0]["output"] is None
    assert snapshots[1] == historical
    assert "private failure" not in repr(snapshots)


@pytest.mark.parametrize("message_id", [None, "wrong_namespace"])
def test_stream_identity_is_required_at_display_boundary(canonical, message_id):
    factory, task_id = canonical
    with factory() as db:
        row = fact(
            db, task_id, "final_answer_start", "start", {"message_id": message_id}
        )
        db.commit()
        with pytest.raises(
            ValueError, match=f"task_id={task_id}, event_id={row.event_id}.*message_id"
        ):
            load_event_display_snapshot(db, task_id)


def test_question_metadata_validation_precedes_empty_default(canonical):
    from xagent.web.services.task_event_display import load_live_display_event

    factory, task_id = canonical
    with factory() as db:
        row = fact(
            db,
            task_id,
            "agent_message",
            "bad-question",
            {
                "expect_response": True,
                "message": "Which?",
                "metadata": [],
            },
        )
        db.commit()
        with pytest.raises(
            ValueError,
            match=f"task_id={task_id}, event_id={row.event_id}.*invalid question",
        ):
            load_event_display_snapshot(db, task_id)
        with pytest.raises(
            ValueError,
            match=f"task_id={task_id}, event_id={row.event_id}.*invalid question",
        ):
            load_live_display_event(db, task_id, "bad-question")


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
async def test_real_failure_notice_keeps_committed_identity_after_new_run(
    canonical, resume, monkeypatch
):
    from unittest.mock import AsyncMock

    from xagent.web.services.task_event_display import publish_task_result
    from xagent.web.services.task_execution import (
        _settle_resumed_task_lease,
        create_terminal_task_error_event,
    )
    from xagent.web.services.task_lease_service import acquire_task_lease
    from xagent.web.services.task_orchestrator import settle_task_lease_isolated

    factory, task_id = canonical
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)
    notice = create_terminal_task_error_event(task_id, "Task failed")
    settle = _settle_resumed_task_lease if resume else settle_task_lease_isolated
    assert settle(lease, error_message="private failure", terminal_event_state=notice)
    with factory() as db:
        historical = load_event_display_snapshot(db, task_id).events[-1]
        version = db.get(Task, task_id).state_version
        replacement = acquire_task_lease(db, task_id, new_run=True)
        assert replacement.run_id != lease.run_id
    sink = AsyncMock()
    monkeypatch.setattr(task_events, "_task_event_sink", sink)
    await publish_task_result(notice, task_id)
    assert sink.await_args_list[0].args[0] == historical
    terminal = sink.await_args_list[1].args[0]
    assert terminal["run_id"] == lease.run_id
    assert terminal["state_version"] == version
    assert terminal["message_id"] == historical["data"]["message_id"]
    assert "private failure" not in repr(sink.await_args_list)


@pytest.mark.asyncio
async def test_external_cancel_live_and_replay_share_committed_identity(
    canonical, monkeypatch
):
    from unittest.mock import AsyncMock

    from xagent.web.models.agent import Agent
    from xagent.web.services import external_task_cancel, task_execution
    from xagent.web.services.task_lease_service import acquire_task_lease

    factory, task_id = canonical
    monkeypatch.setattr(external_task_cancel, "get_session_local", lambda: factory)
    monkeypatch.setattr(
        task_execution.background_task_manager, "cancel_task", AsyncMock()
    )
    with factory() as db:
        task = db.get(Task, task_id)
        agent = Agent(user_id=task.user_id, name="Cancellation display agent")
        db.add(agent)
        db.flush()
        agent_id = int(agent.id)
        task.agent_id = agent_id
        task.source = "external"
        db.commit()
        lease = acquire_task_lease(db, task_id, new_run=True)
        db.refresh(task)
        version = task.state_version
    sink = AsyncMock()
    monkeypatch.setattr(task_events, "_task_event_sink", sink)
    for _ in range(2):
        await external_task_cancel.cancel_external_task_unserialized(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id=lease.run_id,
            expected_state_version=version,
        )
    with factory() as db:
        historical = load_event_display_snapshot(db, task_id).events[-1]
        committed_version = db.get(Task, task_id).state_version
    assert len(sink.await_args_list) == 4
    for display, terminal in zip(sink.await_args_list[::2], sink.await_args_list[1::2]):
        assert display.args[0] == historical
        assert terminal.args[0]["run_id"] == lease.run_id
        assert terminal.args[0]["state_version"] == committed_version
        assert terminal.args[0]["message_id"] == historical["data"]["message_id"]


@pytest.mark.asyncio
async def test_prelease_business_error_does_not_require_a_settlement(
    canonical, monkeypatch
):
    from unittest.mock import AsyncMock

    from xagent.web.services.task_event_display import publish_task_result
    from xagent.web.services.task_execution import create_terminal_task_error_event

    _, task_id = canonical
    notice = create_terminal_task_error_event(task_id, "Operation failed")
    sink = AsyncMock()
    monkeypatch.setattr(task_events, "_task_event_sink", sink)
    await publish_task_result(notice, task_id)
    sink.assert_awaited_once_with(notice, task_id)


@pytest.mark.asyncio
async def test_orchestrator_setup_failure_publishes_canonical_error_and_code(
    canonical, monkeypatch
):
    from unittest.mock import AsyncMock

    from xagent.web.services import task_orchestrator
    from xagent.web.services.llm_utils import AutoModelUnavailableError

    factory, task_id = canonical
    with factory() as db:
        user_id = int(db.get(Task, task_id).user_id)

    def fail_setup(*args, **kwargs):
        raise AutoModelUnavailableError("No eligible model")

    monkeypatch.setattr(task_orchestrator, "load_task_setup_snapshot_sync", fail_setup)
    sink = AsyncMock()
    monkeypatch.setattr(task_events, "_task_event_sink", sink)
    await task_orchestrator._schedule_bg(
        task_id=task_id,
        task_owner_user_id=user_id,
        task_source="web",
        payload=task_orchestrator.TaskTurnPayload("hello"),
        force_fresh=False,
        context=None,
    )
    with factory() as db:
        historical = load_event_display_snapshot(db, task_id).events[-1]
    assert len(sink.await_args_list) == 2
    assert sink.await_args_list[0].args[0] == historical
    terminal = sink.await_args_list[1].args[0]
    assert terminal["type"] == "task_error"
    assert terminal["code"] == "auto_model_unavailable"
    assert terminal["message_id"] == historical["data"]["message_id"]


@pytest.mark.parametrize("kind", ["input_accepted", "assistant_message"])
@pytest.mark.parametrize("field", ["content", "message_type"])
@pytest.mark.parametrize("value", [0, False, [], {}])
def test_message_validation_rejects_falsey_non_strings(canonical, kind, field, value):
    from xagent.web.services.task_event_display import load_live_display_event

    factory, task_id = canonical
    with factory() as db:
        payload = {
            "content": "hello",
            "message_type": "user"
            if kind == "input_accepted"
            else "assistant_response",
            "protocol_event_id": "invalid-message",
        }
        payload[field] = value
        row = fact(
            db, task_id, kind, "invalid-message", payload, flat=True, turn_id="turn"
        )
        db.commit()
        error = f"task_id={task_id}, event_id={row.event_id}.*invalid message content or type"
        with pytest.raises(ValueError, match=error):
            load_event_display_snapshot(db, task_id)
        with pytest.raises(ValueError, match=error):
            load_live_display_event(db, task_id, "invalid-message")


@pytest.mark.parametrize("kind", ["input_accepted", "assistant_message"])
@pytest.mark.parametrize(
    "payload",
    [{}, {"content": None, "message_type": None}, {"content": "", "message_type": ""}],
)
def test_message_empty_defaults_remain_supported(canonical, kind, payload):
    from xagent.web.services.client_error_messages import CLIENT_SAFE_TASK_FAILURE
    from xagent.web.services.task_event_display import load_live_display_event

    factory, task_id = canonical
    with factory() as db:
        fact(
            db,
            task_id,
            kind,
            "empty-message",
            {**payload, "protocol_event_id": "empty-message"},
            flat=True,
            turn_id="turn",
        )
        db.commit()
        snapshot = load_event_display_snapshot(db, task_id)
        expected = "" if kind == "input_accepted" else CLIENT_SAFE_TASK_FAILURE
        assert snapshot.messages[0]["content"] == expected
        assert snapshot.messages[0]["message_type"] == ""
        assert (
            load_live_display_event(db, task_id, "empty-message")["data"]["content"]
            == expected
        )
