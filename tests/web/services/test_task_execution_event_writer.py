from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.web.services.test_task_execution_event_store import engine as engine_fixture
from tests.web.services.test_task_execution_event_store import (
    task_id as task_id_fixture,
)
from xagent.core.agent.checkpoint import TraceCheckpointStore
from xagent.core.agent.runtime import PatternRuntime
from xagent.core.agent.trace import (
    ExecutionEventPersistenceError,
    TraceAction,
    TraceCategory,
    TraceEventType,
    TraceScope,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.services.chat_history_service import persist_user_message_no_commit
from xagent.web.services.managed_task_lease import finalize_managed_task_lease_result
from xagent.web.services.task_execution_controller import (
    TaskControlState,
    apply_task_control_transition,
)
from xagent.web.services.task_lease_service import acquire_task_lease
from xagent.web.tracing import ExecutionEventTraceAdapter, create_task_tracer

engine = engine_fixture
task_id = task_id_fixture


@pytest.fixture
def canonical(engine, task_id, monkeypatch):
    factory = sessionmaker(engine)
    with factory() as db:
        task = db.get(Task, task_id)
        task.conversation_storage_version = 2
        db.commit()
    monkeypatch.setattr("xagent.web.models.database.get_session_local", lambda: factory)
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.get_db", lambda: iter([factory()])
    )
    return factory, task_id


def facts(db, task_id):
    return list(
        db.scalars(
            sa.select(TaskExecutionEvent)
            .where(
                TaskExecutionEvent.task_id == task_id,
            )
            .order_by(TaskExecutionEvent.sequence)
        )
    )


def test_acceptance_and_compatibility_row_share_transaction(canonical):
    factory, task_id = canonical
    with factory() as db:
        task = db.get(Task, task_id)
        user_id = task.user_id
        persist_user_message_no_commit(
            db, task_id, user_id, "hello", turn_id="t1", attachments=[]
        )
        assert [e.kind for e in facts(db, task_id)] == ["input_accepted"]
        db.rollback()
        assert facts(db, task_id) == []
        assert db.query(TaskChatMessage).count() == 0
        first = persist_user_message_no_commit(
            db, task_id, user_id, "hello", turn_id="t1", attachments=[]
        )
        db.commit()
        second = persist_user_message_no_commit(
            db, task_id, user_id, "hello", turn_id="t1", attachments=[]
        )
        db.commit()
        assert first.id == second.id
        assert len(facts(db, task_id)) == 1
        assert second.attachments == []


@pytest.mark.asyncio
async def test_factory_commits_recoverable_state_before_observers(canonical):
    factory, task_id = canonical
    tracer = create_task_tracer(task_id)
    assert tracer.records_execution_events
    assert any(isinstance(h, ExecutionEventTraceAdapter) for h in tracer.handlers)
    observer = AsyncMock()
    tracer.handlers = [observer]
    payload = {
        "execution_id": "run-root",
        "pattern": "ReActPattern",
        "label": "after_llm",
        "context": {
            "messages": [
                {"role": "user", "content": "hello", "metadata": {"turn_id": "t1"}}
            ]
        },
        "pattern_state": {"adopted_plan": "完整计划" * 10000, "pending_tool_calls": []},
    }
    await TraceCheckpointStore(tracer).save(payload)
    with factory() as db:
        rows = facts(db, task_id)
        assert [e.kind for e in rows] == ["recovery_state", "input_applied"]
        assert rows[0].payload["data"]["snapshot"] == payload
        assert rows[1].payload["recovery_event_id"] == rows[0].event_id
        assert db.query(TraceEvent).count() == 1
    observer.handle_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_fact_commit_stops_runtime_before_broadcast(
    canonical, monkeypatch
):
    _, task_id = canonical
    tracer = create_task_tracer(task_id)
    observer = AsyncMock()
    tracer.handlers = [observer]

    def fail(*args, **kwargs):
        raise OSError("database unavailable")

    monkeypatch.setattr(
        "xagent.web.services.task_execution_event_writer.append_task_execution_event_no_commit",
        fail,
    )
    runtime = PatternRuntime(tracer=TraceCheckpointStore(tracer))
    with pytest.raises(ExecutionEventPersistenceError):
        await runtime.on_tool_start(
            tool_call={
                "id": "call1",
                "name": "write",
                "args": {},
                "tool_attempt_id": "attempt1",
                "assistant_message_id": "batch1",
            }
        )
    observer.handle_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_observer_failure_does_not_invalidate_fact(canonical):
    factory, task_id = canonical
    tracer = create_task_tracer(task_id)
    tracer.handlers = [
        AsyncMock(handle_event=AsyncMock(side_effect=OSError("socket closed")))
    ]
    await tracer.trace_event(
        TraceEventType(TraceScope.TASK, TraceAction.START, TraceCategory.GENERAL),
        task_id=str(task_id),
        data={"input": "hello"},
        require_persisted=True,
    )
    with factory() as db:
        assert len(facts(db, task_id)) == 1


@pytest.mark.asyncio
async def test_attempt_result_keeps_batch_identity_and_blocks_blind_replay(canonical):
    factory, task_id = canonical
    tracer = create_task_tracer(task_id)
    tracer.handlers = []
    runtime = PatternRuntime(tracer=TraceCheckpointStore(tracer))
    call = {
        "id": "provider-duplicate-id",
        "name": "write",
        "args": {"value": 1},
        "tool_attempt_id": "attempt1",
        "assistant_message_id": "batch1",
    }
    await runtime.on_tool_start(tool_call=call)
    await runtime.on_tool_end(
        tool_call=call, result={"success": True, "output": "长结果" * 10000}
    )
    with pytest.raises(ExecutionEventPersistenceError):
        await runtime.on_tool_start(tool_call=call)
    with factory() as db:
        rows = facts(db, task_id)
        assert len(rows) == 2
        assert {r.assistant_message_id for r in rows} == {"batch1"}
        assert {r.tool_attempt_id for r in rows} == {"attempt1"}
        assert rows[1].payload["data"]["result"]["output"] == "长结果" * 10000


@pytest.mark.parametrize(
    "status",
    [
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.PAUSED,
        TaskStatus.WAITING_FOR_USER,
    ],
)
def test_channel_outcome_and_transcript_commit_with_lease(canonical, status):
    factory, task_id = canonical
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)
        assert lease is not None
        assert finalize_managed_task_lease_result(
            db,
            lease,
            status=status,
            assistant_content="result",
            execution_result={"output": "result"},
        )
        rows = facts(db, task_id)
        assert rows[-1].kind == "execution_settled"
        assert rows[-1].payload["status"] == status.value
        assert any(row.kind == "assistant_message" for row in rows)
        assert db.get(Task, task_id).runner_id is None


def test_control_and_event_rollback_together(canonical):
    factory, task_id = canonical
    with factory() as db:
        task = db.get(Task, task_id)
        apply_task_control_transition(
            task, TaskControlState.PAUSED, status=TaskStatus.PAUSED
        )
        assert facts(db, task_id)[-1].kind == "control_state_changed"
        db.rollback()
        assert facts(db, task_id) == []
        assert db.get(Task, task_id).status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_real_react_loop_records_batch_before_tool_and_retains_identity(
    canonical,
):
    from tests.core.agent.test_react import FakeLLM, FakeTool
    from xagent.core.agent import ExecutionContext, ReActPattern

    factory, task_id = canonical
    tracer = create_task_tracer(task_id)
    tracer.handlers = []
    runtime = PatternRuntime(tracer=TraceCheckpointStore(tracer))
    context = ExecutionContext(system_prompt="Calculate")
    context.add_user_message("2+2", metadata={"turn_id": "real-turn"})
    tool = FakeTool()
    pattern = ReActPattern(max_iterations=3)
    result = await pattern.run(
        context=context,
        tools=[tool],
        runtime=runtime,
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
    assert result["success"]
    assert len(tool.calls) == 1
    with factory() as db:
        rows = facts(db, task_id)
        tools = [row for row in rows if row.kind.startswith("tool_execution_")]
        state = next(
            row
            for row in rows
            if row.kind == "recovery_state"
            and row.payload["data"]["snapshot"]["label"] == "after_llm"
        )
        snapshot = state.payload["data"]["snapshot"]
        saved_call = snapshot["pattern_state"]["pending_tool_calls"][0]
        assert state.sequence < tools[0].sequence < tools[1].sequence
        assert {row.tool_attempt_id for row in tools} == {saved_call["tool_attempt_id"]}
        assert {row.assistant_message_id for row in tools} == {
            saved_call["assistant_message_id"]
        }
        restored = ReActPattern()
        restored.load_state(snapshot["pattern_state"])
        assert (
            restored.pending_tool_calls[0]["tool_attempt_id"]
            == saved_call["tool_attempt_id"]
        )


@pytest.mark.asyncio
async def test_outbound_stream_is_committed_before_websocket_and_failure_is_strict(
    canonical, monkeypatch
):
    from xagent.web.api import websocket

    factory, task_id = canonical
    monkeypatch.setattr(websocket, "get_db", lambda: iter([factory()]))
    broadcasts = []

    async def broadcast(event, task_id):
        with factory() as db:
            assert facts(db, task_id)[-1].payload["data"]["delta"] == "hello"
        broadcasts.append(event)

    monkeypatch.setattr(websocket.manager, "broadcast_to_task", broadcast)
    handler = websocket.make_agent_outbound_handler(task_id, authoritative=True)
    payload = {"type": "final_answer_delta", "delta": "hello", "stream_id": "stream1"}
    await handler(payload)
    assert len(broadcasts) == 1

    def fail(*args, **kwargs):
        raise OSError("write failure")

    monkeypatch.setattr(
        "xagent.web.services.task_execution_event_writer.append_task_execution_event_no_commit",
        fail,
    )
    with pytest.raises(ExecutionEventPersistenceError):
        await handler(payload)
    assert len(broadcasts) == 1


def test_compatibility_failure_rolls_back_outcome_and_retains_lease(
    canonical, monkeypatch
):
    factory, task_id = canonical
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)

        def fail(*args, **kwargs):
            raise OSError("event storage unavailable")

        monkeypatch.setattr(
            "xagent.web.services.task_execution_event_writer.append_task_execution_event_no_commit",
            fail,
        )
        with pytest.raises(OSError):
            finalize_managed_task_lease_result(
                db, lease, status=TaskStatus.COMPLETED, assistant_content="result"
            )
        db.expire_all()
        task = db.get(Task, task_id)
        assert task.status == TaskStatus.RUNNING
        assert task.runner_id == lease.runner_id
        assert db.query(TaskChatMessage).count() == 0
        assert facts(db, task_id) == []


@pytest.mark.asyncio
async def test_replaced_lease_cannot_append_or_broadcast(canonical):
    from xagent.web.services.task_lease_service import bind_task_lease_context

    factory, task_id = canonical
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)
        db.get(Task, task_id).run_id = "replacement-run"
        db.commit()
    tracer = create_task_tracer(task_id)
    observer = AsyncMock()
    tracer.handlers = [observer]
    with bind_task_lease_context(lease):
        with pytest.raises(ExecutionEventPersistenceError):
            await tracer.trace_event(
                TraceEventType(
                    TraceScope.TASK, TraceAction.START, TraceCategory.GENERAL
                ),
                task_id=str(task_id),
            )
    observer.handle_event.assert_not_awaited()
    with factory() as db:
        assert facts(db, task_id) == []


def test_assistant_projection_replay_has_one_fact_and_one_row(canonical):
    from xagent.web.services.chat_history_service import (
        persist_assistant_message_no_commit,
    )

    factory, task_id = canonical
    with factory() as db:
        task = db.get(Task, task_id)
        task.run_id = "run1"
        task.status = TaskStatus.COMPLETED
        db.commit()
        for _ in range(2):
            persist_assistant_message_no_commit(
                db,
                task_id,
                task.user_id,
                "done",
                message_type="assistant_response",
                content_is_reconciled=True,
            )
            db.commit()
        rows = facts(db, task_id)
        assert len(rows) == 1
        assert db.query(TaskChatMessage).one().execution_event_id == rows[0].event_id


def test_command_fact_and_inbox_are_one_transaction(canonical):
    from xagent.web.models.task_command import TaskExecutionCommand
    from xagent.web.services.task_command_transport import (
        TaskCommandKind,
        stage_task_command,
    )

    factory, task_id = canonical
    with factory() as db:
        task = db.get(Task, task_id)
        stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=task.user_id,
            command_id="answer1",
            kind=TaskCommandKind.RESUME,
            payload={"response": "yes"},
        )
        assert facts(db, task_id)[-1].payload["payload"] == {"response": "yes"}
        db.rollback()
        assert facts(db, task_id) == []
        assert db.query(TaskExecutionCommand).count() == 0


def test_pre_runner_failure_is_a_fact_and_failed_commit_is_not_broadcastable(
    canonical, monkeypatch
):
    from xagent.web.api.websocket import _terminal_task_error_payload

    factory, task_id = canonical
    monkeypatch.setattr("xagent.web.api.websocket.get_session_local", lambda: factory)
    _terminal_task_error_payload(task_id, "sandbox unavailable")
    with factory() as db:
        rows = facts(db, task_id)
        assert rows[-1].kind == "execution_settled"
        assert rows[-1].payload["result"]["error"] == "sandbox unavailable"
        assert rows[-1].payload["status"] == TaskStatus.FAILED.value

    def fail(*args, **kwargs):
        raise OSError("commit failure")

    monkeypatch.setattr(
        "xagent.web.services.task_execution_event_writer.append_task_execution_event_no_commit",
        fail,
    )
    with pytest.raises(ExecutionEventPersistenceError):
        _terminal_task_error_payload(task_id, "sandbox unavailable")


def test_settlement_serializes_execution_context_without_losing_state(canonical):
    from xagent.core.agent import ExecutionContext
    from xagent.web.services.task_execution_event_writer import (
        stage_result_fact_no_commit,
    )

    factory, task_id = canonical
    context = ExecutionContext(execution_id="run-context")
    context.add_assistant_message("完整回复" * 10000)
    with factory() as db:
        task = db.get(Task, task_id)
        task.status = TaskStatus.COMPLETED
        stage_result_fact_no_commit(db, task, {"agent_result": {"context": context}})
        db.commit()
        assert (
            facts(db, task_id)[0].payload["result"]["agent_result"]["context"]
            == context.to_dict()
        )
        with pytest.raises(TypeError, match="Unsupported execution fact"):
            stage_result_fact_no_commit(db, task, {"unknown": object()})
