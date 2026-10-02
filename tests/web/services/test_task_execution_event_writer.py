from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.web.services.test_task_execution_event_store import engine as engine_fixture
from tests.web.services.test_task_execution_event_store import (
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
    # Use this fixture's session factory, not a prior test's global async engine.
    # The runtime compatibility test below supplies both backends explicitly.
    monkeypatch.setenv("XAGENT_ASYNC_TRACE_DB_ENABLED", "false")
    factory = sessionmaker(engine)
    with factory() as db:
        task = db.get(Task, task_id)
        task.conversation_storage_version = 2
        db.commit()
    monkeypatch.setattr("xagent.web.models.database.get_session_local", lambda: factory)
    monkeypatch.setattr(
        "xagent.web.services.trace_handlers.get_db", lambda: iter([factory()])
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
    tracer.handlers.append(observer)
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
    tracer.handlers = [
        h for h in tracer.handlers if isinstance(h, ExecutionEventTraceAdapter)
    ]
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
    tracer.handlers = [
        h for h in tracer.handlers if isinstance(h, ExecutionEventTraceAdapter)
    ]
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
    from xagent.web.services import task_execution as websocket

    factory, task_id = canonical
    monkeypatch.setattr(websocket, "get_db", lambda: iter([factory()]))
    broadcasts = []

    async def broadcast(event, task_id):
        with factory() as db:
            assert facts(db, task_id)[-1].payload["data"]["content"] == "hello"
        broadcasts.append(event)

    monkeypatch.setattr(websocket, "publish_task_event", broadcast)
    handler = websocket.make_agent_outbound_handler(task_id, authoritative=True)
    payload = {
        "type": "final_answer_end",
        "content": "hello",
        "stream_id": "stream1",
        "message_id": "final_answer_1",
    }
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
    from xagent.web.services.task_execution import _terminal_task_error_payload

    factory, task_id = canonical
    monkeypatch.setattr(
        "xagent.web.services.task_execution.get_session_local", lambda: factory
    )
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


@pytest.mark.asyncio
@pytest.mark.parametrize("native_async", [False, True], ids=["sync", "async"])
async def test_fact_and_checkpoint_projection_use_current_trace_database_runtime(
    canonical, engine, monkeypatch, native_async
):
    from xagent.web.services import trace_handlers
    from xagent.web.services.task_lease_service import bind_task_lease_context
    from xagent.web.services.trace_database import TraceDatabaseRuntime

    factory, task_id = canonical
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
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)
    tracer = create_task_tracer(task_id)
    store = TraceCheckpointStore(tracer)
    payload = {
        "execution_id": "root",
        "context": {},
        "label": "after_llm",
        "pattern_state": {"pending_tool_calls": []},
    }
    try:
        with bind_task_lease_context(lease):
            await store.save(payload)
            assert await tracer.load_latest_checkpoint("root") == payload
        with factory() as db:
            task = db.get(Task, task_id)
            assert task.last_checkpoint_trace_event_id is not None
            row = db.get(TraceEvent, task.last_checkpoint_trace_event_id)
            event = next(e for e in facts(db, task_id) if e.kind == "recovery_state")
            assert row.event_id == event.payload["protocol_event_id"]
            assert event.run_id == lease.run_id
            assert event.payload["data"]["snapshot"] == payload
    finally:
        await database_runtime.close()
        if source is not engine:
            source.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["root", "child", "outbound"])
async def test_replaced_attempt_in_same_run_cannot_write_facts(
    canonical, producer, monkeypatch
):
    from xagent.web.services import task_execution
    from xagent.web.services.task_lease_service import bind_task_lease_context

    factory, task_id = canonical
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)
        task = db.get(Task, task_id)
        task.lease_attempt_id = "replacement-attempt"
        db.commit()
    monkeypatch.setattr(task_execution, "get_db", lambda: iter([factory()]))
    broadcast = AsyncMock()
    monkeypatch.setattr(task_execution, "publish_task_event", broadcast)
    with bind_task_lease_context(lease):
        with pytest.raises(ExecutionEventPersistenceError):
            if producer == "outbound":
                await task_execution.make_agent_outbound_handler(
                    task_id, authoritative=True
                )({"type": "final_answer_delta", "delta": "stale"})
            else:
                adapter = ExecutionEventTraceAdapter(
                    task_id, build_id="child" if producer == "child" else None
                )
                from xagent.core.agent.trace import TraceEvent as CoreTraceEvent

                await adapter.commit_event(
                    CoreTraceEvent(
                        TraceEventType(
                            TraceScope.TASK, TraceAction.START, TraceCategory.GENERAL
                        ),
                        task_id=str(task_id),
                    )
                )
    broadcast.assert_not_awaited()
    with factory() as db:
        assert facts(db, task_id) == []
        assert db.query(TraceEvent).count() == 0


@pytest.mark.asyncio
async def test_outbound_question_preserves_source_identity_in_atomic_projection(
    canonical, monkeypatch
):
    from xagent.web.services import task_execution

    factory, task_id = canonical
    monkeypatch.setattr(task_execution, "get_db", lambda: iter([factory()]))
    monkeypatch.setattr(task_execution, "publish_task_event", AsyncMock())
    await task_execution.make_agent_outbound_handler(task_id, authoritative=True)(
        {
            "event_id": "question-event",
            "message": "Which file?",
            "expect_response": True,
        }
    )
    with factory() as db:
        assert len(facts(db, task_id)) == 1
        row = db.query(TaskChatMessage).one()
        assert row.source_event_id == "question-event"
        event = next(e for e in facts(db, task_id) if e.kind == "agent_message")
        assert row.execution_event_id == event.event_id
        assert event.payload["protocol_event_id"] == "question-event"
        assert db.query(TraceEvent).one().event_id == "question-event"


@pytest.mark.parametrize("status", [TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER])
def test_repeated_resting_settlements_keep_distinct_facts_in_same_run(
    canonical, status
):
    from xagent.web.services.task_execution_event_writer import (
        stage_result_fact_no_commit,
    )

    factory, task_id = canonical
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)
        run_id = lease.run_id
        for index in range(2):
            if index:
                lease = acquire_task_lease(db, task_id, expected_run_id=run_id)
            assert lease.run_id == run_id
            result = {"output": f"rest {index}"}
            assert finalize_managed_task_lease_result(
                db,
                lease,
                status=status,
                assistant_content=result["output"],
                execution_result=result,
            )
            # Replaying the same settlement keeps its original identity.
            stage_result_fact_no_commit(db, db.get(Task, task_id), result)
            db.commit()
        settled = [e for e in facts(db, task_id) if e.kind == "execution_settled"]
        assert len(settled) == 2
        assert [e.payload["result"]["output"] for e in settled] == ["rest 0", "rest 1"]
        assert db.query(TaskChatMessage).count() == 2
        assert db.get(Task, task_id).runner_id is None


def test_racing_message_claim_has_only_one_delivery_owner(canonical, engine):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread

    from xagent.web.services.chat_history_service import (
        claim_user_message_delivery_no_commit,
    )

    factory, task_id = canonical
    waiting = Event()

    def before_execute(connection, cursor, statement, parameters, context, many):
        if (
            current_thread().name.startswith("second-claim")
            and "UPDATE tasks" in statement
            and "conversation_event_sequence" in statement
        ):
            waiting.set()

    sa.event.listen(engine, "before_cursor_execute", before_execute)
    try:
        with (
            factory() as winner,
            ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="second-claim"
            ) as pool,
        ):
            user_id = winner.get(Task, task_id).user_id
            first = claim_user_message_delivery_no_commit(
                winner, task_id, user_id, "one input", turn_id="same-turn"
            )
            assert first.claimed

            def second_claim():
                with factory() as db:
                    claim = claim_user_message_delivery_no_commit(
                        db, task_id, user_id, "one input", turn_id="same-turn"
                    )
                    result = claim.claimed, claim.payload_matches
                    db.commit()
                    return result

            future = pool.submit(second_claim)
            try:
                assert waiting.wait(5), "second claimant did not reach the task lock"
            finally:
                winner.commit()
            assert future.result(timeout=5) == (False, True)
        with factory() as db:
            assert db.query(TaskChatMessage).count() == 1
            assert (
                len([e for e in facts(db, task_id) if e.kind == "input_accepted"]) == 1
            )
    finally:
        sa.event.remove(engine, "before_cursor_execute", before_execute)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_action", [TraceAction.START, TraceAction.END])
async def test_compaction_fact_failure_does_not_fall_back_or_mutate_context(
    failed_action,
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from xagent.core.agent.trace import Tracer

    failure = ExecutionEventPersistenceError("one uncertain compaction fact")
    written = []

    async def writer(event):
        written.append(event)
        if (
            event.event_type.category == TraceCategory.LLM
            and event.event_type.action == failed_action
        ):
            raise failure

    tracer = Tracer()
    tracer.event_writer = writer
    runtime = PatternRuntime(tracer=tracer)
    context = SimpleNamespace(
        execution_id="compaction",
        messages=["preserve"],
        metadata={},
        build_llm_compact_request_if_needed=Mock(
            return_value={"messages": [], "max_tokens": 100, "metadata": {}}
        ),
        compact_with_llm_response=Mock(),
        compact_if_needed=Mock(),
    )
    llm = SimpleNamespace(chat=AsyncMock(return_value={"content": "summary"}))
    with pytest.raises(ExecutionEventPersistenceError) as caught:
        await runtime.compact_context_if_needed(context=context, llm=llm)
    assert caught.value is failure
    context.compact_if_needed.assert_not_called()
    context.compact_with_llm_response.assert_not_called()
    assert context.messages == ["preserve"]
    assert len(written) == (1 if failed_action == TraceAction.START else 2)


@pytest.mark.postgresql
def test_trace_fact_and_command_acceptance_do_not_deadlock(canonical, engine):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread

    from xagent.core.agent.trace import TraceEvent as CoreTraceEvent
    from xagent.web.services.task_command_transport import (
        TaskCommandKind,
        enqueue_task_command,
    )
    from xagent.web.services.task_lease_service import bind_task_lease_context

    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL foreign-key row locks")
    factory, task_id = canonical
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)
        user_id = db.get(Task, task_id).user_id
    trace_locked = Event()
    command_waiting = Event()

    def after_execute(connection, cursor, statement, parameters, context, many):
        if (
            current_thread().name.startswith("trace-fact")
            and "UPDATE tasks" in statement
            and not trace_locked.is_set()
        ):
            trace_locked.set()
            assert command_waiting.wait(5)

    def before_execute(connection, cursor, statement, parameters, context, many):
        if (
            current_thread().name.startswith("accept-command")
            and "UPDATE tasks" in statement
            and "conversation_event_sequence" in statement
        ):
            # The command INSERT has already acquired the task FK KEY SHARE.
            command_waiting.set()

    def write_trace():
        with factory() as db, bind_task_lease_context(lease):
            db.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            ExecutionEventTraceAdapter(task_id)._save_trace_event(
                db,
                CoreTraceEvent(
                    TraceEventType(
                        TraceScope.TASK, TraceAction.START, TraceCategory.GENERAL
                    ),
                    task_id=str(task_id),
                ),
            )
            db.commit()

    def accept_command():
        assert trace_locked.wait(5)
        with factory() as db:
            db.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            enqueue_task_command(
                db,
                task_id=task_id,
                actor_user_id=user_id,
                command_id="concurrent-command",
                kind=TaskCommandKind.RESUME,
                payload={},
            )

    sa.event.listen(engine, "after_cursor_execute", after_execute)
    sa.event.listen(engine, "before_cursor_execute", before_execute)
    try:
        with (
            ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="trace-fact"
            ) as writers,
            ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="accept-command"
            ) as commands,
        ):
            writer = writers.submit(write_trace)
            command = commands.submit(accept_command)
            writer.result(timeout=10)
            command.result(timeout=10)
        with factory() as db:
            assert len(facts(db, task_id)) == 2
            assert db.query(TraceEvent).count() == 1
    finally:
        sa.event.remove(engine, "after_cursor_execute", after_execute)
        sa.event.remove(engine, "before_cursor_execute", before_execute)


def test_withdrawn_input_reuses_acceptance_when_handed_to_new_run(canonical):
    from xagent.web.services.chat_history_service import (
        claim_user_message_delivery_no_commit,
        withdraw_pending_user_message_delivery_sync,
    )

    factory, tid = canonical
    with factory() as db:
        task = db.get(Task, tid)
        task.run_id = "ended-run"
        claim = claim_user_message_delivery_no_commit(
            db, tid, task.user_id, "new input", turn_id="turn"
        )
        assert claim.claimed
        db.commit()
        event_id = claim.message.execution_event_id
        user_id = task.user_id
    assert withdraw_pending_user_message_delivery_sync(tid, "turn")
    with factory() as db:
        db.get(Task, tid).run_id = "new-run"
        row = persist_user_message_no_commit(
            db, tid, user_id, "new input", turn_id="turn", delivery_status="pending"
        )
        db.commit()
        assert row.execution_event_id == event_id
        assert len(facts(db, tid)) == 1
        assert facts(db, tid)[0].run_id == "ended-run"
        assert db.query(TaskChatMessage).count() == 1


def test_orphan_delivery_fact_and_projection_are_atomic(canonical, monkeypatch):
    from xagent.web.services import task_execution_event_writer as writer
    from xagent.web.services.task_lease_recovery import (
        reconcile_orphaned_pending_deliveries_no_commit,
    )
    from xagent.web.services.task_lease_service import utc_now

    factory, tid = canonical
    with factory() as db:
        task = db.get(Task, tid)
        task.status = TaskStatus.COMPLETED
        persist_user_message_no_commit(
            db,
            tid,
            task.user_id,
            "pending input",
            turn_id="turn",
            delivery_status="pending",
        )
        db.commit()
        original = writer.stage_delivery_fact_no_commit

        def fail_after_fact(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError("failed transaction")

        with monkeypatch.context() as failure:
            failure.setattr(writer, "stage_delivery_fact_no_commit", fail_after_fact)
            with pytest.raises(OSError):
                reconcile_orphaned_pending_deliveries_no_commit(
                    db, now=utc_now(), task_id=tid
                )
            db.rollback()
        assert db.query(TaskChatMessage).one().delivery_status == "pending"
        assert len(facts(db, tid)) == 1
        assert (
            reconcile_orphaned_pending_deliveries_no_commit(
                db, now=utc_now(), task_id=tid
            )
            == 1
        )
        db.commit()
        db.expire_all()
        assert db.query(TaskChatMessage).one().delivery_status == "dispatched"
        assert facts(db, tid)[-1].payload == {"status": "dispatched"}
        assert (
            reconcile_orphaned_pending_deliveries_no_commit(
                db, now=utc_now(), task_id=tid
            )
            == 0
        )
        assert len(facts(db, tid)) == 2


@pytest.mark.parametrize("with_checkpoint", [False, True])
def test_expired_lease_outcome_fact_and_task_are_atomic(
    canonical, monkeypatch, with_checkpoint
):
    from datetime import timedelta

    from xagent.core.agent.checkpoint import CHECKPOINT_TYPE
    from xagent.web.services import task_execution_event_writer as writer
    from xagent.web.services.task_lease_recovery import (
        TASK_LEASE_EXPIRED_ERROR,
        recover_task_lease_candidate_no_commit,
    )
    from xagent.web.services.task_lease_service import (
        get_expired_task_lease_candidates,
        utc_now,
    )

    factory, tid = canonical
    with factory() as db:
        task = db.get(Task, tid)
        task.status = TaskStatus.RUNNING
        task.control_state = "running"
        task.run_id = "expired-run"
        task.runner_id = "dead-runner"
        task.lease_expires_at = utc_now() - timedelta(seconds=5)
        if with_checkpoint:
            writer.append_fact_no_commit(
                db,
                task_id=tid,
                kind="recovery_state",
                key="expired-checkpoint",
                run_id="expired-run",
                payload={
                    "protocol_event_id": "expired-checkpoint",
                    "data": {
                        "checkpoint_type": CHECKPOINT_TYPE,
                        "snapshot_schema_version": 1,
                        "execution_id": str(tid),
                        "snapshot": {
                            "execution_id": str(tid),
                            "context": {"messages": []},
                        },
                    },
                },
            )
            task.last_checkpoint_event_id = "expired-checkpoint"
        db.commit()
        candidate = get_expired_task_lease_candidates(db, cutoff=utc_now(), limit=1)[0]
        original = writer.stage_result_fact_no_commit

        def fail_after_fact(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError("failed recovery transaction")

        with monkeypatch.context() as failure:
            failure.setattr(writer, "stage_result_fact_no_commit", fail_after_fact)
            with pytest.raises(OSError, match="failed recovery transaction"):
                recover_task_lease_candidate_no_commit(
                    db, candidate, recovered_at=utc_now()
                )
            db.rollback()
        assert db.get(Task, tid).status == TaskStatus.RUNNING
        assert db.get(Task, tid).runner_id == "dead-runner"
        assert db.get(Task, tid).state_version == candidate.state_version
        assert len(facts(db, tid)) == int(with_checkpoint)

        expected = TaskStatus.PAUSED if with_checkpoint else TaskStatus.FAILED
        assert (
            recover_task_lease_candidate_no_commit(
                db, candidate, recovered_at=utc_now()
            )
            == expected
        )
        db.commit()
        assert db.get(Task, tid).status == expected
        assert db.get(Task, tid).runner_id is None
        assert len(facts(db, tid)) == 1 + int(with_checkpoint)
        outcome = facts(db, tid)[-1]
        assert outcome.kind == "execution_settled"
        assert outcome.run_id == "expired-run"
        assert outcome.payload == {
            "status": expected.value,
            "result": {"error": None if with_checkpoint else TASK_LEASE_EXPIRED_ERROR},
        }
        assert (
            recover_task_lease_candidate_no_commit(
                db, candidate, recovered_at=utc_now()
            )
            is None
        )
        assert len(facts(db, tid)) == 1 + int(with_checkpoint)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,field", [(TraceAction.START, "messages"), (TraceAction.END, "response")]
)
async def test_llm_projection_is_capped_while_fact_stays_complete(
    canonical, monkeypatch, action, field
):
    factory, task_id = canonical
    monkeypatch.setenv("XAGENT_MAX_TRACE_PAYLOAD_BYTES", "50000")
    content = "x" * 200_000
    value = [{"role": "user", "content": content}] if field == "messages" else content
    runtime = PatternRuntime(tracer=create_task_tracer(task_id))
    await runtime._emit_trace_event(
        TraceEventType(TraceScope.ACTION, action, TraceCategory.LLM),
        task_id=str(task_id),
        step_id="llm-step",
        data={field: value, "model": "test-model"},
    )
    with factory() as db:
        fact = facts(db, task_id)[0]
        projection = db.query(TraceEvent).filter_by(task_id=task_id).one()
        assert fact.payload["data"][field] == value
        assert len(json.dumps(projection.data).encode()) <= 50000
        assert projection.data["model"] == "test-model"


@pytest.mark.asyncio
async def test_stream_deltas_publish_without_durable_writes(
    canonical, monkeypatch, engine
):
    from xagent.web.services import task_execution as websocket
    from xagent.web.services.task_lease_service import bind_task_lease_context

    factory, task_id = canonical
    monkeypatch.setattr(websocket, "get_db", lambda: iter([factory()]))
    publish = AsyncMock()
    monkeypatch.setattr(websocket, "publish_task_event", publish)
    handler = websocket.make_agent_outbound_handler(task_id, authoritative=True)
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)
    statements = []

    def record(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    sa.event.listen(engine, "before_cursor_execute", record)
    try:
        with bind_task_lease_context(lease):
            for chunk in ("hello", " ", "world"):
                await handler(
                    {
                        "type": "final_answer_delta",
                        "delta": chunk,
                        "stream_id": "s",
                        "message_id": "final_answer_1",
                    }
                )
    finally:
        sa.event.remove(engine, "before_cursor_execute", record)
    assert [call.args[0]["delta"] for call in publish.await_args_list] == [
        "hello",
        " ",
        "world",
    ]
    assert statements and all(
        statement.lstrip().upper().startswith("SELECT") for statement in statements
    )
    with factory() as db:
        assert facts(db, task_id) == []
    with bind_task_lease_context(lease):
        await handler(
            {
                "type": "final_answer_end",
                "content": "hello world",
                "stream_id": "s",
                "message_id": "final_answer_1",
            }
        )
    with factory() as db:
        assert [row.kind for row in facts(db, task_id)] == ["final_answer_end"]
        assert facts(db, task_id)[0].payload["data"]["content"] == "hello world"
