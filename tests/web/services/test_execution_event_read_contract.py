"""Reader prerequisites exercised through the real V2 persistence adapter."""

from copy import deepcopy

import pytest
import sqlalchemy as sa

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
from xagent.core.agent.checkpoint import ExecutionEventPersistenceError
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
from xagent.web.services.task_execution_event_writer import append_fact_no_commit
from xagent.web.tracing import ExecutionEventTraceAdapter, task_database_handler

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


def compact_event(task_id, data):
    return TraceEvent(
        TraceEventType(TraceScope.ACTION, TraceAction.END, TraceCategory.COMPACT),
        task_id=str(task_id),
        step_id="compact-step",
        data=data,
    )


@pytest.mark.asyncio
async def test_summary_retains_event_coordinate_without_changing_legacy_payload(
    canonical,
):
    factory, task_id = canonical
    with factory() as db:
        append_fact_no_commit(
            db,
            task_id=task_id,
            scope_id="child-1",
            kind="task_start",
            key="child",
            payload={},
        )
        message = persist_user_message_no_commit(
            db, task_id, db.get(Task, task_id).user_id, "earlier", turn_id="turn-1"
        )
        db.flush()
        data = {"summary": "Earlier conversation", "watermark_message_id": message.id}
        anchor = next(row for row in facts(db, task_id) if row.kind == "input_accepted")
        expected = {
            "scope_id": "root",
            "event_id": anchor.event_id,
            "sequence": anchor.sequence,
        }
        assert message.id != anchor.sequence
        persist_user_message_no_commit(
            db, task_id, db.get(Task, task_id).user_id, "later", turn_id="turn-2"
        )
        db.commit()
    original_data = deepcopy(data)
    event = compact_event(task_id, data)
    adapter = ExecutionEventTraceAdapter(task_id)
    await adapter.commit_event(event)
    with factory() as db:
        summary = facts(db, task_id)[-1]
        assert summary.payload["transcript_watermark"] == expected
        assert summary.payload["data"] == original_data
        assert summary.sequence > expected["sequence"] + 1
        legacy = db.query(StoredTraceEvent).filter_by(event_id=event.id).one()
        assert legacy.data == original_data
        db.execute(sa.delete(TaskChatMessage).where(TaskChatMessage.task_id == task_id))
        db.commit()
    # A commit-confirmation retry must not need the old projection again.
    await adapter.commit_event(event)
    assert event.data == original_data
    with factory() as db:
        assert (
            len([row for row in facts(db, task_id) if row.kind == "action_end_compact"])
            == 1
        )
        assert facts(db, task_id)[-1].payload["transcript_watermark"] == expected


@pytest.mark.asyncio
async def test_unresolvable_summary_coordinate_rolls_back_without_publishing_fact(
    canonical,
):
    factory, task_id = canonical
    adapter = ExecutionEventTraceAdapter(task_id)
    event = compact_event(
        task_id, {"summary": "Earlier", "watermark_message_id": 999999}
    )
    with pytest.raises(ExecutionEventPersistenceError):
        await adapter.commit_event(event)
    with factory() as db:
        assert facts(db, task_id) == []
        assert db.query(StoredTraceEvent).count() == 0
        assert db.get(Task, task_id).conversation_event_sequence == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        {"summary": "Earlier"},
        {"watermark_message_id": 1},
        {"summary": "", "watermark_message_id": 1},
        {"summary": "Earlier", "watermark_message_id": True},
    ],
)
async def test_unpositioned_compaction_does_not_invent_coverage(canonical, data):
    factory, task_id = canonical
    await ExecutionEventTraceAdapter(task_id).commit_event(compact_event(task_id, data))
    with factory() as db:
        assert "transcript_watermark" not in facts(db, task_id)[0].payload
        assert facts(db, task_id)[0].payload["data"] == data


@pytest.mark.asyncio
async def test_child_summary_does_not_adopt_root_transcript_coordinate(canonical):
    factory, task_id = canonical
    data = {"summary": "Child context", "watermark_message_id": 999999}
    await ExecutionEventTraceAdapter(task_id, build_id="child-1").commit_event(
        compact_event(task_id, data)
    )
    with factory() as db:
        row = facts(db, task_id)[0]
        assert row.scope_id == "child-1"
        assert "transcript_watermark" not in row.payload


@pytest.mark.asyncio
async def test_preexisting_summary_replay_preserves_older_envelope(canonical):
    factory, task_id = canonical
    event = compact_event(
        task_id, {"summary": "Old summary", "watermark_message_id": 999999}
    )
    with factory() as db:
        prior = append_fact_no_commit(
            db,
            task_id=task_id,
            kind="action_end_compact",
            key=f"runtime:{event.id}",
            payload={
                "data": event.data,
                "step_id": event.step_id,
                "protocol_event_id": event.id,
                "event_type": "action_end_compact",
                "parent_event_id": None,
            },
        )
        prior_id = prior.event_id
        db.commit()
    await ExecutionEventTraceAdapter(task_id).commit_event(event)
    with factory() as db:
        rows = facts(db, task_id)
        assert len(rows) == 1
        assert rows[0].event_id == prior_id
        assert "transcript_watermark" not in rows[0].payload


@pytest.mark.asyncio
async def test_v1_compaction_keeps_legacy_path(engine, task_id, monkeypatch):
    from sqlalchemy.orm import sessionmaker

    factory = sessionmaker(engine)
    monkeypatch.setenv("XAGENT_ASYNC_TRACE_DB_ENABLED", "false")
    monkeypatch.setattr("xagent.web.models.database.get_session_local", lambda: factory)
    monkeypatch.setattr(
        "xagent.web.services.trace_handlers.get_db", lambda: iter([factory()])
    )
    event = compact_event(
        task_id, {"summary": "Old summary", "watermark_message_id": 999999}
    )
    await task_database_handler(task_id).handle_event(event)
    with factory() as db:
        assert db.query(TaskExecutionEvent).count() == 0
        assert db.query(StoredTraceEvent).one().data == event.data


@pytest.mark.asyncio
async def test_applied_turns_survive_compaction_without_legacy_content(canonical):
    from types import SimpleNamespace

    from xagent.core.agent import ContextManager, ExecutionContext
    from xagent.core.agent.checkpoint import TraceCheckpointStore
    from xagent.core.agent.context.execution import ACCEPTED_TURN_IDS_METADATA_KEY
    from xagent.core.agent.runner import AgentRunner, UserMessageInjectionOutcome
    from xagent.web.tracing import create_task_tracer

    factory, task_id = canonical
    tracer = create_task_tracer(task_id)
    tracer.handlers = [task_database_handler(task_id)]
    context = ExecutionContext(execution_id="compacted-inputs")
    manager = ContextManager()
    manager.set_context(context)
    runner = AgentRunner(
        SimpleNamespace(llm=None),
        tracer=TraceCheckpointStore(tracer),
        context_manager=manager,
    )
    for turn, content in [("turn-first", "Choose B"), ("turn-last", "Then C")]:
        with factory() as db:
            persist_user_message_no_commit(
                db, task_id, db.get(Task, task_id).user_id, content, turn_id=turn
            )
            db.commit()
        result = await runner.inject_user_message(
            context.execution_id, content, turn_id=turn, request_interrupt=False
        )
        assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH

    context.compact_with_llm_response("Earlier choice: B")
    await TraceCheckpointStore(tracer).save(
        {
            "execution_id": context.execution_id,
            "context": context.to_dict(),
            "label": "compacted",
        }
    )
    with factory() as db:
        db.execute(sa.delete(TaskChatMessage).where(TaskChatMessage.task_id == task_id))
        db.execute(
            sa.delete(StoredTraceEvent).where(StoredTraceEvent.task_id == task_id)
        )
        db.commit()
    with factory() as db:
        rows = facts(db, task_id)
        applied = {row.turn_id: row for row in rows if row.kind == "input_applied"}
        assert set(applied) == {"turn-first", "turn-last"}
        assert len([row for row in rows if row.kind == "input_applied"]) == 2
        states = {row.event_id: row for row in rows if row.kind == "recovery_state"}
        for turn, application in applied.items():
            accepting_state = states[application.payload["recovery_event_id"]]
            assert accepting_state.sequence < application.sequence
            assert turn in {
                message["metadata"]["turn_id"]
                for message in accepting_state.payload["data"]["snapshot"]["context"][
                    "messages"
                ]
                if message["role"] == "user"
            }
        compacted = rows[-1].payload["data"]["snapshot"]["context"]
        restored = ExecutionContext.from_dict(compacted)
        assert [
            m.metadata.get("turn_id") for m in restored.messages if m.role == "user"
        ] == ["turn-last"]
        assert {
            entry[0] for entry in restored.metadata[ACCEPTED_TURN_IDS_METADATA_KEY]
        } == set(applied)
        assert AgentRunner._accepted_turn_digest(
            restored, "turn-first"
        ) == AgentRunner._turn_content_digest("Choose B")
    # Test-only selection above reads event facts; no production V2 reader is installed.
    cold_manager = ContextManager()
    cold_manager.set_context(restored)
    cold_runner = AgentRunner(
        SimpleNamespace(llm=None),
        tracer=TraceCheckpointStore(tracer),
        context_manager=cold_manager,
    )
    retry = await cold_runner.inject_user_message(
        context.execution_id, "Choose B", turn_id="turn-first", request_interrupt=False
    )
    assert retry.outcome is UserMessageInjectionOutcome.POSTED_REPLAY
    with factory() as db:
        assert len(facts(db, task_id)) == len(rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["root", "delegated-child"])
async def test_dag_facts_retain_adopted_plan_child_state_and_tool_result(
    canonical, scope
):
    from tests.core.agent.test_dag import FakeTool, SequenceLLM, build_plan
    from xagent.core.agent import DAGPattern, ExecutionContext, PatternRuntime, PlanStep
    from xagent.core.agent.checkpoint import TraceCheckpointStore
    from xagent.core.agent.trace import Tracer

    factory, task_id = canonical
    adapter = task_database_handler(
        task_id, build_id=None if scope == "root" else scope
    )
    tracer = Tracer()
    tracer.handlers = [adapter]
    tracer.event_writer = adapter.commit_event
    runtime = PatternRuntime(tracer=TraceCheckpointStore(tracer))
    context = ExecutionContext(execution_id="dag-evidence")
    context.add_user_message("Prepare, then calculate")
    plan = build_plan(
        PlanStep(id="prepare", task="Prepare"),
        PlanStep(id="calc", task="Calculate 6*7", dependencies=["prepare"]),
    )
    pattern = DAGPattern(lambda **_: plan, max_concurrency=1)
    tool = FakeTool()
    result = await pattern.run(
        context=context,
        tools=[tool],
        runtime=runtime,
        llm=SequenceLLM(
            [
                {"content": "Prepared", "done": True},
                {
                    "content": "Calculate",
                    "tool_calls": [
                        {
                            "id": "call",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression":"6*7"}',
                            },
                        }
                    ],
                },
                {"content": "42", "done": True},
            ]
        ),
    )
    assert result["success"]
    assert tool.calls == [{"expression": "6*7"}]
    with factory() as db:
        db.execute(
            sa.delete(StoredTraceEvent).where(StoredTraceEvent.task_id == task_id)
        )
        db.commit()
    with factory() as db:
        rows = facts(db, task_id)
        assert {row.scope_id for row in rows} == {scope}
        state = next(
            row
            for row in rows
            if row.kind == "recovery_state"
            and row.payload["data"]["snapshot"]["label"] == "dag_after_llm"
            and row.payload["data"]["snapshot"]["pattern_state"]["active_step_id"]
            == "calc"
        )
        snapshot = state.payload["data"]["snapshot"]
        restored = DAGPattern(lambda **_: pytest.fail("Must reuse adopted plan"))
        restored.load_state(snapshot["pattern_state"])
        assert [step.id for step in restored.plan.steps] == ["prepare", "calc"]
        assert restored.plan.steps[1].dependencies == ["prepare"]
        assert restored.step_results == {"prepare": "Prepared"}
        call = restored.active_step_pattern_states["calc"]["pending_tool_calls"][0]
        assert call["args"] == {"expression": "6*7"}
        child = ExecutionContext.from_dict(restored.active_step_contexts["calc"])
        assert child.execution_id != context.execution_id
        frames = snapshot["execution_snapshot"]["frames"]
        child_frame = frames["dag-evidence:dag_step:calc"]
        assert child_frame["parent_frame_id"] == "dag-evidence:dag"
        assert child_frame["pattern_state"]["pending_tool_calls"] == [call]
        start, end = [
            row for row in rows if row.tool_attempt_id == call["tool_attempt_id"]
        ]
        assert state.sequence < start.sequence < end.sequence
        assert (
            start.assistant_message_id
            == end.assistant_message_id
            == call["assistant_message_id"]
        )
        assert end.payload["data"]["result"] == {"result": 42}
        # Three crash horizons contain distinct evidence: not started, unknown,
        # and a committed result. Reading a result must not execute the tool.
        assert not [
            row
            for row in rows
            if row.sequence <= state.sequence
            and row.tool_attempt_id == call["tool_attempt_id"]
        ]
        assert [
            row.kind
            for row in rows
            if row.sequence <= start.sequence
            and row.tool_attempt_id == call["tool_attempt_id"]
        ] == ["tool_execution_start"]
        assert [
            row.kind
            for row in rows
            if row.sequence <= end.sequence
            and row.tool_attempt_id == call["tool_attempt_id"]
        ] == ["tool_execution_start", "tool_execution_end"]
    assert tool.calls == [{"expression": "6*7"}]


@pytest.mark.asyncio
async def test_live_stream_and_settlement_identity_is_retained_in_facts(
    canonical, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from xagent.core.agent import ExecutionContext, PatternRuntime
    from xagent.core.agent.tracing import TraceEventCallback
    from xagent.web.models.task import TaskStatus
    from xagent.web.services import task_execution
    from xagent.web.services.managed_task_lease import (
        finalize_managed_task_lease_result,
    )
    from xagent.web.services.task_lease_service import (
        acquire_task_lease,
        bind_task_lease_context,
    )
    from xagent.web.tracing import create_task_tracer

    factory, task_id = canonical
    monkeypatch.setattr(task_execution, "get_db", lambda: iter([factory()]))
    publish = AsyncMock()
    monkeypatch.setattr(task_execution, "publish_task_event", publish)
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)
        for turn in ["same-text-1", "same-text-2"]:
            persist_user_message_no_commit(
                db, task_id, db.get(Task, task_id).user_id, "same text", turn_id=turn
            )
        db.commit()
    tracer = create_task_tracer(task_id)
    tracer.handlers = [task_database_handler(task_id)]
    context = ExecutionContext(execution_id="stream-execution")
    runtime = PatternRuntime(
        execution_id=context.execution_id,
        outbound_message_handler=task_execution.make_agent_outbound_handler(
            task_id, authoritative=True
        ),
    )
    with bind_task_lease_context(lease):
        unfinished = await runtime.start_final_answer_stream()
        await runtime.emit_final_answer_delta(unfinished, "not durable")
        failed = await runtime.start_final_answer_stream()
        await runtime.fail_final_answer_stream(failed, "stream interrupted")
        completed = await runtime.start_final_answer_stream()
        await runtime.emit_final_answer_delta(completed, "answer")
        await runtime.end_final_answer_stream(completed, "answer")
        await TraceEventCallback().on_run_end(
            runner=SimpleNamespace(
                tracer=tracer,
                _active_controls={
                    context.execution_id: SimpleNamespace(runtime=runtime)
                },
            ),
            context=context,
            result={
                "success": True,
                "status": "completed",
                "execution_id": context.execution_id,
                "output": "answer",
            },
        )
        with factory() as db:
            assert finalize_managed_task_lease_result(
                db,
                lease,
                status=TaskStatus.COMPLETED,
                assistant_content="answer",
                execution_result={"success": True, "output": "answer"},
            )
    live = [call.args[0] for call in publish.await_args_list]
    with factory() as db:
        db.execute(sa.delete(TaskChatMessage).where(TaskChatMessage.task_id == task_id))
        db.execute(
            sa.delete(StoredTraceEvent).where(StoredTraceEvent.task_id == task_id)
        )
        db.commit()
    with factory() as db:
        rows = facts(db, task_id)
        accepted = [row for row in rows if row.kind == "input_accepted"]
        assert len({row.event_id for row in accepted}) == 2
        assert len({row.sequence for row in accepted}) == 2
        assert {row.turn_id for row in accepted} == {"same-text-1", "same-text-2"}
        streams = [row for row in rows if row.kind.startswith("final_answer_")]
        assert len(streams) == 5  # Three starts, one error and one end; no deltas.
        for row in streams:
            frame = next(
                frame
                for frame in live
                if frame["event_id"] == row.payload["protocol_event_id"]
            )
            assert row.payload["data"]["message_id"] == frame["message_id"]
            assert row.run_id == lease.run_id
        assert [
            row.kind
            for row in streams
            if row.payload["data"]["message_id"] == unfinished
        ] == ["final_answer_start"]
        assert [
            row.kind for row in streams if row.payload["data"]["message_id"] == failed
        ] == ["final_answer_start", "final_answer_error"]
        completion = next(row for row in rows if row.kind == "task_completion")
        assert completion.payload["data"]["result"]["stream_message_id"] == completed
        message = next(row for row in rows if row.kind == "assistant_message")
        settlement = next(row for row in rows if row.kind == "execution_settled")
        assert completion.run_id == message.run_id == settlement.run_id == lease.run_id
        assert completion.sequence < message.sequence < settlement.sequence
        assert message.payload["content"] == "answer"
