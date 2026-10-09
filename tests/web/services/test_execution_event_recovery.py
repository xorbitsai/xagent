"""V2 recovery must work without consulting compatibility content."""

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

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
from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.checkpoint import (
    CheckpointAccessRefusedError,
    CheckpointCorruptError,
    CheckpointUnavailableError,
    ExecutionEventPersistenceError,
    TraceCheckpointStore,
    UnknownToolEffectError,
)
from xagent.core.agent.trace import Tracer
from xagent.web.models.task import Task
from xagent.web.models.task import TraceEvent as StoredTraceEvent
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.services.task_execution_event_writer import append_fact_no_commit
from xagent.web.tracing import ExecutionEventTraceAdapter

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


def tracer_for(task_id, scope=None):
    adapter = ExecutionEventTraceAdapter(task_id, build_id=scope)
    tracer = Tracer()
    tracer.handlers = [adapter]
    tracer.event_writer = adapter.commit_event
    return tracer


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [None, "child"])
async def test_checkpoint_reads_event_after_legacy_trace_deletion(canonical, scope):
    factory, tid = canonical
    tracer = tracer_for(tid, scope)
    store = TraceCheckpointStore(tracer)
    payload = {
        "execution_id": "execution",
        "context": ExecutionContext(execution_id="execution").to_dict(),
        "pattern_state": {},
        "label": "ready",
    }
    await store.save(payload)
    with factory() as db:
        db.execute(sa.delete(StoredTraceEvent).where(StoredTraceEvent.task_id == tid))
        db.commit()
    assert await store.load_latest_checkpoint("execution") == payload
    assert await store.load_latest_checkpoint("other-execution") is None
    if scope:
        assert (
            await TraceCheckpointStore(tracer_for(tid)).load_latest_checkpoint(
                "execution"
            )
            is None
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("newest", ["execution", "other"])
async def test_checkpoint_scan_fetches_the_newest_state_alone_first(canonical, newest):
    factory, tid = canonical
    store = TraceCheckpointStore(tracer_for(tid))
    for label in range(5):
        await store.save(
            {
                "execution_id": "execution",
                "context": {"messages": []},
                "label": str(label),
            }
        )
    if newest == "other":
        await store.save({"execution_id": "other", "context": {"messages": []}})
    pages: list[int] = []
    scalars = Session.scalars

    def counted(self, statement, *args, **kwargs):
        result = scalars(self, statement, *args, **kwargs)
        if "recovery_state" not in statement.compile().params.values():
            return result
        rows = list(result)
        pages.append(len(rows))
        return rows

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Session, "scalars", counted)
        checkpoint = await store.load_latest_checkpoint("execution")
    assert checkpoint is not None and checkpoint["label"] == "4"
    # Each state inlines a whole snapshot: never page in older ones eagerly.
    assert pages == ([1] if newest == "execution" else [1, 5])


@pytest.mark.asyncio
async def test_corrupt_latest_state_does_not_fall_back_to_trace(canonical):
    factory, tid = canonical
    store = TraceCheckpointStore(tracer_for(tid))
    await store.save({"execution_id": "execution", "context": {"messages": []}})
    with factory() as db:
        event = db.query(TaskExecutionEvent).filter_by(kind="recovery_state").one()
        event.payload_version = 99
        db.commit()
    with pytest.raises(CheckpointCorruptError):
        await store.load_latest_checkpoint("execution")


@pytest.mark.asyncio
async def test_settled_execution_cannot_resume_older_checkpoint(canonical):
    factory, tid = canonical
    store = TraceCheckpointStore(tracer_for(tid))
    await store.save({"execution_id": "execution", "context": {"messages": []}})
    with factory() as db:
        append_fact_no_commit(
            db,
            task_id=tid,
            kind="execution_settled",
            key="settled",
            payload={"status": "failed", "result": {"status": "cancelled"}},
        )
        db.commit()
    with pytest.raises(CheckpointAccessRefusedError):
        await store.load_latest_checkpoint("execution")


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [None, "child"])
@pytest.mark.parametrize("mode", ["react", "auto_react", "auto_dag"])
async def test_resume_reuses_tool_result_committed_before_next_checkpoint(
    canonical, scope, monkeypatch, tmp_path, mode
):
    from tests.core.agent.test_auto import FakeLLM as AutoLLM
    from tests.core.agent.test_auto import (
        decision_tool_response,
        plan_tool_response,
    )
    from tests.core.agent.test_react import FakeLLM, FakeTool
    from tests.core.agent.test_runner import FakeWorkspaceManager
    from xagent.core.agent import (
        Agent,
        AgentRunner,
        AutoPattern,
        DAGPattern,
        LLMPlanGenerator,
    )

    def make_pattern():
        return (
            ReActPattern(max_iterations=3)
            if mode == "react"
            else AutoPattern(dag_pattern=DAGPattern(LLMPlanGenerator()))
        )

    def make_llm(responses, *, resume=False):
        if mode == "react":
            return FakeLLM(responses=responses)
        if not resume:
            action = "react" if mode == "auto_react" else "plan_execute"
            prefix = [decision_tool_response(action, "Calculate")]
            if mode == "auto_dag":
                prefix.append(
                    plan_tool_response([{"id": "calc", "task": "Calculate 2+2"}])
                )
            responses = prefix + responses
        return AutoLLM(responses)

    factory, tid = canonical
    tracer = tracer_for(tid, scope)
    store = TraceCheckpointStore(tracer)
    runtime = PatternRuntime(tracer=store)
    original_checkpoint = runtime.checkpoint

    async def crash_after_tool(label, **kwargs):
        if label.endswith(("after_tool", "after_tool_batch")):
            raise ExecutionEventPersistenceError("simulated crash after result commit")
        return await original_checkpoint(label, **kwargs)

    monkeypatch.setattr(runtime, "checkpoint", crash_after_tool)
    context = ExecutionContext(execution_id="tool-crash")
    context.add_user_message("2+2")
    tool = FakeTool()
    with pytest.raises(ExecutionEventPersistenceError):
        await make_pattern().run(
            context=context,
            tools=[tool],
            runtime=runtime,
            llm=make_llm(
                responses=[
                    {
                        "content": "calculate",
                        "tool_calls": [
                            {
                                "id": "call",
                                "function": {
                                    "name": "calculator",
                                    "arguments": '{"expression":"2+2"}',
                                },
                            }
                        ],
                    }
                ]
            ),
        )
    assert len(tool.calls) == 1
    with factory() as db:
        db.execute(sa.delete(StoredTraceEvent).where(StoredTraceEvent.task_id == tid))
        db.commit()
    runner = AgentRunner(
        agent=Agent(
            name="recovered",
            patterns=[make_pattern()],
            tools=[tool],
            llm=make_llm(responses=[{"content": "4", "done": True}], resume=True),
        ),
        tracer=store,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    result = await runner.resume("tool-crash", task="2+2")
    assert result["success"]
    assert len(tool.calls) == 1
    with factory() as db:
        assert (
            len([row for row in facts(db, tid) if row.kind == "tool_execution_start"])
            == 1
        )
        assert (
            len([row for row in facts(db, tid) if row.kind == "tool_execution_end"])
            == 1
        )


@pytest.mark.asyncio
async def test_unknown_tool_effect_is_not_reexecuted(canonical):
    from tests.core.agent.test_react import FakeTool

    _, tid = canonical
    runtime = PatternRuntime(tracer=TraceCheckpointStore(tracer_for(tid)))
    call = {
        "id": "call",
        "name": "calculator",
        "args": {"expression": "2+2"},
        "assistant_message_id": "batch",
        "tool_attempt_id": "attempt",
    }
    await runtime.on_tool_start(tool_call=call)
    tool = FakeTool()
    with pytest.raises(UnknownToolEffectError, match="automatic replay is unsafe"):
        await ReActPattern()._execute_tool_safely(
            call, [tool], runtime, context=ExecutionContext()
        )
    assert tool.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["ReActPattern", "AutoReact", "AutoDAG"])
async def test_reader_preflights_all_pending_effects_before_scheduling(
    canonical, pattern
):
    _, tid = canonical
    tracer = tracer_for(tid)
    store = TraceCheckpointStore(tracer)
    calls = [
        {
            "id": f"call-{i}",
            "name": "calculator",
            "args": {"expression": "2+2"},
            "assistant_message_id": "batch",
            "tool_attempt_id": f"attempt-{i}",
        }
        for i in range(2)
    ]
    state = {"pending_tool_calls": calls}
    if pattern == "AutoReact":
        state = {"decision": {"action": "react"}, "react_state": state}
    elif pattern == "AutoDAG":
        state = {
            "decision": {"action": "plan_execute"},
            "dag_state": {"active_step_pattern_states": {"step": state}},
        }
    await store.save(
        {
            "execution_id": "execution",
            "context": {"messages": []},
            "pattern": "AutoPattern" if pattern.startswith("Auto") else pattern,
            "pattern_state": state,
        }
    )
    await PatternRuntime(tracer=store).on_tool_start(tool_call=calls[1])
    with pytest.raises(UnknownToolEffectError, match="automatic replay is unsafe"):
        await store.load_latest_checkpoint("execution")


@pytest.mark.asyncio
async def test_interaction_anchor_and_question_do_not_read_legacy_content(
    canonical, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from xagent.web.services import task_execution
    from xagent.web.services.task_interaction_anchor import resolve_interaction_anchor
    from xagent.web.services.task_interaction_read import (
        get_pending_interaction_question,
    )
    from xagent.web.services.task_interaction_service import (
        _resolve_read_direction_anchor,
    )
    from xagent.web.services.task_lease_service import (
        acquire_task_lease,
        bind_task_lease_context,
    )
    from xagent.web.services.task_setup_snapshot import (
        load_task_reconstruction_snapshot_sync,
    )

    factory, tid = canonical
    monkeypatch.setattr(task_execution, "get_db", lambda: iter([factory()]))
    monkeypatch.setattr(task_execution, "publish_task_event", AsyncMock())
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    with bind_task_lease_context(lease):
        await TraceCheckpointStore(tracer_for(tid)).save(
            {"execution_id": str(tid), "context": {"messages": []}}
        )
        await task_execution.make_agent_outbound_handler(tid, authoritative=True)(
            {
                "message": "Choose?",
                "expect_response": True,
                "metadata": {"interactions": [{"field": "choice", "type": "text"}]},
            }
        )
    with factory() as db:
        task = db.get(Task, tid)
        anchor = resolve_interaction_anchor(db, task)
        assert anchor is not None
        row = SimpleNamespace(
            id=1,
            task_id=tid,
            resume_trace_event_id=anchor.trace_event_id,
            resume_event_id=anchor.resume_event_id,
            resume_execution_id=anchor.resume_execution_id,
            resume_run_partition=anchor.resume_run_partition,
        )
        # Scrub compatibility content but keep identity/FK control storage.
        for legacy in db.query(StoredTraceEvent).all():
            legacy.data = {"unreadable": True}
        db.commit()

        def no_legacy_read(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().lower().startswith("select") and (
                "trace_events" in statement.lower()
                or "task_chat_messages" in statement.lower()
                or "dag_executions" in statement.lower()
            ):
                raise AssertionError("Legacy content read")

        sa.event.listen(db.bind, "before_cursor_execute", no_legacy_read)
        try:
            assert _resolve_read_direction_anchor(db, row) is None
            assert resolve_interaction_anchor(db, task) == anchor
            assert get_pending_interaction_question(db, task) == (
                "Choose?",
                [{"field": "choice", "type": "text"}],
            )
            assert load_task_reconstruction_snapshot_sync(db, tid).has_history
            row.resume_execution_id = "wrong-execution"
            assert _resolve_read_direction_anchor(db, row).reason == "anchor_dangling"
        finally:
            sa.event.remove(db.bind, "before_cursor_execute", no_legacy_read)


@pytest.mark.asyncio
async def test_stale_lease_cannot_read_checkpoint_or_reuse_result(canonical):
    from xagent.web.services.task_lease_service import (
        acquire_task_lease,
        bind_task_lease_context,
    )

    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    store = TraceCheckpointStore(tracer_for(tid))
    call = {
        "id": "call",
        "name": "calculator",
        "args": {},
        "assistant_message_id": "batch",
        "tool_attempt_id": "attempt",
    }
    with bind_task_lease_context(lease):
        await store.save({"execution_id": "execution", "context": {"messages": []}})
        await PatternRuntime(tracer=store).on_tool_start(tool_call=call)
        await PatternRuntime(tracer=store).on_tool_end(tool_call=call, result=4)
    with factory() as db:
        task = db.get(Task, tid)
        task.runner_id = "replacement-owner"
        db.commit()
    with bind_task_lease_context(lease):
        with pytest.raises(CheckpointAccessRefusedError):
            await store.load_latest_checkpoint("execution")
        with pytest.raises(CheckpointAccessRefusedError):
            await store.load_committed_tool_outcome(call)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [None, "child"])
async def test_dag_resume_keeps_finished_step_and_reuses_child_result(
    canonical, scope, monkeypatch
):
    from tests.core.agent.test_dag import FakeTool, SequenceLLM, build_plan
    from xagent.core.agent import DAGPattern, PlanStep

    factory, tid = canonical
    store = TraceCheckpointStore(tracer_for(tid, scope))
    runtime = PatternRuntime(tracer=store)
    original_checkpoint = runtime.checkpoint

    async def crash_after_tool(label, **kwargs):
        if label in {"dag_after_tool", "dag_after_tool_batch"}:
            raise ExecutionEventPersistenceError("crash after child result")
        return await original_checkpoint(label, **kwargs)

    monkeypatch.setattr(runtime, "checkpoint", crash_after_tool)
    context = ExecutionContext(execution_id="dag-recovery")
    context.add_user_message("Prepare, calculate")
    plan = build_plan(
        PlanStep(id="prepare", task="Prepare"),
        PlanStep(id="calc", task="Calculate", dependencies=["prepare"]),
    )
    tool = FakeTool()
    with pytest.raises(ExecutionEventPersistenceError):
        await DAGPattern(lambda **_: plan, max_concurrency=1).run(
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
                ]
            ),
        )
    with factory() as db:
        db.execute(sa.delete(StoredTraceEvent).where(StoredTraceEvent.task_id == tid))
        db.commit()
    snapshot = await store.load_latest_checkpoint("dag-recovery")
    pattern = DAGPattern(lambda **_: pytest.fail("must reuse adopted plan"))
    pattern.load_state(snapshot["pattern_state"])
    assert pattern.step_results == {"prepare": "Prepared"}
    result = await pattern.run(
        context=ExecutionContext.from_dict(snapshot["context"]),
        tools=[tool],
        runtime=PatternRuntime(tracer=store),
        llm=SequenceLLM([{"content": "42", "done": True}]),
    )
    assert result["success"]
    assert tool.calls == [{"expression": "6*7"}]
    assert result["step_results"] == {"prepare": "Prepared", "calc": "42"}


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["failed", "waiting", "cancelled"])
async def test_tool_outcome_recovery_preserves_failure_and_waiting(canonical, outcome):
    from tests.core.agent.test_react import FakeTool

    _, tid = canonical
    runtime = PatternRuntime(tracer=TraceCheckpointStore(tracer_for(tid)))
    call = {
        "id": "call",
        "name": "calculator",
        "args": {"credential": "private"},
        "assistant_message_id": "batch",
        "tool_attempt_id": "attempt",
    }
    await runtime.on_tool_start(tool_call=call)
    tool = FakeTool()
    tool.sanitize_tool_args_for_trace = lambda args: {"credential": "redacted"}
    pattern = ReActPattern()
    if outcome == "cancelled":
        await runtime.on_tool_cancelled(tool_call=call)
        with pytest.raises(UnknownToolEffectError):
            await pattern._execute_tool_safely(
                call, [tool], runtime, context=ExecutionContext()
            )
    else:
        result = (
            {"success": False, "error": "failed"}
            if outcome == "failed"
            else {"status": "waiting_for_user", "message": "Authorize"}
        )
        await runtime.on_tool_end(tool_call=call, result=result)
        assert (
            await pattern._execute_tool_safely(
                call, [tool], runtime, context=ExecutionContext()
            )
            == result
        )
        assert pattern._record_for_tool_call_id("call").status == (
            "failed" if outcome == "failed" else "waiting_for_user"
        )
        assert pattern._record_for_tool_call_id("call").args == {
            "credential": "redacted"
        }
    assert tool.calls == []


@pytest.mark.asyncio
async def test_owner_replaced_during_read_is_rechecked(canonical, monkeypatch):
    from xagent.web.services import task_execution_event_recovery as recovery
    from xagent.web.services.task_lease_service import (
        acquire_task_lease,
        bind_task_lease_context,
    )

    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    store = TraceCheckpointStore(tracer_for(tid))
    original = recovery.read_event_checkpoint

    def replace_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        with factory() as db:
            db.execute(
                sa.update(Task).where(Task.id == tid).values(runner_id="new-owner")
            )
            db.commit()
        return result

    with bind_task_lease_context(lease):
        await store.save({"execution_id": str(tid), "context": {"messages": []}})
        monkeypatch.setattr(recovery, "read_event_checkpoint", replace_after_read)
        with pytest.raises(CheckpointAccessRefusedError):
            await store.load_latest_checkpoint(str(tid))


@pytest.mark.asyncio
async def test_cold_injection_preserves_compacted_application_and_pending_acceptance(
    canonical,
):
    from types import SimpleNamespace

    from xagent.core.agent.context import ContextManager
    from xagent.core.agent.runner import AgentRunner, UserMessageInjectionOutcome
    from xagent.web.models.chat_message import TaskChatMessage
    from xagent.web.services.chat_history_service import persist_user_message_no_commit

    factory, tid = canonical
    store = TraceCheckpointStore(tracer_for(tid))
    context = ExecutionContext(execution_id="input-recovery")
    manager = ContextManager()
    manager.set_context(context)
    runner = AgentRunner(
        SimpleNamespace(llm=None), tracer=store, context_manager=manager
    )
    with factory() as db:
        for turn, text in [("applied", "choose B"), ("pending", "choose C")]:
            persist_user_message_no_commit(
                db, tid, db.get(Task, tid).user_id, text, turn_id=turn
            )
        db.commit()
    posted = await runner.inject_user_message(
        context.execution_id, "choose B", turn_id="applied", request_interrupt=False
    )
    assert posted.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    context.add_assistant_message("B selected")
    context.compact_with_llm_response("B selected")
    await store.save(
        {"execution_id": context.execution_id, "context": context.to_dict()}
    )
    with factory() as db:
        db.execute(sa.delete(StoredTraceEvent).where(StoredTraceEvent.task_id == tid))
        db.execute(sa.delete(TaskChatMessage).where(TaskChatMessage.task_id == tid))
        db.commit()
        before = len(facts(db, tid))
    cold = AgentRunner(SimpleNamespace(llm=None), tracer=store)
    replay = await cold.inject_user_message(
        context.execution_id, "choose B", turn_id="applied", request_interrupt=False
    )
    assert replay.outcome is UserMessageInjectionOutcome.POSTED_REPLAY
    with factory() as db:
        assert len(facts(db, tid)) == before
        assert [
            row.turn_id for row in facts(db, tid) if row.kind == "input_applied"
        ] == ["applied"]
    snapshot = await store.load_latest_checkpoint(context.execution_id)
    assert not any(
        message.get("content") == "choose C"
        for message in snapshot["context"]["messages"]
    )


@pytest.mark.asyncio
async def test_database_failure_is_unavailable_not_absence(canonical, monkeypatch):
    from xagent.web.services import task_execution_event_recovery as recovery

    _, tid = canonical
    store = TraceCheckpointStore(tracer_for(tid))
    await store.save({"execution_id": "execution", "context": {"messages": []}})

    def unavailable(*args, **kwargs):
        raise sa.exc.OperationalError("SELECT", {}, RuntimeError("offline"))

    monkeypatch.setattr(recovery, "read_event_checkpoint", unavailable)
    with pytest.raises(CheckpointUnavailableError):
        await store.load_latest_checkpoint("execution")
    monkeypatch.setattr(recovery, "read_committed_tool_outcome", unavailable)
    with pytest.raises(CheckpointUnavailableError):
        await store.load_committed_tool_outcome({"tool_attempt_id": "attempt"})


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ["checkpoint", "tool"])
async def test_terminal_status_after_read_blocks_return_with_same_lease(
    canonical, monkeypatch, reader
):
    from xagent.web.models.task import TaskStatus
    from xagent.web.services import task_execution_event_recovery as recovery
    from xagent.web.services.task_lease_service import (
        acquire_task_lease,
        bind_task_lease_context,
    )

    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    store = TraceCheckpointStore(tracer_for(tid))
    runtime = PatternRuntime(tracer=store, execution_id=str(tid))
    call = {
        "id": "call",
        "name": "calculator",
        "arguments": {"expression": "2+2"},
        "assistant_message_id": "batch",
        "tool_attempt_id": "attempt",
    }
    original_check = recovery.check_recovery_owner
    checks = 0

    def cancel_at_return(db, task_id):
        nonlocal checks
        checks += 1
        if checks == 2:
            # Preserve the coordinator's lease identity, as external cancellation
            # can do. Its settlement is beyond the checkpoint reader's horizon.
            with factory() as writer:
                writer.execute(
                    sa.update(Task)
                    .where(Task.id == tid)
                    .values(status=TaskStatus.FAILED)
                )
                append_fact_no_commit(
                    writer,
                    task_id=tid,
                    kind="execution_settled",
                    key="late-cancel",
                    run_id=lease.run_id,
                    payload={"status": "failed", "result": {"status": "cancelled"}},
                )
                writer.commit()
        return original_check(db, task_id)

    with bind_task_lease_context(lease):
        await store.save({"execution_id": str(tid), "context": {"messages": []}})
        await runtime.on_tool_start(tool_call=call)
        await runtime.on_tool_end(
            tool_call=call, result={"success": True, "result": "4"}
        )
        monkeypatch.setattr(recovery, "check_recovery_owner", cancel_at_return)
        with pytest.raises(CheckpointAccessRefusedError, match="already ended"):
            if reader == "checkpoint":
                await store.load_latest_checkpoint(str(tid))
            else:
                await store.load_committed_tool_outcome(call)
    assert checks == 2


@pytest.mark.parametrize("metadata", [None, []])
def test_waiting_question_rejects_malformed_persisted_metadata(canonical, metadata):
    from xagent.web.services.task_interaction_read import (
        get_pending_interaction_question,
    )

    factory, tid = canonical
    with factory() as db:
        append_fact_no_commit(
            db,
            task_id=tid,
            kind="agent_message",
            key="question",
            payload={
                "data": {
                    "message": "Choose?",
                    "expect_response": True,
                    "metadata": metadata,
                }
            },
        )
        db.commit()
        if metadata is None:
            assert get_pending_interaction_question(db, db.get(Task, tid)) == (
                "Choose?",
                None,
            )
        else:
            with pytest.raises(CheckpointCorruptError, match="invalid metadata"):
                get_pending_interaction_question(db, db.get(Task, tid))


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol_id", [None, 42])
async def test_reconstruction_requires_persisted_protocol_identity(
    canonical, protocol_id
):
    from xagent.web.services.task_setup_snapshot import (
        load_task_reconstruction_snapshot_sync,
    )

    factory, tid = canonical
    await TraceCheckpointStore(tracer_for(tid)).save(
        {"execution_id": str(tid), "context": {"messages": []}}
    )
    with factory() as db:
        event = next(row for row in facts(db, tid) if row.kind == "recovery_state")
        payload = dict(event.payload)
        if protocol_id is None:
            payload.pop("protocol_event_id")
        else:
            payload["protocol_event_id"] = protocol_id
        event.payload = payload
        db.commit()
        with pytest.raises(CheckpointCorruptError, match="protocol event ID"):
            load_task_reconstruction_snapshot_sync(db, tid)


@pytest.mark.asyncio
async def test_missing_task_protocol_anchor_never_looks_up_literal_none(
    canonical, monkeypatch
):
    from xagent.web.services import task_execution_event_recovery as recovery
    from xagent.web.services.task_interaction_anchor import resolve_interaction_anchor
    from xagent.web.services.task_lease_service import (
        acquire_task_lease,
        bind_task_lease_context,
    )

    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    with bind_task_lease_context(lease):
        await TraceCheckpointStore(tracer_for(tid)).save(
            {"execution_id": str(tid), "context": {"messages": []}}
        )

    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("No protocol ID is available for lookup")

    monkeypatch.setattr(recovery, "find_event_checkpoint_anchor", unexpected_lookup)
    with factory() as db:
        task = db.get(Task, tid)
        assert task.last_checkpoint_trace_event_id is not None
        task.last_checkpoint_event_id = None
        assert resolve_interaction_anchor(db, task) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_unknown_effect_settles_expired_task(canonical, cancelled):
    from datetime import timedelta

    from xagent.web.models.task import TaskStatus
    from xagent.web.services.task_lease_recovery import (
        recover_task_lease_candidate_no_commit,
    )
    from xagent.web.services.task_lease_service import (
        acquire_task_lease,
        bind_task_lease_context,
        get_expired_task_lease_candidates,
        utc_now,
    )

    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    call = {
        "id": "call",
        "name": "calculator",
        "args": {},
        "assistant_message_id": "batch",
        "tool_attempt_id": "attempt",
    }
    with bind_task_lease_context(lease):
        store = TraceCheckpointStore(tracer_for(tid))
        runtime = PatternRuntime(tracer=store)
        await store.save(
            {
                "execution_id": str(tid),
                "context": {"messages": []},
                "pattern_state": {"pending_tool_calls": [call]},
            }
        )
        await runtime.on_tool_start(tool_call=call)
        if cancelled:
            await runtime.on_tool_cancelled(tool_call=call)
    with factory() as db:
        task = db.get(Task, tid)
        task.lease_expires_at = utc_now() - timedelta(seconds=5)
        db.commit()
        candidate = get_expired_task_lease_candidates(db, cutoff=utc_now(), limit=1)[0]
        assert (
            recover_task_lease_candidate_no_commit(
                db, candidate, recovered_at=utc_now()
            )
            == TaskStatus.FAILED
        )
        db.commit()
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.runner_id is None
        assert task.lease_expires_at is None
        settled = [
            event for event in facts(db, tid) if event.kind == "execution_settled"
        ]
        assert len(settled) == 1
        assert settled[0].payload["status"] == "failed"
        assert task.error_message == (
            "Task execution lease expired with an unknown tool effect; "
            "automatic replay is unsafe."
        )
        assert settled[0].payload["result"]["error"] == task.error_message
        assert not get_expired_task_lease_candidates(db, cutoff=utc_now(), limit=1)
        assert (
            recover_task_lease_candidate_no_commit(
                db, candidate, recovered_at=utc_now()
            )
            is None
        )
        db.commit()
        assert (
            len(
                [event for event in facts(db, tid) if event.kind == "execution_settled"]
            )
            == 1
        )


@pytest.mark.asyncio
async def test_corrupt_sibling_execution_is_isolated(canonical):
    factory, tid = canonical
    store = TraceCheckpointStore(tracer_for(tid, "child"))
    target = {"execution_id": "target", "context": {"messages": []}}
    await store.save(target)
    await store.save({"execution_id": "other", "context": {"messages": []}})
    with factory() as db:
        event = (
            db.query(TaskExecutionEvent)
            .filter_by(kind="recovery_state")
            .order_by(TaskExecutionEvent.sequence.desc())
            .first()
        )
        event.payload_version = 99
        db.commit()
    assert await store.load_latest_checkpoint("target") == target


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["typed", "storage_version", "facts"])
async def test_unavailable_lease_read_is_visible_until_next_clean_sweep(
    canonical, monkeypatch, failure
):
    from datetime import timedelta

    from xagent.web.models.task import TaskStatus
    from xagent.web.services import task_execution_event_recovery as reader
    from xagent.web.services.ops_signals import (
        CHECKPOINT_RECOVERY_UNAVAILABLE,
        active_degradations,
        clear_degradation,
    )
    from xagent.web.services.task_lease_recovery import (
        recover_expired_task_leases_until_cutoff,
    )
    from xagent.web.services.task_lease_service import utc_now

    factory, tid = canonical
    monkeypatch.setattr(
        "xagent.web.models.database.get_engine", lambda: factory.kw["bind"]
    )
    with factory() as db:
        task = db.get(Task, tid)
        task.status = TaskStatus.RUNNING
        task.run_id = "expired-run"
        task.runner_id = "dead-runner"
        task.lease_expires_at = utc_now() - timedelta(seconds=5)
        db.commit()

    def unavailable(*args, **kwargs):
        raise CheckpointUnavailableError("temporarily unavailable")

    def database_failure(conn, cursor, statement, parameters, context, executemany):
        target = (
            "SELECT tasks.conversation_storage_version"
            if failure == "storage_version"
            else "FROM task_execution_events"
        )
        if target in statement:
            raise sa.exc.OperationalError(
                statement, parameters, RuntimeError("offline")
            )

    clear_degradation(CHECKPOINT_RECOVERY_UNAVAILABLE)
    try:
        with monkeypatch.context() as patch:
            if failure == "typed":
                patch.setattr(reader, "read_event_checkpoint", unavailable)
            else:
                sa.event.listen(
                    factory.kw["bind"], "before_cursor_execute", database_failure
                )
            assert (
                await recover_expired_task_leases_until_cutoff(
                    cutoff=utc_now(), batch_size=10
                )
                == 0
            )
        if failure != "typed":
            sa.event.remove(
                factory.kw["bind"], "before_cursor_execute", database_failure
            )
        assert CHECKPOINT_RECOVERY_UNAVAILABLE in active_degradations()
        assert "SELECT" not in active_degradations()[CHECKPOINT_RECOVERY_UNAVAILABLE]
        with factory() as db:
            assert db.get(Task, tid).status == TaskStatus.RUNNING
            assert db.get(Task, tid).runner_id == "dead-runner"
        assert (
            await recover_expired_task_leases_until_cutoff(
                cutoff=utc_now(), batch_size=10
            )
            == 1
        )
        assert CHECKPOINT_RECOVERY_UNAVAILABLE not in active_degradations()
    finally:
        if sa.event.contains(
            factory.kw["bind"], "before_cursor_execute", database_failure
        ):
            sa.event.remove(
                factory.kw["bind"], "before_cursor_execute", database_failure
            )
        clear_degradation(CHECKPOINT_RECOVERY_UNAVAILABLE)


@pytest.mark.asyncio
async def test_unknown_effect_rechecks_widened_partition(canonical, monkeypatch):
    from copy import deepcopy

    from xagent.web.services.task_lease_service import (
        acquire_task_lease,
        bind_task_lease_context,
    )

    factory, tid = canonical
    store = TraceCheckpointStore(tracer_for(tid))
    call = {
        "id": "call",
        "name": "calculator",
        "args": {},
        "assistant_message_id": "batch",
        "tool_attempt_id": "attempt",
    }
    await store.save(
        {
            "execution_id": str(tid),
            "context": {"messages": []},
            "pattern_state": {"pending_tool_calls": [call]},
        }
    )
    await PatternRuntime(tracer=store).on_tool_start(tool_call=call)
    with factory() as db:
        old = db.query(TaskExecutionEvent).filter_by(kind="recovery_state").one()
        assert old.run_id is None
        payload = deepcopy(old.payload)
        lease = acquire_task_lease(db, tid, new_run=True)
    adapter = ExecutionEventTraceAdapter(tid)
    probe = adapter._task_has_run_tagged_checkpoint
    observations = []

    def commit_tagged_checkpoint_after_probe(db):
        answer = probe(db)
        observations.append(answer)
        if len(observations) == 1:
            assert answer is False
            payload["protocol_event_id"] = "fresh-tagged"
            payload["data"]["snapshot"]["pattern_state"] = {}
            with factory() as writer:
                append_fact_no_commit(
                    writer,
                    task_id=tid,
                    scope_id="root",
                    run_id=lease.run_id,
                    kind="recovery_state",
                    key="runtime:fresh-tagged",
                    payload=payload,
                )
                writer.commit()
        return answer

    monkeypatch.setattr(
        adapter,
        "_task_has_run_tagged_checkpoint",
        commit_tagged_checkpoint_after_probe,
    )
    with bind_task_lease_context(lease):
        with pytest.raises(CheckpointUnavailableError, match="stale snapshot"):
            await adapter.load_latest_checkpoint(str(tid))
        assert observations == [False, True]
        assert (await adapter.load_latest_checkpoint(str(tid)))["pattern_state"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [None, {}, {"action": "unknown"}])
@pytest.mark.parametrize("child_slot", ["react_state", "dag_state"])
async def test_auto_child_state_requires_recognized_decision(
    canonical, decision, child_slot
):
    _, tid = canonical
    store = TraceCheckpointStore(tracer_for(tid))
    call = {
        "id": "call",
        "name": "calculator",
        "args": {},
        "assistant_message_id": "batch",
        "tool_attempt_id": "attempt",
    }
    child = {"pending_tool_calls": [call]}
    if child_slot == "dag_state":
        child = {"active_step_pattern_states": {"step": child}}
    await store.save(
        {
            "execution_id": str(tid),
            "context": {"messages": []},
            "pattern": "AutoPattern",
            "pattern_state": {"decision": decision, child_slot: child},
        }
    )
    await PatternRuntime(tracer=store).on_tool_start(tool_call=call)
    with pytest.raises(CheckpointCorruptError, match="Auto child state"):
        await store.load_latest_checkpoint(str(tid))


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [None, {"action": "final_answer"}])
async def test_auto_without_child_state_remains_readable(canonical, decision):
    _, tid = canonical
    store = TraceCheckpointStore(tracer_for(tid))
    snapshot = {
        "execution_id": str(tid),
        "context": {"messages": []},
        "pattern": "AutoPattern",
        "pattern_state": {
            "decision": decision,
            "react_state": None,
            "dag_state": None,
        },
    }
    await store.save(snapshot)
    assert await store.load_latest_checkpoint(str(tid)) == snapshot


@pytest.mark.asyncio
async def test_no_checkpoint_keeps_generic_expiry_reason(canonical):
    from datetime import timedelta

    from xagent.web.models.task import TaskStatus
    from xagent.web.services.task_lease_recovery import (
        TASK_LEASE_EXPIRED_ERROR,
        recover_task_lease_candidate_no_commit,
    )
    from xagent.web.services.task_lease_service import (
        acquire_task_lease,
        get_expired_task_lease_candidates,
        utc_now,
    )

    factory, tid = canonical
    with factory() as db:
        acquire_task_lease(db, tid, new_run=True)
        db.get(Task, tid).lease_expires_at = utc_now() - timedelta(seconds=5)
        db.commit()
        candidate = get_expired_task_lease_candidates(db, cutoff=utc_now(), limit=1)[0]
        assert (
            recover_task_lease_candidate_no_commit(
                db, candidate, recovered_at=utc_now()
            )
            == TaskStatus.FAILED
        )
        db.commit()
        task = db.get(Task, tid)
        settled = db.query(TaskExecutionEvent).filter_by(kind="execution_settled").one()
        assert task.error_message == TASK_LEASE_EXPIRED_ERROR
        assert settled.payload["result"]["error"] == TASK_LEASE_EXPIRED_ERROR
