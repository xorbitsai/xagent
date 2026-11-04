from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tests.core.agent.test_react import (
    FakeLLM,
    SettlementApprovalTool,
    final_answer_response,
)
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
from xagent.core.agent import ExecutionContext, ReActPattern
from xagent.core.agent.checkpoint import (
    ExecutionEventPersistenceError,
    TraceCheckpointStore,
)
from xagent.core.agent.runtime import PatternRuntime
from xagent.core.tools.user_interaction import ToolInteractionSettlement
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.services import task_execution_event_writer
from xagent.web.services.task_event_context_service import load_task_event_context
from xagent.web.services.task_event_display import load_event_display_snapshot
from xagent.web.services.task_lease_service import (
    acquire_task_lease,
    bind_task_lease_context,
)
from xagent.web.tracing import ExecutionEventTraceAdapter, create_task_tracer

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture

WAITING = {"success": False, "status": "waiting_for_user"}


def _tool_call(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": arguments}}


def _runtime(task_id: int) -> PatternRuntime:
    tracer = create_task_tracer(task_id)
    tracer.handlers = [
        h for h in tracer.handlers if isinstance(h, ExecutionEventTraceAdapter)
    ]
    return PatternRuntime(tracer=TraceCheckpointStore(tracer))


def _new_run(factory, task_id):
    """Claim the next run of the task, as a resume after a pause does."""
    with factory() as db:
        task = db.get(Task, task_id)
        if task.status == TaskStatus.RUNNING:
            task.status = TaskStatus.WAITING_FOR_USER
            db.commit()
        lease = acquire_task_lease(db, task_id, new_run=True)
    assert lease is not None
    return lease


def _settlement_facts(rows):
    return [
        row
        for row in rows
        if row.kind.startswith("tool_execution_")
        and row.payload["data"].get("settlement_delivery") is True
    ]


@pytest.mark.parametrize(
    "settlement",
    [
        ToolInteractionSettlement.succeeded(
            {"success": True, "post_urn": "urn:li:share:123"}
        ),
        ToolInteractionSettlement.rejected(error="The user declined to publish."),
    ],
    ids=["succeeded", "rejected"],
)
@pytest.mark.asyncio
async def test_resumed_tool_interaction_settlement_is_readable_on_next_turn(
    canonical, settlement
):
    """A V2 task must read its history after a tool interaction settles."""

    factory, task_id = canonical
    context = ExecutionContext(system_prompt="Publish")
    context.add_user_message("Publish the post.", metadata={"turn_id": "turn-1"})
    pattern = ReActPattern(max_iterations=3)
    with bind_task_lease_context(_new_run(factory, task_id)):
        waiting = await pattern.run(
            context=context,
            tools=[SettlementApprovalTool()],
            llm=FakeLLM(
                [
                    {
                        "tool_calls": [
                            _tool_call(
                                "publish-call", "approval_gate", '{"expression":"go"}'
                            )
                        ]
                    }
                ]
            ),
            runtime=_runtime(task_id),
        )
    assert waiting["status"] == "waiting_for_user"

    # The resume rebuilds the pattern from its checkpointed state.
    resumed_pattern = ReActPattern(max_iterations=3)
    resumed_pattern.load_state(json.loads(json.dumps(pattern.get_state())))
    context.add_user_message("Approve", metadata={"turn_id": "turn-2"})
    with bind_task_lease_context(_new_run(factory, task_id)):
        resumed = await resumed_pattern.run(
            context=context,
            tools=[SettlementApprovalTool(resume_result=settlement)],
            llm=FakeLLM(
                [
                    {
                        "tool_calls": [
                            _tool_call(
                                "final-call",
                                "final_answer",
                                '{"response_language":"English",'
                                '"answer":"Done.","outcome":"completed"}',
                            )
                        ]
                    }
                ]
            ),
            runtime=_runtime(task_id),
        )
    assert resumed["success"] is True

    with factory() as db:
        rows = facts(db, task_id)
        settled = _settlement_facts(rows)
        original = [
            row
            for row in rows
            if row.kind.startswith("tool_execution_") and row not in settled
        ]
        # The settlement pair belongs to the original attempt, from a later run.
        assert [row.kind for row in settled] == [
            "tool_execution_start",
            "tool_execution_end"
            if settlement.status == "succeeded"
            else "tool_execution_failed",
        ]
        assert {row.tool_attempt_id for row in settled} == {original[0].tool_attempt_id}
        assert {row.assistant_message_id for row in settled} == {
            original[0].assistant_message_id
        }
        assert {row.run_id for row in settled}.isdisjoint(
            {row.run_id for row in original}
        )
        # The next turn sees one call whose observation is the settled result.
        loaded = load_task_event_context(db, task_id)
        calls = [m for m in loaded.messages if m.get("tool_calls")]
        assert [c["id"] for m in calls for c in m["tool_calls"]] == ["publish-call"]
        [observation] = [m for m in loaded.messages if m["role"] == "tool"]
        assert observation["raw_result"] == settled[-1].payload["data"]["result"]
        assert observation["raw_result"] != WAITING
        load_event_display_snapshot(db, task_id)


async def _waiting_call(factory, task_id, pattern):
    """Commit a waiting tool attempt in its own run; return its ledger row."""
    [call] = pattern._normalize_tool_calls(
        [{"id": "gate-call", "function": {"name": "approval_gate", "arguments": "{}"}}]
    )
    call.update(tool_attempt_id="attempt-1", assistant_message_id="batch-1")
    with bind_task_lease_context(_new_run(factory, task_id)):
        runtime = _runtime(task_id)
        await runtime.on_tool_start(tool_call=call)
        await runtime.on_tool_end(tool_call=call, result=WAITING)
    pattern._record_tool_call(
        call,
        status="completed",
        result={"success": True},
        settlement_status="succeeded",
        settlement_trace_pending=True,
    )
    [record] = pattern.tool_ledger.values()
    return record


async def _trace_settlement(pattern, record, task_id):
    await pattern._trace_tool_interaction_settlement(
        record=record,
        status="succeeded",
        result={"success": True},
        error=None,
        runtime=_runtime(task_id),
    )


@pytest.mark.parametrize("end_committed", [True, False], ids=["both", "start_only"])
@pytest.mark.asyncio
async def test_settlement_replay_from_a_later_run_is_idempotent(
    canonical, end_committed
):
    """A replay in another run adds only the settlement facts still missing."""

    factory, task_id = canonical
    pattern = ReActPattern()
    record = await _waiting_call(factory, task_id, pattern)
    with bind_task_lease_context(_new_run(factory, task_id)):
        await _trace_settlement(pattern, record, task_id)
    with factory() as db:
        rows = facts(db, task_id)
        assert len(rows) == 4
        if not end_committed:
            # Only the settlement start committed before the run lost it.
            end = rows[-1]
            db.query(TraceEvent).filter(
                TraceEvent.task_id == task_id,
                TraceEvent.event_id == end.payload["protocol_event_id"],
            ).delete()
            db.delete(end)
            db.commit()
        committed = [row.event_id for row in facts(db, task_id)]

    # The ledger still owes the settlement trace.
    pattern.tool_ledger[pattern._ledger_key(record)] = record
    with bind_task_lease_context(_new_run(factory, task_id)):
        await _trace_settlement(pattern, record, task_id)
    with factory() as db:
        rows = facts(db, task_id)
        assert [row.event_id for row in rows][: len(committed)] == committed
        assert [row.kind for row in _settlement_facts(rows)] == [
            "tool_execution_start",
            "tool_execution_end",
        ]
        # A replayed end comes from yet another run than its settlement start.
        start, end = _settlement_facts(rows)
        assert (start.run_id != end.run_id) is not end_committed
        [observation] = [
            m
            for m in load_task_event_context(db, task_id).messages
            if m["role"] == "tool"
        ]
        assert observation["raw_result"] == {"success": True}
    assert not pattern.tool_ledger[pattern._ledger_key(record)].settlement_trace_pending


@pytest.mark.asyncio
async def test_settlement_from_a_ledger_without_identity_stays_unreadable(
    canonical,
):
    """Ledgers checkpointed before the identity was kept fail closed."""

    factory, task_id = canonical
    pattern = ReActPattern()
    record = replace(
        await _waiting_call(factory, task_id, pattern),
        tool_attempt_id=None,
        assistant_message_id=None,
    )
    with bind_task_lease_context(_new_run(factory, task_id)):
        await _trace_settlement(pattern, record, task_id)
    with factory() as db:
        assert {
            row.tool_attempt_id for row in _settlement_facts(facts(db, task_id))
        } == {None}
        with pytest.raises(ValueError, match="no attempt/batch identity"):
            load_task_event_context(db, task_id)


@pytest.mark.asyncio
async def test_settlement_commit_failure_stops_the_run_until_it_replays(
    canonical, monkeypatch
):
    """A failed settlement commit stops the run; the resumed run replays it."""

    factory, task_id = canonical
    context = ExecutionContext(system_prompt="Publish")
    context.add_user_message("Publish the post.", metadata={"turn_id": "turn-1"})
    pattern = ReActPattern(max_iterations=3)
    with bind_task_lease_context(_new_run(factory, task_id)):
        waiting = await pattern.run(
            context=context,
            tools=[SettlementApprovalTool()],
            llm=FakeLLM(
                [
                    {
                        "tool_calls": [
                            _tool_call(
                                "publish-call", "approval_gate", '{"expression":"go"}'
                            )
                        ]
                    }
                ]
            ),
            runtime=_runtime(task_id),
        )
    assert waiting["status"] == "waiting_for_user"

    append = task_execution_event_writer.append_fact_no_commit

    def fail_settlement_facts(db, *, key, **kwargs):
        if ":settlement:" in key:
            raise RuntimeError("database connection lost")
        return append(db, key=key, **kwargs)

    monkeypatch.setattr(
        task_execution_event_writer, "append_fact_no_commit", fail_settlement_facts
    )
    context.add_user_message("Approve", metadata={"turn_id": "turn-2"})
    tool = SettlementApprovalTool(
        resume_result=ToolInteractionSettlement.succeeded(
            {"success": True, "post_urn": "urn:li:share:123"}
        )
    )
    llm = FakeLLM([final_answer_response("Published.")])
    with bind_task_lease_context(_new_run(factory, task_id)):
        with pytest.raises(ExecutionEventPersistenceError):
            await pattern.run(
                context=context, tools=[tool], llm=llm, runtime=_runtime(task_id)
            )
    # The run stopped before asking the model anything with the settled result.
    assert llm.calls == []
    with factory() as db:
        rows = facts(db, task_id)
        assert _settlement_facts(rows) == []
        state = next(row for row in reversed(rows) if row.kind == "recovery_state")
        pattern_state = state.payload["data"]["snapshot"]["pattern_state"]
    # The delivery was checkpointed before its trace: nothing is left to
    # deliver, and the settled row still owes the trace.
    assert pattern_state["pending_tool_interaction_responses"] == []
    assert [
        row["settlement_trace_pending"] for row in pattern_state["tool_ledger"].values()
    ] == [True]

    # The database recovers; the resumed run replays the owed settlement.
    monkeypatch.setattr(task_execution_event_writer, "append_fact_no_commit", append)
    resumed = ReActPattern(max_iterations=3)
    resumed.load_state(json.loads(json.dumps(pattern_state)))
    with bind_task_lease_context(_new_run(factory, task_id)):
        result = await resumed.run(
            context=context,
            tools=[tool],
            llm=FakeLLM([final_answer_response("Published.")]),
            runtime=_runtime(task_id),
        )
    assert result["success"] is True
    # The resume callback ran once: the replay does not dispatch again.
    assert len(tool.resume_calls) == 1
    with factory() as db:
        assert [row.kind for row in _settlement_facts(facts(db, task_id))] == [
            "tool_execution_start",
            "tool_execution_end",
        ]
        [observation] = [
            m
            for m in load_task_event_context(db, task_id).messages
            if m["role"] == "tool"
        ]
        assert observation["raw_result"]["post_urn"] == "urn:li:share:123"
