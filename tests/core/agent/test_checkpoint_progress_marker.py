from __future__ import annotations

import copy
from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, PatternRuntime
from xagent.core.agent.checkpoint import checkpoint_progress_marker
from xagent.core.agent.pattern.dag.dag import DAGPattern
from xagent.core.agent.pattern.dag.plan_generator import ExecutionPlan, PlanStep
from xagent.core.agent.pattern.react.react import ReActPattern, ToolCallRecord


async def _before_llm(
    runtime: PatternRuntime, context: ExecutionContext, pattern: Any
) -> dict[str, Any]:
    return await runtime.checkpoint("before_llm", context=context, pattern=pattern)


def _record(call_id: str) -> ToolCallRecord:
    return ToolCallRecord(
        tool_call_id=call_id,
        tool_name="search",
        args={"q": call_id},
        args_hash=call_id,
        status="completed",
        result="ok",
    )


@pytest.mark.asyncio
async def test_rewriting_before_llm_at_same_position_keeps_marker() -> None:
    context = ExecutionContext(execution_id="exec-marker")
    context.add_user_message("Find it")
    pattern = ReActPattern(max_iterations=5)
    pattern.current_iteration = 2
    pattern._store_tool_record(_record("call-1"))

    first = await _before_llm(
        PatternRuntime(execution_id="exec-marker"), context, pattern
    )

    # A resumed run restores the same state into a fresh pattern and a fresh
    # runtime, then rewrites ``before_llm`` before calling the LLM again.
    resumed = ReActPattern(max_iterations=5)
    resumed.load_state(copy.deepcopy(first["pattern_state"]))
    resumed.status = "thinking"
    resumed_context = ExecutionContext.from_dict(copy.deepcopy(first["context"]))
    second = await _before_llm(
        PatternRuntime(execution_id="exec-marker"), resumed_context, resumed
    )
    second["metadata"] = {"sequence": 99}

    assert checkpoint_progress_marker(first) == "m1:i2:t1:p0:s0"
    assert checkpoint_progress_marker(second) == checkpoint_progress_marker(first)


@pytest.mark.asyncio
async def test_react_advances_change_marker() -> None:
    runtime = PatternRuntime(execution_id="exec-react")
    context = ExecutionContext(execution_id="exec-react")
    context.add_user_message("Find it")
    pattern = ReActPattern(max_iterations=5)
    baseline = checkpoint_progress_marker(await _before_llm(runtime, context, pattern))

    context.add_assistant_message("thinking")
    after_message = checkpoint_progress_marker(
        await _before_llm(runtime, context, pattern)
    )
    pattern._store_tool_record(_record("call-1"))
    after_tool = checkpoint_progress_marker(
        await _before_llm(runtime, context, pattern)
    )
    pattern.current_iteration += 1
    after_iteration = checkpoint_progress_marker(
        await _before_llm(runtime, context, pattern)
    )

    markers = [baseline, after_message, after_tool, after_iteration]
    assert len(set(markers)) == len(markers)


def test_compaction_changes_marker() -> None:
    checkpoint = {
        "context": {"messages": [{"role": "user"}] * 12},
        "pattern_state": {"current_iteration": 4, "tool_ledger": {"a": {}}},
    }
    compacted = copy.deepcopy(checkpoint)
    compacted["context"]["messages"] = [{"role": "system"}, {"role": "user"}]

    assert checkpoint_progress_marker(compacted) != checkpoint_progress_marker(
        checkpoint
    )


def test_dag_plan_and_completed_steps_change_marker() -> None:
    pattern = DAGPattern(lambda **_: None)
    no_plan = {"context": {"messages": []}, "pattern_state": pattern.get_state()}

    pattern.plan = ExecutionPlan(
        steps=[PlanStep(id="a", task="A"), PlanStep(id="b", task="B")]
    )
    planned = {"context": {"messages": []}, "pattern_state": pattern.get_state()}

    pattern.plan.steps[0].status = "completed"
    pattern.step_results["a"] = "done"
    one_done = {"context": {"messages": []}, "pattern_state": pattern.get_state()}

    assert checkpoint_progress_marker(no_plan) == "m0:i0:t0:p0:s0"
    assert checkpoint_progress_marker(planned) == "m0:i0:t0:p2:s0"
    assert checkpoint_progress_marker(one_done) == "m0:i0:t0:p2:s1"


def test_dag_active_step_progress_counts_once() -> None:
    step_state = {"current_iteration": 3, "tool_ledger": {"x": {}, "y": {}}}
    step_context = {"messages": [{}, {}, {}]}
    checkpoint = {
        "context": {"messages": [{}]},
        "pattern_state": {
            "plan": {"steps": [{"id": "a", "status": "running"}]},
            # Legacy singular fields duplicate the dict entry for "a".
            "active_step_id": "a",
            "active_step_pattern_state": step_state,
            "active_step_context": step_context,
            "active_step_pattern_states": {"a": step_state},
            "active_step_contexts": {"a": step_context},
        },
    }
    assert checkpoint_progress_marker(checkpoint) == "m4:i3:t2:p1:s0"

    legacy_only = copy.deepcopy(checkpoint)
    del legacy_only["pattern_state"]["active_step_pattern_states"]
    del legacy_only["pattern_state"]["active_step_contexts"]
    assert checkpoint_progress_marker(legacy_only) == "m4:i3:t2:p1:s0"

    advanced = copy.deepcopy(checkpoint)
    advanced["pattern_state"]["active_step_pattern_states"]["a"]["tool_ledger"][
        "z"
    ] = {}
    assert checkpoint_progress_marker(advanced) != checkpoint_progress_marker(
        checkpoint
    )


def test_auto_nested_child_state_is_counted() -> None:
    checkpoint = {
        "context": {"messages": [{}, {}]},
        "pattern_state": {
            "status": "running",
            "selected_pattern": "react",
            "react_state": {"current_iteration": 1, "tool_ledger": {"a": {}}},
            "dag_state": None,
            "last_result": {"success": False},
        },
    }
    assert checkpoint_progress_marker(checkpoint) == "m2:i1:t1:p0:s0"


@pytest.mark.parametrize(
    "checkpoint",
    [
        None,
        {},
        {"context": None, "pattern_state": None},
        {"context": {"messages": "not-a-list"}, "pattern_state": "legacy"},
        {
            "context": {},
            "pattern_state": {
                "current_iteration": "3",
                "tool_ledger": [],
                "plan": {"steps": None},
                "active_step_pattern_states": ["bad"],
                "active_step_id": 7,
            },
        },
    ],
)
def test_legacy_or_partial_checkpoints_do_not_raise(checkpoint: Any) -> None:
    assert checkpoint_progress_marker(checkpoint) == "m0:i0:t0:p0:s0"


def test_volatile_fields_do_not_affect_marker() -> None:
    checkpoint = {
        "type": "checkpoint",
        "label": "before_llm",
        "execution_id": "exec-1",
        "status": "thinking",
        "metadata": {"sequence": 1},
        "context": {"messages": [{"role": "user"}]},
        "pattern_state": {"current_iteration": 1, "status": "thinking"},
    }
    rewritten = copy.deepcopy(checkpoint)
    rewritten.update(
        label="invalid_tool_protocol",
        execution_id="exec-2",
        status="failed",
        metadata={"sequence": 7},
    )
    rewritten["pattern_state"]["status"] = "failed"

    assert checkpoint_progress_marker(rewritten) == checkpoint_progress_marker(
        checkpoint
    )
