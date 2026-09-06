"""Forced-answer turns whose evidence this turn's compaction destroyed.

A ReAct turn can be forced back to ``final_answer`` alone while the very same
turn's compaction removes the tool observations that answer was supposed to
rest on. The engine must then either hand the dropped tools back for one
re-fetch turn, or say plainly that the observations are unavailable -- never
ask for an answer "using the accumulated conversation and tool results" that
are gone.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import xagent.core.agent.pattern.react.react as react_module
from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.pattern.react.react import (
    FORCED_ANSWER_FOLLOWUP_NO_EVIDENCE,
    FORCED_ANSWER_FOLLOWUP_REFETCHED,
    FORCED_ANSWER_FOLLOWUPS,
    FORCED_ANSWER_REASON_COMPACTION_RECOVERY,
    FORCED_ANSWER_REASON_CONTROL_TOOL_DISABLED,
    FORCED_ANSWER_REASON_EMPTY_FINAL_ANSWER,
    FORCED_ANSWER_REASON_FINALIZE_AFTER_TOOL,
    FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION,
    FORCED_ANSWER_REASONS,
    FORCED_ANSWER_REASONS_EXEMPT_FROM_RECOVERY,
    FORCED_ANSWER_RECOVERY_BUDGET,
    FORCED_ANSWER_RECOVERY_MIN_REMAINING_ITERATIONS,
)

REACT_SOURCE_PATH = Path(react_module.__file__)

# The sentence the ordinary forced instruction opens with. On a turn whose
# evidence was destroyed it is an instruction to invent, so it must be gone.
STALE_EVIDENCE_PHRASE = "the accumulated conversation and tool results"
HONEST_PHRASE = "naming the tool observations that were removed from context"
COMPLETED_OFFERED_PHRASE = "Set outcome=completed only when"
COMPLETED_WITHHELD_PHRASE = "outcome=completed is not available on this turn"
RECOVERY_CHECKPOINT = "forced_answer_evidence_dropped_recovered"


class _EmptyArgs(BaseModel):
    pass


class NamedTool:
    """Minimal work tool whose schema name the test chooses."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []

        class Metadata:
            pass

        self.metadata = Metadata()
        self.metadata.name = name  # type: ignore[attr-defined]
        self.metadata.description = f"Read {name}."  # type: ignore[attr-defined]

    def args_type(self) -> type[BaseModel]:
        return _EmptyArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        return {"output": f"{self.name} result"}


class RecordingLLM:
    """Chat LLM that records every call and replays scripted responses."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses:
            return {"content": "fallback answer", "done": True}
        return self.responses.pop(0)


class ScriptedCompactionRuntime(PatternRuntime):
    """Runtime whose compaction returns a caller-supplied result per turn.

    Used only for the metadata shapes real compaction cannot be steered into
    reliably (an unreadable name mapping, say). The shapes it stands in for are
    anchored to production by
    ``test_real_compaction_reports_the_keys_the_gate_reads``.
    """

    def __init__(self, results: list[Any]) -> None:
        super().__init__()
        self.scripted_compactions = list(results)

    async def compact_context_if_needed(self, **kwargs: Any) -> Any:
        if not self.scripted_compactions:
            return None
        return self.scripted_compactions.pop(0)


def final_answer_response(answer: str = "Here is what remains.") -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": "call_final",
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "arguments": (
                        '{"response_language":"English","outcome":"partial",'
                        f'"answer":"{answer}"}}'
                    ),
                },
            }
        ],
        "done": False,
    }


def tool_call_response(tool_name: str, call_id: str = "call_work") -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
            }
        ],
        "done": False,
    }


def build_context(
    *,
    tool_name: str = "calculator",
    observations: int = 3,
    max_messages: int = 4,
    result: Any | None = None,
    execution_id: str = "forced-answer-compaction",
) -> ExecutionContext:
    """Context primed to compact on every turn.

    ``max_messages`` decides whether the message-dropping fallback actually
    removes anything: a window wider than the transcript keeps every message
    and reports zero dropped observations even though it reports
    ``compacted=True``.
    """
    context = ExecutionContext(execution_id=execution_id)
    context.compact_config.threshold = 1
    context.compact_config.max_messages = max_messages
    context.add_user_message(f"Use {tool_name} and report every value.")
    for index in range(observations):
        call_id = f"seed_{index}"
        context.add_assistant_message(
            "",
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name},
                }
            ],
        )
        context.add_tool_result(
            tool_name,
            {"output": "x" * 400} if result is None else result,
            tool_call_id=call_id,
        )
    return context


def instruction_of(llm: RecordingLLM, index: int = 0) -> str:
    return str(llm.calls[index]["messages"][0].get("content", ""))


def whole_prompt_of(llm: RecordingLLM, index: int = 0) -> str:
    return "\n".join(
        str(message.get("content", "")) for message in llm.calls[index]["messages"]
    )


def tool_names_of(llm: RecordingLLM, index: int = 0) -> list[str]:
    return [tool["function"]["name"] for tool in (llm.calls[index].get("tools") or [])]


def checkpoint_labels(runtime: PatternRuntime) -> list[str]:
    return [str(entry.get("label")) for entry in runtime.checkpoints]


@pytest.mark.asyncio
async def test_real_compaction_reports_the_keys_the_gate_reads() -> None:
    """Anchor the gate's signal to what compaction actually writes.

    Every scripted-metadata test below claims a shape this one proves the two
    real compacting paths produce, so a rename or a semantic change upstream
    cannot leave those tests passing against a fiction.
    """
    summary_context = build_context(max_messages=40)
    summary_runtime = PatternRuntime()
    summary_result = await summary_runtime.compact_context_if_needed(
        context=summary_context,
        llm=RecordingLLM([{"content": "A summary of the run."}]),
    )
    assert summary_result.compacted is True
    assert summary_result.strategy == "llm_summary"
    assert summary_result.metadata["dropped_tool_result_count"] == 3
    assert summary_result.metadata["dropped_tool_results_by_name"] == {"calculator": 3}

    drop_context = build_context(max_messages=4)
    drop_runtime = PatternRuntime()
    drop_result = await drop_runtime.compact_context_if_needed(
        context=drop_context,
        llm=RecordingLLM([{"content": ""}]),
    )
    assert drop_result.compacted is True
    assert drop_result.strategy == "truncate"
    assert drop_result.metadata["dropped_tool_result_count"] == 1
    assert drop_result.metadata["dropped_tool_results_by_name"] == {"calculator": 1}
    # The tail window kept a later successful observation of the same tool.
    # "Is there still tool evidence in context" would answer yes here, which is
    # why the engine never asks that question.
    assert any(message.role == "tool" for message in drop_context.messages)


def compact_result(
    *,
    compacted: bool = True,
    count: Any = 1,
    by_name: Any = None,
    strategy: str = "llm_summary",
) -> Any:
    from xagent.core.agent.context.execution import CompactResult

    metadata: dict[str, Any] = {}
    if count is not None:
        metadata["dropped_tool_result_count"] = count
    if by_name is not None:
        metadata["dropped_tool_results_by_name"] = by_name
    return CompactResult(
        compacted=compacted,
        original_count=9,
        final_count=2,
        strategy=strategy,
        metadata=metadata,
    )


DECLINE_CASES: list[tuple[str, bool]] = [
    # Nothing was destroyed, so the forced turn proceeds exactly as before.
    ("compaction_did_not_run", False),
    ("truncate_removed_no_evidence", False),
    ("summary_unusable_removed_no_evidence", False),
    ("summary_error_removed_no_evidence", False),
    ("dropped_only_failed_tool_messages", False),
    ("dropped_only_control_pseudo_tools", False),
    # Evidence was destroyed but recovery is refused, so the turn must say so.
    ("names_absent_from_base", True),
    ("whitespace_normalized_name", True),
    ("tool_choice_none", True),
    ("unreadable_name_mapping", True),
    ("control_tool_disabled_is_exempt", True),
    ("legacy_checkpoint_without_reason", True),
    ("no_iteration_budget", True),
    ("budget_already_spent", True),
]


async def run_declined_turn(case: str) -> tuple[ReActPattern, RecordingLLM, Any]:
    """Drive one forced turn that the recovery gate refuses."""
    tools: list[Any] = [NamedTool("calculator")]
    llm = RecordingLLM([final_answer_response()])
    runtime: PatternRuntime = PatternRuntime()
    pattern = ReActPattern(max_iterations=6)
    pattern.force_final_answer_next = True
    pattern.forced_answer_reason = FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION
    compact_llm: Any = RecordingLLM([{"content": "A summary of the run."}])
    context = build_context(max_messages=40)

    if case == "compaction_did_not_run":
        context.compact_config.threshold = 10**9
    elif case == "truncate_removed_no_evidence":
        # A wide tail window keeps every message: compacted, nothing dropped.
        compact_llm = RecordingLLM([{"content": ""}])
    elif case == "summary_unusable_removed_no_evidence":
        compact_llm = RecordingLLM([{"content": "   "}])
    elif case == "summary_error_removed_no_evidence":

        class ExplodingCompactLLM:
            async def chat(self, **kwargs: Any) -> Any:
                raise RuntimeError("compaction model unavailable")

        compact_llm = ExplodingCompactLLM()
    elif case == "dropped_only_failed_tool_messages":
        context = build_context(max_messages=40, result={"success": False})
    elif case == "dropped_only_control_pseudo_tools":
        context = build_context(max_messages=40, tool_name="load_skill")
    elif case == "names_absent_from_base":
        # The dropped observation names a tool this run can no longer call.
        context = build_context(max_messages=40, tool_name="list_clients")
    elif case == "whitespace_normalized_name":
        # The dropped-name key collapses interior whitespace; the schema name
        # does not, so the two namespaces miss each other.
        tools = [NamedTool("list  clients")]
        context = build_context(max_messages=40, tool_name="list  clients")
    elif case == "tool_choice_none":
        pattern.tool_choice = "none"
    elif case == "unreadable_name_mapping":
        runtime = ScriptedCompactionRuntime(
            [compact_result(count=4, by_name=["calculator"])]
        )
    elif case == "control_tool_disabled_is_exempt":
        pattern.forced_answer_reason = FORCED_ANSWER_REASON_CONTROL_TOOL_DISABLED
    elif case == "legacy_checkpoint_without_reason":
        pattern.forced_answer_reason = None
    elif case == "no_iteration_budget":
        pattern.max_iterations = 2
        pattern.current_iteration = 1
    elif case == "budget_already_spent":
        pattern.forced_answer_compaction_recoveries = FORCED_ANSWER_RECOVERY_BUDGET
    else:  # pragma: no cover - guards against a typo in the case table
        raise AssertionError(f"unknown case {case}")

    spent_before = pattern.forced_answer_compaction_recoveries
    await pattern.run(
        context=context,
        tools=tools,
        llm=llm,
        compact_llm=compact_llm,
        runtime=runtime,
    )
    return pattern, llm, (runtime, spent_before)


@pytest.mark.asyncio
@pytest.mark.parametrize(("case", "expects_evidence_clause"), DECLINE_CASES)
async def test_recovery_is_declined_and_the_turn_stays_on_final_answer(
    case: str, expects_evidence_clause: bool
) -> None:
    """Every refusal keeps the single-tool turn and picks the right words.

    Refusing recovery is not refusing to be honest: when the gate saw evidence
    destroyed and still declined, the turn must ask for an answer that names
    what is missing rather than one drawn from results that are gone.
    """
    pattern, llm, (runtime, spent_before) = await run_declined_turn(case)

    assert tool_names_of(llm) == ["final_answer"]
    assert llm.calls[0]["tool_choice"] == "required"
    # Declining spends nothing: the budget is only for turns that really did
    # hand the tools back.
    assert pattern.forced_answer_compaction_recoveries == spent_before
    assert RECOVERY_CHECKPOINT not in checkpoint_labels(runtime)

    instruction = instruction_of(llm)
    assert (HONEST_PHRASE in instruction) is expects_evidence_clause
    if expects_evidence_clause:
        # The whole prompt, not just the instruction: a stale copy of the same
        # advice anywhere in the turn undoes it.
        assert STALE_EVIDENCE_PHRASE not in whole_prompt_of(llm)
        assert COMPLETED_OFFERED_PHRASE not in instruction
        assert COMPLETED_WITHHELD_PHRASE in instruction
    else:
        assert STALE_EVIDENCE_PHRASE in instruction
        assert COMPLETED_OFFERED_PHRASE in instruction


@pytest.mark.asyncio
async def test_declining_recovery_records_the_refusal_in_pattern_state() -> None:
    """The refusal is state, not just a loop local.

    Without this the decision to be honest lives only in a variable that no
    checkpoint carries, and a resume between the refusal and the answer would
    restore the forcing while forgetting the reason for it.
    """
    pattern, _llm, _rest = await run_declined_turn("budget_already_spent")

    # The run ended through final_answer, which clears the marker on its way
    # out, so the refusal is observed while the turn is still live below.
    assert pattern.forced_answer_recovery_followup is None

    live_pattern = ReActPattern(max_iterations=6)
    live_pattern.force_final_answer_next = True
    live_pattern.forced_answer_reason = FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION
    live_pattern.forced_answer_compaction_recoveries = FORCED_ANSWER_RECOVERY_BUDGET
    llm = RecordingLLM([{"content": "plain text, not done", "done": False}])
    await live_pattern.run(
        context=build_context(max_messages=40),
        tools=[NamedTool("calculator")],
        llm=llm,
        compact_llm=RecordingLLM([{"content": "A summary of the run."}]),
        runtime=PatternRuntime(),
    )
    assert HONEST_PHRASE in instruction_of(llm)


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "the repeated-tool decision writes its own 'answer from the accumulated "
        "tool results' guidance into the context; rewriting that text belongs to "
        "the decision site and is left for separate work. Delete this marker "
        "when it is done."
    ),
)
async def test_stale_completion_guidance_survives_a_drop_window() -> None:
    """A repeated-tool decision leaves its own "answer from the results" line.

    That guidance is an ordinary context message, so the message-dropping path
    can keep it in the tail window while removing the very results it points
    at. The forced turn then carries the honest instruction and a stale
    contradiction of it at once. Recorded here as an executable note rather
    than prose, so the day it is fixed this test says so.
    """
    pattern = ReActPattern(max_iterations=6)
    pattern.repeated_tool_decision = {
        "tool_name": "calculator",
        "consecutive_calls": 4,
        "reason": "same tool four times",
    }
    pattern.forced_answer_compaction_recoveries = FORCED_ANSWER_RECOVERY_BUDGET
    decision = {
        "content": "",
        "tool_calls": [
            {
                "id": "decision_1",
                "type": "function",
                "function": {
                    "name": "react_decision",
                    "arguments": (
                        '{"action":"final_answer","reason":"enough data",'
                        '"missing_verification":""}'
                    ),
                },
            }
        ],
    }
    llm = RecordingLLM([decision, final_answer_response()])

    await pattern.run(
        context=build_context(max_messages=4),
        tools=[NamedTool("calculator")],
        llm=llm,
        compact_llm=RecordingLLM([{"content": ""}]),
        runtime=PatternRuntime(),
    )

    forced_call = llm.calls[-1]
    assert [tool["function"]["name"] for tool in (forced_call.get("tools") or [])] == [
        "final_answer"
    ]
    assert HONEST_PHRASE in str(forced_call["messages"][0].get("content", ""))
    assert STALE_EVIDENCE_PHRASE not in "\n".join(
        str(message.get("content", "")) for message in forced_call["messages"]
    )


# Sentinels for the reload table below.
_DROP_KEY = object()  # the checkpoint has no such key at all
_UNCHANGED = object()  # the stored value survives the round trip as written


@pytest.mark.parametrize(
    ("key", "stored", "expected"),
    [
        ("forced_answer_reason", _DROP_KEY, None),
        ("forced_answer_reason", None, None),
        ("forced_answer_reason", "some_future_reason", None),
        ("forced_answer_reason", FORCED_ANSWER_REASON_EMPTY_FINAL_ANSWER, _UNCHANGED),
        ("forced_answer_compaction_recoveries", _DROP_KEY, 0),
        ("forced_answer_compaction_recoveries", "1", FORCED_ANSWER_RECOVERY_BUDGET),
        ("forced_answer_compaction_recoveries", 1, _UNCHANGED),
        ("forced_answer_recovery_followup", _DROP_KEY, None),
        ("forced_answer_recovery_followup", None, None),
        ("forced_answer_recovery_followup", ["refetched"], None),
        (
            "forced_answer_recovery_followup",
            FORCED_ANSWER_FOLLOWUP_NO_EVIDENCE,
            _UNCHANGED,
        ),
        (
            "forced_answer_recovery_followup",
            FORCED_ANSWER_FOLLOWUP_REFETCHED,
            _UNCHANGED,
        ),
    ],
)
def test_forced_answer_recovery_state_round_trips_through_checkpoint(
    key: str, stored: Any, expected: Any
) -> None:
    """Reload defaults differ per key, and each direction is deliberate.

    A reason or a marker this build cannot read must not act -- doing less is
    safe. The recovery counter is the opposite: treating a missing counter as
    spent would permanently close the recovery path for every run checkpointed
    before the counter existed, and that path is itself the safety net.

    Each case reshapes one key and asserts all three, because the keys are
    normalized independently and crossing them yields no further behaviour.
    """
    expected_attributes: dict[str, Any] = {
        "forced_answer_reason": FORCED_ANSWER_REASON_EMPTY_FINAL_ANSWER,
        "forced_answer_compaction_recoveries": 1,
        "forced_answer_recovery_followup": FORCED_ANSWER_FOLLOWUP_REFETCHED,
    }

    source = ReActPattern(max_iterations=6)
    source.forced_answer_reason = FORCED_ANSWER_REASON_EMPTY_FINAL_ANSWER
    source.forced_answer_compaction_recoveries = 1
    source.forced_answer_recovery_followup = FORCED_ANSWER_FOLLOWUP_REFETCHED
    state = source.get_state()

    assert {name: state[name] for name in expected_attributes} == expected_attributes

    expected_attributes[key] = stored if expected is _UNCHANGED else expected
    if stored is _DROP_KEY:
        del state[key]
    else:
        state[key] = stored

    restored = ReActPattern(max_iterations=6)
    restored.load_state(state)

    for attribute, value in expected_attributes.items():
        assert getattr(restored, attribute) == value


def test_a_forcing_reason_can_outlive_the_flag_that_carried_it() -> None:
    """Cleared forcing plus a leftover reason is a legal state, not a crash.

    Reason is only ever read while the flag is true, and every site that raises
    the flag rewrites the reason, so a stale reason is unreachable rather than
    scrubbed.
    """
    pattern = ReActPattern(max_iterations=6)
    pattern.force_final_answer_next = False
    pattern.forced_answer_reason = FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION

    restored = ReActPattern(max_iterations=6)
    restored.load_state(pattern.get_state())

    assert restored.force_final_answer_next is False
    assert restored.forced_answer_reason == FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("marker", "expects_forced_turn", "expects_evidence_clause"),
    [
        (FORCED_ANSWER_FOLLOWUP_NO_EVIDENCE, True, True),
        (FORCED_ANSWER_FOLLOWUP_REFETCHED, True, False),
        (None, False, False),
    ],
)
async def test_a_reloaded_marker_decides_the_next_turn(
    marker: str | None, expects_forced_turn: bool, expects_evidence_clause: bool
) -> None:
    """The marker's three readings drive three different next turns."""
    source = ReActPattern(max_iterations=6)
    source.forced_answer_recovery_followup = marker
    restored = ReActPattern(max_iterations=6)
    restored.load_state(source.get_state())

    llm = RecordingLLM([final_answer_response(), {"content": "done", "done": True}])
    context = build_context(max_messages=40)
    context.compact_config.threshold = 10**9
    await restored.run(
        context=context,
        tools=[NamedTool("calculator")],
        llm=llm,
        compact_llm=None,
        runtime=PatternRuntime(),
    )

    forced = tool_names_of(llm) == ["final_answer"]
    assert forced is expects_forced_turn
    assert (HONEST_PHRASE in instruction_of(llm)) is expects_evidence_clause
    if expects_forced_turn:
        assert restored.forced_answer_reason == FORCED_ANSWER_REASON_COMPACTION_RECOVERY


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["single_call_drop_window", "multi_turn_summary"])
async def test_a_declined_turn_still_speaks_honestly_after_a_resume(
    shape: str,
) -> None:
    """Interrupt a declined turn, reload it, and it must still be honest.

    The two shapes fail differently without the refusal in state. Under the
    message-dropping path the reloaded turn is still forced -- the tail window
    kept a later successful observation, so the engine recomputes "evidence is
    fine" and asks for an answer drawn from results it no longer has. Under the
    summary path nothing tool-shaped survives, so a run that forces itself from
    the latest tool result stops being a forced turn at all and burns its last
    iteration without answering.
    """
    if shape == "single_call_drop_window":
        pattern = ReActPattern(max_iterations=2, finalize_after_tool_result=True)
        pattern.current_iteration = 1
        context = build_context(max_messages=4)
        compact_llm: Any = RecordingLLM([{"content": ""}])
    else:
        pattern = ReActPattern(max_iterations=6)
        pattern.current_iteration = 1
        pattern.force_final_answer_next = True
        pattern.forced_answer_reason = FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION
        pattern.forced_answer_compaction_recoveries = FORCED_ANSWER_RECOVERY_BUDGET
        context = build_context(max_messages=40)
        compact_llm = RecordingLLM([{"content": "A summary of the run."}])

    runtime = PatternRuntime()
    interrupting_llm = RecordingLLM([])

    async def interrupt_before_answering(**kwargs: Any) -> Any:
        interrupting_llm.calls.append(kwargs)
        runtime.request_interrupt("user stop")
        return {"content": "", "done": False}

    interrupting_llm.chat = interrupt_before_answering  # type: ignore[method-assign]
    await pattern.run(
        context=context,
        tools=[NamedTool("calculator")],
        llm=interrupting_llm,
        compact_llm=compact_llm,
        runtime=runtime,
    )
    state = pattern.get_state()
    assert (
        state["forced_answer_recovery_followup"] == FORCED_ANSWER_FOLLOWUP_NO_EVIDENCE
    )
    spent_before = state["forced_answer_compaction_recoveries"]

    resumed = ReActPattern(max_iterations=pattern.max_iterations)
    resumed.load_state(state)
    resumed_runtime = PatternRuntime()
    resumed_llm = RecordingLLM([final_answer_response()])
    # The context has already been compacted once and is now under threshold,
    # so this turn's compaction returns nothing and the gate never runs.
    context.compact_config.threshold = 10**9
    await resumed.run(
        context=context,
        tools=[NamedTool("calculator")],
        llm=resumed_llm,
        compact_llm=None,
        runtime=resumed_runtime,
    )

    assert resumed_llm.calls[0]["tool_choice"] == "required"
    assert tool_names_of(resumed_llm) == ["final_answer"]
    assert HONEST_PHRASE in instruction_of(resumed_llm)
    assert STALE_EVIDENCE_PHRASE not in instruction_of(resumed_llm)
    assert resumed.forced_answer_compaction_recoveries == spent_before
    assert RECOVERY_CHECKPOINT not in checkpoint_labels(resumed_runtime)
    if shape == "single_call_drop_window":
        assert resumed.max_iterations - resumed.current_iteration == 1


@pytest.mark.asyncio
async def test_a_recoverable_turn_still_speaks_honestly_after_a_resume() -> None:
    """The gate can agree to recover on a build that cannot recover yet.

    Handing the dropped tools back is not implemented here, so a turn the gate
    cleared for recovery is left exactly where a refused one is: still forced
    to final_answer, with observations it can no longer read. It must therefore
    record the same refusal in state. Setup below is the refused summary shape
    with one line removed -- the spent-budget override -- which is the only
    thing that separates the two paths on this build.
    """
    pattern = ReActPattern(max_iterations=6)
    pattern.current_iteration = 1
    pattern.force_final_answer_next = True
    pattern.forced_answer_reason = FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION
    context = build_context(max_messages=40)
    compact_llm = RecordingLLM([{"content": "A summary of the run."}])

    runtime = PatternRuntime()
    interrupting_llm = RecordingLLM([])

    async def interrupt_before_answering(**kwargs: Any) -> Any:
        interrupting_llm.calls.append(kwargs)
        runtime.request_interrupt("user stop")
        return {"content": "", "done": False}

    interrupting_llm.chat = interrupt_before_answering  # type: ignore[method-assign]
    await pattern.run(
        context=context,
        tools=[NamedTool("calculator")],
        llm=interrupting_llm,
        compact_llm=compact_llm,
        runtime=runtime,
    )

    # The live turn was honest, so the checkpoint has to say so too.
    assert HONEST_PHRASE in instruction_of(interrupting_llm)
    state = pattern.get_state()
    assert (
        state["forced_answer_recovery_followup"] == FORCED_ANSWER_FOLLOWUP_NO_EVIDENCE
    )

    resumed = ReActPattern(max_iterations=pattern.max_iterations)
    resumed.load_state(state)
    resumed_runtime = PatternRuntime()
    resumed_llm = RecordingLLM([final_answer_response()])
    # The context has already been compacted once and is now under threshold,
    # so this turn's compaction returns nothing and the gate never runs.
    context.compact_config.threshold = 10**9
    await resumed.run(
        context=context,
        tools=[NamedTool("calculator")],
        llm=resumed_llm,
        compact_llm=None,
        runtime=resumed_runtime,
    )

    assert tool_names_of(resumed_llm) == ["final_answer"]
    assert resumed_llm.calls[0]["tool_choice"] == "required"
    assert HONEST_PHRASE in instruction_of(resumed_llm)
    assert STALE_EVIDENCE_PHRASE not in whole_prompt_of(resumed_llm)
    assert RECOVERY_CHECKPOINT not in checkpoint_labels(resumed_runtime)


@pytest.mark.asyncio
async def test_a_declined_turn_clears_its_marker_once_it_has_run() -> None:
    """The refusal applies to exactly one turn.

    Reaching the clearing point at all takes a declined turn that ends with a
    completed tool batch, and a declined turn only offers final_answer. The one
    route there is the escape hatch the engine already has: the model names a
    tool outside the narrowed set, the same-turn retry hands the full set back
    and drops the forcing with it, and the work tool the model then calls runs
    to completion.

    Without the loop-local half of the refusal the marker survives that turn
    and turns the next ordinary turn into a second forced one, repeating a
    warning about missing evidence that was already delivered.
    """
    pattern = ReActPattern(max_iterations=6)
    pattern.force_final_answer_next = True
    pattern.forced_answer_reason = FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION
    pattern.forced_answer_compaction_recoveries = FORCED_ANSWER_RECOVERY_BUDGET
    llm = RecordingLLM(
        [
            # Names a tool the narrowed set does not carry.
            tool_call_response("calculator", call_id="stray"),
            # The retry has the full set back, so this one actually runs.
            tool_call_response("calculator", call_id="work"),
            final_answer_response(),
        ]
    )
    context = build_context(max_messages=40)

    await pattern.run(
        context=context,
        tools=[NamedTool("calculator")],
        llm=llm,
        compact_llm=RecordingLLM(
            [{"content": f"Summary {index}."} for index in range(4)]
        ),
        runtime=PatternRuntime(),
    )

    assert HONEST_PHRASE in instruction_of(llm, 0)
    assert pattern.forced_answer_recovery_followup is None
    # The turn after the completed batch is an ordinary one: the refusal was
    # spent, so nothing forces this turn and nothing repeats the warning.
    assert tool_names_of(llm, 2) != ["final_answer"]
    assert HONEST_PHRASE not in instruction_of(llm, 2)


@pytest.mark.asyncio
async def test_a_finished_run_leaves_no_follow_up_marker_behind() -> None:
    """Finishing clears the marker so a later reload is an ordinary turn."""
    pattern = ReActPattern(max_iterations=6)
    pattern.forced_answer_recovery_followup = FORCED_ANSWER_FOLLOWUP_NO_EVIDENCE
    context = build_context(max_messages=40)
    context.compact_config.threshold = 10**9
    llm = RecordingLLM([final_answer_response()])

    result = await pattern.run(
        context=context,
        tools=[NamedTool("calculator")],
        llm=llm,
        compact_llm=None,
        runtime=PatternRuntime(),
    )

    assert result["success"] is True
    assert pattern.get_state()["forced_answer_recovery_followup"] is None

    resumed = ReActPattern(max_iterations=6)
    resumed.load_state(pattern.get_state())
    resumed.status = "thinking"
    resumed.current_iteration = 0
    resumed_llm = RecordingLLM([final_answer_response()])
    await resumed.run(
        context=build_context(max_messages=40, execution_id="after-final"),
        tools=[NamedTool("calculator")],
        llm=resumed_llm,
        compact_llm=None,
        runtime=PatternRuntime(),
    )
    assert tool_names_of(resumed_llm) != ["final_answer"]
    assert HONEST_PHRASE not in instruction_of(resumed_llm)


def _forced_answer_set_nodes(tree: ast.Module) -> list[tuple[ast.AST, ast.Assign]]:
    """Every ``self.force_final_answer_next = True`` with its enclosing block."""
    found: list[tuple[ast.AST, ast.Assign]] = []
    for parent in ast.walk(tree):
        for field_name in parent._fields:
            body = getattr(parent, field_name, None)
            if not isinstance(body, list):
                continue
            for node in body:
                if not isinstance(node, ast.Assign):
                    continue
                if not _assigns_attribute(node, "force_final_answer_next"):
                    continue
                if isinstance(node.value, ast.Constant) and node.value.value is True:
                    found.append((parent, node))
    return found


def _assigns_attribute(node: ast.Assign, attribute: str) -> bool:
    for target in node.targets:
        if (
            isinstance(target, ast.Attribute)
            and target.attr == attribute
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            return True
    return False


def _reason_names_in_block(parent: ast.AST) -> list[str]:
    names: list[str] = []
    for field_name in parent._fields:
        body = getattr(parent, field_name, None)
        if not isinstance(body, list):
            continue
        for node in body:
            if not isinstance(node, ast.Assign):
                continue
            if not _assigns_attribute(node, "forced_answer_reason"):
                continue
            if isinstance(node.value, ast.Name):
                names.append(node.value.id)
    return names


def test_every_forced_answer_set_site_records_its_reason() -> None:
    """Raising the forcing flag and naming the reason are one action.

    The recovery gate refuses to undo a forcing it cannot name, so a site that
    raises the flag without a reason does not break loudly -- it silently turns
    the whole recovery path off for that source.

    Bounds worth knowing about this check:
      1. It reads this one module. Another module assigning the attribute on a
         pattern object is invisible here; the repository-wide search below is
         what covers that, and only for this literal spelling.
      2. It recognises only ``ast.Assign`` of the literal ``True``. Assigning a
         variable, a ``bool(...)`` call, or going through ``setattr`` slips past.
      3. It reads statement blocks, not runtime reachability. A reason assigned
         in the same block and overwritten later still counts as paired; the
         behavioural tests above cover that layer.
      4. It reads the syntax tree, never the source text. This module writes the
         attribute name into comments and docstrings, and a text count would go
         red for reasons that have nothing to do with a missing reason.
    """
    source = REACT_SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    pairs = _forced_answer_set_nodes(tree)

    assert len(pairs) >= 4
    for parent, node in pairs:
        reason_names = _reason_names_in_block(parent)
        assert reason_names, (
            "force_final_answer_next is set to True at line "
            f"{node.lineno} without a forced_answer_reason in the same block"
        )
        for name in reason_names:
            assert name.startswith("FORCED_ANSWER_REASON_")
            assert getattr(react_module, name) in FORCED_ANSWER_REASONS

    assert FORCED_ANSWER_REASONS_EXEMPT_FROM_RECOVERY <= FORCED_ANSWER_REASONS
    assert FORCED_ANSWER_FOLLOWUPS == {
        FORCED_ANSWER_FOLLOWUP_REFETCHED,
        FORCED_ANSWER_FOLLOWUP_NO_EVIDENCE,
    }
    assert FORCED_ANSWER_RECOVERY_MIN_REMAINING_ITERATIONS == 2
    assert FORCED_ANSWER_REASON_FINALIZE_AFTER_TOOL in FORCED_ANSWER_REASONS


def test_the_forcing_flag_is_only_written_inside_the_react_module() -> None:
    """Nothing outside this module raises the flag behind the gate's back."""
    src_root = REACT_SOURCE_PATH.parents[4]
    pattern = re.compile(r"force_final_answer_next\s*=")
    offenders = [
        path
        for path in src_root.rglob("*.py")
        if path != REACT_SOURCE_PATH
        and pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
