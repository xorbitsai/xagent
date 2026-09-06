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
from xagent.core.agent.context.execution import (
    COMPACT_DROPPED_TOOL_NAME_MAX_CHARS,
    COMPACT_DROPPED_TOOL_NOTICE_MAX_CHARS,
    COMPACT_DROPPED_TOOL_NOTICE_MAX_NAMES,
)
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
from xagent.core.model.chat.exceptions import LLMToolProtocolError

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
async def test_a_recovery_turn_resumes_into_a_recovery_turn() -> None:
    """A pause inside a recovery turn must not resume into a forced answer.

    The gate clears this turn for recovery, which undoes the forcing and hands
    the dropped tool back. Both halves of that decision have to be undone
    together: the loop local picks this turn's instruction, and the marker is
    what a resume reads. A marker still saying "answer without the evidence"
    would restore the forcing this turn had just dropped, and the resumed turn
    would answer without the values it was on its way to fetch. Setup below is
    the refused summary shape with one line removed -- the spent-budget
    override -- which is the only thing that separates the two paths.
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

    # The live turn recovered: the forcing is gone and the dropped tool is
    # back, so the marker that would reinstate the forcing has to be gone too.
    assert RECOVERY_CHECKPOINT in checkpoint_labels(runtime)
    assert tool_names_of(interrupting_llm) != ["final_answer"]
    state = pattern.get_state()
    assert state["forced_answer_recovery_followup"] is None

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

    # Resuming lands in the turn the pause interrupted, not in a forced answer.
    assert tool_names_of(resumed_llm) != ["final_answer"]


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


class FailingTool(NamedTool):
    """Work tool whose result is a structured failure."""

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        return {"success": False, "error": f"{self.name} is unavailable"}


def state_at(
    runtime: PatternRuntime, label: str, occurrence: int = 0
) -> dict[str, Any]:
    matching = [entry for entry in runtime.checkpoints if entry.get("label") == label]
    return dict(matching[occurrence].get("pattern_state") or {})


def recovery_runtime(**metadata: Any) -> ScriptedCompactionRuntime:
    """Runtime that destroys evidence on the first turn and nothing after.

    Only the first turn needs to lose evidence. Letting every turn compact
    would make the recovery turn's own re-fetch the next casualty, which is a
    different scenario from the one under test.
    """
    return ScriptedCompactionRuntime([compact_result(**metadata)])


FORCING_SOURCES = [
    "repeated_tool_decision",
    "finalize_after_tool_result",
    "latest_tool_result_inline",
    "empty_final_answer_rejected",
]


def forced_pattern(source: str, **kwargs: Any) -> ReActPattern:
    if source == "latest_tool_result_inline":
        # No flag: the loop derives the forcing from the last tool result.
        return ReActPattern(finalize_after_tool_result=True, **kwargs)
    pattern = ReActPattern(**kwargs)  # type: ignore[arg-type]
    pattern.force_final_answer_next = True
    pattern.forced_answer_reason = {
        "repeated_tool_decision": FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION,
        "finalize_after_tool_result": FORCED_ANSWER_REASON_FINALIZE_AFTER_TOOL,
        "empty_final_answer_rejected": FORCED_ANSWER_REASON_EMPTY_FINAL_ANSWER,
    }[source]
    return pattern


@pytest.mark.asyncio
@pytest.mark.parametrize("forcing_source", FORCING_SOURCES)
@pytest.mark.parametrize("compact_strategy", ["llm_summary", "truncate"])
async def test_recovery_hands_back_the_tools_whose_results_were_dropped(
    forcing_source: str, compact_strategy: str
) -> None:
    """The turn stops being a forced answer and becomes a re-fetch instead.

    Undoing the forcing is the whole point: the model is handed exactly the
    tools whose observations went missing, plus final_answer so it can still
    finish, and nothing else. Tools that lost nothing stay out, and the two
    tools that talk to the user rather than read anything stay out on every
    path.

    The run picks ``tool_choice="auto"`` so the forcing override is visible: a
    forced turn pins the choice to "required" regardless, and a recovery turn
    has to be back on the run's own setting.
    """
    dropped = NamedTool("list_clients")
    untouched = NamedTool("read_ledger")
    pattern = forced_pattern(forcing_source, max_iterations=6, tool_choice="auto")
    runtime = recovery_runtime(
        by_name={"list_clients": 3}, count=3, strategy=compact_strategy
    )
    llm = RecordingLLM([tool_call_response("list_clients"), final_answer_response()])

    await pattern.run(
        context=build_context(tool_name="list_clients", max_messages=40),
        tools=[dropped, untouched],
        llm=llm,
        compact_llm=None,
        runtime=runtime,
    )

    recovery_tools = tool_names_of(llm, 0)
    assert sorted(recovery_tools) == ["final_answer", "list_clients"]
    assert "read_ledger" not in recovery_tools
    assert "send_message" not in recovery_tools
    assert "ask_user_question" not in recovery_tools
    assert llm.calls[0]["tool_choice"] == "auto"
    assert pattern.forced_answer_compaction_recoveries == 1
    assert checkpoint_labels(runtime).count(RECOVERY_CHECKPOINT) == 1
    # The recovery turn cleared the forcing outright rather than deferring it.
    assert state_at(runtime, RECOVERY_CHECKPOINT)["force_final_answer_next"] is False
    assert state_at(runtime, RECOVERY_CHECKPOINT)["forced_answer_reason"] is None
    # The turn after it inherits a verdict, whichever way the re-fetch went.
    assert (
        state_at(runtime, "before_llm", 1)["forced_answer_recovery_followup"]
        in FORCED_ANSWER_FOLLOWUPS
    )


@pytest.mark.asyncio
async def test_a_recovery_turn_is_followed_by_a_forced_answer_turn() -> None:
    """A successful re-fetch buys exactly one ordinary forced answer turn."""
    pattern = ReActPattern(max_iterations=6)
    pattern.force_final_answer_next = True
    pattern.forced_answer_reason = FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION
    runtime = recovery_runtime(by_name={"list_clients": 2}, count=2)
    llm = RecordingLLM([tool_call_response("list_clients"), final_answer_response()])

    await pattern.run(
        context=build_context(tool_name="list_clients", max_messages=40),
        tools=[NamedTool("list_clients")],
        llm=llm,
        compact_llm=None,
        runtime=runtime,
    )

    assert (
        state_at(runtime, "before_llm", 1)["forced_answer_recovery_followup"]
        == FORCED_ANSWER_FOLLOWUP_REFETCHED
    )
    assert tool_names_of(llm, 1) == ["final_answer"]
    assert llm.calls[1]["tool_choice"] == "required"
    # The evidence came back, so the ordinary wording is correct again.
    assert HONEST_PHRASE not in instruction_of(llm, 1)
    assert STALE_EVIDENCE_PHRASE in instruction_of(llm, 1)
    assert pattern.forced_answer_reason == FORCED_ANSWER_REASON_COMPACTION_RECOVERY


@pytest.mark.asyncio
async def test_a_stale_success_left_in_the_tail_does_not_count_as_recovered() -> None:
    """The message-dropping path keeps a decoy, and the verdict ignores it.

    Its tail window can hold an older successful observation of the same tool
    while removing the one the answer needed. Asking "is there tool evidence in
    context" answers yes here and is exactly the wrong question.
    """
    pattern = ReActPattern(max_iterations=6)
    pattern.force_final_answer_next = True
    pattern.forced_answer_reason = FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION
    runtime = recovery_runtime(
        by_name={"list_clients": 1}, count=1, strategy="truncate"
    )
    llm = RecordingLLM([tool_call_response("list_clients"), final_answer_response()])
    context = build_context(tool_name="list_clients", max_messages=40)

    await pattern.run(
        context=context,
        tools=[FailingTool("list_clients")],
        llm=llm,
        compact_llm=None,
        runtime=runtime,
    )

    assert any(message.role == "tool" for message in context.messages)
    assert HONEST_PHRASE in instruction_of(llm, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "second_forcing_from_repeated_decision",
        "recovery_followup_never_chains",
    ],
)
async def test_the_recovery_budget_is_spent_once_per_pattern(scenario: str) -> None:
    """One re-fetch per pattern, and the turn it buys can never buy another.

    Two separate things hold the line. The budget stops a second recovery in
    the same run, and the follow-up turn's own reason sits in the exempt set so
    that even with budget to spare it cannot recover again -- which is what
    stops recovery from chaining indefinitely.
    """
    pattern = ReActPattern(max_iterations=8)
    pattern.force_final_answer_next = True
    pattern.forced_answer_reason = FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION
    if scenario == "recovery_followup_never_chains":
        # Budget deliberately widened: whatever refuses the second recovery
        # here, it is not the budget.
        pattern.forced_answer_compaction_recoveries = -1
    runtime = ScriptedCompactionRuntime(
        [
            compact_result(by_name={"list_clients": 2}, count=2),
            compact_result(by_name={"list_clients": 2}, count=2),
        ]
    )
    llm = RecordingLLM([tool_call_response("list_clients"), final_answer_response()])

    await pattern.run(
        context=build_context(tool_name="list_clients", max_messages=40),
        tools=[NamedTool("list_clients")],
        llm=llm,
        compact_llm=None,
        runtime=runtime,
    )

    # Turn 0 recovered. Turn 1 lost evidence again and must not recover.
    assert checkpoint_labels(runtime).count(RECOVERY_CHECKPOINT) == 1
    assert tool_names_of(llm, 1) == ["final_answer"]
    assert HONEST_PHRASE in instruction_of(llm, 1)


@pytest.mark.parametrize(
    ("name_count", "name_length"),
    [(0, 8), (1, 8), (20, 8), (21, 8), (1, 65), (20, 64)],
)
def test_the_recovery_message_authorizes_a_bounded_refetch(
    name_count: int, name_length: int
) -> None:
    """Four things must be said, and the tool names must stay bounded.

    The names come from dynamic server configuration and this message enters
    the context on every recovery, so it mirrors all three bounds the
    compaction notice already applies rather than inventing its own.
    """
    pattern = ReActPattern(max_iterations=6)
    names = {
        f"tool_{index}".ljust(name_length, "z")[:name_length]
        for index in range(name_count)
    }
    message = pattern._forced_answer_recovery_message(names)

    # 1: which observations went missing.
    assert "removed from context by compaction" in message
    # 2: re-fetching is expected, and it outranks the standing advice.
    assert "expected action" in message
    assert "takes priority over any earlier" in message
    assert "not to repeat completed tool work" in message
    # 3: only reading tools may be re-run.
    assert "Only re-run tools that read" in message
    assert "writes, sends, executes, or otherwise changes state" in message
    # 4: no answer before the values are back, and no stand-in for them.
    assert "Do not give the final answer before" in message
    assert "reconstructing, estimating, or illustrating" in message

    listed = [line for line in message.splitlines() if line.startswith("- tool_")]
    assert len(listed) <= COMPACT_DROPPED_TOOL_NOTICE_MAX_NAMES
    assert all(len(line) - 2 <= COMPACT_DROPPED_TOOL_NAME_MAX_CHARS for line in listed)
    assert len(message) <= COMPACT_DROPPED_TOOL_NOTICE_MAX_CHARS + max(
        (len(line) + 1 for line in message.splitlines()), default=0
    )
    if name_count > len(listed):
        assert "additional tool name" in message
    if name_count:
        assert listed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        "no_evidence_dropped",
        "refetch_succeeds",
        "refetch_fails",
        "model_answers_without_refetching",
        "model_answers_in_plain_text",
        "model_calls_out_of_set_tool",
        "single_call_cannot_recover",
        "recovery_at_the_exact_iteration_floor",
    ],
)
async def test_an_evidence_loss_still_ends_the_run_with_an_answer(
    outcome: str,
) -> None:
    """However the recovery turn ends, the run delivers an answer.

    Spending a turn on re-fetching only helps if the run can still afford the
    answer afterwards, which is what the remaining-iterations floor buys. A run
    that recovers and then hits its iteration ceiling would have been better
    off never recovering.
    """
    tools: list[Any] = [NamedTool("list_clients")]
    pattern = ReActPattern(max_iterations=6)
    pattern.force_final_answer_next = True
    pattern.forced_answer_reason = FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION
    scripted: list[Any] = [compact_result(by_name={"list_clients": 2}, count=2)]
    responses: list[Any] = [
        tool_call_response("list_clients"),
        final_answer_response(),
    ]

    if outcome == "no_evidence_dropped":
        scripted = [compact_result(count=0, by_name={})]
        responses = [final_answer_response()]
    elif outcome == "refetch_fails":
        tools = [FailingTool("list_clients")]
    elif outcome == "model_answers_without_refetching":
        responses = [final_answer_response()]
    elif outcome == "model_answers_in_plain_text":
        responses = [{"content": "Answering from the summary.", "done": True}]
    elif outcome == "model_calls_out_of_set_tool":
        tools = [NamedTool("list_clients"), NamedTool("read_ledger")]
        responses = [
            tool_call_response("read_ledger"),
            final_answer_response(),
        ]
    elif outcome == "single_call_cannot_recover":
        pattern = ReActPattern(max_iterations=2, finalize_after_tool_result=True)
        pattern.current_iteration = 1
        pattern.force_final_answer_next = True
        pattern.forced_answer_reason = FORCED_ANSWER_REASON_FINALIZE_AFTER_TOOL
        responses = [final_answer_response()]
    elif outcome == "recovery_at_the_exact_iteration_floor":
        # Exactly two iterations left: one to re-fetch, one to answer.
        pattern = ReActPattern(max_iterations=4)
        pattern.current_iteration = 2
        pattern.force_final_answer_next = True
        pattern.forced_answer_reason = FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION
        tools = [FailingTool("list_clients")]

    runtime = ScriptedCompactionRuntime(scripted)
    llm = RecordingLLM(responses)
    result = await pattern.run(
        context=build_context(tool_name="list_clients", max_messages=40),
        tools=tools,
        llm=llm,
        compact_llm=None,
        runtime=runtime,
    )

    assert result["success"] is True
    assert isinstance(result["response"], str) and result["response"].strip()
    assert result.get("status") != "max_iterations"
    assert "max_iterations" not in checkpoint_labels(runtime)

    if outcome == "single_call_cannot_recover":
        assert checkpoint_labels(runtime).count(RECOVERY_CHECKPOINT) == 0
        assert HONEST_PHRASE in instruction_of(llm, 0)
    elif outcome == "no_evidence_dropped":
        assert checkpoint_labels(runtime).count(RECOVERY_CHECKPOINT) == 0
        assert HONEST_PHRASE not in instruction_of(llm, 0)
    else:
        assert checkpoint_labels(runtime).count(RECOVERY_CHECKPOINT) == 1

    if outcome == "recovery_at_the_exact_iteration_floor":
        # The floor is the only reason this run had room for both turns.
        assert pattern.forced_answer_compaction_recoveries == 1
        assert HONEST_PHRASE in instruction_of(llm, 1)
    if outcome == "refetch_succeeds":
        assert HONEST_PHRASE not in instruction_of(llm, 1)
    if outcome == "refetch_fails":
        assert HONEST_PHRASE in instruction_of(llm, 1)
        assert tool_names_of(llm, 1) == ["final_answer"]
        assert STALE_EVIDENCE_PHRASE not in whole_prompt_of(llm, 1)
    if outcome == "model_calls_out_of_set_tool":
        # The model fetched something, but not what went missing.
        assert HONEST_PHRASE in instruction_of(llm, 1)


class SucceedsThenFailsTool(NamedTool):
    """Work tool that succeeds on its first call and fails afterwards."""

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        if len(self.calls) == 1:
            return {"output": f"{self.name} first result"}
        return {"success": False, "error": f"{self.name} is unavailable now"}


@pytest.mark.asyncio
async def test_an_older_successful_call_does_not_pass_for_a_recovered_one() -> None:
    """The ledger outlives compaction, so the verdict cannot go by name.

    The observation compaction dropped was itself a successful call of a
    recovery-set tool, and its ledger entry survives -- compaction removes
    messages, not ledger entries. Asking "has a tool of this name ever
    succeeded" therefore answers yes before the recovery turn even starts.
    Here the same tool succeeds on the first turn and fails on the re-fetch,
    so the two readings disagree and only the turn-scoped one is right.
    """
    tool = SucceedsThenFailsTool("list_clients")
    pattern = ReActPattern(max_iterations=6, finalize_after_tool_result=True)
    runtime = ScriptedCompactionRuntime(
        [None, compact_result(by_name={"list_clients": 1}, count=1)]
    )
    llm = RecordingLLM(
        [
            tool_call_response("list_clients", call_id="first"),
            tool_call_response("list_clients", call_id="refetch"),
            final_answer_response(),
        ]
    )
    context = ExecutionContext(execution_id="ledger-scope")
    context.add_user_message("List every client.")

    await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        compact_llm=None,
        runtime=runtime,
    )

    assert len(tool.calls) == 2
    assert checkpoint_labels(runtime).count(RECOVERY_CHECKPOINT) == 1
    assert "first" in pattern.tool_ledger
    assert pattern.tool_ledger["first"].status == "completed"
    assert (
        state_at(runtime, "before_llm", 2)["forced_answer_recovery_followup"]
        == FORCED_ANSWER_FOLLOWUP_NO_EVIDENCE
    )
    assert HONEST_PHRASE in instruction_of(llm, 2)


COMPLETE_SET_PHRASE = "Re-decide this turn using the complete current tool set"
TURN_SET_PHRASE = "which is the set available on this turn"


class ProtocolErrorLLM(RecordingLLM):
    """Chat LLM that raises a provider protocol error on a chosen call."""

    def __init__(self, responses: list[Any], *, fail_on: int, code: str) -> None:
        super().__init__(responses)
        self.fail_on = fail_on
        self.code = code

    async def chat(self, **kwargs: Any) -> Any:
        index = len(self.calls)
        self.calls.append(kwargs)
        if index == self.fail_on:
            raise LLMToolProtocolError(
                provider="deepseek",
                code=self.code,
                message=f"provider returned {self.code}",
            )
        if not self.responses:
            return {"content": "fallback answer", "done": True}
        return self.responses.pop(0)


async def run_protocol_retry(
    turn_kind: str,
    protocol_error: str,
    *,
    user_interaction_enabled: bool = True,
) -> tuple[ReActPattern, ProtocolErrorLLM, ScriptedCompactionRuntime, set[str]]:
    """Drive one turn whose first LLM call fails the tool protocol."""
    tools = [NamedTool("list_clients"), NamedTool("read_ledger")]
    pattern = ReActPattern(
        max_iterations=6, user_interaction_enabled=user_interaction_enabled
    )
    scripted: list[Any] = [None]
    recovery_set: set[str] = set()

    if turn_kind == "recovery_turn":
        pattern.force_final_answer_next = True
        pattern.forced_answer_reason = FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION
        scripted = [compact_result(by_name={"list_clients": 2}, count=2)]
        recovery_set = {"list_clients", "final_answer"}
    elif turn_kind == "forced_turn":
        pattern.force_final_answer_next = True
        pattern.forced_answer_reason = FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION

    llm = ProtocolErrorLLM(
        [tool_call_response("list_clients"), final_answer_response()],
        fail_on=0,
        code=(
            "unavailable_tool_call"
            if protocol_error == "unavailable_tool_call"
            else "invalid_tool_protocol"
        ),
    )
    runtime = ScriptedCompactionRuntime(scripted)
    await pattern.run(
        context=build_context(tool_name="list_clients", max_messages=40),
        tools=tools,
        llm=llm,
        compact_llm=None,
        runtime=runtime,
    )
    return pattern, llm, runtime, recovery_set


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_kind", ["recovery_turn", "ordinary_turn", "forced_turn"])
@pytest.mark.parametrize(
    "protocol_error", ["unavailable_tool_call", "invalid_protocol"]
)
async def test_the_protocol_retry_uses_this_turns_own_tool_set(
    turn_kind: str, protocol_error: str
) -> None:
    """A retry must not quietly widen the turn it is repairing.

    The retry exists to let the model re-decide with the full picture, so on an
    ordinary or forced turn it keeps handing back the run's whole tool set,
    unchanged. A recovery turn is the one case where that set has to be
    trimmed: it deliberately narrowed the tools, and the two tools that contact
    the user rather than read anything have no business reappearing through a
    repair path.
    """
    _pattern, llm, _runtime, recovery_set = await run_protocol_retry(
        turn_kind, protocol_error
    )

    base_names = {
        "list_clients",
        "read_ledger",
        "final_answer",
        "send_message",
        "ask_user_question",
    }
    retry_names = set(tool_names_of(llm, 1))

    if turn_kind == "recovery_turn":
        assert retry_names == base_names - {"send_message", "ask_user_question"}
        # Stated as properties rather than a copied name list: the retry set
        # must be able to reach everything the recovery turn offered, and must
        # not smuggle in a tool that talks to the user.
        assert recovery_set <= retry_names
        assert retry_names & {"send_message", "ask_user_question"} == set()
    elif turn_kind == "ordinary_turn":
        assert retry_names == base_names
    elif protocol_error == "unavailable_tool_call":
        # The forced turn's whole-set recovery semantics are untouched.
        assert retry_names == base_names
    else:
        assert retry_names == {"final_answer"}

    retry_instruction = instruction_of(llm, 1)
    if protocol_error == "unavailable_tool_call":
        if turn_kind == "recovery_turn":
            assert TURN_SET_PHRASE in retry_instruction
            assert COMPLETE_SET_PHRASE not in retry_instruction
        else:
            assert COMPLETE_SET_PHRASE in retry_instruction


@pytest.mark.asyncio
async def test_narrowing_the_retry_set_removes_nothing_when_asking_is_off() -> None:
    """With user interaction off the trim is already done and takes no work."""
    _pattern, llm, _runtime, _recovery_set = await run_protocol_retry(
        "recovery_turn", "unavailable_tool_call", user_interaction_enabled=False
    )
    assert set(tool_names_of(llm, 1)) == {
        "list_clients",
        "read_ledger",
        "final_answer",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("turn_kind", "protocol_error", "expects_honest_retry"),
    [
        ("declined_forced_turn", "empty_final_answer", True),
        ("declined_forced_turn", "malformed_tool_arguments", True),
        ("consumption_turn", "empty_final_answer", True),
        ("declined_forced_turn", "unavailable_tool_call", False),
    ],
)
async def test_the_retry_that_produces_the_answer_is_honest_too(
    turn_kind: str, protocol_error: str, expects_honest_retry: bool
) -> None:
    """The retry is the call that reaches the user, so it needs the wording.

    The first call of a turn can carry the honest instruction and still be
    thrown away: a blank final_answer or malformed arguments makes the retry
    the one that actually produces the answer. If the retry rebuilds its own
    prompt without knowing the evidence is gone, it hands back the very
    sentence the turn was supposed to replace.

    The last case is the deliberate exception. There the model named a tool
    outside the narrowed set, the retry hands the full set back, and the right
    move is to fetch the missing values -- not to be told they are unreachable.
    """
    pattern = ReActPattern(max_iterations=6)
    scripted: list[Any] = [compact_result(by_name={"list_clients": 2}, count=2)]
    if turn_kind == "consumption_turn":
        # This turn is forced because the previous recovery turn came back
        # empty-handed; nothing is dropped on the turn itself.
        pattern.forced_answer_recovery_followup = FORCED_ANSWER_FOLLOWUP_NO_EVIDENCE
        scripted = [None]
    else:
        pattern.force_final_answer_next = True
        pattern.forced_answer_reason = FORCED_ANSWER_REASON_REPEATED_TOOL_DECISION
        pattern.forced_answer_compaction_recoveries = FORCED_ANSWER_RECOVERY_BUDGET

    llm: RecordingLLM
    if protocol_error == "empty_final_answer":
        blank = final_answer_response()
        blank["tool_calls"][0]["function"]["arguments"] = (
            '{"response_language":"English","answer":"   "}'
        )
        llm = RecordingLLM([blank, final_answer_response()])
    else:
        llm = ProtocolErrorLLM(
            [final_answer_response()],
            fail_on=0,
            code=(
                "unavailable_tool_call"
                if protocol_error == "unavailable_tool_call"
                else "malformed_tool_arguments"
            ),
        )

    await pattern.run(
        context=build_context(tool_name="list_clients", max_messages=40),
        tools=[NamedTool("list_clients")],
        llm=llm,
        compact_llm=None,
        runtime=ScriptedCompactionRuntime(scripted),
    )

    assert HONEST_PHRASE in instruction_of(llm, 0)
    retry_instruction = instruction_of(llm, 1)
    assert (HONEST_PHRASE in retry_instruction) is expects_honest_retry
    if expects_honest_retry:
        assert STALE_EVIDENCE_PHRASE not in retry_instruction
        assert tool_names_of(llm, 1) == ["final_answer"]
    else:
        assert COMPLETE_SET_PHRASE in retry_instruction
