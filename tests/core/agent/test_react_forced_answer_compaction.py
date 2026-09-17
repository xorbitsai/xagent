"""The forced-answer turn keeps its evidence, and says so when it lost some.

A turn whose schema is already down to ``final_answer`` does not compact, so
the values it answers from are still there; and once any compaction on this
context removed an observation, every prompt that answers -- or decides to --
without tools says so instead of claiming the results accumulated.
"""

from __future__ import annotations

import ast
import json
import logging
import pathlib
import re
from typing import Any

import pytest

from tests.core.agent.forced_answer_harness import (
    OBSERVATION_MARKER,
    CalculatorTool,
    CompactingLLM,
    ScriptedLLM,
    WindowlessCompactingLLM,
    build_context,
    prompt_text,
)
from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.context import ContextManager
from xagent.core.agent.context.execution import (
    TOOL_EVIDENCE_REMOVED_METADATA_KEY as KEY,
)
from xagent.core.agent.context.execution import (
    CompactResult,
    note_compaction_evidence_loss,
    tool_evidence_state,
)
from xagent.core.agent.grounding import (
    EVIDENCE_REMOVED_FACTS,
    EVIDENCE_UNKNOWN_FACTS,
    evidence_facts,
    grounding_rule,
)
from xagent.core.agent.pattern.auto.auto import (
    DECISION_TOOL_NAME,
    AutoAction,
    AutoPattern,
)
from xagent.core.agent.pattern.dag.dag import DAGPattern
from xagent.core.model.chat.exceptions import (
    LLMContextLengthError,
    LLMToolProtocolError,
)

FACTS_HEAD = EVIDENCE_REMOVED_FACTS.split(".")[0]
UNKNOWN_FACTS_HEAD = EVIDENCE_UNKNOWN_FACTS.split(".")[0]

STATES = [
    pytest.param("intact", id="intact"),
    pytest.param("removed", id="removed"),
    pytest.param("unknown", id="unknown"),
]


def apply_state(context: ExecutionContext, state: str) -> None:
    """Put a context into one evidence state, absence included."""
    if state == "unknown":
        context.metadata.pop(KEY, None)
    else:
        context.metadata[KEY] = state == "removed"


def facts_head(state: str) -> str:
    """The head fragment expected in text for one evidence state, empty for intact."""
    if state == "intact":
        return ""
    if state == "unknown":
        return UNKNOWN_FACTS_HEAD
    return FACTS_HEAD


def tool_call(name: str, arguments: str = "{}") -> dict[str, Any]:
    call = {
        "id": f"c-{name}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    return {"content": "", "tool_calls": [call]}


def final_answer_call() -> dict[str, Any]:
    return tool_call("final_answer", '{"answer": "done", "outcome": "completed"}')


def empty_final_answer_call() -> dict[str, Any]:
    return tool_call("final_answer", '{"answer": "", "outcome": "completed"}')


async def run_one_turn(
    *,
    context: ExecutionContext,
    forced: bool,
    llm: Any | None = None,
    compact_llm: Any | None = None,
    pattern: ReActPattern | None = None,
) -> tuple[ReActPattern, ScriptedLLM]:
    """Drive exactly one ReAct iteration, forced or ordinary."""
    llm = llm or ScriptedLLM([final_answer_call()])
    pattern = pattern or ReActPattern(max_iterations=1)
    pattern.force_final_answer_next = forced
    runtime = PatternRuntime(execution_id=context.execution_id)
    await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=runtime,
        compact_llm=compact_llm or CompactingLLM(),
    )
    return pattern, llm


# --------------------------------------------------------------------------
# Invariant A -- a forced turn does not compact
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forced_turn_keeps_every_observation_and_does_not_compact() -> None:
    """The raw observation text reaches the model on a forced turn."""
    context = build_context(observations=6, threshold=500)
    before = len(context.messages)

    _, llm = await run_one_turn(context=context, forced=True)

    sent = prompt_text(llm.calls[0]["messages"])
    assert OBSERVATION_MARKER.format(index=0) in sent
    assert OBSERVATION_MARKER.format(index=5) in sent
    assert len(context.messages) >= before  # nothing was taken away
    assert "Compacted conversation summary:" not in sent
    assert context.metadata[KEY] is False


@pytest.mark.asyncio
async def test_ordinary_turn_still_compacts_and_latches() -> None:
    """The skip is confined to the forced turn."""
    context = build_context(observations=6, threshold=500)
    before = len(context.messages)

    await run_one_turn(context=context, forced=False)

    assert len(context.messages) < before
    assert context.metadata[KEY] is True


@pytest.mark.asyncio
async def test_route_and_call_prompts_match_only_on_the_forced_turn() -> None:
    """Routing and the real call see the same messages when nothing compacts.

    Both prompts are locals inside the loop, so ``_messages_for_llm`` is wrapped
    to record what it returned. Asserting only "called twice" would pass under
    every implementation, including the broken one.
    """
    built: list[list[dict[str, Any]]] = []

    def record(pattern: ReActPattern) -> ReActPattern:
        original = pattern._messages_for_llm

        def wrapper(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            built.append(original(*args, **kwargs))
            return built[-1]

        pattern._messages_for_llm = wrapper  # type: ignore[method-assign]
        return pattern

    for forced, same in ((True, True), (False, False)):
        built.clear()
        await run_one_turn(
            context=build_context(observations=6, threshold=500),
            forced=forced,
            pattern=record(ReActPattern(max_iterations=1)),
        )
        assert (built[0] == built[1]) is same


class ProtocolErrorOnFirstCallLLM(ScriptedLLM):
    """Replays queued responses, but raises a protocol error on the first call.

    One of the two exits that clear the forced flag is only reachable this
    way: the code that says the model called a tool the narrowed schema no
    longer offers arrives as an adapter exception, not inside a response body.
    """

    def __init__(self, responses: list[Any], *, code: str) -> None:
        super().__init__(responses)
        self.code = code

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise LLMToolProtocolError(
                provider="fake", code=self.code, message="tool is unavailable"
            )
        if not self.responses:
            return final_answer_call()
        return self.responses.pop(0)


def _forced_next(pattern: ReActPattern) -> ReActPattern:
    """Force the next turn through ``force_final_answer_next`` directly."""
    pattern.force_final_answer_next = True
    return pattern


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # Built per run rather than held in this list: a scripted model is spent
    # once it has replayed its queue.
    "make_llm, make_pattern, tools, skips_expected, compactions_expected",
    [
        (
            lambda: ScriptedLLM([final_answer_call()]),
            lambda: _forced_next(ReActPattern(max_iterations=2)),
            [],
            1,
            0,
        ),
        (
            lambda: ScriptedLLM([{"content": "plain text answer", "tool_calls": []}]),
            lambda: _forced_next(ReActPattern(max_iterations=2)),
            [],
            1,
            0,
        ),
        (
            lambda: ScriptedLLM([empty_final_answer_call(), empty_final_answer_call()]),
            lambda: _forced_next(ReActPattern(max_iterations=2)),
            [],
            1,
            0,
        ),
        (
            lambda: ScriptedLLM([tool_call("list_clients"), tool_call("list_clients")]),
            lambda: _forced_next(ReActPattern(max_iterations=2)),
            [],
            1,
            1,
        ),
        (
            lambda: ProtocolErrorOnFirstCallLLM(
                [tool_call("list_clients")], code="unavailable_tool_call"
            ),
            lambda: _forced_next(ReActPattern(max_iterations=2)),
            [],
            1,
            1,
        ),
        (
            lambda: ProtocolErrorOnFirstCallLLM(
                [tool_call("calculator", '{"expression": "1+1"}')],
                code="unavailable_tool_call",
            ),
            lambda: ReActPattern(max_iterations=4, finalize_after_tool_result=True),
            [CalculatorTool()],
            2,
            0,
        ),
    ],
    ids=[
        "final_answer",
        "plain_text",
        "empty_answer_then_repair_fails",
        "out_of_schema_tool",
        "unavailable_tool_call",
        "single_call_after_escape",
    ],
)
async def test_the_skip_lasts_exactly_as_long_as_its_source_forces(
    caplog: pytest.LogCaptureFixture,
    make_llm: Any,
    make_pattern: Any,
    tools: list[Any],
    skips_expected: int,
    compactions_expected: int,
) -> None:
    """The skip lasts exactly as long as its source keeps forcing -- one turn or two.

    Two turns are offered to every shape. What is asserted is how many turns
    skipped compaction -- counted off the production log line, one per skip --
    and how many turns compacted. Asserting ``force_final_answer_next is
    False`` instead would hold under every implementation, right or wrong:
    ``_finalize_success`` clears that field on any run that ends normally, so
    it cannot tell an exit that cleared the flag itself from a run that merely
    finished.

    The two exits that clear the flag are separate lines in separate handlers,
    so they get one shape each; deleting either one leaves the other green.
    That guarantee is scoped to ``force_final_answer_next``: the last shape
    forces through ``finalize_after_tool_result`` (single_call mode) instead,
    which those two exits do not clear, so a turn that escapes the narrowed
    schema can still be forced again immediately after -- two skips, not one.
    """
    context = build_context(observations=6, threshold=500)
    runtime = PatternRuntime(execution_id=context.execution_id)
    compactions: list[Any] = []
    compact = runtime.compact_context_if_needed

    async def counted(**kwargs: Any) -> Any:
        compactions.append(kwargs.get("metadata"))
        return await compact(**kwargs)

    runtime.compact_context_if_needed = counted  # type: ignore[method-assign]
    pattern = make_pattern()

    with caplog.at_level(logging.INFO, logger="xagent.core.agent.pattern.react.react"):
        await pattern.run(
            context=context,
            tools=tools,
            llm=make_llm(),
            runtime=runtime,
            compact_llm=CompactingLLM(),
        )

    skipped = [r for r in caplog.records if "did not compact" in r.getMessage()]
    assert len(skipped) == skips_expected
    assert len(compactions) == compactions_expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forced, threshold, expected",
    [
        (True, 500, "over_threshold=True"),
        (True, 10**6, "over_threshold=False"),
        (False, 500, None),
    ],
    ids=["forced_over", "forced_under", "ordinary_turn"],
)
async def test_the_skip_is_logged_with_numbers_only(
    caplog: pytest.LogCaptureFixture, forced: bool, threshold: int, expected: str | None
) -> None:
    """One info line per skipped compaction, ids and numbers only.

    ``over_threshold`` is what separates a skip that mattered from one on a
    turn that was never going to compact anyway. The logged ``context_tokens``
    number is asserted to be this turn's real payload -- the persisted
    messages plus the tool schemas and forced-answer system prompt actually
    sent -- and not just the persisted message list, which undercounts it.
    """
    context = build_context(observations=6, threshold=threshold)
    persisted_only = context.estimate_context_tokens()
    with caplog.at_level(logging.INFO, logger="xagent.core.agent.pattern.react.react"):
        await run_one_turn(context=context, forced=forced)

    lines = [
        r.getMessage() for r in caplog.records if "did not compact" in r.getMessage()
    ]
    if expected is None:
        assert lines == []
        return
    assert len(lines) == 1
    assert f"execution_id={context.execution_id}" in lines[0]
    assert "iteration=0" in lines[0]
    assert f"threshold={threshold}" in lines[0]
    assert expected in lines[0]
    assert OBSERVATION_MARKER.format(index=0) not in lines[0]
    assert "list_clients" not in lines[0]

    logged = int(re.search(r"context_tokens=(\d+)", lines[0]).group(1))
    # The logged number is this turn's real payload, not the persisted list.
    assert logged > persisted_only


# --------------------------------------------------------------------------
# Invariant B -- latching, and what each compaction shape does to the marker
# --------------------------------------------------------------------------


def _result(**metadata: Any) -> CompactResult:
    return CompactResult(
        compacted=True,
        original_count=10,
        final_count=2,
        strategy="llm_summary",
        metadata=dict(metadata),
    )


@pytest.mark.parametrize(
    "result, expected, warns",
    [
        (_result(dropped_tool_result_count=6), True, False),
        (_result(dropped_tool_result_count=1), True, False),
        (_result(removed_count=11, dropped_tool_result_count=0), False, False),
        (_result(removed_count=0), False, False),
        (CompactResult(False, 3, 3, "none", {}), False, False),
        (None, False, False),
        (_result(fallback_suppressed=True), False, False),
        (
            _result(dropped_context_ref_count=4, dropped_tool_result_count=0),
            False,
            False,
        ),
        (_result(dropped_tool_result_count=True), False, True),
        (_result(dropped_tool_result_count="6"), False, True),
        (_result(llm_compact_error="boom", dropped_tool_result_count=3), True, False),
        (_result(llm_summary_unusable=True, dropped_tool_result_count=3), True, False),
    ],
    ids="summary_dropped_six truncate_dropped_one assistant_text_only "
    "tail_window_kept_all under_threshold runtime_returned_none "
    "compact_window_unknown image_refs_only count_is_a_bool count_is_a_string "
    "summary_raised_then_truncated summary_unusable_then_truncated".split(),
)
def test_latch_reads_dropped_observations_not_compacted(
    caplog: pytest.LogCaptureFixture, result: Any, expected: bool, warns: bool
) -> None:
    """Only a removed observation latches the marker; only a malformed count warns.

    ``under_threshold`` and ``runtime_returned_none`` both carry no
    ``dropped_tool_result_count`` at all -- the first because the metadata is
    genuinely empty, the second because a ``None`` result never reaches that
    metadata check in the first place -- and neither warns: absence is an
    ordinary outcome of this call, not a malformed one. Each malformed-count
    row also pins what the warning may and may not say: it names the value's
    type and the execution id, and it never repeats the value itself.
    """
    context = ContextManager().create_context(execution_id="latch")
    with caplog.at_level(logging.WARNING, logger="xagent.core.agent.context.execution"):
        note_compaction_evidence_loss(context, result)

    assert context.metadata[KEY] is expected
    warnings = [
        r
        for r in caplog.records
        if r.name == "xagent.core.agent.context.execution"
        and r.levelno == logging.WARNING
    ]
    assert len(warnings) == (1 if warns else 0)
    if warns:
        message = warnings[0].getMessage()
        dropped_value = result.metadata["dropped_tool_result_count"]
        assert type(dropped_value).__name__ in message
        assert f"execution_id={context.execution_id}" in message
        assert str(dropped_value) not in message


def test_latch_is_monotonic() -> None:
    """A later lossless compaction does not clear an earlier loss."""
    context = ContextManager().create_context(execution_id="latch-monotonic")
    note_compaction_evidence_loss(context, _result(dropped_tool_result_count=3))
    note_compaction_evidence_loss(context, _result(dropped_tool_result_count=0))
    assert context.metadata[KEY] is True


def _marker_names(tree: ast.AST) -> set[str]:
    """Every local name in this module that stands for the marker key."""
    names = {"TOOL_EVIDENCE_REMOVED_METADATA_KEY"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "TOOL_EVIDENCE_REMOVED_METADATA_KEY"
            )
    return names


def _names_the_marker(node: ast.AST, names: set[str]) -> bool:
    if isinstance(node, ast.Constant):
        return node.value == KEY
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Attribute):
        return node.attr in names
    return False


def _writes_false(node: ast.AST, names: set[str]) -> bool:
    def is_false(value: ast.AST) -> bool:
        return isinstance(value, ast.Constant) and value.value is False

    if isinstance(node, ast.Assign) and is_false(node.value):
        return any(
            isinstance(target, ast.Subscript) and _names_the_marker(target.slice, names)
            for target in node.targets
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "setdefault"
        and len(node.args) == 2
        and is_false(node.args[1])
    ):
        return _names_the_marker(node.args[0], names)
    return False


def test_only_one_place_in_src_writes_the_marker_false() -> None:
    """Exactly one write node in the whole tree sets the marker to False.

    Recorded as ``(file, lineno)`` rather than just the file, because the
    invariant is about write actions, not files: a second ``False`` write
    added later in ``manager.py`` -- above or below the existing one -- would
    otherwise collapse into the same one-file set and pass unnoticed. Line
    numbers are not asserted literally (any edit above ``manager.py``'s write
    would shift the number and make the assertion fragile for no reason);
    only the file each write node lands in is checked, after the count of
    distinct write nodes has already been pinned to one.

    Read as syntax, not as text: a substring scan of one line misses a
    ``setdefault(KEY, False)``, an assignment wrapped across lines, and a
    write through an aliased import of the key.
    """
    src = pathlib.Path(__file__).resolve().parents[3] / "src" / "xagent"
    writers: set[tuple[str, int]] = set()
    for path in src.rglob("*.py"):
        tree = ast.parse(path.read_text())
        names = _marker_names(tree)
        for node in ast.walk(tree):
            if _writes_false(node, names):
                writers.add((str(path.relative_to(src)), node.lineno))
    assert len(writers) == 1, sorted(writers)
    assert {path for path, _ in writers} == {"core/agent/context/manager.py"}


def _is_compaction_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compact_context_if_needed"
    )


def _is_latch_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "note_compaction_evidence_loss"
    )


def _enclosing_function(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef], lineno: int
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The innermost function definition whose range holds this line.

    A function nested inside another both hold the line, and the nested one
    always starts later, so the containing definition with the largest start
    line is the innermost one.
    """
    holding = [
        func
        for func in functions
        if func.lineno <= lineno <= (func.end_lineno or func.lineno)
    ]
    if not holding:
        return None
    return max(holding, key=lambda func: func.lineno)


def test_every_real_compaction_call_site_latches_the_marker() -> None:
    """What this guard proves, and the gap it leaves open.

    (1) It proves that within any one function body, the number of latch
    calls is never fewer than the number of compaction calls -- a call-count
    floor, not a zero/non-zero check.
    (2) It does not prove call-by-call pairing: two compaction calls and two
    latch calls in the same function still pass even if both latches actually
    follow the same one compaction call while the other compaction call's
    loss goes unlatched -- counts alone cannot tell those two shapes apart.
    (3) Any function literally named ``compact_context_if_needed`` is skipped
    entirely, which is what lets the two forwarding wrappers
    (``dag.py:287-294`` and ``auto.py:355-362``, both of which relay to
    ``self.parent.compact_context_if_needed(...)``) through; being skipped by
    name is not evidence that those two functions latch correctly, only that
    this guard does not check them.
    (4) There is deliberately no separate exclusion keyed on the receiver
    being named ``parent``: that would also hide a genuine, unlatched call
    written through a ``self.parent.`` receiver inside an ordinarily named
    function, which is exactly the shape this test exists to catch.
    """
    src = pathlib.Path(__file__).resolve().parents[3] / "src" / "xagent"
    call_sites: set[tuple[str, str]] = set()
    compaction_counts: dict[tuple[str, str], int] = {}
    latch_counts: dict[tuple[str, str], int] = {}
    for path in src.rglob("*.py"):
        tree = ast.parse(path.read_text())
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        rel = str(path.relative_to(src))
        for node in ast.walk(tree):
            if _is_compaction_call(node):
                owner = _enclosing_function(functions, node.lineno)
                if owner is None or owner.name == "compact_context_if_needed":
                    continue
                key = (rel, owner.name)
                call_sites.add(key)
                compaction_counts[key] = compaction_counts.get(key, 0) + 1
            elif _is_latch_call(node):
                owner = _enclosing_function(functions, node.lineno)
                if owner is None or owner.name == "compact_context_if_needed":
                    continue
                key = (rel, owner.name)
                latch_counts[key] = latch_counts.get(key, 0) + 1

    under_latched = {
        key: (count, latch_counts.get(key, 0))
        for key, count in compaction_counts.items()
        if latch_counts.get(key, 0) < count
    }

    assert under_latched == {}
    assert call_sites >= {
        ("core/agent/pattern/react/react.py", "_run_tool_calling_loop"),
        ("core/agent/pattern/auto/auto.py", "_decide"),
    }


@pytest.mark.asyncio
async def test_truncate_path_latches_without_writing_a_notice() -> None:
    """The truncate path stays silent; the marker is what carries it.

    Reached with no summarizer at all, the only way to get it: the ReAct call
    site substitutes the main model when no compact model is set.
    """
    context = build_context(observations=6, threshold=500)
    runtime = PatternRuntime(execution_id=context.execution_id)
    result = await runtime.compact_context_if_needed(context=context, llm=None)
    note_compaction_evidence_loss(context, result)

    assert result.strategy == "truncate"
    assert context.metadata[KEY] is True
    body = "\n".join(message.content for message in context.messages)
    assert "Compacted conversation summary:" not in body
    assert "dropped by this compaction" not in body


@pytest.mark.asyncio
async def test_windowless_compact_model_does_not_latch() -> None:
    """A compact model with no window makes the runtime refuse to compact, so nothing is lost and the marker stays False."""
    context = build_context(observations=6, threshold=500)
    await run_one_turn(
        context=context, forced=False, compact_llm=WindowlessCompactingLLM()
    )
    assert context.metadata[KEY] is False
    assert len(context.messages) > 2


@pytest.mark.asyncio
async def test_loss_in_an_earlier_turn_reaches_the_later_forced_turn() -> None:
    """The cross-turn cell -- compaction one turn, forced answer the next."""
    context = build_context(observations=6, threshold=500)
    await run_one_turn(context=context, forced=False)
    assert context.metadata[KEY] is True

    _, llm = await run_one_turn(context=context, forced=True)
    sent = prompt_text(llm.calls[0]["messages"])
    assert FACTS_HEAD in sent
    assert "accumulated" not in sent


class RoutingLLM(ScriptedLLM):
    """A routing model: declares a window, and always picks ``react``."""

    context_window = 200_000

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return tool_call(
            DECISION_TOOL_NAME,
            json.dumps(
                {
                    "action": "react",
                    "reason": "the request needs tools",
                    "requires_current_or_external_facts": True,
                    "existing_context_sufficient": False,
                }
            ),
        )


@pytest.mark.asyncio
async def test_auto_routing_loss_reaches_the_react_forced_turn() -> None:
    """The loss Auto's own compaction caused is stated downstream.

    Driven through ``AutoPattern._decide`` rather than by calling the compact
    runtime and the latch by hand: what is under test is that the routing
    stage latches at all, so the latch has to come from production code. A
    cell that latched inside its own body would stay green with that call
    site deleted.
    """
    context = build_context(observations=6, threshold=500)
    runtime = PatternRuntime(execution_id=context.execution_id)
    routing_llm = RoutingLLM()

    decision = await AutoPattern()._decide(
        context=context,
        tools=[],
        llm=routing_llm,
        compact_llm=CompactingLLM(),
        runtime=runtime,
    )

    assert decision.decision.action is AutoAction.REACT
    assert context.metadata[KEY] is True
    routed = prompt_text(routing_llm.calls[0]["messages"])
    assert FACTS_HEAD in routed
    assert "accumulated" not in routed

    # Auto hands the same context object to the pattern it routed to.
    _, llm = await run_one_turn(context=context, forced=True)
    assert FACTS_HEAD in prompt_text(llm.calls[0]["messages"])


# --------------------------------------------------------------------------
# Invariant C -- the four prompts that answer, or decide to answer, tool-less
# --------------------------------------------------------------------------


def _react_forced(context: ExecutionContext) -> str:
    return prompt_text(
        ReActPattern()._messages_for_llm(
            context,
            has_tools=True,
            force_final_answer=True,
            tool_names=["final_answer"],
        )
    )


def _react_decision(context: ExecutionContext) -> str:
    # The prompt this builder writes is the last message; the ones before it
    # are history, which gap 8 covers and this cell does not.
    return ReActPattern()._messages_for_repeated_tool_decision(
        context, {"tool_name": "list_clients", "consecutive_tool_calls": 6}
    )[-1]["content"]


def _dag_assessment(context: ExecutionContext) -> str:
    return DAGPattern(lambda **_: None)._completion_assessment_messages(context)[0][
        "content"
    ]


def _auto_decision(context: ExecutionContext) -> str:
    return AutoPattern()._decision_prompt(
        [], evidence_state=tool_evidence_state(context)
    )


# Main's wording at the spots this change splits into branches, quoted whole
# rather than probed for a few words: probing catches a deleted phrase but not
# a rewritten one, and "only when" turned into "whenever" inverts the rule
# while leaving every probe satisfied. The quote stops where main's own shared
# fragments begin -- grounding_rule and the file-reference and language rules
# are not this change's text, and pinning them here would redden this cell for
# an edit made somewhere else, whose cheapest repair is to edit this value.
MAIN_FORCED_ANSWER_OPENING = (
    "Produce the final user-facing answer by calling the final_answer "
    "control tool exactly once using the accumulated conversation and tool "
    "results. Do not call any other tool and do not output tool-call markup "
    "as plain text. Set outcome=completed only when every requested action "
    "or verification succeeded; otherwise set outcome=partial or "
    "outcome=blocked and say what remains. If a previous ask_user_question "
    "narrowed the request to a selected subset of items or resources, the "
    "final answer must cover only that subset — leave out anything outside "
    "it even if an earlier tool call already returned data about it. "
)
MAIN_DECISION_COMPLETION_CLAUSE = (
    "Use this as the controlling request when deciding whether the "
    "accumulated tool results have completed the user's requested work."
)
MAIN_DECISION_SUFFICIENCY_CLAUSE = (
    "Choose final_answer when the conversation and accumulated tool results "
    "are sufficient to answer the latest user request."
)
# For these two the quote is not the whole prompt but main's two sentences
# that sit adjacent across this change's insertion point: whatever the marker
# renders goes between them, so the pair is present verbatim exactly when the
# marker renders nothing. It stops at the colon for the same reason as above --
# the shared rule after it is not this change's text.
MAIN_DAG_ASSESSMENT_SPLICE = (
    "Put status before answer in the tool arguments. When writing the answer "
    "field, including any content carried over from candidate_output or "
    "step_results:"
)
MAIN_AUTO_DECISION_SPLICE = (
    "Put action before answer in the tool arguments. When writing that answer field:"
)


CONSUMERS = [
    pytest.param(
        _react_forced, (MAIN_FORCED_ANSWER_OPENING,), True, id="react_forced_answer"
    ),
    pytest.param(
        _react_decision,
        (MAIN_DECISION_COMPLETION_CLAUSE, MAIN_DECISION_SUFFICIENCY_CLAUSE),
        False,
        id="react_repeated_tool_decision",
    ),
    pytest.param(
        _dag_assessment,
        (MAIN_DAG_ASSESSMENT_SPLICE,),
        True,
        id="dag_completion_assessment",
    ),
    pytest.param(
        _auto_decision, (MAIN_AUTO_DECISION_SPLICE,), True, id="auto_routing_decision"
    ),
]


# The colon that introduces the shared grounding rule at each site. Kept out
# of CONSUMERS so the two cells with no lead-in are not forced to carry an
# empty placeholder value.
RULE_LEAD_INS = {
    _dag_assessment: (
        "When writing the answer field, including any content carried over "
        "from candidate_output or step_results:"
    ),
    _auto_decision: "When writing that answer field:",
}


@pytest.mark.parametrize("build, _main_fragments, _carries_grounding_rule", CONSUMERS)
@pytest.mark.parametrize("state", STATES)
def test_every_toolless_answer_prompt_states_the_loss(
    build: Any,
    _main_fragments: tuple[str, ...],
    _carries_grounding_rule: bool,
    state: str,
) -> None:
    """Every tool-less answer prompt carries the facts for its state.

    None of the three non-intact states claims accumulated results. The scope
    of that assertion is the prompt this builder writes for this turn, not
    the whole payload: a guidance line written into the context on an earlier
    turn can still carry main's wording, which is an acknowledged gap no
    per-turn builder can reach.
    """
    context = build_context(observations=2, threshold=500)
    apply_state(context, state)
    text = build(context)
    assert evidence_facts(state) == "" or facts_head(state) in text
    if state == "intact":
        assert FACTS_HEAD not in text
        assert UNKNOWN_FACTS_HEAD not in text
    else:
        assert "accumulated" not in text


@pytest.mark.parametrize("build, _main_fragments, carries_grounding_rule", CONSUMERS)
@pytest.mark.parametrize("state", ["removed", "unknown"])
def test_the_loss_is_stated_before_the_shared_grounding_rule(
    build: Any,
    _main_fragments: tuple[str, ...],
    carries_grounding_rule: bool,
    state: str,
) -> None:
    """Which of the two comes first is a decision, so it is pinned at every site.

    The three prompts that carry both state what is no longer readable before
    the wider rule about answering from sources, in either non-intact state.
    The fourth builder carries no shared rule at all; that exclusion is
    asserted rather than left silent, so adding the rule there without
    picking an order reddens this cell.

    For the two sites with a colon lead-in into the shared rule, nothing may
    be spliced between that colon and the rule it introduces -- moving the
    lead-in ahead of the facts (rather than reordering facts and rule) would
    satisfy the plain ordering check above while still breaking what the
    colon points at.
    """
    context = build_context(observations=2, threshold=500)
    apply_state(context, state)
    text = build(context)
    head = facts_head(state)
    grounding_head = grounding_rule(can_call_tools=False).split(".")[0]
    assert head in text
    if not carries_grounding_rule:
        assert grounding_head not in text
        return
    assert text.index(head) < text.index(grounding_head)

    lead_in = RULE_LEAD_INS.get(build)
    if lead_in is None:
        return
    # The colon introduces the shared rule; nothing may be spliced between them.
    assert text.index(head) < text.index(lead_in)
    after = text[text.index(lead_in) + len(lead_in) :]
    assert after.lstrip().startswith(grounding_head)


@pytest.mark.parametrize("build, main_fragments, _carries_grounding_rule", CONSUMERS)
def test_a_run_that_never_lost_anything_reads_exactly_like_main(
    build: Any, main_fragments: tuple[str, ...], _carries_grounding_rule: bool
) -> None:
    """The marker-false branch is word-for-word main's wording.

    The context must come from ``ContextManager.create_context``. One built
    directly carries no key, reads as "evidence removed", and would make this
    cell fail for a reason that has nothing to do with the wording.
    """
    stamped = build_context(observations=2, threshold=500)
    assert stamped.metadata[KEY] is False
    text = build(stamped)
    assert FACTS_HEAD not in text
    for fragment in main_fragments:
        assert fragment in text


@pytest.mark.parametrize("state", ["removed", "unknown"])
def test_the_outcome_rule_stays_a_conditional(state: str) -> None:
    """A dropped result message is not a dropped action, in either non-intact state."""
    context = build_context(observations=2, threshold=500)
    apply_state(context, state)
    text = _react_forced(context)
    assert "outcome=completed" in text
    assert "Do not rest an outcome=completed claim on an observation" in text
    assert "outcome=completed is unavailable" not in text
    if state == "unknown":
        assert "you cannot read in this context" in text
        assert "that was removed" not in text
    else:
        assert "that was removed" in text
        assert "you cannot read in this context" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["removed", "intact", "unknown"])
async def test_the_guidance_written_into_the_context_matches_the_marker(
    state: str,
) -> None:
    """This guidance is history, so it must agree with the next prompt."""
    context = build_context(observations=2, threshold=500)
    apply_state(context, state)
    pattern = ReActPattern()
    pattern.repeated_tool_decision = {
        "tool_name": "list_clients",
        "consecutive_tool_calls": 6,
    }
    decision = tool_call(
        "react_decision",
        '{"action": "final_answer", "reason": "r", "missing_verification": ""}',
    )
    await pattern._run_repeated_tool_decision(
        context=context,
        llm=ScriptedLLM([decision]),
        runtime=PatternRuntime(execution_id=context.execution_id),
    )
    guidance = context.messages[-1].content
    assert guidance.startswith("Repeated tool decision completion guidance")
    if state == "removed":
        assert "accumulated" not in guidance
        assert "Compaction has removed tool observations from this run" in guidance
        # End to end: the forced turn that reads this message back must not
        # find main's claim sitting in its own payload.
        _, llm = await run_one_turn(context=context, forced=True)
        assert "accumulated" not in prompt_text(llm.calls[0]["messages"])
    elif state == "unknown":
        assert "accumulated" not in guidance
        assert "cannot be determined" in guidance
        assert "Compaction has removed" not in guidance
        _, llm = await run_one_turn(context=context, forced=True)
        assert "accumulated" not in prompt_text(llm.calls[0]["messages"])
    else:
        assert guidance.endswith(
            "The repeated-tool decision selected final_answer, so the next "
            "normal ReAct step must produce the final user-facing answer from "
            "the accumulated conversation and tool results. Do not call more "
            "tools in that final step. Do not send a progress update or "
            "promise future work as the final answer; if the accumulated "
            "results are insufficient or show the task is incomplete, say "
            "that directly."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["removed", "unknown"])
async def test_protocol_repair_retry_inherits_the_same_wording(state: str) -> None:
    """The repair retry inside a forced turn carries the same wording.

    The retry is reached, not simulated: the first response calls
    final_answer with an empty answer field, which the pattern treats as a
    protocol violation and repairs once. The second call is identified by the
    repair instruction only that path writes, so a cell that merely rebuilt
    the ordinary forced prompt could not pass as this one.
    """
    context = build_context(observations=2, threshold=500)
    apply_state(context, state)
    llm = ScriptedLLM([empty_final_answer_call(), final_answer_call()])
    pattern = ReActPattern(max_iterations=1)
    pattern.force_final_answer_next = True

    await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=PatternRuntime(execution_id=context.execution_id),
        compact_llm=CompactingLLM(),
    )

    assert len(llm.calls) == 2
    repair = prompt_text(llm.calls[1]["messages"])
    assert "called final_answer with an empty answer field" in repair
    assert facts_head(state) in repair
    assert "accumulated" not in repair


@pytest.mark.asyncio
async def test_a_dag_step_child_carries_its_own_loss_and_the_root_stays_intact(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A step's loss drives that step's own forced turn, and only that one.

    The child is built the way ``_execute_step_impl`` builds it, then driven
    through a real forced answer turn: its own marker reaches its own wording
    and its own compaction skip. The root keeps its messages and its
    completion assessment says nothing about a loss, which is why the marker
    is not raised to the root.
    """
    root = build_context(observations=6, threshold=500)
    root_messages_before = len(root.messages)
    child = root.create_child_context(metadata={"dag_step_id": "step-1"})

    await run_one_turn(context=child, forced=False)
    assert child.metadata[KEY] is True
    assert root.metadata[KEY] is False

    with caplog.at_level(logging.INFO, logger="xagent.core.agent.pattern.react.react"):
        _, llm = await run_one_turn(context=child, forced=True)

    assert [r for r in caplog.records if "did not compact" in r.getMessage()]
    assert FACTS_HEAD in prompt_text(llm.calls[0]["messages"])
    assert len(root.messages) == root_messages_before
    assert FACTS_HEAD not in _dag_assessment(root)


@pytest.mark.asyncio
async def test_a_lossy_step_result_reaches_the_root_under_the_shared_rule() -> None:
    """A step's lost evidence travels to the root as step output, not as the marker.

    The root's own messages are untouched, so the root still does not latch --
    that half of the invariant above still holds. But the step's forced-turn
    answer reaches the root's completion-assessment payload through
    ``step_results`` and ``candidate_output``, which are read unconditionally
    and are not filtered by the marker. What stands between that carried-over
    content and the assessment model treating it as fresh accumulated
    evidence is not the marker -- the root's is false -- it is the
    unconditional rule about content carried over from candidate_output or
    step_results. This test pins that as the current shape of this known
    limit, not as a guarantee that nothing can go wrong here.
    """
    root = build_context(observations=6, threshold=500)
    child = root.create_child_context(metadata={"dag_step_id": "step-1"})
    await run_one_turn(context=child, forced=False)
    assert child.metadata[KEY] is True

    step_answer_llm = ScriptedLLM(
        [
            tool_call(
                "final_answer",
                '{"answer": "STEP_OUTPUT_MARKER_7f3", "outcome": "partial"}',
            )
        ]
    )
    await run_one_turn(context=child, forced=True, llm=step_answer_llm)
    step_output = child.messages[-1].content
    assert step_output == "STEP_OUTPUT_MARKER_7f3"

    pattern = DAGPattern(lambda **_: None)
    pattern.step_results = {"step-1": step_output}
    messages = pattern._completion_assessment_messages(root)
    system, payload = messages[0]["content"], messages[1]["content"]

    assert step_output in payload
    assert '"candidate_output"' in payload
    assert '"step_results"' in payload
    assert root.metadata[KEY] is False
    assert FACTS_HEAD not in system
    assert UNKNOWN_FACTS_HEAD not in system
    assert (
        "When writing the answer field, including any content carried over "
        "from candidate_output or step_results:" in system
    )
    assert grounding_rule(can_call_tools=False).split(".")[0] in system


# --------------------------------------------------------------------------
# Failure path
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_skipped_compaction_that_overflows_fails_the_run_cleanly() -> None:
    """Overflow ends the run -- no retry, no fallback compaction, no answer."""

    class OverflowLLM(ScriptedLLM):
        async def chat(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            raise LLMContextLengthError(
                "This model's maximum context length is 8192 tokens"
            )

    context = build_context(observations=6, threshold=500)
    assistants_before = sum(1 for m in context.messages if m.role == "assistant")
    llm = OverflowLLM()
    pattern = ReActPattern(max_iterations=2)
    pattern.force_final_answer_next = True

    with pytest.raises(LLMContextLengthError):
        await pattern.run(
            context=context,
            tools=[],
            llm=llm,
            runtime=PatternRuntime(execution_id=context.execution_id),
            compact_llm=CompactingLLM(),
        )

    assert len(llm.calls) == 1
    assert sum(1 for m in context.messages if m.role == "assistant") == (
        assistants_before
    )
    assert context.metadata[KEY] is False
