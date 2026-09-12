"""A run remembers which tool observations compaction destroyed.

Compaction can destroy tool observations in the middle of a run, and the turn
that loses them is often not the turn that later has to answer without them.
This file tests that ``ReActPattern`` keeps one durable record of that loss --
``lost_tool_evidence`` -- on the pattern instance for the life of the run, that
the record accumulates regardless of whether the turn that lost the evidence
was itself a forced-answer turn, that it survives a checkpoint round trip, and
that it is cleared once the run actually finishes. It also covers how a later
forced-answer turn reads this record back and changes the instruction it sends
to the model: reporting that observations were removed, instead of asking for
an answer "using the accumulated conversation and tool results" that are gone.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.context.execution import (
    COMPACT_DROPPED_TOOL_NAME_MAX_CHARS,
    COMPACT_DROPPED_TOOL_NOTICE_MAX_CHARS,
    COMPACT_DROPPED_TOOL_NOTICE_MAX_NAMES,
    bounded_notice_lines,
)
from xagent.core.agent.grounding import VALUE_KINDS, grounding_rule
from xagent.core.agent.language import final_answer_language_rule
from xagent.core.agent.pattern.react import react as react_module
from xagent.core.agent.pattern.react.react import (
    LOST_TOOL_EVIDENCE_GRANULARITY,
    LOST_TOOL_EVIDENCE_MAX_CALL_IDS,
    LOST_TOOL_NAME_OMISSION_ALLOWANCE,
    CompactionLoss,
    LostToolEvidence,
    ToolCallRecord,
)
from xagent.core.file_ref import final_deliverable_file_reference_instructions
from xagent.core.model.chat.exceptions import LLMToolProtocolError

# The exact prefix the gate's one log line opens with.
EVIDENCE_DROPPED_LOG_PHRASE = "Forced answer turn is missing tool evidence"

# The sentence the ordinary forced instruction opens with. On a turn whose
# evidence was destroyed it is an instruction to invent, so it must be gone.
STALE_EVIDENCE_PHRASE = "the accumulated conversation and tool results"
HONEST_PHRASE = "Compaction removed tool observations from this run's context"
CONDITIONAL_SUMMARY_PHRASE = "If a compaction summary stands above"
COMPLETED_OFFERED_PHRASE = "Set outcome=completed only when"
COMPLETED_CONDITIONAL_PHRASE = (
    "Do not rest an outcome=completed claim on an observation that was removed"
)


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
    """Chat LLM that records every call and replays scripted responses.

    ``context_window`` is required for real compaction to pick the summary
    path at all: without it, ``compact_context_if_needed`` cannot bound the
    summary request and reports the request as unfittable rather than
    running it.
    """

    context_window = 128_000

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses:
            return {"content": "fallback answer", "done": True}
        return self.responses.pop(0)


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


def empty_final_answer_response(call_id: str = "call_final") -> dict[str, Any]:
    """A ``final_answer`` call whose ``answer`` field is blank.

    ``_response_requires_tool_protocol_retry`` rejects this and routes it
    into the pattern's single repair retry, whatever the retry's own
    response turns out to be.
    """
    return {
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "arguments": (
                        '{"response_language":"English","outcome":"partial","answer":""}'
                    ),
                },
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


def state_at(
    runtime: PatternRuntime, label: str, occurrence: int = 0
) -> dict[str, Any]:
    matching = [entry for entry in runtime.checkpoints if entry.get("label") == label]
    return dict(matching[occurrence].get("pattern_state") or {})


def compact_result(
    *,
    compacted: bool = True,
    count: Any = 1,
    by_name: Any = None,
    call_ids: Any = None,
    without_call_id: Any = None,
    strategy: str = "llm_summary",
) -> Any:
    """Build a ``CompactResult`` with only the metadata keys under test set.

    ``call_ids`` and ``without_call_id`` let a test script the two keys this
    branch writes for identifying individual dropped calls, including
    leaving them absent or malformed -- shapes real compaction is not known
    to produce, but that ``_dropped_tool_evidence`` must still not
    misinterpret.
    """
    from xagent.core.agent.context.execution import CompactResult

    metadata: dict[str, Any] = {}
    if count is not None:
        metadata["dropped_tool_result_count"] = count
    if by_name is not None:
        metadata["dropped_tool_results_by_name"] = by_name
    if call_ids is not None:
        metadata["dropped_tool_result_call_ids"] = call_ids
    if without_call_id is not None:
        metadata["dropped_tool_results_without_call_id"] = without_call_id
    return CompactResult(
        compacted=compacted,
        original_count=9,
        final_count=2,
        strategy=strategy,
        metadata=metadata,
    )


class ScriptedCompactionRuntime(PatternRuntime):
    """Runtime whose compaction returns a caller-supplied result per turn.

    Used for the interleavings the honest-forcing gate has to handle that
    real threshold-triggered compaction cannot be steered into reliably --
    in particular, a loss on one turn with several ordinary, nothing-lost
    turns running in between before a forced turn. The metadata shapes it
    plays back are anchored to what real compaction actually writes by
    ``test_real_compaction_reports_the_keys_the_gate_reads`` above.

    ``pattern`` and ``force_after_call`` together let a test flip a
    pattern's sticky ``force_final_answer_next`` flag partway through a run,
    without the test code ever getting control back between turns: right
    after the compaction call at 1-based count ``force_after_call`` returns,
    which is late enough that call's own turn has already computed its own
    (unforced) instruction, and early enough that the next turn reads the
    flag as already set.
    """

    def __init__(
        self,
        results: list[Any],
        *,
        pattern: "ReActPattern | None" = None,
        force_after_call: int | None = None,
    ) -> None:
        super().__init__()
        self.scripted_compactions = list(results)
        self._pattern = pattern
        self._force_after_call = force_after_call
        self._compact_calls = 0

    async def compact_context_if_needed(self, **kwargs: Any) -> Any:
        self._compact_calls += 1
        if self._compact_calls == self._force_after_call and self._pattern is not None:
            self._pattern.force_final_answer_next = True
        if not self.scripted_compactions:
            return None
        return self.scripted_compactions.pop(0)


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


def instruction_of(llm: RecordingLLM, index: int = 0) -> str:
    return str(llm.calls[index]["messages"][0].get("content", ""))


def whole_prompt_of(llm: RecordingLLM, index: int = 0) -> str:
    return "\n".join(
        str(message.get("content", "")) for message in llm.calls[index]["messages"]
    )


def evidence_dropped_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if EVIDENCE_DROPPED_LOG_PHRASE in record.getMessage()
    ]


@pytest.mark.asyncio
async def test_real_compaction_reports_the_keys_the_gate_reads() -> None:
    """Anchor every scripted-metadata test in this file to what compaction
    actually writes.

    Every ``compact_result()`` call elsewhere in this file claims a metadata
    shape that this test proves the two real compacting paths -- the LLM
    summary and the message-dropping backstop -- actually produce, so a
    rename or a semantic change upstream in ``ExecutionContext`` cannot leave
    the rest of this file passing against a fiction.
    """
    summary_context = build_context(max_messages=40)
    summary_result = await PatternRuntime().compact_context_if_needed(
        context=summary_context,
        llm=RecordingLLM([{"content": "A summary of the run."}]),
    )
    assert summary_result.compacted is True
    assert summary_result.strategy == "llm_summary"
    assert summary_result.metadata["dropped_tool_result_count"] == 3
    assert summary_result.metadata["dropped_tool_results_by_name"] == {"calculator": 3}
    assert set(summary_result.metadata["dropped_tool_result_call_ids"]) == {
        "seed_0",
        "seed_1",
        "seed_2",
    }
    assert summary_result.metadata["dropped_tool_results_without_call_id"] == 0

    drop_context = build_context(max_messages=4)
    drop_result = await PatternRuntime().compact_context_if_needed(
        context=drop_context,
        llm=RecordingLLM([{"content": ""}]),
    )
    assert drop_result.compacted is True
    assert drop_result.strategy == "truncate"
    assert drop_result.metadata["dropped_tool_result_count"] == 1
    assert drop_result.metadata["dropped_tool_results_by_name"] == {"calculator": 1}
    assert set(drop_result.metadata["dropped_tool_result_call_ids"]) == {"seed_0"}
    assert drop_result.metadata["dropped_tool_results_without_call_id"] == 0
    # The tail window kept a later successful observation of the same tool.
    # "Is there still tool evidence in context" would answer yes here, which
    # is exactly why the record identifies losses by call id instead of by
    # tool name.
    assert any(message.role == "tool" for message in drop_context.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_summary", [True, False], ids=["llm_summary", "truncate"])
@pytest.mark.parametrize("forced", [True, False], ids=["forced", "unforced"])
async def test_an_unforced_turn_still_records_what_it_lost(
    use_summary: bool, forced: bool
) -> None:
    """A turn folds compaction's loss into the run's record either way.

    Checked at the ``after_llm`` checkpoint, which is written right after
    compaction folds the loss into ``pattern.lost_tool_evidence`` but before
    the ``final_answer`` control tool's own finalize step -- which
    deliberately clears the record once the run is over, and is exactly what
    ``test_a_finished_run_leaves_no_lost_evidence_behind`` checks below. A
    live post-run read would show an empty record for every cell here
    precisely because that clearing step ran, which is why this test reads
    the mid-run checkpoint instead.

    Mutation this test catches: if the fold-in call in
    ``_run_tool_calling_loop`` were moved inside
    ``if force_final_answer_now:``, the unforced cell's checkpointed
    ``call_ids`` would read back empty here; with the actual unconditional
    fold-in it does not.
    """
    max_messages = 40 if use_summary else 4
    scripted_summary_content = "A summary of the run." if use_summary else ""

    reference_context = build_context(max_messages=max_messages)
    reference_result = await PatternRuntime().compact_context_if_needed(
        context=reference_context,
        llm=RecordingLLM([{"content": scripted_summary_content}]),
    )
    expected_call_ids = sorted(
        reference_result.metadata["dropped_tool_result_call_ids"]
    )
    assert expected_call_ids

    context = build_context(max_messages=max_messages)
    tools: list[Any] = [NamedTool("calculator")]
    llm = RecordingLLM([final_answer_response()])
    compact_llm = RecordingLLM([{"content": scripted_summary_content}])
    pattern = ReActPattern(max_iterations=6)
    pattern.force_final_answer_next = forced
    runtime = PatternRuntime()

    await pattern.run(
        context=context,
        tools=tools,
        llm=llm,
        compact_llm=compact_llm,
        runtime=runtime,
    )

    recorded = state_at(runtime, "after_llm")["lost_tool_evidence"]
    assert recorded["call_ids"] == expected_call_ids
    assert recorded["call_ids"] != []


@pytest.mark.parametrize(
    (
        "source_call_ids",
        "source_unnameable",
        "expected_call_ids",
        "expected_unnameable",
    ),
    [
        pytest.param(set(), False, set(), False, id="empty"),
        pytest.param({"call_1"}, False, {"call_1"}, False, id="one_id"),
        pytest.param(
            {"call_1", "call_2", "call_3"},
            False,
            {"call_1", "call_2", "call_3"},
            False,
            id="several_ids",
        ),
        pytest.param(set(), True, set(), True, id="unnameable_alone"),
        pytest.param(
            {"call_1", "call_2"},
            True,
            {"call_1", "call_2"},
            True,
            id="ids_plus_unnameable",
        ),
        pytest.param(
            {
                f"call_{index:04d}"
                for index in range(LOST_TOOL_EVIDENCE_MAX_CALL_IDS + 5)
            },
            False,
            set(
                sorted(
                    f"call_{index:04d}"
                    for index in range(LOST_TOOL_EVIDENCE_MAX_CALL_IDS + 5)
                )[:LOST_TOOL_EVIDENCE_MAX_CALL_IDS]
            ),
            True,
            id="more_ids_than_cap",
        ),
    ],
)
def test_the_lost_evidence_record_round_trips_through_a_checkpoint(
    source_call_ids: set[str],
    source_unnameable: bool,
    expected_call_ids: set[str],
    expected_unnameable: bool,
) -> None:
    """``get_state()``/``load_state()`` must reproduce the record on the
    other side.

    Without a correct round trip, a run resumed from a checkpoint would
    silently start with a wrong lost-evidence record: either forgetting real
    losses (a missing-key regression) or inventing ones that were never
    there.

    The over-cap cell does not expect the exact same call ids back: writing
    more ids than a read-back accepts is a legitimate outcome of a long run's
    own accumulation over many compactions, and the cap's truncation-on-read
    is what the dedicated over-cap test below pins down in detail. Included
    here only to confirm that ``get_state()`` itself never truncates -- only
    ``from_state()`` does -- so the source's full set survives the write side
    of the round trip even though the read side then shortens it.
    """
    source = ReActPattern()
    source.lost_tool_evidence = LostToolEvidence(
        call_ids=set(source_call_ids), unnameable=source_unnameable
    )
    state = source.get_state()
    assert len(state["lost_tool_evidence"]["call_ids"]) == len(source_call_ids)

    destination = ReActPattern()
    destination.load_state(state)

    assert destination.lost_tool_evidence.call_ids == expected_call_ids
    assert destination.lost_tool_evidence.unnameable is expected_unnameable


@pytest.mark.parametrize(
    "raw",
    [
        [],
        "x",
        {"call_ids": "x"},
        {"granularity": LOST_TOOL_EVIDENCE_GRANULARITY, "call_ids": [1]},
        {"granularity": "tool_name", "call_ids": []},
        # Both of these carry a call id rather than an empty list, and the
        # second carries a falsy value. Without either, a build that kept no
        # guard at all and simply coerced the field would land on the same
        # answer these cells expect, and the cell would pass against both.
        {
            "granularity": LOST_TOOL_EVIDENCE_GRANULARITY,
            "call_ids": ["call_a"],
            "unnameable": "yes",
        },
        {
            "granularity": LOST_TOOL_EVIDENCE_GRANULARITY,
            "call_ids": ["call_a"],
            "unnameable": 0,
        },
    ],
    ids=[
        "not_a_dict_list",
        "not_a_dict_str",
        "missing_granularity",
        "non_string_call_id",
        "wrong_granularity",
        "unnameable_not_bool_str",
        # bool is a subclass of int in Python, so a guard written as
        # isinstance(raw, int) would wave this one through.
        "unnameable_not_bool_int",
    ],
)
def test_an_unreadable_record_reads_as_evidence_still_missing(raw: Any) -> None:
    """Three separate facts ``load_state`` must never confuse with one
    another: a key that was never written, a value that was written and
    reads back cleanly, and a value that was written but cannot be read.

    Mutation this test catches: collapsing the missing-key branch and the
    unreadable-value branch of ``load_state`` into one conditional expression
    makes the missing-key cell report ``unnameable=True``; with the two
    branches kept separate it correctly reports ``False``.
    """
    base_state = ReActPattern().get_state()

    missing_key_state = dict(base_state)
    missing_key_state.pop("lost_tool_evidence", None)
    missing_key_pattern = ReActPattern()
    missing_key_pattern.load_state(missing_key_state)
    assert missing_key_pattern.lost_tool_evidence.call_ids == set()
    assert missing_key_pattern.lost_tool_evidence.unnameable is False
    assert missing_key_pattern.lost_tool_evidence.still_missing() is False

    wellformed_state = dict(base_state)
    wellformed_state["lost_tool_evidence"] = {
        "granularity": LOST_TOOL_EVIDENCE_GRANULARITY,
        "call_ids": ["call_a", "call_b"],
        "unnameable": False,
    }
    wellformed_pattern = ReActPattern()
    wellformed_pattern.load_state(wellformed_state)
    assert wellformed_pattern.lost_tool_evidence.call_ids == {"call_a", "call_b"}
    assert wellformed_pattern.lost_tool_evidence.unnameable is False

    unreadable_state = dict(base_state)
    unreadable_state["lost_tool_evidence"] = raw
    unreadable_pattern = ReActPattern()
    unreadable_pattern.load_state(unreadable_state)
    assert unreadable_pattern.lost_tool_evidence.call_ids == set()
    assert unreadable_pattern.lost_tool_evidence.unnameable is True
    assert unreadable_pattern.lost_tool_evidence.still_missing() is True


def test_an_over_long_record_is_truncated_and_says_so() -> None:
    """A read-back over the cap is shortened, not silently accepted whole.

    Without the cap, a payload naming an unbounded number of ids -- whether
    from a run that genuinely lost that many observations over its lifetime,
    or from a build with a different limit -- would be accepted at whatever
    size it arrives, defeating the reason the cap exists.
    """
    ids = sorted(
        f"call_{index:04d}" for index in range(LOST_TOOL_EVIDENCE_MAX_CALL_IDS + 5)
    )
    raw = {
        "granularity": LOST_TOOL_EVIDENCE_GRANULARITY,
        "call_ids": ids,
        "unnameable": False,
    }

    evidence = LostToolEvidence.from_state(raw)

    assert len(evidence.call_ids) == LOST_TOOL_EVIDENCE_MAX_CALL_IDS
    assert evidence.call_ids == set(ids[:LOST_TOOL_EVIDENCE_MAX_CALL_IDS])
    assert evidence.unnameable is True


@pytest.mark.parametrize("times", [1, 3, 10])
def test_an_unnameable_loss_is_recorded_once_however_often_it_is_seen(
    times: int,
) -> None:
    """``unnameable`` is a flag, not a tally, so repeats must not change the
    resulting state.

    Without this, replaying the same turn after an interrupt -- which folds
    the same compaction loss into the record a second time -- would make the
    record depend on how many times a turn happened to be replayed rather
    than on what compaction actually destroyed.
    """
    loss = CompactionLoss(call_ids=(), unaccounted=2)

    repeated = ReActPattern()
    for _ in range(times):
        repeated._record_lost_tool_evidence(loss)

    once = ReActPattern()
    once._record_lost_tool_evidence(loss)

    assert (
        repeated.get_state()["lost_tool_evidence"]
        == once.get_state()["lost_tool_evidence"]
    )
    assert repeated.lost_tool_evidence.unnameable is True
    assert repeated.lost_tool_evidence.call_ids == set()


@pytest.mark.parametrize(
    ("call_ids_value", "without_call_id_value", "expected_call_ids"),
    [
        pytest.param(
            ["call_a", "call_b"],
            1,
            {"call_a", "call_b"},
            id="readable_ids_plus_uncounted",
        ),
        pytest.param(None, 0, set(), id="call_id_list_absent"),
        pytest.param("not-a-list", 0, set(), id="call_id_list_not_a_list"),
        pytest.param([1, 2], 0, set(), id="call_id_list_non_string_element"),
        pytest.param(["", "call_a"], 0, set(), id="call_id_list_empty_string_element"),
    ],
)
def test_an_observation_without_a_call_id_is_not_given_one(
    call_ids_value: Any,
    without_call_id_value: Any,
    expected_call_ids: set[str],
) -> None:
    """A destroyed observation this turn cannot name gets no invented
    identity.

    These call-id shapes -- an absent list, a non-list, a non-string
    element, an empty-string element -- have no known way to occur through a
    production entry point today: ``Message.tool_call_id`` is typed
    ``str | None`` and every wired writer fills it before a message reaches
    compaction. They are pinned here directly against
    ``ReActPattern._dropped_tool_evidence`` rather than forced through an
    end-to-end fake that would not occur in production.

    Without this behavior, a malformed or absent id list could be repaired
    by inventing a placeholder id -- letting a later answer believe one
    specific call's evidence came back when nobody ever identified it -- or
    by falling back to the tool's name, which would wrongly let any later
    call to that same tool count as bringing the lost observation back. Both
    are worse than admitting the loss cannot be named.
    """
    pattern = ReActPattern()
    result = compact_result(
        count=3, call_ids=call_ids_value, without_call_id=without_call_id_value
    )

    loss = pattern._dropped_tool_evidence(result)
    assert set(loss.call_ids) == expected_call_ids

    pattern._record_lost_tool_evidence(loss)
    assert pattern.lost_tool_evidence.call_ids == expected_call_ids
    assert pattern.lost_tool_evidence.unnameable is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("llm_responses", "max_iterations", "expected_label", "expected_success"),
    [
        # A final_answer call reaches _finalize_outcome (checkpoint "final")
        # through _execute_pending_tool_calls, whose caller then writes one
        # more checkpoint labeled after the control result's own status.
        pytest.param(
            [final_answer_response()],
            6,
            "completed",
            True,
            id="final_answer_tool_call",
        ),
        pytest.param(
            [{"content": "Here is the answer.", "tool_calls": [], "done": True}],
            6,
            "final",
            True,
            id="plain_assistant_text",
        ),
        # Same control-tool path as final_answer_tool_call above: the
        # retry's final_answer call also finishes through
        # _execute_pending_tool_calls.
        pytest.param(
            [empty_final_answer_response(), final_answer_response()],
            6,
            "completed",
            True,
            id="rejected_then_finishes",
        ),
        # A work-tool call, never a final_answer: with max_iterations=1 the
        # loop executes this one batch and then exhausts its range without
        # ever asking the model for an answer.
        pytest.param(
            [tool_call_response("calculator")],
            1,
            "max_iterations",
            False,
            id="max_iterations",
        ),
        pytest.param(
            [empty_final_answer_response(), empty_final_answer_response()],
            6,
            "invalid_tool_protocol",
            False,
            id="invalid_tool_protocol",
        ),
    ],
)
async def test_a_finished_run_leaves_no_lost_evidence_behind(
    llm_responses: list[Any],
    max_iterations: int,
    expected_label: str,
    expected_success: bool,
) -> None:
    """However a run ends, the checkpoint it leaves behind carries no
    leftover loss.

    A run can end five ways here: it answers on the first turn with a
    ``final_answer`` tool call, it answers with plain assistant text, its
    first answer is rejected for an empty ``answer`` field and the one
    repair retry then answers, it runs out of iterations without ever
    answering, or that one repair retry also comes back empty and the run is
    abandoned. The last two do not go through ``_finalize_outcome`` at all --
    they write a checkpoint labeled ``max_iterations`` or
    ``invalid_tool_protocol`` instead of ``final`` -- but a resume can load
    from either checkpoint just as it can load from ``final``, so a leftover
    record there is exactly as stale as one left behind on a normal finish.

    Every cell starts from the same context, whose three seeded tool
    observations this run's own first-turn compaction destroys, so the
    lost-evidence record genuinely holds something part-way through the run;
    without that, the assertions below would pass vacuously against a record
    nothing ever populated.

    Mutation this test catches: dropping
    ``_clear_lost_tool_evidence_at_run_end()`` from the ``max_iterations``
    path or from ``_invalid_tool_protocol_result`` leaves that one cell's
    ``call_ids`` still holding the seeded run's lost ids, while the three
    cells that finish through ``_finalize_outcome`` keep passing regardless,
    because that path clears the record on its own.
    """
    context = build_context(max_messages=4)
    pattern = ReActPattern(max_iterations=max_iterations)
    runtime = PatternRuntime()

    result = await pattern.run(
        context=context,
        tools=[NamedTool("calculator")],
        llm=RecordingLLM(llm_responses),
        compact_llm=RecordingLLM([{"content": ""}]),
        runtime=runtime,
    )

    assert result["success"] is expected_success
    last_checkpoint = runtime.checkpoints[-1]
    assert last_checkpoint.get("label") == expected_label
    final_state = dict(last_checkpoint.get("pattern_state") or {})
    assert final_state["lost_tool_evidence"]["call_ids"] == []
    assert final_state["lost_tool_evidence"]["unnameable"] is False
    assert pattern.lost_tool_evidence.call_ids == set()
    assert pattern.lost_tool_evidence.unnameable is False


def make_ledger_record(
    tool_call_id: str,
    *,
    tool_name: str,
    args_hash: str = "hash-a",
    status: str = "completed",
    result: Any = None,
) -> ToolCallRecord:
    """Build a ledger entry with only the fields the discharge rule reads.

    ``_discharge_lost_tool_evidence`` reads a record's ``tool_name``,
    ``args_hash``, ``status``, and ``result`` -- never its ``args`` or
    ``turn_id`` -- so those two are left at their dataclass defaults.
    """
    return ToolCallRecord(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        args={},
        args_hash=args_hash,
        status=status,
        result=result,
    )


# These tests exercise ``_discharge_lost_tool_evidence`` directly rather than
# through ``pattern.run()``. The rule it implements reads only
# ``self.tool_ledger`` and ``self.lost_tool_evidence`` and takes a plain list
# of call ids -- it never touches the message transcript, the LLM, or actual
# tool execution. Reproducing a lost entry whose id is genuinely resolvable in
# ``self.tool_ledger`` end-to-end would require a real run to execute a tool
# call (so it lands in the ledger), then a real compaction pass to drop
# exactly that call's message on a later turn, then a further scripted LLM
# turn to re-issue a specific tool call shape (success, failure, cancellation,
# or different arguments) -- none of which the existing scaffolding in this
# file produces for free (``build_context``'s seeded tool calls are written
# straight into the context's messages, never through the pattern's own
# ``tool_ledger``, so they can never be discharged). That machinery would
# only be replicating this method's own fingerprint-matching logic through an
# expensive detour. Driving the method directly keeps each test's cause and
# effect visible in one place.


@pytest.mark.parametrize(
    ("args_hash", "status", "result", "batch_call_ids"),
    [
        pytest.param(
            "hash-a",
            "failed",
            {"success": False, "error": "boom", "tool_name": "calculator"},
            ["batch_1"],
            id="tool_raised",
        ),
        pytest.param(
            "hash-a",
            "failed",
            {"success": False, "error": "explicit failure"},
            ["batch_1"],
            id="structured_failure",
        ),
        pytest.param(
            "hash-a",
            "cancelled",
            {"success": False, "status": "cancelled", "error": "discarded"},
            ["batch_1"],
            id="cancelled",
        ),
        pytest.param(
            "hash-b",
            "completed",
            {"output": "a different value"},
            ["batch_1"],
            id="different_arguments",
        ),
        pytest.param(None, None, None, [], id="empty_batch"),
    ],
)
def test_a_refetch_that_does_not_resolve_the_entry_discharges_nothing(
    args_hash: str | None,
    status: str | None,
    result: Any,
    batch_call_ids: list[str],
) -> None:
    """A refetch discharges nothing unless it is a completed, successful,
    same-fingerprint (same tool name and args_hash) call named in this
    batch.

    Four ways a same-tool refetch can fail to qualify, each pinned to its
    own mutation:

    - ``tool_raised`` / ``structured_failure`` / ``cancelled``: the refetch
      shares "hash-a" with the lost entry but did not succeed -- it raised,
      returned a structured failure, or was cancelled -- so its ledger
      record carries a status other than "completed". Mutation this test
      catches: relaxing the discharge condition from "completed and
      successful" down to merely "a matching call ran" would empty the
      record in all three of these cells; the real condition leaves it
      unchanged in all three.
    - ``different_arguments``: the refetch succeeds but under "hash-b", a
      different observation from the one lost under "hash-a". Mutation
      this test catches: matching on tool name alone, dropping the
      args_hash half of the comparison, would empty the record here; with
      the real fingerprint match it stays unchanged.
    - ``empty_batch``: no refetch happened at all, so an empty batch must
      never be read as "everything came back". Mutation this test catches:
      treating an empty ``batch_call_ids`` as a signal that the whole
      record is now satisfied would empty it here; the real rule finds no
      completed, successful call to match against and therefore removes
      nothing.
    """
    pattern = ReActPattern()
    pattern.tool_ledger["lost_1"] = make_ledger_record(
        "lost_1", tool_name="calculator", args_hash="hash-a"
    )
    pattern.lost_tool_evidence = LostToolEvidence(call_ids={"lost_1"})
    if args_hash is not None:
        pattern.tool_ledger["batch_1"] = make_ledger_record(
            "batch_1",
            tool_name="calculator",
            args_hash=args_hash,
            status=status,
            result=result,
        )

    pattern._discharge_lost_tool_evidence(batch_call_ids)

    assert pattern.lost_tool_evidence.call_ids == {"lost_1"}
    assert pattern.lost_tool_evidence.unnameable is False


def test_a_matching_successful_refetch_discharges_its_entry() -> None:
    """The happy path: a completed, successful, same-fingerprint call
    discharges exactly its own entry and no other.

    Mutation this test catches: a discharge rule that clears the whole
    record instead of just the matched id(s) would also drop "lost_other"
    here; with the real per-fingerprint removal it survives.
    """
    pattern = ReActPattern()
    pattern.tool_ledger["lost_1"] = make_ledger_record(
        "lost_1", tool_name="calculator", args_hash="hash-a"
    )
    pattern.tool_ledger["lost_other"] = make_ledger_record(
        "lost_other", tool_name="search", args_hash="hash-c"
    )
    pattern.lost_tool_evidence = LostToolEvidence(call_ids={"lost_1", "lost_other"})
    pattern.tool_ledger["batch_1"] = make_ledger_record(
        "batch_1",
        tool_name="calculator",
        args_hash="hash-a",
        status="completed",
        result={"output": "the recovered value"},
    )

    pattern._discharge_lost_tool_evidence(["batch_1"])

    assert pattern.lost_tool_evidence.call_ids == {"lost_other"}
    assert pattern.lost_tool_evidence.unnameable is False


@pytest.mark.parametrize("successful_call_count", [1, 3, 10])
def test_an_unnameable_loss_survives_every_successful_call(
    successful_call_count: int,
) -> None:
    """``unnameable`` never clears, no matter how many successful calls a
    batch contains.

    ``unnameable`` stands for a destroyed observation with no call id at
    all, so there is nothing in any batch -- one call or ten -- that could
    ever be shown to be that same observation coming back.

    Mutation this test catches: clearing ``unnameable`` alongside the
    matched ids (or unconditionally, whenever the method runs at all) would
    make ``still_missing()`` read ``False`` here; the real rule leaves it
    untouched, so it stays ``True`` regardless of how many calls succeed.
    """
    pattern = ReActPattern()
    pattern.lost_tool_evidence = LostToolEvidence(call_ids=set(), unnameable=True)
    batch_call_ids = []
    for index in range(successful_call_count):
        call_id = f"batch_{index}"
        pattern.tool_ledger[call_id] = make_ledger_record(
            call_id,
            tool_name="calculator",
            args_hash=f"hash-{index}",
            status="completed",
            result={"output": "value"},
        )
        batch_call_ids.append(call_id)

    pattern._discharge_lost_tool_evidence(batch_call_ids)

    assert pattern.lost_tool_evidence.call_ids == set()
    assert pattern.lost_tool_evidence.unnameable is True
    assert pattern.lost_tool_evidence.still_missing() is True


def test_an_unnameable_loss_survives_a_batch_that_does_discharge_an_entry() -> None:
    """``unnameable`` also survives the path that actually removes an id.

    The test above starts from an empty ``call_ids``, and an empty record
    is sent straight back out of the method before it reaches any removal.
    This one records both an id and an id-less loss, and the finished batch
    resolves the id -- so the removal step really runs -- and pins that the
    id-less loss is still outstanding afterwards.

    Mutation this test catches: clearing ``unnameable`` at the end of the
    removal step would make ``still_missing()`` read ``False`` here, even
    though nothing in the batch could be the id-less observation coming
    back.
    """
    pattern = ReActPattern()
    pattern.tool_ledger["lost_1"] = make_ledger_record(
        "lost_1", tool_name="calculator", args_hash="hash-a"
    )
    pattern.lost_tool_evidence = LostToolEvidence(call_ids={"lost_1"}, unnameable=True)
    pattern.tool_ledger["batch_1"] = make_ledger_record(
        "batch_1",
        tool_name="calculator",
        args_hash="hash-a",
        status="completed",
        result={"output": "the recovered value"},
    )

    pattern._discharge_lost_tool_evidence(["batch_1"])

    assert pattern.lost_tool_evidence.call_ids == set()
    assert pattern.lost_tool_evidence.unnameable is True
    assert pattern.lost_tool_evidence.still_missing() is True


def test_a_stale_success_from_an_earlier_batch_discharges_nothing() -> None:
    """A matching successful call sitting in the ledger from an earlier,
    already-finished batch does not discharge an entry.

    Discharge is scoped to the batch that just finished: the ledger holds a
    completed, successful, fingerprint-matching record, but its id is not
    among ``batch_call_ids`` for this call, so it must not count.

    Mutation this test catches: scanning the whole ledger for any matching
    successful record instead of only the ids passed in would empty the
    record here; the real rule only considers ``batch_call_ids``.
    """
    pattern = ReActPattern()
    pattern.tool_ledger["lost_1"] = make_ledger_record(
        "lost_1", tool_name="calculator", args_hash="hash-a"
    )
    pattern.lost_tool_evidence = LostToolEvidence(call_ids={"lost_1"})
    # A successful, fingerprint-matching record already sits in the ledger
    # from an earlier batch, but it is not part of the batch just finished.
    pattern.tool_ledger["earlier_batch_call"] = make_ledger_record(
        "earlier_batch_call",
        tool_name="calculator",
        args_hash="hash-a",
        status="completed",
        result={"output": "an old success"},
    )

    pattern._discharge_lost_tool_evidence(["some_other_unrelated_call"])

    assert pattern.lost_tool_evidence.call_ids == {"lost_1"}
    assert pattern.lost_tool_evidence.unnameable is False


def test_every_tool_batch_completion_discharges_the_record() -> None:
    """Every point in react.py that awaits a tool batch must also discharge
    lost evidence.

    This is a structural check over the source, not a behavioral one -- the
    discharge rule's own behavior is covered by the tests above. What this
    pins down is the wiring: every call site that awaits
    ``self._execute_pending_tool_calls`` is a tool-batch completion point,
    and each one must also call ``self._discharge_lost_tool_evidence`` in
    its own enclosing statement block, so a batch that completed work is
    never left unchecked for recoverable losses just because of which call
    site it went through.

    The "exactly 2" count is not a number this test is trying to freeze for
    its own sake. It exists to force whoever adds a third call site to
    ``_execute_pending_tool_calls`` to make a conscious decision about
    whether that new site also needs a discharge call, instead of silently
    inheriting whatever the count happened to be. If react.py grows a third
    call site, update this test deliberately, with the new site's discharge
    call added or a stated reason it does not need one.

    Mutation this test catches: deleting the discharge call that follows
    either await fails that call site's own assertion immediately; a
    hypothetical third call site added later without a discharge call, or
    without updating this test, fails the count assertion.
    """
    tree = ast.parse(Path(react_module.__file__).read_text())
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]

    def is_execute_pending_await(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Await)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "_execute_pending_tool_calls"
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "self"
        )

    def is_discharge_call(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_discharge_lost_tool_evidence"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        )

    def containing_block(stmt: ast.stmt) -> list[ast.stmt]:
        parent = getattr(stmt, "parent")
        for field_name in ("body", "orelse", "finalbody"):
            value = getattr(parent, field_name, None)
            if isinstance(value, list) and stmt in value:
                return value
        raise AssertionError(
            f"could not locate the enclosing statement block for {ast.dump(stmt)}"
        )

    call_sites: list[ast.stmt] = []
    for node in ast.walk(tree):
        if not is_execute_pending_await(node):
            continue
        enclosing_statement: ast.AST = node
        while not isinstance(enclosing_statement, ast.stmt):
            enclosing_statement = getattr(enclosing_statement, "parent")
        call_sites.append(enclosing_statement)

    assert len(call_sites) == 2, (
        "expected exactly 2 statements awaiting _execute_pending_tool_calls "
        f"in react.py; found {len(call_sites)} -- see this test's docstring "
        "before changing this number"
    )

    for statement in call_sites:
        block = containing_block(statement)
        discharge_calls = [
            inner
            for sibling in block
            for inner in ast.walk(sibling)
            if is_discharge_call(inner)
        ]
        assert discharge_calls, (
            "a statement awaiting _execute_pending_tool_calls has no "
            "_discharge_lost_tool_evidence call in its enclosing block: "
            f"{ast.dump(statement)}"
        )


@pytest.mark.asyncio
async def test_a_call_that_arrived_without_an_id_still_discharges_its_entry() -> None:
    """A batch call with no id of its own at plan time still discharges the
    lost-evidence entry its refetch resolves.

    Reproduced at the loop level rather than by driving the model to omit an
    id: the model-facing path always normalizes a tool call's id before it
    ever reaches ``pending_tool_calls``, so a real end-to-end run cannot
    produce one that is missing there. A checkpoint resume can, though --
    ``pending_tool_calls`` is restored from state verbatim, with whatever
    shape it was written in -- and this test starts from exactly that shape:
    a pending call with no ``id`` key at all, the same as a resumed batch a
    pre-normalization checkpoint might carry. ``_execute_tool_safely``
    stamps an id onto that same dict, in place, the moment it actually runs
    the call, which is the id the discharge lookup has to use.

    Mutation this test catches: reading ``batch_call_ids`` from
    ``self.pending_tool_calls`` (or from ``tool_calls``) before awaiting
    ``_execute_pending_tool_calls`` instead of after makes this call's id
    read back empty, so the fingerprint match never happens and the entry
    is left in the record even though its value came back.
    """
    pattern = ReActPattern(max_iterations=3)
    args_hash = pattern._args_hash({})
    pattern.tool_ledger["lost_call"] = make_ledger_record(
        "lost_call", tool_name="calculator", args_hash=args_hash
    )
    pattern.lost_tool_evidence = LostToolEvidence(call_ids={"lost_call"})
    pattern.status = "acting"
    tool_call: dict[str, Any] = {"name": "calculator", "args": {}}
    pattern.pending_tool_calls = [tool_call]

    context = ExecutionContext(execution_id="no-id-discharge")
    context.add_user_message("Answer now.")
    llm = RecordingLLM([final_answer_response()])
    runtime = PatternRuntime()

    result = await pattern.run(
        context=context,
        tools=[NamedTool("calculator")],
        llm=llm,
        compact_llm=None,
        runtime=runtime,
    )

    assert result["success"] is True
    # The stamp landed on the same dict this test built, and the ledger now
    # holds the call under that stamped id.
    stamped_id = tool_call["id"]
    assert stamped_id
    assert pattern.tool_ledger[stamped_id].tool_name == "calculator"
    # Checked mid-run, at the "before_llm" checkpoint the next iteration
    # writes right after the batch's discharge and well before the run
    # finishes: the run's own end-of-run clear (``_finalize_outcome`` also
    # empties this record) would otherwise mask a discharge that never
    # actually matched anything.
    mid_run_state = state_at(runtime, "before_llm")
    assert mid_run_state["lost_tool_evidence"]["call_ids"] == []
    assert pattern.lost_tool_evidence.call_ids == set()


# The tests below cover how a forced-answer turn reads ``lost_tool_evidence``
# back and logs what it finds. The tests further down in this file cover how
# that same read changes the forced-answer instruction text itself.


@pytest.mark.asyncio
@pytest.mark.parametrize("turns_between", [1, 2, 3])
@pytest.mark.parametrize(
    "strategy",
    ["llm_summary", "truncate"],
    ids=["summary_strategy", "truncate_strategy"],
)
@pytest.mark.parametrize("forcing", ["sticky_flag", "finalize_after_tool_result"])
async def test_a_loss_on_an_earlier_turn_still_forces_an_honest_answer(
    turns_between: int,
    strategy: str,
    forcing: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A forced turn warns about evidence an earlier turn's compaction
    destroyed, even when several ordinary turns ran in between and the
    forced turn's own compaction destroys nothing.

    Covers both ways a turn becomes forced -- the sticky
    ``force_final_answer_next`` flag, and ``finalize_after_tool_result``
    deriving the same instruction inline from the last successful tool
    result -- and both real compaction strategies, since the two report the
    same metadata keys but under different ``strategy`` labels.

    Mutation this test catches: a gate reading
    ``dropped_tool_result_count > 0`` off this turn's own ``CompactResult``,
    instead of the run's durable ``lost_tool_evidence`` record, emits zero
    warnings on every cell here, because the forced turn's own compaction
    always reports nothing dropped in this scenario. The record-reading gate
    emits exactly one.
    """
    tool = NamedTool("calculator")
    no_op_response = {"content": "", "tool_calls": [], "done": False}
    loss_result = compact_result(count=1, call_ids=["lost_1"], strategy=strategy)
    nothing_result = compact_result(count=0, by_name={}, strategy=strategy)

    pattern = ReActPattern(max_iterations=turns_between + 5)
    if forcing == "finalize_after_tool_result":
        pattern.finalize_after_tool_result = True

    responses: list[Any] = [no_op_response]
    scripted: list[Any] = [loss_result]
    for step in range(1, turns_between + 1):
        scripted.append(nothing_result)
        if forcing == "finalize_after_tool_result" and step == turns_between:
            responses.append(tool_call_response("calculator", call_id=f"mid_{step}"))
        else:
            responses.append(no_op_response)
    scripted.append(nothing_result)
    responses.append(final_answer_response())

    force_after_call = turns_between + 1 if forcing == "sticky_flag" else None
    runtime = ScriptedCompactionRuntime(
        scripted, pattern=pattern, force_after_call=force_after_call
    )
    llm = RecordingLLM(responses)
    context = ExecutionContext(execution_id="earlier-loss")
    context.add_user_message("Use calculator and report every value.")

    with caplog.at_level(logging.WARNING, logger=react_module.__name__):
        result = await pattern.run(
            context=context,
            tools=[tool],
            llm=llm,
            compact_llm=None,
            runtime=runtime,
        )

    assert result["success"] is True
    assert len(evidence_dropped_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_a_forced_turn_with_nothing_missing_says_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A forced turn -- forced for any reason -- with an empty lost-evidence
    record stays silent. This is the ordinary case: most forced turns have
    lost nothing at all.
    """
    pattern = ReActPattern(max_iterations=3)
    pattern.force_final_answer_next = True
    runtime = ScriptedCompactionRuntime([compact_result(count=0, by_name={})])
    llm = RecordingLLM([final_answer_response()])
    context = ExecutionContext(execution_id="nothing-missing")
    context.add_user_message("Answer now.")

    with caplog.at_level(logging.WARNING, logger=react_module.__name__):
        result = await pattern.run(
            context=context, tools=[], llm=llm, compact_llm=None, runtime=runtime
        )

    assert result["success"] is True
    assert evidence_dropped_warnings(caplog) == []


@pytest.mark.asyncio
async def test_an_unforced_turn_says_nothing_however_much_is_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unforced turn holds the whole tool set and can fetch the value
    itself, so there is nothing to warn about yet, however much the record
    already holds.
    """
    pattern = ReActPattern(max_iterations=3)
    pattern.lost_tool_evidence = LostToolEvidence(
        call_ids={"lost_1", "lost_2"}, unnameable=True
    )
    runtime = ScriptedCompactionRuntime([compact_result(count=0, by_name={})])
    llm = RecordingLLM(
        [{"content": "Answering directly.", "tool_calls": [], "done": True}]
    )
    context = ExecutionContext(execution_id="unforced-missing")
    context.add_user_message("Answer now.")

    with caplog.at_level(logging.WARNING, logger=react_module.__name__):
        result = await pattern.run(
            context=context, tools=[], llm=llm, compact_llm=None, runtime=runtime
        )

    assert result["success"] is True
    assert evidence_dropped_warnings(caplog) == []


@pytest.mark.asyncio
async def test_an_unnameable_loss_alone_still_trips_the_gate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A record holding only ``unnameable=True`` and no call ids still forces
    the honest turn to say so, and the log must read ``named=unknown``, not
    ``named=[]``.

    Mutation this test catches: printing the empty names list makes the
    field read ``named=[]``, which reads as an intact run with nothing
    actually missing.
    """
    pattern = ReActPattern(max_iterations=3)
    pattern.force_final_answer_next = True
    pattern.lost_tool_evidence = LostToolEvidence(call_ids=set(), unnameable=True)
    runtime = ScriptedCompactionRuntime([compact_result(count=0, by_name={})])
    llm = RecordingLLM([final_answer_response()])
    context = ExecutionContext(execution_id="unnameable-only")
    context.add_user_message("Answer now.")

    with caplog.at_level(logging.WARNING, logger=react_module.__name__):
        result = await pattern.run(
            context=context, tools=[], llm=llm, compact_llm=None, runtime=runtime
        )

    assert result["success"] is True
    warnings = evidence_dropped_warnings(caplog)
    assert len(warnings) == 1
    assert "named=unknown" in warnings[0]
    assert "named_count=0" in warnings[0]
    assert "named=[]" not in warnings[0]


@pytest.mark.parametrize(
    ("name_count", "name_length"),
    [(20, 64), (20, 200), (21, 8), (1, 2000), (20, 63)],
    ids=["at_cap", "long_names", "one_over_cap", "single_long_name", "regression_1176"],
)
def test_the_named_list_in_the_log_stays_bounded(
    name_count: int, name_length: int
) -> None:
    """``_bounded_tool_names`` never returns a result whose lines, joined
    with the trailing omission line it appends, exceed the shared notice
    budget -- however many names there are, how long each one runs, or
    whether anything was actually left out.

    ``regression_1176`` is the measured counter-example that used to slip
    past this budget: 20 names of 63 characters each fit under
    ``COMPACT_DROPPED_TOOL_NOTICE_MAX_CHARS`` on their own, but appending
    the omission line after the fact pushed the assembled result to 1176
    characters against a 1152 budget.

    Mutation this test catches: dropping ``chars_used`` from the
    ``bounded_notice_lines`` call inside ``_bounded_tool_names`` -- or
    dropping the total character-budget skip inside ``bounded_notice_lines``
    itself, keeping only the per-name length clamp and the entry-count cap
    -- lets the joined length, omission line included, exceed
    ``COMPACT_DROPPED_TOOL_NOTICE_MAX_CHARS``.
    """
    names = [f"tool_{index}".rjust(name_length, "x") for index in range(name_count)]

    result = ReActPattern._bounded_tool_names(names)

    assert len("\n".join(result)) <= COMPACT_DROPPED_TOOL_NOTICE_MAX_CHARS

    expected_lines, omitted = bounded_notice_lines(
        names,
        render=lambda name: name[:COMPACT_DROPPED_TOOL_NAME_MAX_CHARS],
        max_chars=COMPACT_DROPPED_TOOL_NOTICE_MAX_CHARS,
        max_entries=COMPACT_DROPPED_TOOL_NOTICE_MAX_NAMES,
        chars_used=LOST_TOOL_NAME_OMISSION_ALLOWANCE,
    )
    if omitted:
        assert result[:-1] == expected_lines
        assert re.match(r"^\.\.\. \d+ more names? omitted$", result[-1])
    else:
        assert result == expected_lines


def test_a_name_the_ledger_cannot_resolve_counts_as_unnamed() -> None:
    """A call id the ledger cannot resolve is absent from the returned names
    but still marks the result unnameable, rather than being silently
    dropped.

    Dropping it instead of flagging it would let a record that still holds a
    real entry report back as though nothing were missing.
    """
    pattern = ReActPattern()
    pattern.lost_tool_evidence = LostToolEvidence(call_ids={"ghost_call"})

    names, unnamed = pattern._lost_tool_names()

    assert names == []
    assert unnamed is True


@pytest.mark.asyncio
async def test_a_matching_refetch_stops_the_honest_forcing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A real re-fetch of the same tool with the same arguments discharges
    the lost-evidence entry through the actual loop wiring, so a later
    forced turn stays silent.

    This drives a real tool call into ``self.tool_ledger``, scripts a
    compaction that reports that exact call id destroyed, drives a second
    real call to the same tool with the same (empty) arguments, and only
    then forces the answer -- proving the batch-id capture at the in-turn
    tool-call call site (the ``if tool_calls:`` branch, not the loop-top
    resume branch) feeds ``_discharge_lost_tool_evidence`` correctly. A test
    that called ``_discharge_lost_tool_evidence`` directly, as the tests
    above it in this file do, would not exercise that wiring.
    """
    tool = NamedTool("list_clients")
    pattern = ReActPattern(max_iterations=6)
    scripted_compactions: list[Any] = [
        None,
        compact_result(count=1, call_ids=["first"], by_name={"list_clients": 1}),
        compact_result(count=0, by_name={}),
    ]
    runtime = ScriptedCompactionRuntime(
        scripted_compactions, pattern=pattern, force_after_call=2
    )
    llm = RecordingLLM(
        [
            tool_call_response("list_clients", call_id="first"),
            tool_call_response("list_clients", call_id="refetch"),
            final_answer_response(),
        ]
    )
    context = ExecutionContext(execution_id="matching-refetch")
    context.add_user_message("List every client.")

    with caplog.at_level(logging.WARNING, logger=react_module.__name__):
        result = await pattern.run(
            context=context,
            tools=[tool],
            llm=llm,
            compact_llm=None,
            runtime=runtime,
        )

    assert result["success"] is True
    assert len(tool.calls) == 2
    assert pattern.lost_tool_evidence.call_ids == set()
    assert evidence_dropped_warnings(caplog) == []


# The tests below cover the wording of the forced-answer instruction itself:
# what it asks for when a turn's evidence is intact, and how it changes when
# ``lost_tool_evidence`` still holds something compaction destroyed.


async def drive_forced_turn_missing_evidence(
    *,
    strategy: str = "truncate",
    named: dict[str, str] | None = None,
    unnameable_ids: Sequence[str] = (),
    execution_id: str = "forced-missing-evidence",
) -> RecordingLLM:
    """Run one forced-answer turn whose lost-evidence record holds exactly
    the given call ids, and return the ``RecordingLLM`` that captured every
    prompt the turn sent.

    ``named`` maps a call id to the tool name it should resolve to: each one
    is pre-seeded into the pattern's own tool ledger so ``_lost_tool_names``
    can look it up for the gate's log line, the same way a real run's ledger
    would after actually executing that call. ``unnameable_ids`` are left
    out of the ledger entirely, so the same lookup fails. Passing neither
    leaves the record empty, which is the ordinary case: a forced turn with
    nothing missing.
    """
    named = dict(named or {})
    pattern = ReActPattern(max_iterations=4)
    for call_id, tool_name in named.items():
        pattern.tool_ledger[call_id] = make_ledger_record(call_id, tool_name=tool_name)
    dropped_ids = [*named.keys(), *unnameable_ids]
    scripted: list[Any] = [
        compact_result(count=len(dropped_ids), call_ids=dropped_ids, strategy=strategy)
        if dropped_ids
        else compact_result(count=0, by_name={}, strategy=strategy)
    ]
    pattern.force_final_answer_next = True
    runtime = ScriptedCompactionRuntime(scripted)
    llm = RecordingLLM([final_answer_response()])
    context = ExecutionContext(execution_id=execution_id)
    context.add_user_message("Answer now.")

    result = await pattern.run(
        context=context, tools=[], llm=llm, compact_llm=None, runtime=runtime
    )
    assert result["success"] is True
    return llm


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
                provider="test-provider",
                code=self.code,
                message=f"provider returned {self.code}",
            )
        if not self.responses:
            return {"content": "fallback answer", "done": True}
        return self.responses.pop(0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy",
    ["llm_summary", "truncate"],
    ids=["summary_strategy", "truncate_strategy"],
)
@pytest.mark.parametrize("shape", ["named_only", "unnameable_only", "both"])
async def test_the_honest_instruction_replaces_the_stale_one(
    strategy: str, shape: str
) -> None:
    """A forced turn whose record still holds something compaction
    destroyed is told so, whichever compaction path destroyed it and
    whatever mix of ledger-resolvable and unresolvable ids the record
    holds.

    Mutation this test catches: dropping the ``evidence_dropped`` argument
    from the ``_messages_for_llm`` call in ``_run_tool_calling_loop`` sends
    the ordinary forced instruction, whose opening asks the model to answer
    from tool results that are gone.
    """
    named: dict[str, str] = {}
    unnameable_ids: list[str] = []
    if shape in ("named_only", "both"):
        named["lost_named"] = "list_clients"
    if shape in ("unnameable_only", "both"):
        unnameable_ids.append("lost_orphan")

    llm = await drive_forced_turn_missing_evidence(
        strategy=strategy, named=named, unnameable_ids=unnameable_ids
    )

    assert HONEST_PHRASE in instruction_of(llm, 0)
    # Checked against the whole prompt, not just the instruction: a stale
    # copy of the same advice sitting in any other message would undo the
    # honest wording just as much as leaving it in the instruction would.
    assert STALE_EVIDENCE_PHRASE not in whole_prompt_of(llm, 0)


@pytest.mark.asyncio
async def test_the_honest_instruction_is_conditional_on_a_summary_being_present() -> (
    None
):
    """The instruction is worded as a conditional on a summary being
    present, never as a branch on which compaction strategy destroyed the
    evidence.

    The message-dropping backstop only trims a tail window, so a summary an
    earlier turn's own compaction already wrote can still be standing above
    a forced turn whose own compaction this turn only dropped messages --
    wording that branched on this turn's own strategy would wrongly tell
    the model there is no summary in that case.

    Mutation this test catches: branching the opening on which compaction
    strategy ran makes the two instructions below differ; the real wording
    is identical either way because it never reads the strategy at all.
    """
    summary_llm = await drive_forced_turn_missing_evidence(
        strategy="llm_summary", named={"lost_named": "list_clients"}
    )
    truncate_llm = await drive_forced_turn_missing_evidence(
        strategy="truncate", named={"lost_named": "list_clients"}
    )

    summary_instruction = instruction_of(summary_llm, 0)
    truncate_instruction = instruction_of(truncate_llm, 0)

    assert summary_instruction == truncate_instruction
    assert CONDITIONAL_SUMMARY_PHRASE in summary_instruction
    assert CONDITIONAL_SUMMARY_PHRASE in truncate_instruction


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["llm_summary", "truncate"])
async def test_the_honest_instruction_reuses_the_shared_value_kinds_list(
    strategy: str,
) -> None:
    """The kinds of value the instruction enumerates come from the shared
    ``VALUE_KINDS`` constant, interpolated, never a hand-written fourth
    copy of the same list.

    Mutation this test catches: replacing the interpolated constant with a
    hand-written, shorter list of value kinds makes ``VALUE_KINDS in
    instruction`` false.
    """
    llm = await drive_forced_turn_missing_evidence(
        strategy=strategy, named={"lost_named": "list_clients"}
    )
    assert VALUE_KINDS in instruction_of(llm, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "record_tools",
    [("send_message",), ("list_clients",), ("send_message", "list_clients")],
    ids=["write_only", "read_only", "write_and_read"],
)
async def test_a_removed_write_observation_does_not_ban_completed_outright(
    record_tools: tuple[str, ...],
) -> None:
    """The completion rule the honest instruction states is a conditional,
    not a flat ban, whether the removed observation came from a tool that
    writes, one that reads, or a mix of both.

    A removed observation is not always a removed read: a write whose
    result message compaction later discarded may already have succeeded,
    so a flat ban would make the model under-report work that genuinely
    finished.

    Mutation this test catches: replacing the conditional rule with a flat
    ban makes the instruction contain "outcome=completed is not available
    on this turn".
    """
    named = {f"lost_{name}": name for name in record_tools}
    llm = await drive_forced_turn_missing_evidence(named=named)

    instruction = instruction_of(llm, 0)
    assert COMPLETED_CONDITIONAL_PHRASE in instruction
    assert "outcome=completed is not available on this turn" not in instruction


@pytest.mark.asyncio
async def test_a_turn_with_nothing_missing_keeps_the_ordinary_instruction() -> None:
    """A forced turn whose lost-evidence record is empty is given the same
    words, in the same order, that every forced turn is given.

    The wording below is held as one contiguous literal on purpose. The
    source builds it from an opening and a shared tail so that a turn
    missing evidence can swap the opening, and splitting a string that way
    is exactly how a sentence quietly changes places with its neighbour.

    Mutation this test catches: moving one sentence past another while
    carving the opening out. Every phrase-level assertion above still
    passes on the reordered text; this one stops finding the literal.
    """
    llm = await drive_forced_turn_missing_evidence()

    instruction = instruction_of(llm, 0)
    assert STALE_EVIDENCE_PHRASE in instruction
    assert COMPLETED_OFFERED_PHRASE in instruction
    assert HONEST_PHRASE not in instruction

    ordinary_wording = (
        "Produce the final user-facing answer by calling the final_answer "
        "control tool exactly once using the accumulated conversation and "
        "tool results. Do not call any other tool and do not output "
        "tool-call markup as plain text. Set outcome=completed only when "
        "every requested action or verification succeeded; otherwise set "
        "outcome=partial or outcome=blocked and say what remains. If a "
        "previous ask_user_question narrowed the request to a selected "
        "subset of items or resources, the final answer must cover only "
        "that subset — leave out anything outside it even if an earlier "
        "tool call already returned data about it. "
    )
    # ExecutionContext's own system message (turn-start time, file-reference
    # rules) sits ahead of this instruction in the same message; the
    # instruction itself is appended verbatim, so it is the exact suffix.
    assert instruction.endswith(
        f"{ordinary_wording}"
        f"{grounding_rule(can_call_tools=False)}\n\n"
        f"{final_deliverable_file_reference_instructions(can_lookup=False)}\n\n"
        f"{final_answer_language_rule()}"
    )


@pytest.mark.asyncio
async def test_the_protocol_repair_keeps_the_honest_wording() -> None:
    """The retry that repairs a broken tool-protocol response is often the
    call that actually produces the answer the user sees, so it needs the
    same honest wording as the first call it replaces.

    Mutation this test catches: not forwarding ``evidence_dropped`` into
    ``_retry_tool_protocol_response`` makes the retry rebuild its prompt
    without it, so its instruction falls back to ``STALE_EVIDENCE_PHRASE``;
    with the real forwarding it carries ``HONEST_PHRASE`` instead, same as
    the call it is repairing.
    """
    pattern = ReActPattern(max_iterations=4)
    pattern.tool_ledger["lost_named"] = make_ledger_record(
        "lost_named", tool_name="list_clients"
    )
    pattern.force_final_answer_next = True
    runtime = ScriptedCompactionRuntime(
        [compact_result(count=1, call_ids=["lost_named"], strategy="truncate")]
    )
    llm = ProtocolErrorLLM(
        [final_answer_response()],
        fail_on=0,
        code="invalid_tool_protocol",
    )
    context = ExecutionContext(execution_id="protocol-repair-honest")
    context.add_user_message("Answer now.")

    result = await pattern.run(
        context=context, tools=[], llm=llm, compact_llm=None, runtime=runtime
    )

    assert result["success"] is True
    assert len(llm.calls) >= 2
    first_instruction = instruction_of(llm, 0)
    retry_instruction = instruction_of(llm, 1)
    assert HONEST_PHRASE in first_instruction
    assert HONEST_PHRASE in retry_instruction
    assert STALE_EVIDENCE_PHRASE not in retry_instruction
