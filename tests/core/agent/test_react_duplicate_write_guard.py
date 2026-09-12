"""Same-turn duplicate-write guard on the ReAct tool-execution path.

Covers xorbitsai/xagent#2217: a write-category tool call whose (tool,
arguments) pair already succeeded earlier in the same turn must not execute
again; the model receives a structured suppression envelope carrying the
prior result instead.

Enrollment is explicit-only and read off ``tool.metadata`` (wrappers such as
the sandbox tool wrapper forward only ``.metadata``): an MCP wire
declaration carried as ``metadata.mcp_non_idempotent_write`` (classified by
``classify_non_idempotent_write`` — covered by the wire-level tests in
``tests/core/tools/adapters/vibe/test_mcp_adapter.py``), or an internal
tool's ``non_idempotent = True`` marker. Undeclared tools stay exempt, so
legitimate repeated reads (status polling with identical args) keep
executing. The guard is strictly per-turn: it only fires for calls stamped
with a turn_id, and only against records from the same turn.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from xagent.core.agent import ExecutionContext, ReActPattern
from xagent.core.agent.pattern.react.duplicate_write_guard import (
    DUPLICATE_WRITE_SUPPRESSED_KEY,
    build_suppression_envelope,
    tool_requires_duplicate_write_guard,
)

TURN_1 = {"turn_id": "turn-1"}
TURN_2 = {"turn_id": "turn-2"}


class CreateRecordArgs(BaseModel):
    title: str
    amount: int = 0


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeWriteTool:
    """A create-style tool whose metadata declaration is configurable."""

    def __init__(
        self,
        *,
        mcp_non_idempotent_write: Any = True,
        fail_first: bool = False,
        name: str = "create_record",
        concurrency_safe: bool = False,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fail_first = fail_first

        class Metadata:
            description = "Create a record in the external system."
            non_idempotent = False

        Metadata.name = name
        Metadata.mcp_non_idempotent_write = mcp_non_idempotent_write
        Metadata.concurrency_safe = concurrency_safe
        self.metadata = Metadata()

    def args_type(self) -> type[BaseModel]:
        return CreateRecordArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        self.calls.append(dict(args))
        if self._fail_first and len(self.calls) == 1:
            return {"success": False, "error": "transient provider error"}
        return {"success": True, "record_id": f"rec-{len(self.calls)}"}


class FakeNonIdempotentInternalTool(FakeWriteTool):
    """Internal (non-MCP) tool explicitly marked non-idempotent."""

    non_idempotent = True

    def __init__(self) -> None:
        super().__init__(mcp_non_idempotent_write=None, name="submit_form")


def _tool_call_response(name: str, args: dict[str, Any], call_id: str) -> dict:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
        "done": False,
    }


def _pattern(**overrides: Any) -> ReActPattern:
    kwargs: dict[str, Any] = {
        "max_iterations": 8,
        "repeated_tool_decision_after_consecutive_tool_calls": None,
        "repeated_tool_decision_after_consecutive_work_tool_calls": None,
    }
    kwargs.update(overrides)
    return ReActPattern(**kwargs)


def _turn_context(message: str = "Create the record") -> ExecutionContext:
    context = ExecutionContext()
    context.add_user_message(message, metadata=dict(TURN_1))
    return context


def _run_twice_llm(
    name: str,
    first_args: dict[str, Any],
    second_args: dict[str, Any],
) -> FakeLLM:
    return FakeLLM(
        responses=[
            _tool_call_response(name, first_args, "call_1"),
            _tool_call_response(name, second_args, "call_2"),
            {"content": "Done.", "done": True},
        ]
    )


def _tool_results(context: ExecutionContext) -> list[Any]:
    return [
        message.metadata["raw_result"]
        for message in context.messages
        if message.role == "tool"
    ]


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------


def test_only_explicit_declarations_require_the_guard() -> None:
    assert tool_requires_duplicate_write_guard(
        FakeWriteTool(mcp_non_idempotent_write=True)
    )
    assert tool_requires_duplicate_write_guard(FakeNonIdempotentInternalTool())
    # Undeclared MCP tools (None) and explicitly-idempotent ones (False) are
    # exempt: an unannotated tool may be a legitimate identical-args poll
    # loop, and deduplication fails open.
    assert not tool_requires_duplicate_write_guard(
        FakeWriteTool(mcp_non_idempotent_write=None)
    )
    assert not tool_requires_duplicate_write_guard(
        FakeWriteTool(mcp_non_idempotent_write=False)
    )


def test_metadata_level_internal_marker_enrolls() -> None:
    # AbstractBaseTool.metadata carries the tool's non_idempotent marker, and
    # wrappers forward metadata — the guard must honor it there too.
    tool = FakeWriteTool(mcp_non_idempotent_write=None)
    tool.metadata.non_idempotent = True
    assert tool_requires_duplicate_write_guard(tool)


def test_non_boolean_marker_on_the_tool_does_not_enroll() -> None:
    # A tool author writing a truthy non-boolean must not be enrolled by
    # accident. Only the tool-object attribute is checked here: the metadata
    # fields are typed on a pydantic model, which coerces a truthy value to
    # True before the guard ever sees it, so there is no such shape to pin.
    tool = FakeWriteTool(mcp_non_idempotent_write=None)
    tool.non_idempotent = "yes"  # type: ignore[attr-defined]
    assert not tool_requires_duplicate_write_guard(tool)


def test_tool_without_metadata_is_exempt() -> None:
    class Bare:
        pass

    assert not tool_requires_duplicate_write_guard(Bare())


# ---------------------------------------------------------------------------
# Suppression through the ReAct loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_write_is_suppressed() -> None:
    args = {"title": "invoice", "amount": 7}
    llm = _run_twice_llm("create_record", args, dict(args))
    tool = FakeWriteTool()
    context = _turn_context("Create the invoice record")

    result = await _pattern().run(context=context, tools=[tool], llm=llm)

    assert result["success"] is True
    assert tool.calls == [args]

    tool_results = _tool_results(context)
    assert len(tool_results) == 2
    envelope = tool_results[1]
    assert envelope[DUPLICATE_WRITE_SUPPRESSED_KEY] is True
    assert envelope["success"] is True
    assert envelope["suppressed_duplicate_of"] == "call_1"
    assert envelope["result"] == {"success": True, "record_id": "rec-1"}


@pytest.mark.asyncio
async def test_key_order_does_not_defeat_the_guard() -> None:
    # The args hash canonicalizes via json.dumps(sort_keys=True): the same
    # arguments serialized in a different key order are still one write.
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "create_record",
                            "arguments": '{"title": "invoice", "amount": 7}',
                        },
                    }
                ],
                "done": False,
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {
                            "name": "create_record",
                            "arguments": '{"amount": 7, "title": "invoice"}',
                        },
                    }
                ],
                "done": False,
            },
            {"content": "Done.", "done": True},
        ]
    )
    tool = FakeWriteTool()
    context = _turn_context()

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert len(tool.calls) == 1


@pytest.mark.asyncio
async def test_different_args_both_execute() -> None:
    llm = _run_twice_llm(
        "create_record",
        {"title": "invoice", "amount": 7},
        {"title": "invoice", "amount": 8},
    )
    tool = FakeWriteTool()
    context = _turn_context("Create both records")

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert len(tool.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "declaration",
    [None, False],
    ids=["undeclared", "explicitly_idempotent"],
)
async def test_unenrolled_tools_repeat_identical_calls(declaration: Any) -> None:
    args = {"title": "status-poll"}
    llm = _run_twice_llm("create_record", args, dict(args))
    tool = FakeWriteTool(mcp_non_idempotent_write=declaration)
    context = _turn_context("Poll twice")

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert len(tool.calls) == 2


@pytest.mark.asyncio
async def test_failed_first_write_is_retried() -> None:
    args = {"title": "invoice"}
    llm = _run_twice_llm("create_record", args, dict(args))
    tool = FakeWriteTool(fail_first=True)
    context = _turn_context()

    await _pattern().run(context=context, tools=[tool], llm=llm)

    # Only a call that *succeeded* earlier in the turn suppresses a repeat.
    assert len(tool.calls) == 2


@pytest.mark.asyncio
async def test_non_idempotent_internal_tool_is_guarded() -> None:
    args = {"title": "form"}
    llm = _run_twice_llm("submit_form", args, dict(args))
    tool = FakeNonIdempotentInternalTool()
    context = _turn_context("Submit the form")

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert tool.calls == [args]


@pytest.mark.asyncio
async def test_provider_id_reuse_does_not_defeat_the_guard() -> None:
    # Provider-supplied tool_call ids are not guaranteed unique. When the
    # model reuses the completed call's id for the identical repeat, the
    # guard must still find the completed record (the scan runs before the
    # repeat writes any ledger entry) and must not overwrite the genuine
    # result with the suppression envelope, so a third reuse still attaches
    # the original result.
    args = {"title": "invoice"}
    llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", args, "call_1"),
            _tool_call_response("create_record", dict(args), "call_1"),
            _tool_call_response("create_record", dict(args), "call_1"),
            {"content": "Done.", "done": True},
        ]
    )
    tool = FakeWriteTool()
    context = _turn_context()

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert len(tool.calls) == 1
    envelopes = _tool_results(context)[1:]
    assert len(envelopes) == 2
    for envelope in envelopes:
        assert envelope[DUPLICATE_WRITE_SUPPRESSED_KEY] is True
        assert envelope["suppressed_duplicate_of"] == "call_1"
        assert envelope["result"] == {"success": True, "record_id": "rec-1"}


@pytest.mark.asyncio
async def test_third_call_attaches_the_original_result() -> None:
    args = {"title": "invoice"}
    llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", args, "call_1"),
            _tool_call_response("create_record", dict(args), "call_2"),
            _tool_call_response("create_record", dict(args), "call_3"),
            {"content": "Done.", "done": True},
        ]
    )
    tool = FakeWriteTool()
    context = _turn_context()

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert len(tool.calls) == 1
    envelopes = _tool_results(context)[1:]
    # Both suppressions must point at the genuine execution, never at a
    # previous suppression envelope.
    for envelope in envelopes:
        assert envelope["suppressed_duplicate_of"] == "call_1"
        assert envelope["result"] == {"success": True, "record_id": "rec-1"}


@pytest.mark.asyncio
async def test_unknown_tool_bypasses_the_guard_gracefully() -> None:
    # A call naming a tool that is not mounted must not trip the guard's
    # lookup; it flows to the normal not-found error path.
    llm = FakeLLM(
        responses=[
            _tool_call_response("missing_tool", {"title": "x"}, "call_1"),
            {"content": "Done.", "done": True},
        ]
    )
    tool = FakeWriteTool()
    context = _turn_context("Use a missing tool")

    result = await _pattern().run(context=context, tools=[tool], llm=llm)

    assert result["success"] is True
    assert tool.calls == []
    first_result = _tool_results(context)[0]
    assert first_result.get("success") is False


# ---------------------------------------------------------------------------
# Turn scoping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unstamped_turns_are_never_guarded() -> None:
    # Fail-closed on turn identity: without a stamped turn_id the guard
    # cannot bound suppression to one turn, so it does not fire at all.
    args = {"title": "invoice"}
    llm = _run_twice_llm("create_record", args, dict(args))
    tool = FakeWriteTool()
    context = ExecutionContext()
    context.add_user_message("Create the record")  # no turn metadata

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert len(tool.calls) == 2


@pytest.mark.asyncio
async def test_intra_turn_resume_suppresses_under_the_same_turn_id() -> None:
    args = {"title": "invoice"}
    first_llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", args, "call_1"),
            {"content": "Created.", "done": True},
        ]
    )
    first_tool = FakeWriteTool()
    first_context = _turn_context()
    first_pattern = _pattern()
    await first_pattern.run(context=first_context, tools=[first_tool], llm=first_llm)
    assert len(first_tool.calls) == 1

    resumed_pattern = _pattern()
    resumed_pattern.load_state(json.loads(json.dumps(first_pattern.get_state())))
    resumed_llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", dict(args), "call_2"),
            {"content": "Done.", "done": True},
        ]
    )
    resumed_tool = FakeWriteTool()
    resumed_context = _turn_context()

    await resumed_pattern.run(
        context=resumed_context, tools=[resumed_tool], llm=resumed_llm
    )

    assert resumed_tool.calls == []
    envelope = _tool_results(resumed_context)[0]
    assert envelope[DUPLICATE_WRITE_SUPPRESSED_KEY] is True
    assert envelope["suppressed_duplicate_of"] == "call_1"


@pytest.mark.asyncio
async def test_identical_write_in_a_later_turn_executes() -> None:
    # The guard must be strictly per-turn (#2217): an explicit user request
    # in a later turn of the same execution repeats the identical write.
    # The runner stamps a fresh turn_id on every user message; the pattern
    # re-resolves it at each pattern start.
    args = {"title": "invoice"}
    pattern = _pattern()
    tool = FakeWriteTool()
    context = ExecutionContext()

    context.add_user_message("Create the record", metadata=dict(TURN_1))
    first_llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", args, "call_1"),
            {"content": "Created.", "done": True},
        ]
    )
    await pattern.run(context=context, tools=[tool], llm=first_llm)
    assert len(tool.calls) == 1

    context.add_user_message(
        "Create the exact same record again", metadata=dict(TURN_2)
    )
    second_llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", dict(args), "call_2"),
            {"content": "Created again.", "done": True},
        ]
    )
    await pattern.run(context=context, tools=[tool], llm=second_llm)

    assert len(tool.calls) == 2


@pytest.mark.asyncio
async def test_pre_upgrade_checkpoint_records_never_match_a_stamped_turn() -> None:
    # Checkpoints written before ToolCallRecord.turn_id existed restore with
    # turn_id=None; against a stamped turn they must fail the equality check
    # and err toward executing.
    args = {"title": "invoice"}
    seed_pattern = _pattern()
    seed_llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", args, "call_1"),
            {"content": "Created.", "done": True},
        ]
    )
    seed_tool = FakeWriteTool()
    await seed_pattern.run(context=_turn_context(), tools=[seed_tool], llm=seed_llm)

    state = json.loads(json.dumps(seed_pattern.get_state()))
    for record in state["tool_ledger"].values():
        record.pop("turn_id", None)  # simulate the pre-upgrade shape

    resumed_pattern = _pattern()
    resumed_pattern.load_state(state)
    resumed_tool = FakeWriteTool()
    resumed_llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", dict(args), "call_2"),
            {"content": "Done.", "done": True},
        ]
    )

    await resumed_pattern.run(
        context=_turn_context(), tools=[resumed_tool], llm=resumed_llm
    )

    assert len(resumed_tool.calls) == 1


# ---------------------------------------------------------------------------
# Ledger-order and concurrency edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_envelope_first_ledger_order_still_attaches_the_genuine_result() -> None:
    # load_state rebuilds the ledger in the checkpoint's stored order, so an
    # envelope record can precede its genuine record. The scan must skip the
    # envelope and attach the genuine execution's result.
    genuine_result = {"success": True, "record_id": "rec-1"}
    envelope_record = {
        "tool_call_id": "call_2",
        "tool_name": "create_record",
        "args": {"title": "invoice", "amount": 0},
        "args_hash": "",
        "status": "completed",
        "result": build_suppression_envelope(
            tool_name="create_record",
            prior_tool_call_id="call_1",
            prior_result=genuine_result,
        ),
        "error": None,
        "turn_id": "turn-1",
    }
    genuine_record = {
        "tool_call_id": "call_1",
        "tool_name": "create_record",
        "args": {"title": "invoice", "amount": 0},
        "args_hash": "",
        "status": "completed",
        "result": genuine_result,
        "error": None,
        "turn_id": "turn-1",
    }

    seed_pattern = _pattern()
    seed_llm = FakeLLM(
        responses=[
            _tool_call_response(
                "create_record", {"title": "invoice", "amount": 0}, "call_1"
            ),
            {"content": "Created.", "done": True},
        ]
    )
    await seed_pattern.run(
        context=_turn_context(), tools=[FakeWriteTool()], llm=seed_llm
    )
    state = json.loads(json.dumps(seed_pattern.get_state()))
    real_hash = state["tool_ledger"]["call_1"]["args_hash"]
    envelope_record["args_hash"] = real_hash
    genuine_record["args_hash"] = real_hash
    state["tool_ledger"] = {"call_2": envelope_record, "call_1": genuine_record}

    pattern = _pattern()
    pattern.load_state(state)
    tool = FakeWriteTool()
    llm = FakeLLM(
        responses=[
            _tool_call_response(
                "create_record", {"title": "invoice", "amount": 0}, "call_3"
            ),
            {"content": "Done.", "done": True},
        ]
    )
    context = _turn_context()

    await pattern.run(context=context, tools=[tool], llm=llm)

    assert tool.calls == []
    envelope = _tool_results(context)[0]
    assert envelope["suppressed_duplicate_of"] == "call_1"
    assert envelope["result"] == genuine_result


@pytest.mark.asyncio
async def test_concurrent_identical_writes_are_a_disclosed_boundary() -> None:
    # Documents the disclosed check-then-record window: when an operator
    # marks a non-idempotent tool concurrency_safe (contradicting that
    # flag's idempotency meaning), two identical calls in one concurrent
    # batch are invisible to each other (neither is "completed" during the
    # other's scan) and both execute. If this test ever fails because only
    # one call ran, the window was closed — update the guard's docstring and
    # the PR-facing disclosure.
    args = {"title": "invoice"}
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "create_record",
                            "arguments": json.dumps(args),
                        },
                    },
                    {
                        "id": "call_2",
                        "function": {
                            "name": "create_record",
                            "arguments": json.dumps(args),
                        },
                    },
                ],
                "done": False,
            },
            {"content": "Done.", "done": True},
        ]
    )
    tool = FakeWriteTool(concurrency_safe=True)
    context = _turn_context()
    pattern = _pattern(tool_parallel_enabled=True, tool_max_concurrency=2)

    await pattern.run(context=context, tools=[tool], llm=llm)

    assert len(tool.calls) == 2


# ---------------------------------------------------------------------------
# Envelope shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_envelope_strips_reserved_transport_keys() -> None:
    # The ledger stores the raw execution return; reserved transport keys
    # must not reach the model nested inside the suppression envelope.
    #
    # Only a *valid* scope is exercised, deliberately: a value the scope
    # parser rejects cannot reach this code at all, because the genuine
    # call's own add_tool_result raises ValueError on the same value
    # (context_ref.py) and nothing on the ReAct loop's backfill path catches
    # it, so the turn ends before any duplicate can be issued.
    args = {"title": "invoice"}

    class ReservedKeysTool(FakeWriteTool):
        async def run_json_async(self, tool_args: dict[str, Any]) -> Any:
            self.calls.append(dict(tool_args))
            return {
                "success": True,
                "record_id": "rec-1",
                "_xagent_context_refs": [],
                "_xagent_supersedes_scope": "records",
            }

    llm = _run_twice_llm("create_record", args, dict(args))
    tool = ReservedKeysTool()
    context = _turn_context()

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert len(tool.calls) == 1
    envelope = _tool_results(context)[1]
    assert envelope[DUPLICATE_WRITE_SUPPRESSED_KEY] is True
    assert "_xagent_context_refs" not in envelope["result"]
    assert "_xagent_supersedes_scope" not in envelope["result"]
    assert envelope["result"]["record_id"] == "rec-1"


def test_suppression_envelope_shape() -> None:
    envelope = build_suppression_envelope(
        tool_name="create_record",
        prior_tool_call_id="call_1",
        prior_result={"success": True, "record_id": "rec-1"},
    )
    assert envelope["success"] is True
    assert envelope[DUPLICATE_WRITE_SUPPRESSED_KEY] is True
    assert envelope["tool_name"] == "create_record"
    assert envelope["suppressed_duplicate_of"] == "call_1"
    assert envelope["result"] == {"success": True, "record_id": "rec-1"}
    assert "already succeeded earlier in this turn" in envelope["message"]


# ---------------------------------------------------------------------------
# Metering: a suppressed duplicate must not be billed
# ---------------------------------------------------------------------------


async def _metered_run(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Run a duplicate-write scenario and return (metered_counts, tool_calls)."""
    metered: list[int] = []
    monkeypatch.setattr(
        "xagent.core.model.chat.token_context.add_tool_call_usage",
        lambda count=1: metered.append(count),
    )
    args = {"title": "invoice", "amount": 7}
    llm = _run_twice_llm("create_record", args, dict(args))
    tool = FakeWriteTool()
    context = _turn_context("Create the invoice record")
    await _pattern().run(context=context, tools=[tool], llm=llm)
    return metered, tool.calls


@pytest.mark.asyncio
async def test_suppressed_duplicate_write_is_not_metered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The suppression path keeps trace visibility but skips billing.

    Regression for xorbitsai/xagent#2231: before the fix, the suppressed
    repeat still consumed one billable action even though the tool never
    ran — only the genuine first call consumes an execution round.
    """
    metered, calls = await _metered_run(monkeypatch)

    assert calls == [{"title": "invoice", "amount": 7}]
    # Exactly one metered invocation: the genuine execution. The suppressed
    # duplicate must not add a second billable action.
    assert metered == [1]


@pytest.mark.asyncio
async def test_metered_default_still_bills_each_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two distinct executions in one turn still bill two actions."""
    metered: list[int] = []
    monkeypatch.setattr(
        "xagent.core.model.chat.token_context.add_tool_call_usage",
        lambda count=1: metered.append(count),
    )
    llm = _run_twice_llm(
        "create_record",
        {"title": "invoice", "amount": 7},
        {"title": "invoice", "amount": 8},
    )
    tool = FakeWriteTool()
    context = _turn_context("Create both records")

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert len(tool.calls) == 2
    assert metered == [1, 1]


@pytest.mark.asyncio
async def test_metered_flag_is_a_call_site_contract_not_payload_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool-controlled payload cannot dodge metering.

    The exemption is an explicit call-site parameter on
    PatternRuntime.on_tool_start; a tool_call dict carrying a
    duplicate_write_suppressed marker (as a hostile model or tool wrapper
    might inject) must still be billed.
    """
    metered: list[int] = []
    monkeypatch.setattr(
        "xagent.core.model.chat.token_context.add_tool_call_usage",
        lambda count=1: metered.append(count),
    )
    pattern = _pattern()
    tool_call = {
        "id": "call_h",
        "name": "create_record",
        "args": {"title": "spoofed"},
        "duplicate_write_suppressed": True,
    }
    # The runtime would normally be PatternRuntime; exercising the hook with
    # a forged payload must not exempt it from metering.
    from xagent.core.agent import PatternRuntime

    runtime = PatternRuntime(execution_id="test")
    await runtime.on_tool_start(tool_call=tool_call)

    assert metered == [1]
