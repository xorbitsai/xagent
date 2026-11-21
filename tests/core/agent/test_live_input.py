"""Live user input: mailbox, safe-point admission and hand-back at the end."""

from __future__ import annotations

import asyncio
import copy
import gc
import itertools
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from tests.core.agent.test_agent_service_post_user_message import (
    FakeWorkspaceManager,
    _build_service,
)
from xagent.config import get_agent_pattern_for_execution_mode
from xagent.core.agent import (
    Agent,
    AgentExecutionAdapter,
    AgentExecutionConfig,
    ExecutionContext,
    PatternRuntime,
    ReActPattern,
)
from xagent.core.agent import runner as runner_module
from xagent.core.agent.context.execution import context_checkpoint_gate
from xagent.core.agent.live_input import (
    LiveInputClaim,
    LiveInputMailbox,
    LiveInputOffer,
    LiveInputResult,
    LiveUserInput,
)
from xagent.core.agent.pattern.react import react as react_module
from xagent.core.agent.runner import (
    AgentRunner,
    ExecutionControl,
    UserMessageInjectionConflictError,
    UserMessageInjectionOutcome,
    UserMessageInjectionRejectedError,
    UserMessageInjectionResult,
    _record_injection_outcome,
)
from xagent.core.agent.runtime import LLMCallPreempted
from xagent.core.guardrails.input import InputReviewRejected
from xagent.core.model.chat.types import ChunkType, StreamChunk
from xagent.web.services.db_runtime import run_db_io_cancellation_safe

EID = "live-exec"
LIVE_TEXT = "INJECTED"
RUN_TIMEOUT_SECONDS = 8


class _NoArgs(BaseModel):
    pass


class RecordingAdmission:
    """Admission that records every call it receives."""

    def __init__(
        self,
        claim: LiveInputClaim | BaseException = LiveInputClaim.CLAIMED,
        *,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.claim_result = claim
        self.events: list[Any] = []
        self.gate = gate

    async def claim(self) -> LiveInputClaim:
        self.events.append("claim")
        if self.gate is not None:
            await self.gate.wait()
        if isinstance(self.claim_result, BaseException):
            raise self.claim_result
        return self.claim_result

    async def finish(self, result: LiveInputResult) -> None:
        self.events.append(result)


class ProbeTool:
    def __init__(self, name: str, *, safe: bool = False) -> None:
        self.metadata = SimpleNamespace(
            name=name, description="probe", concurrency_safe=safe
        )
        self.calls: list[dict[str, Any]] = []
        self.on_start: Any = None

    def args_type(self) -> type[BaseModel]:
        return _NoArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        if self.on_start is not None:
            await self.on_start(self.metadata.name)
        await asyncio.sleep(0.005)
        return {"ok": True}


def tool_calls(*names: str) -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {"id": f"call_{name}", "function": {"name": name, "arguments": "{}"}}
            for name in names
        ],
    }


def final_answer(outcome: str | None = None) -> dict[str, Any]:
    args = {"answer": "done", "response_language": "English"}
    if outcome is not None:
        args["outcome"] = outcome
    arguments = json.dumps(args)
    return {
        "content": "",
        "tool_calls": [
            {
                "id": "call_final",
                "function": {"name": "final_answer", "arguments": arguments},
            }
        ],
    }


class ScriptedLLM:
    """Chat model that replays a script; a callable entry runs before replying."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[list[dict[str, Any]]] = []
        # Requests that were cancelled while they were being answered.
        self.cancelled = 0

    async def chat(self, **kwargs: Any) -> Any:
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        entry = self.script.pop(0)
        if callable(entry):
            try:
                entry = await entry()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        return entry


class StreamingScriptedLLM(ScriptedLLM):
    """Native-streaming model: an entry is a list of chunks and async hooks."""

    async def stream_chat(self, **kwargs: Any) -> Any:
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        for part in self.script.pop(0):
            if not callable(part):
                yield part
                continue
            try:
                await part()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise


@dataclass
class Rig:
    runner: AgentRunner
    llm: ScriptedLLM
    tools: dict[str, ProbeTool]
    runtimes: list[PatternRuntime] = field(default_factory=list)

    def offer(
        self,
        admission: RecordingAdmission,
        *,
        turn_id: str = "t1",
        text: str = LIVE_TEXT,
    ) -> LiveInputOffer:
        control = self.runner._active_controls[EID]
        if control.runtime not in self.runtimes:
            self.runtimes.append(control.runtime)
        return self.runner.offer_live_user_message(
            EID,
            execution_message=text,
            display_message=text,
            turn_id=turn_id,
            admission=admission,
        )

    def start(self, **kwargs: Any) -> asyncio.Task[dict[str, Any]]:
        # A run that cannot finish fails the test instead of hanging it.
        return asyncio.create_task(
            asyncio.wait_for(
                self.runner.run("hello", execution_id=EID, **kwargs),
                timeout=RUN_TIMEOUT_SECONDS,
            )
        )


def make_rig(
    script: list[Any],
    tool_names: tuple[str, ...] = ("w1", "w2"),
    *,
    safe: tuple[str, ...] = (),
    pattern: Any = None,
    llm_class: type[ScriptedLLM] = ScriptedLLM,
) -> Rig:
    tools = {name: ProbeTool(name, safe=name in safe) for name in tool_names}
    llm = llm_class(script)
    pattern = pattern or ReActPattern(
        tool_parallel_enabled=True, tool_max_concurrency=2
    )
    agent = Agent(name="a", patterns=[pattern], llm=llm, tools=list(tools.values()))  # type: ignore[arg-type]
    return Rig(AgentRunner(agent=agent), llm, tools)


# Segment kinds: the tool calls of one model turn and how they are scheduled.
SEGMENTS = {
    "serial": ("w1", "w2"),
    "parallel": ("p1", "p2"),
    "control-serial": ("send_message", "w1"),
    "parallel-serial": ("p1", "p2", "w1"),
}
# After which checkpoint, counted per label, the "between segments" moment fires.
BETWEEN = {
    "serial": ("before_tool", 1),
    "parallel": ("after_tool_batch", 0),
    "control-serial": ("before_tool", 0),
    "parallel-serial": ("before_tool", 0),
}
MOMENTS = ["before-first", "first-running", "between", "last-running"]


async def run_scenario(
    monkeypatch: pytest.MonkeyPatch, segment: str, moment: str
) -> tuple[Rig, RecordingAdmission, dict[str, Any]]:
    calls = SEGMENTS[segment]
    rig_tools = tuple(n for n in calls if n != "send_message")
    adm = RecordingAdmission()
    fired: list[bool] = []

    def fire() -> None:
        if not fired:
            fired.append(True)
            assert rig.offer(adm) is LiveInputOffer.QUEUED

    async def first_reply() -> dict[str, Any]:
        return tool_calls(*calls)

    rig = make_rig(
        [first_reply, final_answer()],
        rig_tools,
        safe=tuple(n for n in rig_tools if n.startswith("p")),
    )
    first_tool, last_tool = rig_tools[0], rig_tools[-1]

    async def on_start(name: str) -> None:
        if (moment, name) in {
            ("first-running", first_tool),
            ("last-running", last_tool),
        }:
            fire()

    for tool in rig.tools.values():
        tool.on_start = on_start
    label, index = BETWEEN[segment]
    seen: dict[str, int] = {}
    original = PatternRuntime.checkpoint

    async def checkpoint(self: PatternRuntime, name: str, *a: Any, **kw: Any) -> Any:
        count = seen.get(name, 0)
        seen[name] = count + 1
        if moment == "between" and (name, count) == (label, index):
            fire()
        return await original(self, name, *a, **kw)

    monkeypatch.setattr(PatternRuntime, "checkpoint", checkpoint)
    # "before-first": the first model response is complete and no tool has
    # started. An offer during the request itself would cancel the request.
    original_llm_end = PatternRuntime.on_llm_end

    async def on_llm_end(self: PatternRuntime, *a: Any, **kw: Any) -> Any:
        if moment == "before-first":
            fire()
        return await original_llm_end(self, *a, **kw)

    monkeypatch.setattr(PatternRuntime, "on_llm_end", on_llm_end)
    result = await rig.start()
    assert fired, "the offer moment was never reached"
    return rig, adm, result


def pairs_intact(messages: list[dict[str, Any]]) -> bool:
    for index, message in enumerate(messages):
        ids = {call["id"] for call in message.get("tool_calls") or []}
        if not ids:
            continue
        following = []
        for later in messages[index + 1 :]:
            if later["role"] != "tool":
                break
            following.append(later["tool_call_id"])
        if set(following) != ids or len(following) != len(ids):
            return False
    return True


# --------------------------------------------------------------------------
# Mailbox
# --------------------------------------------------------------------------


def make_item(turn_id: str = "t1", text: str = "m") -> LiveUserInput:
    return LiveUserInput(turn_id, text, text, RecordingAdmission())


def test_mailbox_orders_dedupes_and_closes() -> None:
    mailbox = LiveInputMailbox()
    assert mailbox.offer(make_item("a")) is LiveInputOffer.QUEUED
    assert mailbox.offer(make_item("a")) is LiveInputOffer.ALREADY_QUEUED
    assert mailbox.offer(make_item("a", "other")) is LiveInputOffer.CONFLICT
    assert mailbox.offer(make_item("b")) is LiveInputOffer.QUEUED
    assert mailbox.pending()
    first = mailbox.take_next()
    assert first is not None and first.turn_id == "a"
    rest = mailbox.close()
    assert [item.turn_id for item in rest] == ["b"]
    assert mailbox.close() == []
    assert not mailbox.pending()
    assert mailbox.take_next() is None
    assert mailbox.offer(make_item("c")) is LiveInputOffer.NO_LIVE_RUN


def test_mailbox_answers_for_a_taken_message_until_settled() -> None:
    mailbox = LiveInputMailbox()
    assert mailbox.offer(make_item("a")) is LiveInputOffer.QUEUED
    taken = mailbox.take_next()
    assert taken is not None
    assert not mailbox.pending()
    assert mailbox.offer(make_item("a")) is LiveInputOffer.ALREADY_QUEUED
    assert mailbox.offer(make_item("a", "other")) is LiveInputOffer.CONFLICT
    # Only the object that was taken can settle it.
    mailbox.settle(make_item("a"))
    assert mailbox.offer(make_item("a", "other")) is LiveInputOffer.CONFLICT
    mailbox.settle(taken)
    assert mailbox.offer(make_item("a")) is LiveInputOffer.QUEUED


# --------------------------------------------------------------------------
# Offer
# --------------------------------------------------------------------------


def manual_runner(*, mailbox: bool = True) -> tuple[AgentRunner, PatternRuntime]:
    runner = AgentRunner(agent=Agent(name="a", patterns=[], llm=None))
    runtime = PatternRuntime(execution_id=EID)
    if mailbox:
        runtime.live_input = LiveInputMailbox()
    runner._active_controls[EID] = ExecutionControl(
        runtime=runtime, task=None, event_loop=asyncio.get_running_loop()
    )
    runner.context_manager.create_context(EID)
    return runner, runtime


async def test_offer_answers_before_queueing() -> None:
    runner, runtime = manual_runner()
    context = runner.context_manager.get_context(EID)
    assert context is not None
    context.add_user_message("seen", metadata={"turn_id": "old"})

    def offer(turn_id: str, text: str) -> LiveInputOffer:
        return runner.offer_live_user_message(
            EID,
            execution_message=text,
            display_message=text,
            turn_id=turn_id,
            admission=RecordingAdmission(),
        )

    with pytest.raises(ValueError):
        offer("   ", "x")
    assert offer("old", "seen") is LiveInputOffer.ALREADY_ACCEPTED
    assert offer("old", "different") is LiveInputOffer.CONFLICT
    assert offer("new", "x") is LiveInputOffer.QUEUED
    assert offer("new", "x") is LiveInputOffer.ALREADY_QUEUED
    assert offer("new", "y") is LiveInputOffer.CONFLICT
    runtime.live_input = None
    assert offer("n2", "x") is LiveInputOffer.UNSUPPORTED
    del runner._active_controls[EID]
    assert offer("n3", "x") is LiveInputOffer.NO_LIVE_RUN


async def test_offer_refuses_finishing_run_and_compacted_turn() -> None:
    runner, _ = manual_runner()
    context = runner.context_manager.get_context(EID)
    assert context is not None

    def offer(turn_id: str, text: str) -> LiveInputOffer:
        return runner.offer_live_user_message(
            EID,
            execution_message=text,
            display_message=text,
            turn_id=turn_id,
            admission=RecordingAdmission(),
        )

    # The message itself is gone (compacted); only the accepted-turn record stays.
    gone = context.add_user_message("gone", metadata={"turn_id": "old"})
    runner._record_accepted_turn(context, gone)
    context.messages.remove(gone)
    assert offer("old", "gone") is LiveInputOffer.ALREADY_ACCEPTED
    assert offer("old", "other") is LiveInputOffer.CONFLICT
    context_checkpoint_gate(context).run_finishing = True
    assert offer("new", "x") is LiveInputOffer.NO_LIVE_RUN


async def test_offer_rejects_foreign_event_loop() -> None:
    runner, _ = manual_runner()
    other = asyncio.new_event_loop()
    try:
        runner._active_controls[EID].event_loop = other
        with pytest.raises(RuntimeError, match="event loop"):
            runner.offer_live_user_message(
                EID,
                execution_message="x",
                display_message="x",
                turn_id="t",
                admission=RecordingAdmission(),
            )
    finally:
        other.close()


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("balanced", LiveInputOffer.QUEUED),
        ("graph", LiveInputOffer.QUEUED),
        ("auto", LiveInputOffer.UNSUPPORTED),
        ("think", LiveInputOffer.UNSUPPORTED),
        ("flash", LiveInputOffer.UNSUPPORTED),
    ],
)
async def test_offer_by_pattern(mode: str, expected: LiveInputOffer) -> None:
    config = AgentExecutionConfig(
        name="a", llm=None, pattern=get_agent_pattern_for_execution_mode(mode)
    )
    pattern, _ = AgentExecutionAdapter(config)._build_pattern()
    adm = RecordingAdmission()
    if mode in {"balanced", "graph", "flash"}:
        rig = make_rig([], pattern=pattern)

        async def first_reply() -> dict[str, Any]:
            assert rig.offer(adm) is expected
            return final_answer() if mode == "flash" else tool_calls("w1")

        rig.llm.script = [first_reply, final_answer()]
    else:
        release = asyncio.Event()

        async def blocked(**_: Any) -> dict[str, Any]:
            await release.wait()
            return {"success": True, "output": "done"}

        pattern.run = blocked
        rig = make_rig([], pattern=pattern)
    task = rig.start()
    if mode in {"auto", "think"}:
        while EID not in rig.runner._active_controls:
            await asyncio.sleep(0)
        assert rig.offer(adm) is expected
        release.set()
    await task
    if expected is LiveInputOffer.QUEUED:
        assert adm.events == ["claim", LiveInputResult.APPLIED]
        assert LIVE_TEXT in str(rig.llm.requests[-1])
    else:
        assert adm.events == []


async def test_offer_during_react_preamble(monkeypatch: pytest.MonkeyPatch) -> None:
    reached, release = asyncio.Event(), asyncio.Event()

    async def enrich(**_: Any) -> None:
        reached.set()
        await release.wait()

    monkeypatch.setattr(react_module, "enrich_context_with_memory", enrich)
    rig = make_rig([tool_calls("w1"), final_answer()])
    adm = RecordingAdmission()
    task = rig.start()
    await reached.wait()
    assert rig.offer(adm) is LiveInputOffer.QUEUED
    release.set()
    await task
    assert adm.events == ["claim", LiveInputResult.APPLIED]
    assert LIVE_TEXT in str(rig.llm.requests[1])


async def test_offer_while_a_taken_message_is_admitted() -> None:
    gate = asyncio.Event()
    first = RecordingAdmission(gate=gate)
    same = RecordingAdmission()
    other = RecordingAdmission()
    answers: list[LiveInputOffer] = []
    background: list[asyncio.Future[None]] = []
    rig = make_rig([tool_calls("w1"), final_answer()], ("w1",))

    async def repeat_offers() -> None:
        for _ in range(500):
            if "claim" in first.events:
                break
            await asyncio.sleep(0.005)
        assert "claim" in first.events
        answers.append(rig.offer(same, turn_id="dup"))
        answers.append(rig.offer(other, turn_id="dup", text=LIVE_TEXT + "-other"))
        gate.set()

    async def on_start(_: str) -> None:
        assert rig.offer(first, turn_id="dup") is LiveInputOffer.QUEUED
        background.append(asyncio.ensure_future(repeat_offers()))

    rig.tools["w1"].on_start = on_start
    result = await rig.start()
    await asyncio.gather(*background)
    assert result["success"] is True
    assert answers == [LiveInputOffer.ALREADY_QUEUED, LiveInputOffer.CONFLICT]
    assert first.events == ["claim", LiveInputResult.APPLIED]
    assert same.events == []
    assert other.events == []


@pytest.mark.parametrize("compacted", [False, True])
@pytest.mark.parametrize("text", ["seen", "other"])
async def test_offer_and_injection_classify_a_known_turn_alike(
    compacted: bool, text: str
) -> None:
    runner, _ = manual_runner()
    context = runner.context_manager.get_context(EID)
    assert context is not None
    seen = context.add_user_message("seen", metadata={"turn_id": "old"})
    runner._record_accepted_turn(context, seen)
    if compacted:
        context.messages.remove(seen)
    offered = runner.offer_live_user_message(
        EID,
        execution_message=text,
        display_message=text,
        turn_id="old",
        admission=RecordingAdmission(),
    )
    if text == "seen":
        assert offered is LiveInputOffer.ALREADY_ACCEPTED
        result = await runner.inject_user_message(
            EID,
            execution_message=text,
            display_message=text,
            turn_id="old",
            request_interrupt=False,
        )
        assert result.outcome is UserMessageInjectionOutcome.POSTED_REPLAY
    else:
        assert offered is LiveInputOffer.CONFLICT
        with pytest.raises(UserMessageInjectionConflictError):
            await runner.inject_user_message(
                EID,
                execution_message=text,
                display_message=text,
                turn_id="old",
                request_interrupt=False,
            )


# --------------------------------------------------------------------------
# Safe point
# --------------------------------------------------------------------------


@pytest.mark.parametrize("segment", list(SEGMENTS))
@pytest.mark.parametrize("moment", MOMENTS)
async def test_live_input_waits_for_the_whole_batch(
    monkeypatch: pytest.MonkeyPatch, segment: str, moment: str
) -> None:
    rig, adm, result = await run_scenario(monkeypatch, segment, moment)
    assert result["success"] is True
    for tool in rig.tools.values():
        assert len(tool.calls) == 1
    assert all(not runtime._interrupt_requested for runtime in rig.runtimes)
    assert adm.events == ["claim", LiveInputResult.APPLIED]
    messages = rig.llm.requests[-1]
    assert pairs_intact(messages)
    live_index = next(
        i for i, m in enumerate(messages) if LIVE_TEXT in str(m.get("content"))
    )
    assert live_index > max(i for i, m in enumerate(messages) if m["role"] == "tool")
    context = result["context"]
    assert len(context._sanitize_tool_message_pairs(list(context.messages))) == len(
        context.messages
    )


@pytest.mark.parametrize("call_site", ["settled", "pending-tool-call", "tool-running"])
async def test_admit_requires_settled_tools(call_site: str) -> None:
    runtime = PatternRuntime(execution_id=EID)
    runtime.live_input = LiveInputMailbox()
    taken: list[str] = []

    async def injector(item: LiveUserInput) -> LiveInputResult:
        taken.append(item.turn_id)
        return LiveInputResult.APPLIED

    runtime._live_input_injector = injector
    runtime.live_input.offer(make_item())
    if call_site == "settled":
        assert await runtime.admit_live_user_messages(None, tools_settled=True) == 1
        assert taken == ["t1"]
        return
    if call_site == "tool-running":
        runtime._active_tool_tasks.add(asyncio.get_running_loop().create_future())
    with pytest.raises(RuntimeError):
        await runtime.admit_live_user_messages(
            None, tools_settled=call_site != "pending-tool-call"
        )
    assert runtime.live_input.pending() and taken == []


@pytest.mark.parametrize("cause", ["fence", "limit", "flag"])
async def test_forced_answer_turn_keeps_mailbox(
    monkeypatch: pytest.MonkeyPatch, cause: str
) -> None:
    pattern = ReActPattern(max_iterations=1 if cause == "limit" else 20)
    closing = final_answer()
    if cause == "limit":
        closing["tool_calls"][0]["function"]["arguments"] = json.dumps(
            {"answer": "done", "outcome": "partial", "response_language": "English"}
        )
    rig = make_rig([tool_calls("w1"), closing], ("w1",), pattern=pattern)
    adm = RecordingAdmission()

    async def on_start(_: str) -> None:
        rig.offer(adm)
        if cause == "flag":
            pattern.force_final_answer_next = True

    rig.tools["w1"].on_start = on_start
    if cause == "fence":
        monkeypatch.setattr(
            ReActPattern,
            "_settlement_fence_active",
            lambda self, runtime: bool(rig.tools["w1"].calls),
        )
    result = await rig.start()
    assert result["success"] is True
    assert adm.events == [LiveInputResult.RETURNED]
    assert LIVE_TEXT not in str(rig.llm.requests[-1])
    # The forced answer turn is sent as it is: a waiting message neither
    # cancels it nor makes the loop skip it.
    assert len(rig.llm.requests) == 2
    assert rig.llm.cancelled == 0


async def test_taken_message_is_settled_when_its_admission_raises() -> None:
    runtime = PatternRuntime(execution_id=EID)
    runtime.live_input = LiveInputMailbox()

    async def injector(item: LiveUserInput) -> LiveInputResult:
        raise asyncio.CancelledError()

    runtime._live_input_injector = injector
    assert runtime.live_input.offer(make_item("t1")) is LiveInputOffer.QUEUED
    with pytest.raises(asyncio.CancelledError):
        await runtime.admit_live_user_messages(None, tools_settled=True)
    assert runtime.live_input.offer(make_item("t1")) is LiveInputOffer.QUEUED


# --------------------------------------------------------------------------
# Exactly-once hand-over
# --------------------------------------------------------------------------


async def test_leftover_live_inputs_returned_before_runner_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline: list[Any] = []

    class Timed(RecordingAdmission):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        async def finish(self, result: LiveInputResult) -> None:
            # A real delivery awaits I/O; a hand-back deferred to a background
            # task would land after the run returned.
            await asyncio.sleep(0.01)
            timeline.append((self.name, result))

    rig = make_rig([final_answer()])
    original_llm_end = PatternRuntime.on_llm_end

    async def on_llm_end(self: PatternRuntime, *a: Any, **kw: Any) -> Any:
        # The final answer is complete; the run is about to finish.
        assert rig.offer(Timed("first"), turn_id="a") is LiveInputOffer.QUEUED
        assert rig.offer(Timed("second"), turn_id="b") is LiveInputOffer.QUEUED
        return await original_llm_end(self, *a, **kw)

    monkeypatch.setattr(PatternRuntime, "on_llm_end", on_llm_end)
    await rig.start()
    timeline.append("run-returned")
    assert timeline == [
        ("first", LiveInputResult.RETURNED),
        ("second", LiveInputResult.RETURNED),
        "run-returned",
    ]


async def test_live_input_finished_exactly_once_when_cancelled() -> None:
    hold = asyncio.Event()
    adm = RecordingAdmission()

    offered = asyncio.Event()

    async def on_start(_: str) -> None:
        # A running tool keeps the message waiting in the mailbox.
        assert rig.offer(adm) is LiveInputOffer.QUEUED
        offered.set()
        await hold.wait()

    rig = make_rig([tool_calls("w1"), final_answer()], ("w1",))
    rig.tools["w1"].on_start = on_start
    task = rig.start()
    await offered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.gather(*rig.runner._live_input_tasks)
    assert adm.events == [LiveInputResult.RETURNED]


async def test_offer_after_the_run_ended_is_refused_untouched() -> None:
    rig = make_rig([final_answer()])
    await rig.start()
    late = RecordingAdmission()
    assert (
        rig.runner.offer_live_user_message(
            EID,
            execution_message="x",
            display_message="x",
            turn_id="late",
            admission=late,
        )
        is LiveInputOffer.NO_LIVE_RUN
    )
    assert late.events == []


@pytest.mark.parametrize(
    "claim,result",
    [
        (LiveInputClaim.TAKEN, LiveInputResult.TAKEN),
        (LiveInputClaim.UNAVAILABLE, LiveInputResult.RETURNED),
        (LiveInputClaim.UNKNOWN, LiveInputResult.UNKNOWN),
        (RuntimeError("claim exploded"), LiveInputResult.UNKNOWN),
    ],
)
async def test_unclaimed_input_is_never_injected(
    claim: LiveInputClaim | BaseException, result: LiveInputResult
) -> None:
    adm = RecordingAdmission(claim)

    async def reply() -> dict[str, Any]:
        assert rig.offer(adm) is LiveInputOffer.QUEUED
        return tool_calls("w1")

    rig = make_rig([reply, final_answer()], ("w1",))
    await rig.start()
    assert adm.events == ["claim", result]
    assert LIVE_TEXT not in str(rig.llm.requests[-1])


@pytest.mark.parametrize(
    "recorded,posted,error,expected",
    [
        (UserMessageInjectionOutcome.POSTED_FRESH, None, None, LiveInputResult.APPLIED),
        (UserMessageInjectionOutcome.NOT_POSTED, None, None, LiveInputResult.RETURNED),
        (
            UserMessageInjectionOutcome.NOT_POSTED,
            None,
            UserMessageInjectionRejectedError("x"),
            LiveInputResult.RETURNED,
        ),
        (
            UserMessageInjectionOutcome.OUTCOME_UNKNOWN,
            None,
            None,
            LiveInputResult.UNKNOWN,
        ),
        (
            UserMessageInjectionOutcome.NOT_POSTED,
            None,
            InputReviewRejected("no"),
            LiveInputResult.REFUSED,
        ),
        (
            UserMessageInjectionOutcome.NOT_POSTED,
            None,
            UserMessageInjectionConflictError("c"),
            LiveInputResult.CONFLICT,
        ),
        (
            UserMessageInjectionOutcome.NOT_POSTED,
            None,
            RuntimeError("other"),
            LiveInputResult.RETURNED,
        ),
    ],
)
async def test_injection_dispositions_map_to_results(
    monkeypatch: pytest.MonkeyPatch,
    recorded: UserMessageInjectionOutcome,
    posted: None,
    error: BaseException | None,
    expected: LiveInputResult,
) -> None:
    runner = AgentRunner(agent=Agent(name="a", patterns=[], llm=None))

    async def inject(*args: Any, **kwargs: Any) -> UserMessageInjectionResult:
        assert kwargs["request_interrupt"] is False
        _record_injection_outcome(recorded)
        if error is not None:
            raise error
        return UserMessageInjectionResult(context=None, outcome=recorded)

    monkeypatch.setattr(runner, "inject_user_message", inject)
    adm = RecordingAdmission()
    item = LiveUserInput("t1", "m", "m", adm)
    assert await runner._inject_live_input(EID, item) is expected
    assert adm.events == ["claim", expected]


async def test_cancel_while_claiming_finishes_taken_and_queued_items_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "LIVE_INPUT_CANCEL_GRACE_SECONDS", 0.05)
    gate = asyncio.Event()
    first = RecordingAdmission(gate=gate)
    second = RecordingAdmission()

    async def reply() -> dict[str, Any]:
        rig.offer(first, turn_id="a")
        rig.offer(second, turn_id="b")
        return tool_calls("w1")

    rig = make_rig([reply, final_answer()], ("w1",))
    task = rig.start()
    while "claim" not in first.events:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.gather(*rig.runner._live_input_tasks, return_exceptions=True)
    assert first.events == ["claim", LiveInputResult.UNKNOWN]
    assert second.events == [LiveInputResult.RETURNED]


@pytest.mark.parametrize("trigger", ["flag", "checker"])
async def test_stop_before_claim_returns_remaining(trigger: str) -> None:
    adms = [RecordingAdmission() for _ in range(3)]

    async def reply() -> dict[str, Any]:
        for index, adm in enumerate(adms):
            rig.offer(adm, turn_id=f"t{index}")
        return tool_calls("w1")

    rig = make_rig([reply, final_answer()], ("w1",))
    original = adms[0].claim

    async def claim_and_stop() -> LiveInputClaim:
        if trigger == "flag":
            rig.runtimes[0]._interrupt_requested = True
        return await original()

    adms[0].claim = claim_and_stop  # type: ignore[method-assign]
    # The checker is what a quota gate looks like to the runtime: only
    # should_interrupt() consults it, a bare read of the stop flag does not.
    checker = (lambda: "claim" in adms[0].events) if trigger == "checker" else None
    result = await rig.start(interrupt_checker=checker)
    assert result["success"] is False
    assert adms[0].events == ["claim", LiveInputResult.APPLIED]
    assert adms[1].events == [LiveInputResult.RETURNED]
    assert adms[2].events == [LiveInputResult.RETURNED]


async def test_guidance_interrupt_returns_queued_live_inputs() -> None:
    adm = RecordingAdmission()
    release = asyncio.Event()

    rig = make_rig([tool_calls("w1"), final_answer()], ("w1",))

    async def on_start(_: str) -> None:
        assert rig.offer(adm) is LiveInputOffer.QUEUED
        await release.wait()

    rig.tools["w1"].on_start = on_start
    task = rig.start()
    while not rig.tools["w1"].calls:
        await asyncio.sleep(0)
    await rig.runner.inject_user_message(EID, "guidance", request_interrupt=True)
    result = await task
    assert result["status"] == "interrupted"
    assert adm.events == [LiveInputResult.RETURNED]


class AcceptingPattern:
    """Waits until released; declares that it takes live input."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def accepts_live_input(self) -> bool:
        return True

    async def run(self, **_: Any) -> dict[str, Any]:
        self.started.set()
        await self.release.wait()
        return {"success": True, "output": "done"}


@pytest.mark.parametrize("layer", ["service", "adapter", "registry"])
async def test_offer_is_forwarded_through_every_layer(
    tmp_path: Path, layer: str
) -> None:
    service, registry, tracer = _build_service(EID)
    pattern = AcceptingPattern()
    runner = AgentRunner(
        agent=Agent(name="forward", patterns=[pattern]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    handle = registry.start(runner, execution_id=EID, task="Wait")
    await asyncio.wait_for(pattern.started.wait(), timeout=5)
    target = {
        "service": service,
        "adapter": service._execution_adapter,
        "registry": registry,
    }[layer]
    adm = RecordingAdmission()
    offer = target.offer_live_user_message(
        EID,
        execution_message="x",
        display_message="x",
        turn_id="t1",
        admission=adm,
    )
    assert offer is LiveInputOffer.QUEUED
    pattern.release.set()
    assert handle.task is not None
    await handle.task
    assert adm.events == [LiveInputResult.RETURNED]


async def test_live_input_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    from xagent.core.agent import runner as runner_module

    counted: list[tuple[str, dict[str, str] | None]] = []

    def record(name: str, *, attributes: Any = None, **_: Any) -> None:
        counted.append((name, attributes))

    monkeypatch.setattr(runner_module, "increment_counter", record)

    class FailingFinish(RecordingAdmission):
        async def finish(self, result: LiveInputResult) -> None:
            await super().finish(result)
            raise RuntimeError("finish failed")

    adm = FailingFinish()

    async def reply() -> dict[str, Any]:
        assert rig.offer(adm) is LiveInputOffer.QUEUED
        return tool_calls("w1")

    rig = make_rig([reply, final_answer()], ("w1",))
    result = await rig.start()
    # A finish that raises is counted and does not end the run.
    assert result["success"] is True
    assert ("xagent.live_input.offers", {"outcome": "queued"}) in counted
    assert ("xagent.live_input.finishes", {"outcome": "applied"}) in counted
    assert ("xagent.live_input.finish_errors", None) in counted


class StagedPattern:
    """Waits to be released, then fails or answers; may take live input."""

    def __init__(self, *, accepts: bool, fails: bool) -> None:
        self.accepts = accepts
        self.fails = fails
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.on_start: Any = None

    def accepts_live_input(self) -> bool:
        return self.accepts

    async def run(self, **_: Any) -> dict[str, Any]:
        if self.on_start is not None:
            self.on_start()
        self.started.set()
        await self.release.wait()
        if self.fails:
            raise RuntimeError("pattern failed")
        return {"success": True, "output": "done"}


def offer_to_runner(
    runner: AgentRunner, admission: RecordingAdmission, turn_id: str
) -> LiveInputOffer:
    return runner.offer_live_user_message(
        EID,
        execution_message=LIVE_TEXT,
        display_message=LIVE_TEXT,
        turn_id=turn_id,
        admission=admission,
    )


async def test_live_input_stops_at_the_first_pattern_that_does_not_take_it() -> None:
    first = StagedPattern(accepts=True, fails=True)
    second = StagedPattern(accepts=False, fails=True)
    third = StagedPattern(accepts=True, fails=False)
    runner = AgentRunner(
        agent=Agent(name="a", patterns=[first, second, third], llm=None)  # type: ignore[list-item]
    )
    queued = RecordingAdmission()
    late = RecordingAdmission()
    seen_by_second: list[Any] = []
    second.on_start = lambda: seen_by_second.extend(queued.events)
    task = asyncio.create_task(runner.run("hello", execution_id=EID))
    await asyncio.wait_for(first.started.wait(), timeout=5)
    assert offer_to_runner(runner, queued, "t1") is LiveInputOffer.QUEUED
    first.release.set()
    await asyncio.wait_for(second.started.wait(), timeout=5)
    answer_in_second = offer_to_runner(runner, late, "t2")
    second.release.set()
    await asyncio.wait_for(third.started.wait(), timeout=5)
    answer_in_third = offer_to_runner(runner, late, "t3")
    third.release.set()
    result = await task
    assert result["success"] is True
    assert answer_in_second is LiveInputOffer.UNSUPPORTED
    assert answer_in_third is LiveInputOffer.UNSUPPORTED
    assert seen_by_second == [LiveInputResult.RETURNED]
    assert queued.events == [LiveInputResult.RETURNED]
    assert late.events == []


async def test_run_whose_first_pattern_does_not_take_live_input_never_queues() -> None:
    first = StagedPattern(accepts=False, fails=True)
    second = StagedPattern(accepts=True, fails=False)
    runner = AgentRunner(
        agent=Agent(name="a", patterns=[first, second], llm=None)  # type: ignore[list-item]
    )
    adm = RecordingAdmission()
    setup_entered = asyncio.Event()
    setup_release = asyncio.Event()

    async def gated_setup(tools: list[Any], *, task_id: str) -> None:
        setup_entered.set()
        await setup_release.wait()

    runner._setup_tools = gated_setup  # type: ignore[method-assign]
    task = asyncio.create_task(runner.run("hello", execution_id=EID))
    await asyncio.wait_for(setup_entered.wait(), timeout=5)
    assert offer_to_runner(runner, adm, "t1") is LiveInputOffer.UNSUPPORTED
    setup_release.set()
    await asyncio.wait_for(first.started.wait(), timeout=5)
    first.release.set()
    await asyncio.wait_for(second.started.wait(), timeout=5)
    assert offer_to_runner(runner, adm, "t2") is LiveInputOffer.UNSUPPORTED
    second.release.set()
    result = await task
    assert result["success"] is True
    assert adm.events == []


# --------------------------------------------------------------------------
# Cancellation of the run while a taken message is being admitted
#
# Every way a run can be asked to stop (an external stop, a chat cancel, a
# graceful shutdown, a lost lease) reaches this layer as ``cancel()`` on the
# task that runs the execution, so one scripted cancel covers all of them; the
# first test below is the one that pins the shared behavior.
# --------------------------------------------------------------------------


def record_counters(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    counted: list[tuple[str, Any]] = []

    def record(name: str, *, attributes: Any = None, **_: Any) -> None:
        counted.append((name, attributes))

    monkeypatch.setattr(runner_module, "increment_counter", record)
    return counted


def deferred_counts(counted: list[tuple[str, Any]]) -> list[str]:
    return [
        attributes["outcome"]
        for name, attributes in counted
        if name == "xagent.live_input.cancel_deferred"
    ]


def live_text_count(context: Any) -> int:
    return sum(str(message.content).count(LIVE_TEXT) for message in context.messages)


@pytest.mark.parametrize("moment", ["claiming", "writing"])
async def test_cancel_during_live_input_waits_for_its_outcome(
    monkeypatch: pytest.MonkeyPatch, moment: str
) -> None:
    gate = asyncio.Event()
    entered = asyncio.Event()
    release = asyncio.Event()
    seen: dict[str, Any] = {}
    counted = record_counters(monkeypatch)
    adm = RecordingAdmission(gate=gate if moment == "claiming" else None)

    async def reply() -> dict[str, Any]:
        assert rig.offer(adm) is LiveInputOffer.QUEUED
        return tool_calls("w1")

    rig = make_rig([reply, final_answer()], ("w1",))
    original = rig.runner._persist_injected_context

    async def persist(**kwargs: Any) -> None:
        if not entered.is_set():
            seen["candidate"] = kwargs["context"]
            seen["live"] = rig.runner.context_manager.get_context(EID)
            entered.set()
            await release.wait()
        await original(**kwargs)

    monkeypatch.setattr(rig.runner, "_persist_injected_context", persist)
    task = rig.start()
    if moment == "claiming":
        while "claim" not in adm.events:
            await asyncio.sleep(0)
    else:
        await entered.wait()
    task.cancel()
    await asyncio.sleep(0.01)
    # The message is still being admitted, so the stop has not taken effect.
    held_back = not task.done()
    gate.set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    # No background task has been awaited: the outcome is already recorded.
    assert task.cancelled()
    assert adm.events == ["claim", LiveInputResult.APPLIED]
    assert held_back
    assert live_text_count(seen["live"]) == 1
    if moment == "writing":
        assert live_text_count(seen["candidate"]) == 1
    assert len(rig.llm.requests) == 1
    assert deferred_counts(counted) == ["completed"]


@pytest.mark.parametrize(
    "stall,final",
    [
        ("coroutine", LiveInputResult.UNKNOWN),
        ("thread", LiveInputResult.UNKNOWN),
        ("before-write", LiveInputResult.RETURNED),
    ],
)
@pytest.mark.timeout(10)
async def test_stuck_live_input_releases_cancellation_after_grace(
    monkeypatch: pytest.MonkeyPatch, stall: str, final: LiveInputResult
) -> None:
    grace = 0.4
    monkeypatch.setattr(runner_module, "LIVE_INPUT_CANCEL_GRACE_SECONDS", grace)
    counted = record_counters(monkeypatch)
    blocker = threading.Event()
    never = asyncio.Event()

    class ThreadAdmission(RecordingAdmission):
        async def claim(self) -> LiveInputClaim:
            self.events.append("claim")
            await run_db_io_cancellation_safe(lambda: blocker.wait(5))
            return LiveInputClaim.CLAIMED

    if stall == "coroutine":
        adm: RecordingAdmission = RecordingAdmission(gate=never)
    elif stall == "thread":
        adm = ThreadAdmission()
    else:
        adm = RecordingAdmission()

    async def stuck_review(*_: Any, **__: Any) -> None:
        await never.wait()

    async def reply() -> dict[str, Any]:
        if stall == "before-write":
            # The input review runs after the claim and before anything is
            # written; it is switched on only once the run itself has started.
            monkeypatch.setattr(runner_module, "input_review_configured", lambda: True)
            monkeypatch.setattr(runner_module, "review_input", stuck_review)
        assert rig.offer(adm) is LiveInputOffer.QUEUED
        return tool_calls("w1")

    rig = make_rig([reply, final_answer()], ("w1",))
    loop = asyncio.get_running_loop()
    task = rig.start()
    try:
        while "claim" not in adm.events:
            await asyncio.sleep(0)
        if stall == "before-write":
            await asyncio.sleep(0.05)
        started = loop.time()
        task.cancel()
        await asyncio.sleep(grace / 2)
        # Inside the grace period the cancellation is held back.
        assert not task.done()
        assert adm.events == ["claim"]
        assert deferred_counts(counted) == []
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        elapsed = loop.time() - started
        assert grace - 0.01 <= elapsed < grace + 0.5
        assert task.cancelled()
        assert deferred_counts(counted) == ["grace_expired"]
    finally:
        blocker.set()
    await asyncio.wait_for(
        asyncio.gather(*rig.runner._live_input_tasks, return_exceptions=True), 6
    )
    assert adm.events[-1] is final


@pytest.mark.parametrize("ending", ["returns", "raises"])
async def test_deferred_cancel_still_propagates(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    ending: str,
) -> None:
    gate = asyncio.Event()
    adm = RecordingAdmission(gate=gate)
    unretrieved: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _loop, context: unretrieved.append(context))

    async def reply() -> dict[str, Any]:
        assert rig.offer(adm) is LiveInputOffer.QUEUED
        return tool_calls("w1")

    rig = make_rig([reply, final_answer()], ("w1",))
    started = asyncio.Event()
    if ending == "raises":

        async def failing(execution_id: str, item: LiveUserInput) -> LiveInputResult:
            started.set()
            await gate.wait()
            raise RuntimeError("boom")

        monkeypatch.setattr(rig.runner, "_inject_live_input", failing)
    task = rig.start()
    try:
        if ending == "raises":
            await started.wait()
        else:
            while "claim" not in adm.events:
                await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0.01)
        gate.set()
        with caplog.at_level(logging.ERROR, logger=runner_module.logger.name):
            with pytest.raises(asyncio.CancelledError):
                await task
        assert task.cancelled()
        gc.collect()
        await asyncio.sleep(0)
        assert unretrieved == []
    finally:
        loop.set_exception_handler(None)
    if ending == "raises":
        failures = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any(r.exc_info and r.exc_info[0] is RuntimeError for r in failures)
        assert all(LIVE_TEXT not in r.getMessage() for r in failures)
    else:
        assert adm.events == ["claim", LiveInputResult.APPLIED]


async def test_repeated_cancel_keeps_the_first_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "LIVE_INPUT_CANCEL_GRACE_SECONDS", 0.5)
    adm = RecordingAdmission(gate=asyncio.Event())

    async def reply() -> dict[str, Any]:
        assert rig.offer(adm) is LiveInputOffer.QUEUED
        return tool_calls("w1")

    rig = make_rig([reply, final_answer()], ("w1",))
    loop = asyncio.get_running_loop()
    task = rig.start()
    while "claim" not in adm.events:
        await asyncio.sleep(0)
    started = loop.time()
    task.cancel()
    await asyncio.sleep(0.4)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert 0.49 <= loop.time() - started < 0.8
    await asyncio.gather(*rig.runner._live_input_tasks, return_exceptions=True)
    assert adm.events == ["claim", LiveInputResult.UNKNOWN]


async def test_deferred_cancel_is_recognised_by_asyncio_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    adm = RecordingAdmission(gate=gate)

    async def reply() -> dict[str, Any]:
        assert rig.offer(adm) is LiveInputOffer.QUEUED
        asyncio.get_running_loop().call_later(0.2, gate.set)
        return tool_calls("w1")

    rig = make_rig([reply, final_answer()], ("w1",))
    cancelling_at_hand_back: list[int] = []
    hand_back = rig.runner._return_live_inputs_in_background

    def spy(items: list[LiveUserInput]) -> None:
        task = asyncio.current_task()
        assert task is not None
        cancelling_at_hand_back.append(task.cancelling())
        hand_back(items)

    monkeypatch.setattr(rig.runner, "_return_live_inputs_in_background", spy)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await rig.runner.run("hello", execution_id=EID)
    assert adm.events == ["claim", LiveInputResult.APPLIED]
    # The run is still marked as being cancelled while it winds down.
    assert cancelling_at_hand_back == [1]


async def test_cancel_during_first_of_several_live_inputs() -> None:
    gate = asyncio.Event()
    first = RecordingAdmission(gate=gate)
    second = RecordingAdmission()
    third = RecordingAdmission()

    async def reply() -> dict[str, Any]:
        rig.offer(first, turn_id="a")
        rig.offer(second, turn_id="b")
        rig.offer(third, turn_id="c")
        return tool_calls("w1")

    rig = make_rig([reply, final_answer()], ("w1",))
    task = rig.start()
    while "claim" not in first.events:
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0.01)
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert first.events == ["claim", LiveInputResult.APPLIED]
    await asyncio.gather(*rig.runner._live_input_tasks)
    assert second.events == [LiveInputResult.RETURNED]
    assert third.events == [LiveInputResult.RETURNED]


async def test_no_cancellation_records_no_deferral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counted = record_counters(monkeypatch)
    adm = RecordingAdmission()

    async def reply() -> dict[str, Any]:
        assert rig.offer(adm) is LiveInputOffer.QUEUED
        return tool_calls("w1")

    rig = make_rig([reply, final_answer()], ("w1",))
    await rig.start()
    assert adm.events == ["claim", LiveInputResult.APPLIED]
    assert deferred_counts(counted) == []


async def test_cancel_right_after_take_still_admits_the_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adm = RecordingAdmission()

    async def reply() -> dict[str, Any]:
        assert rig.offer(adm) is LiveInputOffer.QUEUED
        mailbox = rig.runner._active_controls[EID].runtime.live_input
        take_next = mailbox.take_next

        def take_and_cancel() -> LiveUserInput | None:
            item = take_next()
            # Takes effect at the next suspension, which must already be
            # inside the protected admission of the item just taken.
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            return item

        monkeypatch.setattr(mailbox, "take_next", take_and_cancel)
        return tool_calls("w1")

    rig = make_rig([reply, final_answer()], ("w1",))
    task = rig.start()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert adm.events == ["claim", LiveInputResult.APPLIED]


# --------------------------------------------------------------------------
# Active turn
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (LiveInputResult.APPLIED, "live-1"),
        (LiveInputResult.RETURNED, "original"),
        (LiveInputResult.REFUSED, "original"),
        (LiveInputResult.UNKNOWN, "original"),
    ],
)
async def test_applied_live_input_becomes_active_turn(
    outcome: LiveInputResult, expected: str
) -> None:
    from xagent.core.agent import ExecutionContext

    # The context already ends with the message of turn "live-1" while the
    # runtime still holds the earlier turn, so only an applied outcome may
    # move the runtime to the context's latest turn.
    context = ExecutionContext(execution_id=EID)
    context.add_user_message("first", metadata={"turn_id": "original"})
    context.add_user_message(LIVE_TEXT, metadata={"turn_id": "live-1"})
    runtime = PatternRuntime(execution_id=EID)
    runtime.live_input = LiveInputMailbox()

    async def injector(item: LiveUserInput) -> LiveInputResult:
        return outcome

    runtime._live_input_injector = injector
    runtime.active_turn_id = "original"
    runtime.live_input.offer(make_item("live-1"))
    await runtime.admit_live_user_messages(context, tools_settled=True)
    assert runtime.active_turn_id == expected


@pytest.mark.parametrize(
    ("claim", "expected_second"),
    [(LiveInputClaim.CLAIMED, "live-1"), (LiveInputClaim.UNAVAILABLE, None)],
)
async def test_tool_after_live_input_runs_under_inserted_turn(
    claim: LiveInputClaim, expected_second: str | None
) -> None:
    holder: list[Rig] = []
    seen: dict[str, str | None] = {}

    async def on_start(name: str) -> None:
        runtime = holder[0].runner._active_controls[EID].runtime
        seen[name] = runtime.active_turn_id
        if name == "w1":
            holder[0].offer(RecordingAdmission(claim), turn_id="live-1")

    rig = make_rig([tool_calls("w1"), tool_calls("w2"), final_answer()])
    holder.append(rig)
    for tool in rig.tools.values():
        tool.on_start = on_start
    result = await rig.start()
    assert result["success"] is True
    assert seen["w1"] and seen["w1"] != "live-1"
    # An input that was not applied leaves the turn the run started under.
    assert seen["w2"] == (expected_second or seen["w1"])


@pytest.mark.parametrize("inserted", [True, False])
async def test_repeat_write_after_live_input_is_not_suppressed(inserted: bool) -> None:
    holder: list[Rig] = []

    async def on_start(name: str) -> None:
        if inserted and len(rig.tools["w1"].calls) == 1:
            holder[0].offer(RecordingAdmission(), turn_id="live-1")

    rig = make_rig(
        [tool_calls("w1"), tool_calls("w1"), final_answer()],
        ("w1",),
        pattern=ReActPattern(
            repeated_tool_decision_after_consecutive_tool_calls=None,
            repeated_tool_decision_after_consecutive_work_tool_calls=None,
        ),
    )
    holder.append(rig)
    rig.tools["w1"].metadata.mcp_non_idempotent_write = True
    rig.tools["w1"].on_start = on_start
    result = await rig.start()
    assert result["success"] is True
    assert len(rig.tools["w1"].calls) == (2 if inserted else 1)


# --------------------------------------------------------------------------
# Iteration allowance
# --------------------------------------------------------------------------

GRANT = react_module.LIVE_INPUT_ITERATION_GRANT
# More than the allowance of three messages, so the max_iterations bound on
# the allowance does not apply to the cases that use it.
UNCAPPED_MAX_ITERATIONS = 3 * GRANT + 1


def rig_with_offered_messages(
    count: int | list[int],
    rest: list[Any],
    *,
    max_iterations: int,
    claim: LiveInputClaim = LiveInputClaim.CLAIMED,
) -> Rig:
    """``count`` messages are queued when the first tool starts; a list queues
    ``count[i]`` when tool call ``i`` starts, so each lands at its own safe
    point. A running tool never has the model request cancelled."""
    counts = [count] if isinstance(count, int) else count
    rig = make_rig(
        [*[tool_calls("w1")] * len(counts), *rest],
        ("w1",),
        pattern=tool_only_pattern(max_iterations),
    )
    started = 0

    async def on_start(_: str) -> None:
        nonlocal started
        position, started = started, started + 1
        if position >= len(counts):
            return
        first_index = sum(counts[:position])
        for index in range(first_index, first_index + counts[position]):
            offer = rig.offer(
                RecordingAdmission(claim),
                turn_id=f"live-{index}",
                text=f"{LIVE_TEXT}{index}",
            )
            assert offer is LiveInputOffer.QUEUED

    rig.tools["w1"].on_start = on_start
    return rig


def tool_only_pattern(max_iterations: int) -> ReActPattern:
    return ReActPattern(
        max_iterations=max_iterations,
        repeated_tool_decision_after_consecutive_tool_calls=None,
        repeated_tool_decision_after_consecutive_work_tool_calls=None,
    )


def live_messages_in(checkpoint: dict[str, Any]) -> int:
    """How many live messages the context saved in ``checkpoint`` holds."""
    messages = (checkpoint.get("context") or {}).get("messages") or []
    return sum(str(message.get("content")).count(LIVE_TEXT) for message in messages)


@pytest.mark.parametrize(
    ("offers", "claim", "granted"),
    [
        pytest.param([0], LiveInputClaim.CLAIMED, 0, id="0"),
        pytest.param([1], LiveInputClaim.CLAIMED, 1, id="1"),
        pytest.param([3], LiveInputClaim.CLAIMED, 3, id="3"),
        pytest.param([1, 1], LiveInputClaim.CLAIMED, 2, id="two_safe_points"),
        pytest.param([1], LiveInputClaim.UNAVAILABLE, 0, id="returned"),
        pytest.param([1], LiveInputClaim.UNKNOWN, 0, id="unknown"),
        pytest.param([1], LiveInputClaim.TAKEN, 0, id="taken"),
    ],
)
async def test_live_input_iteration_grant(
    offers: list[int], claim: LiveInputClaim, granted: int
) -> None:
    max_iterations = UNCAPPED_MAX_ITERATIONS
    rig = rig_with_offered_messages(
        offers,
        [tool_calls("w1")] * (max_iterations + GRANT * 3 + 5),
        max_iterations=max_iterations,
        claim=claim,
    )
    result = await rig.start()
    # Every iteration of this model asks for the tool once, so the tool call
    # count is the number of iterations that ran. Only a message that was
    # written into the context adds to the bound.
    assert len(rig.tools["w1"].calls) == max_iterations + GRANT * granted
    # The run ends at the cap and reports the allowance next to the bound.
    assert result["live_input_iteration_grant"] == GRANT * granted


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(InputReviewRejected("no"), id="refused"),
        pytest.param(UserMessageInjectionConflictError("c"), id="conflict"),
    ],
)
async def test_live_input_failed_injection_adds_no_iteration_grant(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    max_iterations = 3
    rig = rig_with_offered_messages(
        1,
        [tool_calls("w1")] * (max_iterations + GRANT + 5),
        max_iterations=max_iterations,
    )

    async def inject(*args: Any, **kwargs: Any) -> UserMessageInjectionResult:
        raise error

    monkeypatch.setattr(rig.runner, "inject_user_message", inject)
    await rig.start()
    assert rig.runner.agent.patterns[0].live_input_iteration_grant == 0
    assert len(rig.tools["w1"].calls) == max_iterations


async def test_forced_answer_extra_iterations_extend_the_raised_cap() -> None:
    pattern = ReActPattern(max_iterations=GRANT)
    pattern.live_input_iteration_grant = GRANT
    pattern.force_final_answer_next = True
    pattern.forced_answer_extra_iterations = 1
    runtime = PatternRuntime(execution_id=EID)
    cap = 2 * GRANT
    bound = pattern._within_iteration_bound
    assert bound(cap - 1, runtime=runtime, context=None)
    assert bound(cap, runtime=runtime, context=None)
    assert not bound(cap + 1, runtime=runtime, context=None)


@pytest.mark.parametrize("delivery", ["raised", "returned"])
async def test_unavailable_tool_recovery_stops_at_the_raised_cap(delivery: str) -> None:
    from xagent.core.model.chat.exceptions import LLMToolProtocolError
    from xagent.core.model.chat.tool_protocol import (
        ToolProtocolViolation,
        tool_protocol_error_response,
    )

    max_iterations = 3

    async def refused() -> Any:
        if delivery == "raised":
            raise LLMToolProtocolError(
                provider="test",
                code="unavailable_tool_call",
                message="unavailable tool call",
            )
        return tool_protocol_error_response(
            ToolProtocolViolation(
                provider="test",
                code="unavailable_tool_call",
                message="unavailable tool call",
            )
        )

    # Requests 0..2 are iterations 0..2; request 3 is the iteration that sits
    # at max_iterations and is refused. With the allowance in force the
    # refusal is recovered from and the next request still offers work.
    rig = rig_with_offered_messages(
        1,
        [tool_calls("w1"), tool_calls("w1"), refused, tool_calls("w1"), final_answer()],
        max_iterations=max_iterations,
    )
    result = await rig.start()
    assert result["success"] is True
    assert len(rig.tools["w1"].calls) == max_iterations + 1


@pytest.mark.parametrize(
    ("saved", "expected"),
    [
        # A checkpoint written before the allowance existed has no such key.
        pytest.param({}, 0, id="missing"),
        pytest.param({"live_input_iteration_grant": None}, 0, id="none"),
        pytest.param({"live_input_iteration_grant": -4}, 0, id="negative"),
        pytest.param(
            {"live_input_iteration_grant": GRANT + 5}, GRANT, id="above_max_iterations"
        ),
        pytest.param(
            {"live_input_iteration_grant": 5, "max_iterations": 0},
            0,
            id="max_iterations_zero",
        ),
    ],
)
def test_live_input_iteration_grant_loads_within_range(
    saved: dict[str, Any], expected: int
) -> None:
    state = ReActPattern(max_iterations=GRANT).get_state()
    del state["live_input_iteration_grant"]
    state.update(saved)
    # Built with the default max_iterations, so only the saved one can bound
    # the allowance it loads.
    restored = ReActPattern()
    restored.live_input_iteration_grant = 7
    restored.load_state(state)
    assert restored.live_input_iteration_grant == expected


@pytest.mark.parametrize("n_applied", [1, 3])
async def test_live_input_iteration_grant_survives_state_round_trip(
    n_applied: int,
) -> None:
    max_iterations = UNCAPPED_MAX_ITERATIONS
    rig = rig_with_offered_messages(
        n_applied, [final_answer()], max_iterations=max_iterations
    )
    pattern = rig.runner.agent.patterns[0]
    await rig.start()
    assert pattern.live_input_iteration_grant == GRANT * n_applied
    restored = ReActPattern(max_iterations=max_iterations)
    restored.load_state(json.loads(json.dumps(pattern.get_state())))
    assert restored.live_input_iteration_grant == GRANT * n_applied
    assert restored._iteration_cap() == max_iterations + GRANT * n_applied


async def test_live_input_iteration_grant_stops_at_max_iterations() -> None:
    # The second message takes the allowance to max_iterations; the third is
    # still written into the conversation but adds nothing.
    max_iterations = 2 * GRANT - 1
    rig = rig_with_offered_messages(
        [1, 1, 1],
        [tool_calls("w1")] * (3 * max_iterations),
        max_iterations=max_iterations,
    )
    result = await rig.start()
    assert result["live_input_iteration_grant"] == max_iterations
    assert len(rig.tools["w1"].calls) == 2 * max_iterations
    last_request = " ".join(str(m.get("content")) for m in rig.llm.requests[-1])
    assert all(f"{LIVE_TEXT}{index}" in last_request for index in range(3))
    # The third safe point applied a message at the cap and still saved.
    labels = [c["label"] for c in rig.runtimes[0].checkpoints]
    assert labels.count("live_input_applied") == 3


@pytest.mark.parametrize(
    ("offers", "claim", "saves"),
    [
        pytest.param([1], LiveInputClaim.CLAIMED, 1, id="one"),
        pytest.param([3], LiveInputClaim.CLAIMED, 1, id="three_at_one_safe_point"),
        pytest.param([1, 1], LiveInputClaim.CLAIMED, 2, id="two_safe_points"),
        pytest.param([1], LiveInputClaim.UNAVAILABLE, 0, id="returned"),
    ],
)
async def test_live_input_iteration_grant_is_saved_at_the_safe_point(
    offers: list[int], claim: LiveInputClaim, saves: int
) -> None:
    rig = rig_with_offered_messages(
        offers, [final_answer()], max_iterations=UNCAPPED_MAX_ITERATIONS, claim=claim
    )
    await rig.start()
    checkpoints = rig.runtimes[0].checkpoints
    # One save for each safe point that applied a message, none otherwise.
    labels = [checkpoint["label"] for checkpoint in checkpoints]
    assert labels.count("live_input_applied") == saves
    # The first pattern checkpoint to hold a new message is that save, and
    # every pattern checkpoint holds the allowance of the messages it holds.
    held = [live_messages_in(checkpoint) for checkpoint in checkpoints]
    first_to_hold_more = [
        checkpoint["label"]
        for checkpoint, before, now in zip(checkpoints, [0, *held], held)
        if now > before
    ]
    assert first_to_hold_more == ["live_input_applied"] * saves
    assert [
        checkpoint["pattern_state"]["live_input_iteration_grant"]
        for checkpoint in checkpoints
    ] == [GRANT * count for count in held]


async def test_live_input_iteration_grant_survives_resume() -> None:
    max_iterations = UNCAPPED_MAX_ITERATIONS
    rig = rig_with_offered_messages(1, [final_answer()], max_iterations=max_iterations)
    await rig.start()
    saved = next(
        checkpoint
        for checkpoint in rig.runtimes[0].checkpoints
        if checkpoint["label"] == "live_input_applied"
    )
    resumed = tool_only_pattern(max_iterations)
    resumed.load_state(json.loads(json.dumps(saved["pattern_state"])))
    start = resumed.current_iteration
    assert start >= 1
    tool = ProbeTool("w1")
    context = ExecutionContext(execution_id=EID)
    context.add_user_message("hello")
    result = await resumed.run(
        context=context,
        tools=[tool],
        llm=ScriptedLLM([tool_calls("w1")] * (max_iterations + GRANT + 5)),
        runtime=PatternRuntime(execution_id=EID),
    )
    # The resumed run keeps working past max_iterations up to the raised cap.
    assert result["success"] is False
    assert len(tool.calls) == max_iterations + GRANT - start


async def test_iteration_limit_delivery_reports_the_grant() -> None:
    max_iterations = UNCAPPED_MAX_ITERATIONS
    rig = rig_with_offered_messages(
        [1, 1],
        [tool_calls("w1")] * (max_iterations + 2 * GRANT - 2)
        + [final_answer(outcome="partial")],
        max_iterations=max_iterations,
    )
    result = await rig.start()
    # The answer delivered at the cap reports the allowance as the failure
    # result does.
    assert result["success"] is True
    assert result["termination_reason"] == "max_iterations"
    assert result["iterations"] == max_iterations
    assert result["live_input_iteration_grant"] == GRANT * 2
    assert len(rig.tools["w1"].calls) == max_iterations + GRANT * 2


async def test_live_input_mixed_batch_grants_only_the_applied_message() -> None:
    max_iterations = UNCAPPED_MAX_ITERATIONS
    started: list[bool] = []

    async def offer_claimed_then_unavailable(_: str) -> None:
        if started:
            return
        started.append(True)
        for index, claim in enumerate(
            (LiveInputClaim.CLAIMED, LiveInputClaim.UNAVAILABLE)
        ):
            offer = rig.offer(
                RecordingAdmission(claim),
                turn_id=f"live-{index}",
                text=f"{LIVE_TEXT}{index}",
            )
            assert offer is LiveInputOffer.QUEUED

    rig = make_rig(
        [tool_calls("w1")] * (max_iterations + 2 * GRANT + 5),
        ("w1",),
        pattern=tool_only_pattern(max_iterations),
    )
    rig.tools["w1"].on_start = offer_claimed_then_unavailable
    result = await rig.start()
    assert result["live_input_iteration_grant"] == GRANT
    assert len(rig.tools["w1"].calls) == max_iterations + GRANT


async def test_live_input_replay_of_a_turn_written_after_queueing_adds_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    max_iterations = UNCAPPED_MAX_ITERATIONS
    outcomes: list[UserMessageInjectionOutcome] = []
    original_llm_end = PatternRuntime.on_llm_end

    async def on_llm_end(self: PatternRuntime, *a: Any, **kw: Any) -> Any:
        # The first response is complete; no tool has started.
        if not outcomes:
            offer = rig.offer(RecordingAdmission(), turn_id="dup", text=LIVE_TEXT)
            assert offer is LiveInputOffer.QUEUED
            # The same turn reaches the context through the ordinary injection
            # path before the loop takes the queued copy at its safe point.
            written = await rig.runner.inject_user_message(
                EID,
                execution_message=LIVE_TEXT,
                display_message=LIVE_TEXT,
                turn_id="dup",
                request_interrupt=False,
            )
            outcomes.append(written.outcome)
        return await original_llm_end(self, *a, **kw)

    monkeypatch.setattr(PatternRuntime, "on_llm_end", on_llm_end)
    rig = make_rig(
        [tool_calls("w1")] * (max_iterations + 2 * GRANT + 5),
        ("w1",),
        pattern=tool_only_pattern(max_iterations),
    )
    result = await rig.start()
    assert outcomes == [UserMessageInjectionOutcome.POSTED_FRESH]
    # The queued copy finds its turn already accepted: it counts as applied
    # and adds the allowance once, without a second copy of the message.
    assert result["live_input_iteration_grant"] == GRANT
    assert len(rig.tools["w1"].calls) == max_iterations + GRANT
    last_request = " ".join(str(m.get("content")) for m in rig.llm.requests[-1])
    assert last_request.count(LIVE_TEXT) == 1


# --------------------------------------------------------------------------
# Iteration counter
# --------------------------------------------------------------------------


def test_iteration_counter_matches_count_and_repeats_once() -> None:
    plain = react_module._IterationCounter(7)
    assert list(itertools.islice(plain, 50)) == list(
        itertools.islice(itertools.count(7), 50)
    )
    counter = react_module._IterationCounter(4)
    iterator = iter(counter)
    assert [next(iterator), next(iterator)] == [4, 5]
    counter.repeat()
    assert counter.repeats == 1
    assert [next(iterator) for _ in range(3)] == [5, 6, 7]
    assert counter.repeats == 0


# --------------------------------------------------------------------------
# Preempting the main model request
# --------------------------------------------------------------------------


class Frames:
    """Outbound handler that keeps every frame the run emits."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def __call__(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)


def stream_end() -> StreamChunk:
    return StreamChunk(type=ChunkType.END)


def stream_token(text: str) -> StreamChunk:
    return StreamChunk(type=ChunkType.TOKEN, delta=text)


def stream_tool_calls(*names: str) -> list[Any]:
    chunks = [
        StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "index": index,
                    "id": f"call_{name}",
                    "function": {"name": name, "arguments": "{}"},
                }
            ],
        )
        for index, name in enumerate(names)
    ]
    return [*chunks, stream_end()]


def stream_answer_part(arguments: str) -> StreamChunk:
    return StreamChunk(
        type=ChunkType.TOOL_CALL,
        tool_calls=[
            {
                "index": 0,
                "id": "call_final",
                "function": {"name": "final_answer", "arguments": arguments},
            }
        ],
    )


def stream_final_answer_parts(answer: str, *, cut: int, hook: Any = None) -> list[Any]:
    """A final_answer call streamed in two parts, ``cut`` characters into the text.

    With a ``hook`` the stream stops after the first part and runs it, and the
    second part is never produced when the hook does not return.
    """
    arguments = json.dumps({"answer": answer, "response_language": "English"})
    at = arguments.index(answer) + cut
    parts: list[Any] = [stream_answer_part(arguments[:at])]
    if hook is not None:
        return [*parts, hook]
    return [*parts, stream_answer_part(arguments[at:]), stream_end()]


# How long a scripted model request stays open after a message was offered
# during it. A preempted request is cancelled at once; one that is not simply
# carries on and returns ``reply`` after this long.
HOLD_SECONDS = 0.3


def offer_then_wait(
    get_rig: Any,
    admission: RecordingAdmission,
    *,
    turn_id: str = "t1",
    text: str = LIVE_TEXT,
    reply: Any = None,
) -> Any:
    """A model-request step that offers a message and then keeps the request open."""

    async def hook() -> Any:
        assert (
            get_rig().offer(admission, turn_id=turn_id, text=text)
            is LiveInputOffer.QUEUED
        )
        await asyncio.sleep(HOLD_SECONDS)
        return reply

    return hook


class HeldStreamLLM:
    """Native-streaming model that holds its stream open until released."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.gate = asyncio.Event()

    async def stream_chat(self, **_: Any) -> Any:
        yield stream_token("partial")
        self.started.set()
        await self.gate.wait()
        yield stream_token(" rest")
        yield stream_end()


@dataclass
class LlmCallSpy:
    # (runtime method, preemptible argument or None when absent, outcome)
    calls: list[tuple[str, Any, str]] = field(default_factory=list)
    main_starts: int = 0
    repeats: int = 0
    before_llm_checkpoints: int = 0

    def preempted(self, method: str) -> bool:
        return (method, True, "LLMCallPreempted") in self.calls


def spy_llm_calls(monkeypatch: pytest.MonkeyPatch) -> LlmCallSpy:
    spy = LlmCallSpy()

    def wrap(method: str) -> None:
        original = getattr(PatternRuntime, method)

        async def wrapper(self: PatternRuntime, llm: Any, *a: Any, **kw: Any) -> Any:
            try:
                result = await original(self, llm, *a, **kw)
            except BaseException as exc:
                spy.calls.append((method, kw.get("preemptible"), type(exc).__name__))
                raise
            spy.calls.append((method, kw.get("preemptible"), "ok"))
            return result

        monkeypatch.setattr(PatternRuntime, method, wrapper)

    for method in ("run_llm_call", "run_streaming_llm_call", "stream_final_answer"):
        wrap(method)
    original_start = PatternRuntime.on_llm_start

    async def on_llm_start(self: PatternRuntime, *a: Any, **kw: Any) -> Any:
        if "purpose" not in (kw.get("metadata") or {}):
            spy.main_starts += 1
        return await original_start(self, *a, **kw)

    monkeypatch.setattr(PatternRuntime, "on_llm_start", on_llm_start)
    original_checkpoint = PatternRuntime.checkpoint

    async def checkpoint(self: PatternRuntime, name: str, *a: Any, **kw: Any) -> Any:
        if name == "before_llm":
            spy.before_llm_checkpoints += 1
        return await original_checkpoint(self, name, *a, **kw)

    monkeypatch.setattr(PatternRuntime, "checkpoint", checkpoint)
    original_repeat = react_module._IterationCounter.repeat

    def repeat(self: Any) -> None:
        spy.repeats += 1
        original_repeat(self)

    monkeypatch.setattr(react_module._IterationCounter, "repeat", repeat)
    return spy


def arm_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the first compaction check ask the model for a summary."""
    from xagent.core.agent.context.execution import CompactResult, ExecutionContext

    armed = [True]

    def request(self: Any, *, context_window: int | None = None) -> Any:
        if not armed:
            return None
        armed.clear()
        return {
            "messages": [{"role": "user", "content": "summarize"}],
            "max_tokens": 64,
            "original_tokens": 10,
            "metadata": {},
        }

    def apply(
        self: Any, response: Any, *, llm: Any, original_tokens: Any = None
    ) -> Any:
        return CompactResult(
            compacted=True,
            original_count=1,
            final_count=1,
            strategy="llm_summary",
            metadata={},
        )

    monkeypatch.setattr(
        ExecutionContext, "build_llm_compact_request_if_needed", request
    )
    monkeypatch.setattr(ExecutionContext, "compact_with_llm_response", apply)


def live_index(messages: list[dict[str, Any]], text: str = LIVE_TEXT) -> int:
    return next(i for i, m in enumerate(messages) if text in str(m.get("content")))


PREEMPTED_KINDS = {
    "stream": "run_streaming_llm_call",
    "no-tools": "stream_final_answer",
    "chat": "run_streaming_llm_call",
}


def build_preempt_rig(kind: str, adm: RecordingAdmission) -> Rig:
    """A run whose first main request is open when the message is offered."""
    box: list[Rig] = []
    hook = offer_then_wait(lambda: box[0], adm, reply=tool_calls("w1"))
    if kind == "stream":
        rig = make_rig(
            [
                [stream_tool_calls("w1")[0], hook],
                stream_final_answer_parts("ok", cut=1),
            ],
            ("w1",),
            llm_class=StreamingScriptedLLM,
        )
    elif kind == "no-tools":
        rig = make_rig(
            [[stream_token("PARTIAL"), hook], [stream_token("done"), stream_end()]],
            ("w1",),
            pattern=ReActPattern(tool_choice="none"),
            llm_class=StreamingScriptedLLM,
        )
    elif kind == "chat":
        rig = make_rig([hook, final_answer()], ("w1",))
    elif kind == "protocol-retry":
        from xagent.core.model.chat.exceptions import LLMToolProtocolError

        async def refused() -> Any:
            raise LLMToolProtocolError(
                provider="test",
                code="unavailable_tool_call",
                message="unavailable tool call",
            )

        async def retry_reply() -> Any:
            assert box[0].offer(adm) is LiveInputOffer.QUEUED
            await asyncio.sleep(0.01)
            return tool_calls("w1")

        rig = make_rig([refused, retry_reply, final_answer()], ("w1",))
    else:

        async def summary() -> str:
            assert box[0].offer(adm) is LiveInputOffer.QUEUED
            await asyncio.sleep(0.01)
            return "summary"

        rig = make_rig([summary, final_answer()], ("w1",))
    box.append(rig)
    return rig


@pytest.mark.parametrize("kind", ["stream", "no-tools", "chat"])
async def test_preempt_cancels_the_main_call(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    spy = spy_llm_calls(monkeypatch)
    adm = RecordingAdmission()
    rig = build_preempt_rig(kind, adm)
    result = await rig.start(outbound_message_handler=Frames())
    assert result["success"] is True
    assert adm.events == ["claim", LiveInputResult.APPLIED]
    assert all(not runtime._interrupt_requested for runtime in rig.runtimes)
    assert spy.preempted(PREEMPTED_KINDS[kind])
    # The cancelled request and its repeat with the message in its input.
    assert len(rig.llm.requests) == 2
    assert LIVE_TEXT not in str(rig.llm.requests[0])
    assert live_index(rig.llm.requests[-1]) > 0
    assert rig.tools["w1"].calls == []


@pytest.mark.parametrize(
    ("kind", "expected_requests"),
    [
        # The refused request, its retry, then the main request that takes
        # the message in.
        pytest.param("protocol-retry", 3, id="protocol-retry"),
        # The summary request, then the main request that takes the message in.
        pytest.param("compaction", 2, id="compaction"),
    ],
)
async def test_preempt_leaves_non_main_calls_running(
    monkeypatch: pytest.MonkeyPatch, kind: str, expected_requests: int
) -> None:
    spy = spy_llm_calls(monkeypatch)
    if kind == "compaction":
        arm_compaction(monkeypatch)
    adm = RecordingAdmission()
    rig = build_preempt_rig(kind, adm)
    result = await rig.start(outbound_message_handler=Frames())
    assert result["success"] is True
    assert adm.events == ["claim", LiveInputResult.APPLIED]
    assert all(not runtime._interrupt_requested for runtime in rig.runtimes)
    # A retry after a protocol error and the compaction request are not main
    # requests: they finish, and the message waits for the next main request.
    assert not any(call[2] == "LLMCallPreempted" for call in spy.calls)
    assert rig.llm.cancelled == 0
    assert len(rig.llm.requests) == expected_requests
    assert live_index(rig.llm.requests[-1]) > 0
    assert all(LIVE_TEXT not in str(request) for request in rig.llm.requests[:-1])


@pytest.mark.parametrize("gap", ["compaction", "before-llm-checkpoint", "prepare-llm"])
async def test_live_input_in_gap_skips_main_call(
    monkeypatch: pytest.MonkeyPatch, gap: str
) -> None:
    spy = spy_llm_calls(monkeypatch)
    adm = RecordingAdmission()
    box: list[Rig] = []
    fired: list[bool] = []

    def fire() -> None:
        if not fired:
            fired.append(True)
            assert box[0].offer(adm) is LiveInputOffer.QUEUED

    async def summary() -> str:
        fire()
        await asyncio.sleep(0.01)
        return "summary"

    if gap == "compaction":
        arm_compaction(monkeypatch)
    rig = make_rig(
        [summary, final_answer()] if gap == "compaction" else [final_answer()]
    )
    box.append(rig)
    if gap == "before-llm-checkpoint":
        original = PatternRuntime.checkpoint

        async def checkpoint(
            self: PatternRuntime, name: str, *a: Any, **kw: Any
        ) -> Any:
            if name == "before_llm":
                fire()
            return await original(self, name, *a, **kw)

        monkeypatch.setattr(PatternRuntime, "checkpoint", checkpoint)
    if gap == "prepare-llm":
        prepare = react_module.prepare_llm_for_context

        async def prepare_and_offer(**kwargs: Any) -> Any:
            fire()
            return await prepare(**kwargs)

        monkeypatch.setattr(react_module, "prepare_llm_for_context", prepare_and_offer)
    result = await rig.start()
    assert fired and result["success"] is True
    assert adm.events == ["claim", LiveInputResult.APPLIED]
    # The skipped attempt sent nothing and traced nothing; only the repeated
    # one did, and its input carries the message.
    assert spy.main_starts == 1
    assert spy.repeats == 1
    # A skipped attempt writes no before_llm checkpoint, unless the message
    # arrived while that very checkpoint was being written.
    assert spy.before_llm_checkpoints == (2 if gap == "before-llm-checkpoint" else 1)
    assert not any(call[2] == "LLMCallPreempted" for call in spy.calls)
    assert rig.llm.cancelled == 0
    assert len(rig.llm.requests) == (2 if gap == "compaction" else 1)
    assert LIVE_TEXT in str(rig.llm.requests[-1])


async def test_preempts_bounded_by_offers(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 5
    spy = spy_llm_calls(monkeypatch)
    box: list[Rig] = []
    offered: list[int] = []
    # Offer k lands at the first opportunity of kind ``order[k % 3]``.
    order = ["prepare", "checkpoint", "request"]

    def fire(kind: str) -> bool:
        if len(offered) >= n or order[len(offered) % 3] != kind:
            return False
        index = len(offered)
        offered.append(index)
        assert (
            box[0].offer(
                RecordingAdmission(),
                turn_id=f"live-{index}",
                text=f"{LIVE_TEXT}{index}",
            )
            is LiveInputOffer.QUEUED
        )
        return True

    async def reply() -> Any:
        if fire("request"):
            await asyncio.sleep(HOLD_SECONDS)
        return final_answer()

    prepare = react_module.prepare_llm_for_context
    checkpoint = PatternRuntime.checkpoint

    async def prepare_and_offer(**kwargs: Any) -> Any:
        fire("prepare")
        return await prepare(**kwargs)

    async def checkpoint_and_offer(
        self: PatternRuntime, name: str, *a: Any, **kw: Any
    ) -> Any:
        if name == "before_llm":
            fire("checkpoint")
        return await checkpoint(self, name, *a, **kw)

    monkeypatch.setattr(react_module, "prepare_llm_for_context", prepare_and_offer)
    monkeypatch.setattr(PatternRuntime, "checkpoint", checkpoint_and_offer)
    rig = make_rig([reply] * (n + 2))
    box.append(rig)
    result = await rig.start()
    assert result["success"] is True
    assert len(offered) == n
    assert spy.repeats == n
    assert all(f"{LIVE_TEXT}{i}" in str(rig.llm.requests[-1]) for i in range(n))


LIMIT = react_module.LIVE_INPUT_PREEMPT_LIMIT


def offering_requests(count: int, box: list[Rig]) -> list[Any]:
    """Model requests that each offer a new message, hold, then ask for a tool."""
    return [
        offer_then_wait(
            lambda: box[0],
            RecordingAdmission(),
            turn_id=f"live-{index}",
            text=f"{LIVE_TEXT}{index}",
            reply=tool_calls("w1"),
        )
        for index in range(count)
    ]


async def test_message_past_the_preempt_limit_waits_for_the_next_safe_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = spy_llm_calls(monkeypatch)
    box: list[Rig] = []
    # One message more than the limit arrives while the same iteration's
    # request is open; the first LIMIT of them preempt it.
    rig = make_rig([*offering_requests(LIMIT + 1, box), final_answer()], ("w1",))
    box.append(rig)
    result = await rig.start()
    assert result["success"] is True
    assert spy.repeats == LIMIT
    assert rig.llm.cancelled == LIMIT
    # The request the last message arrived in ran to its end, and its tool
    # call ran once.
    assert len(rig.llm.requests) == LIMIT + 2
    assert len(rig.tools["w1"].calls) == 1
    assert f"{LIVE_TEXT}{LIMIT}" not in str(rig.llm.requests[LIMIT])
    assert f"{LIVE_TEXT}{LIMIT - 1}" in str(rig.llm.requests[LIMIT])
    # It was taken in at the next safe point, before the next request.
    assert all(f"{LIVE_TEXT}{i}" in str(rig.llm.requests[-1]) for i in range(LIMIT + 1))


async def test_preempt_count_starts_again_with_the_next_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = spy_llm_calls(monkeypatch)
    box: list[Rig] = []
    # Iteration 0 uses up the limit; the message that arrives in iteration 1
    # preempts that iteration's request again.
    rig = make_rig([*offering_requests(LIMIT + 2, box), final_answer()], ("w1",))
    box.append(rig)
    result = await rig.start()
    assert result["success"] is True
    assert spy.repeats == LIMIT + 1
    assert rig.llm.cancelled == LIMIT + 1
    assert len(rig.llm.requests) == LIMIT + 3
    assert all(f"{LIVE_TEXT}{i}" in str(rig.llm.requests[-1]) for i in range(LIMIT + 2))


@pytest.mark.parametrize("kind", ["stream", "chat"])
async def test_llm_input_keeps_pairs_after_preempt(kind: str) -> None:
    adm = RecordingAdmission()
    box: list[Rig] = []
    hook = offer_then_wait(lambda: box[0], adm, reply=tool_calls("w2"))
    if kind == "stream":
        rig = make_rig(
            [
                stream_tool_calls("w1"),
                [stream_tool_calls("w2")[0], hook],
                stream_final_answer_parts("ok", cut=1),
            ],
            llm_class=StreamingScriptedLLM,
        )
    else:
        rig = make_rig([tool_calls("w1"), hook, final_answer()])
    box.append(rig)
    result = await rig.start()
    assert result["success"] is True
    messages = rig.llm.requests[-1]
    assert len(rig.llm.requests) == 3
    assert pairs_intact(messages)
    assert live_index(messages) > max(
        i for i, m in enumerate(messages) if m["role"] == "tool"
    )
    # The preempted request left no assistant turn behind.
    assert [
        bool(m.get("tool_calls")) for m in messages if m["role"] == "assistant"
    ] == [True]
    context = result["context"]
    assert len(context._sanitize_tool_message_pairs(list(context.messages))) == len(
        context.messages
    )


@pytest.mark.parametrize(
    ("n_applied", "n_preempted", "max_iterations"),
    [
        *[
            pytest.param(applied, 5, UNCAPPED_MAX_ITERATIONS, id=f"{applied}-5")
            for applied in (0, 1, 3)
        ],
        # The allowance stays within max_iterations however many messages
        # preempted a request.
        pytest.param(3, 5, 3, id="3-5-capped"),
    ],
)
async def test_live_input_iteration_grant_under_preempts(
    monkeypatch: pytest.MonkeyPatch,
    n_applied: int,
    n_preempted: int,
    max_iterations: int,
) -> None:
    spy = spy_llm_calls(monkeypatch)
    granted = min(GRANT * n_applied, max_iterations)
    box: list[Rig] = []

    def preempted_request(index: int) -> Any:
        claim = (
            LiveInputClaim.CLAIMED if index < n_applied else LiveInputClaim.UNAVAILABLE
        )

        async def hook() -> Any:
            offer = box[0].offer(
                RecordingAdmission(claim), turn_id=f"live-{index}", text=f"m{index}"
            )
            assert offer is LiveInputOffer.QUEUED
            await asyncio.sleep(HOLD_SECONDS)
            return tool_calls("w1")

        return hook

    rig = make_rig(
        [
            *[preempted_request(i) for i in range(n_preempted)],
            *[tool_calls("w1")] * (max_iterations + GRANT * 3 + 5),
        ],
        ("w1",),
        pattern=tool_only_pattern(max_iterations),
    )
    box.append(rig)
    pattern = rig.runner.agent.patterns[0]
    await rig.start()
    # A preempted request never used up an iteration, so the run still gets
    # exactly the iterations the cap and the applied messages allow.
    assert spy.repeats == n_preempted
    assert len(rig.tools["w1"].calls) == max_iterations + granted
    assert pattern.live_input_iteration_grant == granted
    restored = ReActPattern(max_iterations=max_iterations)
    restored.load_state(json.loads(json.dumps(pattern.get_state())))
    assert restored._iteration_cap() == max_iterations + granted


async def test_preempt_on_last_ordinary_iteration_raises_iteration_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = spy_llm_calls(monkeypatch)
    max_iterations = 3
    adm = RecordingAdmission()
    box: list[Rig] = []

    async def last_ordinary_request() -> Any:
        assert box[0].offer(adm) is LiveInputOffer.QUEUED
        await asyncio.sleep(HOLD_SECONDS)
        return tool_calls("w1")

    rig = make_rig(
        [
            *[tool_calls("w1")] * (max_iterations - 1),
            last_ordinary_request,
            *[tool_calls("w1")] * (max_iterations + GRANT + 5),
        ],
        ("w1",),
        pattern=tool_only_pattern(max_iterations),
    )
    box.append(rig)
    pattern = rig.runner.agent.patterns[0]
    await rig.start()
    granted = min(GRANT, max_iterations)
    assert spy.repeats == 1
    assert rig.llm.cancelled == 1
    assert adm.events == ["claim", LiveInputResult.APPLIED]
    assert pattern.live_input_iteration_grant == granted
    assert len(rig.tools["w1"].calls) == max_iterations + granted


async def test_dag_step_calls_never_preemptible() -> None:
    from xagent.core.agent import ExecutionContext
    from xagent.core.agent.pattern.dag.dag import DAGPattern, _DAGStepRuntime

    seen: list[dict[str, Any]] = []

    class RecordingRuntime(PatternRuntime):
        async def run_streaming_llm_call(self, llm: Any, **kwargs: Any) -> Any:
            seen.append(kwargs)
            return final_answer()

        async def run_llm_call(self, llm: Any, **kwargs: Any) -> Any:
            seen.append(kwargs)
            return final_answer()

    parent = RecordingRuntime(execution_id=EID)
    parent.live_input = LiveInputMailbox()
    runtime = _DAGStepRuntime(
        parent=parent,
        dag_pattern=DAGPattern(lambda **_: None),  # type: ignore[arg-type,return-value]
        root_context=ExecutionContext(execution_id=EID),
        step_id="step",
    )
    context = ExecutionContext(execution_id=f"{EID}:step")
    context.add_user_message("hello")
    await ReActPattern().run(
        context=context, tools=[], llm=ScriptedLLM([]), runtime=runtime
    )
    assert seen
    assert all("preemptible" not in kwargs for kwargs in seen)


# --------------------------------------------------------------------------
# Offering a message preempts the running main request
# --------------------------------------------------------------------------


async def test_offer_does_not_preempt_while_tool_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.agent import runner as runner_module

    counted: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        runner_module,
        "increment_counter",
        lambda name, *, attributes=None, **_: counted.append((name, attributes)),
    )
    runner, runtime = manual_runner()
    llm = HeldStreamLLM()
    request = asyncio.create_task(
        runtime.run_streaming_llm_call(llm, messages=[], preemptible=True)
    )
    await llm.started.wait()
    tool: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    runtime._active_tool_tasks.add(tool)
    offer = runner.offer_live_user_message(
        EID,
        execution_message="x",
        display_message="x",
        turn_id="t1",
        admission=RecordingAdmission(),
    )
    await asyncio.sleep(0)
    assert offer is LiveInputOffer.QUEUED
    assert not request.done() and not tool.cancelled()
    assert ("xagent.live_input.preempts", {"outcome": "nothing_to_cancel"}) in counted
    # Once no tool is running, the next message does cancel the request.
    runtime._active_tool_tasks.discard(tool)
    second = runner.offer_live_user_message(
        EID,
        execution_message="y",
        display_message="y",
        turn_id="t2",
        admission=RecordingAdmission(),
    )
    assert second is LiveInputOffer.QUEUED
    with pytest.raises(LLMCallPreempted):
        await asyncio.wait_for(request, timeout=5)
    assert ("xagent.live_input.preempts", {"outcome": "cancelled"}) in counted


async def test_offer_already_queued_does_not_preempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = manual_runner()
    context = runner.context_manager.get_context(EID)
    assert context is not None
    context.add_user_message("seen", metadata={"turn_id": "old"})
    preempts: list[bool] = []
    monkeypatch.setattr(
        runtime, "request_llm_preempt", lambda: preempts.append(True) or False
    )

    def offer(turn_id: str, text: str) -> LiveInputOffer:
        return runner.offer_live_user_message(
            EID,
            execution_message=text,
            display_message=text,
            turn_id=turn_id,
            admission=RecordingAdmission(),
        )

    assert offer("new", "x") is LiveInputOffer.QUEUED
    assert preempts == [True]
    assert offer("new", "x") is LiveInputOffer.ALREADY_QUEUED
    assert offer("new", "y") is LiveInputOffer.CONFLICT
    assert offer("old", "seen") is LiveInputOffer.ALREADY_ACCEPTED
    runtime.live_input = None
    assert offer("n2", "x") is LiveInputOffer.UNSUPPORTED
    assert preempts == [True]


async def test_interrupt_wins_over_offer_on_same_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = spy_llm_calls(monkeypatch)
    adm = RecordingAdmission()
    box: list[Rig] = []

    async def offer_and_stop() -> Any:
        assert box[0].offer(adm) is LiveInputOffer.QUEUED
        assert box[0].runner.pause(EID, "stop") is True
        await asyncio.sleep(HOLD_SECONDS)

    rig = make_rig(
        [stream_final_answer_parts("PARTIAL answer......", cut=7, hook=offer_and_stop)],
        ("w1",),
        llm_class=StreamingScriptedLLM,
    )
    box.append(rig)
    frames = Frames()
    result = await rig.start(outbound_message_handler=frames)
    assert result["status"] == "interrupted"
    assert adm.events == [LiveInputResult.RETURNED]
    assert spy.repeats == 0
    assert not spy.preempted("run_streaming_llm_call")
    assert len(rig.llm.requests) == 1
    assert rig.llm.cancelled == 1
    assert [e["type"] for e in frames.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_error",
    ]
    assert not [e for e in frames.events if e.get("status") == "interrupted"]


# --------------------------------------------------------------------------
# A preempted answer is closed as interrupted, not as failed
# --------------------------------------------------------------------------


PARTIAL_ANSWER = "PARTIAL sydney"
# The stream guard holds back the last six characters of the text it has seen,
# so only the text before them has been delivered when the answer is cut off.
HELD_BACK = "......"


def interrupted_start_rig(kind: str) -> Rig:
    adm = RecordingAdmission()
    box: list[Rig] = []
    hook = offer_then_wait(lambda: box[0], adm)
    written = PARTIAL_ANSWER + HELD_BACK
    if kind == "with-tools":
        rig = make_rig(
            [
                stream_final_answer_parts(written, cut=len(written), hook=hook),
                stream_final_answer_parts("NEW melbourne answer", cut=7),
            ],
            ("w1",),
            llm_class=StreamingScriptedLLM,
        )
    else:
        rig = make_rig(
            [
                [stream_token(written), hook],
                [stream_token("NEW mel"), stream_token("bourne answer"), stream_end()],
            ],
            ("w1",),
            pattern=ReActPattern(tool_choice="none"),
            llm_class=StreamingScriptedLLM,
        )
    box.append(rig)
    return rig


@pytest.mark.parametrize("kind", ["with-tools", "no-tools"])
async def test_preempted_stream_emits_interrupted_start(kind: str) -> None:
    rig = interrupted_start_rig(kind)
    frames = Frames()
    result = await rig.start(outbound_message_handler=frames)
    assert result["success"] is True
    events = frames.events
    first_id = events[0]["message_id"]
    # The frames the web client already handles correctly, in order.
    assert [{k: v for k, v in e.items() if k != "delta"} for e in events[:3]] == [
        {"type": "final_answer_start", "message_id": first_id, "task_id": EID},
        {"type": "final_answer_delta", "message_id": first_id, "task_id": EID},
        {
            "type": "final_answer_start",
            "message_id": first_id,
            "task_id": EID,
            "status": "interrupted",
        },
    ]
    assert events[1]["delta"] == PARTIAL_ANSWER
    assert set(events[2]) == {"type", "message_id", "task_id", "status"}
    second_id = events[3]["message_id"]
    assert events[3] == {
        "type": "final_answer_start",
        "message_id": second_id,
        "task_id": EID,
    }
    assert second_id != first_id
    later = [e for e in events[4:] if e["message_id"] == second_id]
    assert [e["type"] for e in later][-1] == "final_answer_end"
    assert later[-1]["content"] == "NEW melbourne answer"
    assert "".join(e["delta"] for e in later if e["type"] == "final_answer_delta") == (
        "NEW melbourne answer"
    )
    assert not [e for e in events if e["type"] == "final_answer_error"]


def test_interrupted_start_survives_web_stream_event_builder() -> None:
    from xagent.web.services.task_execution import create_final_answer_stream_event

    payload = {
        "type": "final_answer_start",
        "message_id": "final_answer_1",
        "task_id": EID,
        "status": "interrupted",
    }
    event = create_final_answer_stream_event("final_answer_start", EID, payload)
    assert event["type"] == "final_answer_start"
    assert event["message_id"] == "final_answer_1"
    assert event["status"] == "interrupted"
