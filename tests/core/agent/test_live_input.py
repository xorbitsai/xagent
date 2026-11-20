"""Live user input: mailbox, safe-point admission and hand-back at the end."""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from xagent.config import get_agent_pattern_for_execution_mode
from xagent.core.agent import (
    Agent,
    AgentExecutionAdapter,
    AgentExecutionConfig,
    PatternRuntime,
    ReActPattern,
)
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
    InjectionDisposition,
    UserMessageInjectionConflictError,
    UserMessageInjectionOutcome,
    UserMessageInjectionRejectedError,
    UserMessageInjectionResult,
    _record_injection_outcome,
    classify_injection,
)
from xagent.core.guardrails.input import InputReviewRejected

EID = "live-exec"
LIVE_TEXT = "INJECTED"


class _NoArgs(BaseModel):
    pass


class RecordingAdmission:
    """Admission that records every call, optionally into a shared timeline."""

    def __init__(
        self,
        claim: LiveInputClaim | BaseException = LiveInputClaim.CLAIMED,
        *,
        name: str = "a",
        timeline: list[Any] | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.claim_result = claim
        self.name = name
        self.events: list[Any] = []
        self.timeline = timeline
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
        if self.timeline is not None:
            self.timeline.append((self.name, result))


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


def final_answer() -> dict[str, Any]:
    arguments = json.dumps({"answer": "done", "response_language": "English"})
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

    async def chat(self, **kwargs: Any) -> Any:
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        entry = self.script.pop(0)
        if callable(entry):
            entry = await entry()
        return entry


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

    def start(self) -> asyncio.Task[dict[str, Any]]:
        return asyncio.create_task(self.runner.run("hello", execution_id=EID))


def make_rig(
    script: list[Any],
    tool_names: tuple[str, ...] = ("w1", "w2"),
    *,
    safe: tuple[str, ...] = (),
    pattern: Any = None,
) -> Rig:
    tools = {name: ProbeTool(name, safe=name in safe) for name in tool_names}
    llm = ScriptedLLM(script)
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
        if moment == "before-first":
            fire()
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


# --------------------------------------------------------------------------
# Safe point
# --------------------------------------------------------------------------


@pytest.mark.parametrize("segment", list(SEGMENTS))
@pytest.mark.parametrize("moment", MOMENTS)
async def test_live_input_never_cancels_tools(
    monkeypatch: pytest.MonkeyPatch, segment: str, moment: str
) -> None:
    rig, adm, result = await run_scenario(monkeypatch, segment, moment)
    assert result["success"] is True
    for tool in rig.tools.values():
        assert len(tool.calls) == 1
    assert all(not runtime._interrupt_requested for runtime in rig.runtimes)
    assert adm.events == ["claim", LiveInputResult.APPLIED]


@pytest.mark.parametrize("segment", list(SEGMENTS))
@pytest.mark.parametrize("moment", MOMENTS)
async def test_llm_input_keeps_pairs_after_live_input(
    monkeypatch: pytest.MonkeyPatch, segment: str, moment: str
) -> None:
    rig, _, result = await run_scenario(monkeypatch, segment, moment)
    messages = rig.llm.requests[-1]
    assert pairs_intact(messages)
    roles = [m["role"] for m in messages]
    live_index = next(
        i for i, m in enumerate(messages) if LIVE_TEXT in str(m.get("content"))
    )
    assert live_index > max(i for i, r in enumerate(roles) if r == "tool")
    context = result["context"]
    assert len(context._sanitize_tool_message_pairs(list(context.messages))) == len(
        context.messages
    )


@pytest.mark.parametrize("call_site", ["loop-top", "between-segments", "tool-running"])
async def test_admit_requires_settled_tools(call_site: str) -> None:
    runtime = PatternRuntime(execution_id=EID)
    runtime.live_input = LiveInputMailbox()
    taken: list[str] = []

    async def injector(item: LiveUserInput) -> LiveInputResult:
        taken.append(item.turn_id)
        return LiveInputResult.APPLIED

    runtime._live_input_injector = injector
    runtime.live_input.offer(make_item())
    pattern = ReActPattern()
    if call_site == "loop-top":
        assert await pattern._admit_live_inputs_at_safe_point(runtime, None) == 1
        assert taken == ["t1"]
        return
    if call_site == "between-segments":
        pattern.pending_tool_calls = [{"id": "c", "name": "w", "args": {}}]
    else:
        runtime._active_tool_tasks.add(asyncio.get_running_loop().create_future())
    with pytest.raises(RuntimeError):
        await pattern._admit_live_inputs_at_safe_point(runtime, None)
    assert runtime.live_input.pending() and taken == []


async def test_forced_answer_turn_keeps_mailbox_settlement_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig, adm = await forced_answer_run(monkeypatch, "fence")
    assert adm.events[-1] is LiveInputResult.RETURNED and "claim" not in adm.events
    assert LIVE_TEXT not in str(rig.llm.requests[-1])


async def test_forced_answer_turn_keeps_mailbox_iteration_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig, adm = await forced_answer_run(monkeypatch, "limit")
    assert adm.events == [LiveInputResult.RETURNED]
    assert LIVE_TEXT not in str(rig.llm.requests[-1])


async def forced_answer_run(
    monkeypatch: pytest.MonkeyPatch, cause: str
) -> tuple[Rig, RecordingAdmission]:
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

    rig.tools["w1"].on_start = on_start
    if cause == "fence":
        monkeypatch.setattr(
            ReActPattern,
            "_settlement_fence_active",
            lambda self, runtime: bool(rig.tools["w1"].calls),
        )
    result = await rig.start()
    assert result["success"] is True
    return rig, adm


# --------------------------------------------------------------------------
# Exactly-once hand-over
# --------------------------------------------------------------------------


async def test_leftover_live_inputs_returned_before_runner_returns() -> None:
    timeline: list[Any] = []
    first = RecordingAdmission(name="first", timeline=timeline)
    second = RecordingAdmission(name="second", timeline=timeline)

    async def reply() -> dict[str, Any]:
        assert rig.offer(first, turn_id="a") is LiveInputOffer.QUEUED
        assert rig.offer(second, turn_id="b") is LiveInputOffer.QUEUED
        return final_answer()

    rig = make_rig([reply])
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

    async def reply() -> dict[str, Any]:
        assert rig.offer(adm) is LiveInputOffer.QUEUED
        offered.set()
        await hold.wait()
        return final_answer()

    rig = make_rig([reply])
    task = rig.start()
    await offered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.gather(*rig.runner._live_input_tasks)
    assert adm.events == [LiveInputResult.RETURNED]


async def test_duplicate_turn_id_is_finished_once_and_closed_mailbox_refuses() -> None:
    adm = RecordingAdmission()
    duplicate = RecordingAdmission()

    async def reply() -> dict[str, Any]:
        assert rig.offer(adm) is LiveInputOffer.QUEUED
        assert rig.offer(duplicate) is LiveInputOffer.ALREADY_QUEUED
        return final_answer()

    rig = make_rig([reply])
    await rig.start()
    assert adm.events == [LiveInputResult.RETURNED] and duplicate.events == []
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
    assert classify_injection(recorded, None, error) in set(InjectionDisposition)


async def test_cancel_while_claiming_finishes_taken_and_queued_items_once() -> None:
    gate = asyncio.Event()
    first = RecordingAdmission(gate=gate, name="first")
    second = RecordingAdmission(name="second")

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
    await asyncio.gather(*rig.runner._live_input_tasks)
    assert first.events == ["claim", LiveInputResult.UNKNOWN]
    assert second.events == [LiveInputResult.RETURNED]


async def test_stop_before_claim_returns_remaining() -> None:
    adms = [RecordingAdmission(name=str(i)) for i in range(3)]

    async def reply() -> dict[str, Any]:
        for index, adm in enumerate(adms):
            rig.offer(adm, turn_id=f"t{index}")
        return tool_calls("w1")

    rig = make_rig([reply, final_answer()], ("w1",))
    original = adms[0].claim

    async def claim_and_stop() -> LiveInputClaim:
        rig.runtimes[0]._interrupt_requested = True
        return await original()

    adms[0].claim = claim_and_stop  # type: ignore[method-assign]
    result = await rig.start()
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
