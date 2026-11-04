from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xagent.core.agent.context import ContextManager
from xagent.core.agent.runner import (
    AgentRunner,
    UserMessageInjectionConflictError,
    UserMessageInjectionOutcome,
    UserMessageInjectionRejectedError,
)
from xagent.core.agent.runtime import ExecutionInterrupted, PatternRuntime


@pytest.fixture
def live():
    manager = ContextManager()
    context = manager.create_context("atomic")
    context.add_user_message("original")
    tracer = SimpleNamespace(
        load_latest_checkpoint=AsyncMock(return_value=None), checkpoint=AsyncMock()
    )
    runner = AgentRunner(
        SimpleNamespace(llm=None), tracer=tracer, context_manager=manager
    )
    yield runner, context, tracer


@pytest.mark.asyncio
async def test_candidate_is_invisible_until_persisted(live):
    runner, context, tracer = live
    entered, release = asyncio.Event(), asyncio.Event()

    async def write(**payload):
        assert [m["content"] for m in payload["context"]["messages"]] == [
            "original",
            "new",
        ]
        entered.set()
        await release.wait()

    tracer.checkpoint.side_effect = write
    task = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="turn")
    )
    await entered.wait()
    try:
        assert [m.content for m in context.messages] == ["original"]
        assert "_pending_user_message_trace_turn_id" not in context.metadata
    finally:
        release.set()
        await task
    assert [m.content for m in context.messages] == ["original", "new"]


@pytest.mark.asyncio
async def test_confirmed_failed_write_leaves_no_ghost(live):
    runner, context, tracer = live
    tracer.checkpoint.side_effect = RuntimeError("write failed")
    with pytest.raises(UserMessageInjectionRejectedError, match="write failed"):
        await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert [m.content for m in context.messages] == ["original"]
    tracer.checkpoint.side_effect = None
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert tracer.checkpoint.await_count == 2


@pytest.mark.asyncio
async def test_lost_ack_readback_confirms_message(live):
    runner, context, tracer = live

    async def write(**payload):
        tracer.load_latest_checkpoint.return_value = payload
        raise RuntimeError("lost acknowledgement")

    tracer.checkpoint.side_effect = write
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert [m.content for m in context.messages] == ["original", "new"]


@pytest.mark.asyncio
async def test_uncertain_write_blocks_stale_checkpoint(live):
    runner, context, tracer = live

    async def write(**payload):
        tracer.load_latest_checkpoint.side_effect = RuntimeError("read unavailable")
        raise RuntimeError("write outcome unknown")

    tracer.checkpoint.side_effect = write
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    assert [m.content for m in context.messages] == ["original"]
    runtime = PatternRuntime(execution_id="atomic", tracer=tracer)
    with pytest.raises(ExecutionInterrupted):
        await runtime.checkpoint("late", context=context, pattern=SimpleNamespace())
    assert tracer.checkpoint.await_count == 1


@pytest.mark.asyncio
async def test_ordinary_checkpoint_waits_for_injection_publication(live):
    runner, context, tracer = live
    entered, release = asyncio.Event(), asyncio.Event()
    snapshots = []

    async def write(**payload):
        if payload["label"] == "user_message_injected":
            entered.set()
            await release.wait()
        snapshots.append(payload)

    tracer.checkpoint.side_effect = write
    injection = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="turn")
    )
    await entered.wait()
    runtime = PatternRuntime(execution_id="atomic", tracer=tracer)
    ordinary = asyncio.create_task(
        runtime.checkpoint("ordinary", context=context, pattern=SimpleNamespace())
    )
    await asyncio.sleep(0)
    assert not ordinary.done()
    # The executing pattern can still append output while persistence awaits.
    context.add_assistant_message("concurrent output")
    release.set()
    await asyncio.gather(injection, ordinary)
    assert [m.content for m in context.messages] == [
        "original",
        "concurrent output",
        "new",
    ]
    assert snapshots[-1]["label"] == "ordinary"
    assert snapshots[-1]["context"]["messages"][-1]["content"] == "new"


@pytest.mark.asyncio
async def test_concurrent_duplicate_injection_writes_once(live):
    runner, context, tracer = live
    first, second = await asyncio.gather(
        runner.inject_user_message("atomic", "new", turn_id="turn"),
        runner.inject_user_message("atomic", "new", turn_id="turn"),
    )
    assert {first.outcome, second.outcome} == {
        UserMessageInjectionOutcome.POSTED_FRESH,
        UserMessageInjectionOutcome.POSTED_REPLAY,
    }
    assert tracer.checkpoint.await_count == 1
    assert len(context.messages) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["write", "readback"])
async def test_cancelled_uncertain_injection_blocks_stale_writes(live, cancel_at):
    runner, context, tracer = live
    entered = asyncio.Event()

    async def block():
        entered.set()
        await asyncio.Event().wait()

    async def write(**payload):
        await block()

    # AsyncMock's side_effect must itself await the suspended read.
    async def read(_):
        await block()

    async def fail_write(**payload):
        tracer.load_latest_checkpoint.side_effect = read
        raise RuntimeError("uncertain")

    tracer.checkpoint.side_effect = write if cancel_at == "write" else fail_write
    task = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="turn")
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [m.content for m in context.messages] == ["original"]
    runtime = PatternRuntime(execution_id="atomic", tracer=tracer)
    with pytest.raises(ExecutionInterrupted):
        await runtime.checkpoint("late", context=context, pattern=SimpleNamespace())


@pytest.mark.asyncio
async def test_concurrent_cold_injections_share_context(live):
    runner, context, tracer = live
    payload = {"context": context.to_dict()}
    runner.context_manager.remove_context("atomic")
    reads, both = 0, asyncio.Event()

    async def read(_):
        nonlocal reads
        reads += 1
        if reads == 2:
            both.set()
        await both.wait()
        return payload

    async def write(**written):
        nonlocal payload
        # A real store suspends, letting the other reader publish meanwhile.
        await asyncio.sleep(0)
        payload = written

    tracer.load_latest_checkpoint.side_effect = read
    tracer.checkpoint.side_effect = write
    a, b = await asyncio.gather(
        runner.inject_user_message(
            "atomic", "one", turn_id="one", request_interrupt=False
        ),
        runner.inject_user_message(
            "atomic", "two", turn_id="two", request_interrupt=False
        ),
    )
    assert a.context is b.context
    # No run holds the restored context, so the last injection evicts it.
    assert runner.context_manager.get_context("atomic") is None
    assert {m.content for m in a.context.messages} == {"original", "one", "two"}
    assert {
        m["content"] for m in tracer.checkpoint.call_args.kwargs["context"]["messages"]
    } == {"original", "one", "two"}


@pytest.mark.asyncio
async def test_cancel_before_write_does_not_poison_context(live):
    from xagent.core.agent.context.execution import context_checkpoint_gate

    runner, context, tracer = live
    gate = context_checkpoint_gate(context)
    async with gate.shared():
        task = asyncio.create_task(
            runner.inject_user_message("atomic", "new", turn_id="turn")
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not gate.injection_uncertain
    tracer.checkpoint.assert_not_awaited()
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload", ["malformed", {"context": {}}, {"context": {"messages": ["bad"]}}]
)
async def test_malformed_readback_cannot_prove_absence(live, payload):
    runner, context, tracer = live

    async def write(**_):
        tracer.load_latest_checkpoint.return_value = payload
        raise RuntimeError("unknown write")

    tracer.checkpoint.side_effect = write
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    assert [m.content for m in context.messages] == ["original"]


@pytest.mark.asyncio
async def test_live_unknown_stops_run_and_explicit_resume_reloads_checkpoint(live):
    runner, context, tracer = live
    entered, release = asyncio.Event(), asyncio.Event()
    durable = {"context": context.to_dict()}

    class Pattern:
        async def run(self, *, context, runtime, **kwargs):
            entered.set()
            await release.wait()
            await runtime.checkpoint("after", context=context, pattern=self)
            return {"success": True, "output": "done"}

    runner.agent.patterns = [Pattern()]
    operation = asyncio.create_task(
        runner.run("original", execution_id="atomic", checkpoint=durable)
    )
    await entered.wait()

    async def ambiguous_write(**payload):
        nonlocal durable
        durable = payload
        tracer.load_latest_checkpoint.side_effect = RuntimeError("read unavailable")
        raise RuntimeError("ack lost")

    tracer.checkpoint.side_effect = ambiguous_write
    try:
        outcome = await runner.inject_user_message("atomic", "new", turn_id="turn")
        assert outcome.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
        release.set()
        result = await operation
        assert result["status"] == "interrupted"
        assert tracer.checkpoint.await_count == 1
        tracer.load_latest_checkpoint.side_effect = None
        tracer.load_latest_checkpoint.return_value = durable
        tracer.checkpoint.side_effect = None
        result = await runner.resume("atomic")
        assert result["success"]
        restored = result["context"]
        assert [m.content for m in restored.messages].count("new") == 1
    finally:
        release.set()
        operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["callback", "watermark"])
async def test_post_commit_cancellation_still_interrupts_live_run(live, cancel_at):
    from xagent.core.agent.runner import ExecutionControl

    runner, context, tracer = live
    entered = asyncio.Event()

    async def callback(**kwargs):
        if cancel_at == "callback":
            entered.set()
            await asyncio.Event().wait()
        else:
            context.metadata["_user_message_trace_watermark"] = "turn"

    async def write(**payload):
        if payload["label"] == "user_message_trace_watermark":
            entered.set()
            await asyncio.Event().wait()

    runner.callbacks = [SimpleNamespace(on_user_message_posted=callback)]
    tracer.checkpoint.side_effect = write
    runtime = PatternRuntime(execution_id="atomic", tracer=tracer)
    runner._active_controls["atomic"] = ExecutionControl(runtime=runtime, task=None)
    operation = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="turn")
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert await runtime.should_interrupt()
        assert context.messages[-1].content == "new"
    finally:
        operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_run_completion_waits_for_in_flight_injection(live):
    runner, context, tracer = live
    running, finish, writing, drain = (asyncio.Event() for _ in range(4))

    class Pattern:
        async def run(self, **kwargs):
            running.set()
            await finish.wait()
            return {"success": True, "output": "prior output"}

    runner.agent.patterns = [Pattern()]
    operation = asyncio.create_task(
        runner.run(
            "original", execution_id="atomic", checkpoint={"context": context.to_dict()}
        )
    )
    await running.wait()

    async def write(**payload):
        writing.set()
        await drain.wait()
        tracer.load_latest_checkpoint.side_effect = RuntimeError("read unavailable")
        raise RuntimeError("lost acknowledgement")

    tracer.checkpoint.side_effect = write
    injection = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="turn")
    )
    try:
        await writing.wait()
        finish.set()
        await asyncio.sleep(0)
        assert not operation.done()
        drain.set()
        assert (await injection).outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
        result = await operation
        assert result.get("status") != "interrupted"
        assert result["injection_outcome_unknown"] is True
        assert result["success"]
        assert result["output"] == "prior output"
        assert [
            m.content for m in runner.context_manager.get_context("atomic").messages
        ].count("new") == 0
    finally:
        finish.set()
        drain.set()
        await asyncio.gather(operation, injection, return_exceptions=True)


@pytest.mark.asyncio
async def test_live_input_after_result_publication_defers_until_old_run_finishes(live):
    runner, context, tracer = live
    publishing, finish = asyncio.Event(), asyncio.Event()

    async def on_run_end(**kwargs):
        publishing.set()
        await finish.wait()

    runner.callbacks = [SimpleNamespace(on_run_end=on_run_end)]
    runner.agent.patterns = [
        SimpleNamespace(run=AsyncMock(return_value={"success": True, "output": "done"}))
    ]
    operation = asyncio.create_task(
        runner.run(
            "original", execution_id="atomic", checkpoint={"context": context.to_dict()}
        )
    )
    try:
        await publishing.wait()
        result = await runner.inject_user_message("atomic", "new", turn_id="turn")
        assert result.outcome is UserMessageInjectionOutcome.NOT_POSTED
        tracer.checkpoint.assert_not_awaited()
        finish.set()
        finished = await operation
        assert finished["success"]
        # The finished run's context was evicted; the deferred input restores
        # from the checkpoint.
        assert runner.context_manager.get_context("atomic") is None
        tracer.load_latest_checkpoint.return_value = {
            "context": finished["context"].to_dict()
        }
        # The existing deferred path posts without interrupting the finished run.
        result = await runner.inject_user_message(
            "atomic", "new", turn_id="turn", request_interrupt=False
        )
        assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    finally:
        finish.set()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_explicit_new_input_reloads_uncertain_idle_context(live):
    from xagent.core.agent.context.execution import context_checkpoint_gate

    runner, old_context, tracer = live
    durable = None

    async def write(**payload):
        nonlocal durable
        durable = payload
        tracer.load_latest_checkpoint.side_effect = RuntimeError("unavailable")
        raise RuntimeError("lost ack")

    tracer.checkpoint.side_effect = write
    first = await runner.inject_user_message(
        "atomic", "first", turn_id="first", request_interrupt=False
    )
    assert first.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    tracer.load_latest_checkpoint.side_effect = None
    tracer.load_latest_checkpoint.return_value = durable
    tracer.checkpoint.side_effect = None
    second = await runner.inject_user_message(
        "atomic", "second", turn_id="second", request_interrupt=False
    )
    assert second.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert second.context is not old_context
    assert [m.content for m in second.context.messages] == [
        "original",
        "first",
        "second",
    ]
    assert context_checkpoint_gate(old_context).injection_uncertain
    assert tracer.checkpoint.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("active_run", [False, True])
async def test_fenced_context_rejects_later_input_without_writing(live, active_run):
    from xagent.core.agent.context.execution import context_checkpoint_gate
    from xagent.core.agent.runner import track_user_message_injection

    runner, context, tracer = live

    async def write(**payload):
        tracer.load_latest_checkpoint.side_effect = RuntimeError("unavailable")
        raise RuntimeError("lost ack")

    tracer.checkpoint.side_effect = write
    first = await runner.inject_user_message("atomic", "first", turn_id="first")
    assert first.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    assert context_checkpoint_gate(context).injection_uncertain
    tracer.load_latest_checkpoint.side_effect = None
    tracer.checkpoint.side_effect = None
    if active_run:
        runner._active_controls["atomic"] = SimpleNamespace(
            runtime=SimpleNamespace(last_checkpoint=None)
        )

    with track_user_message_injection() as attempt:
        second = await runner.inject_user_message(
            "atomic",
            "second",
            turn_id="second",
            # A live interrupt is fenced even when no run is active; a deferred
            # input is fenced only while the old execution still runs.
            request_interrupt=not active_run,
        )

    assert second.outcome is UserMessageInjectionOutcome.REJECTED_RETRYABLE
    assert attempt.outcome is UserMessageInjectionOutcome.REJECTED_RETRYABLE
    # Truthy so that no caller mistakes it for a deferrable NOT_POSTED.
    assert second.outcome
    assert second.context is context
    assert runner.context_manager.get_context("atomic") is context
    assert [m.content for m in context.messages] == ["original"]
    assert context_checkpoint_gate(context).injection_uncertain
    assert tracer.checkpoint.await_count == 1


@pytest.mark.asyncio
async def test_confirmed_readback_does_not_leave_transient_unknown_result(live):
    from xagent.core.agent.context.execution import context_checkpoint_gate

    runner, context, tracer = live
    started, interrupt, reading, release = (asyncio.Event() for _ in range(4))
    payload = None

    class Pattern:
        async def run(self, **kwargs):
            started.set()
            await interrupt.wait()
            raise ExecutionInterrupted("user pause")

    runner.agent.patterns = [Pattern()]
    operation = asyncio.create_task(
        runner.run(
            "original", execution_id="atomic", checkpoint={"context": context.to_dict()}
        )
    )
    await started.wait()

    async def read(_):
        reading.set()
        await release.wait()
        return payload

    async def write(**candidate):
        nonlocal payload
        payload = candidate
        tracer.load_latest_checkpoint.side_effect = read
        raise RuntimeError("lost ack")

    tracer.checkpoint.side_effect = write
    injection = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="new")
    )
    try:
        await reading.wait()
        interrupt.set()
        gate = context_checkpoint_gate(runner.context_manager.get_context("atomic"))

        async def finalizer_waiting():
            while not gate._waiters:
                await asyncio.sleep(0)

        await asyncio.wait_for(finalizer_waiting(), 2)
        release.set()
        assert (await injection).outcome is UserMessageInjectionOutcome.POSTED_FRESH
        assert (await operation)["injection_outcome_unknown"] is False
    finally:
        release.set()
        interrupt.set()
        await asyncio.gather(operation, injection, return_exceptions=True)


@pytest.mark.asyncio
async def test_exceptional_run_exit_closes_live_admission(live):
    from xagent.core.agent.checkpoint import CheckpointPersistenceError

    runner, context, tracer = live

    class Pattern:
        async def run(self, **kwargs):
            raise CheckpointPersistenceError("failed checkpoint")

    runner.agent.patterns = [Pattern()]
    with pytest.raises(CheckpointPersistenceError):
        await runner.run(
            "original", execution_id="atomic", checkpoint={"context": context.to_dict()}
        )
    posted = await runner.inject_user_message("atomic", "late", turn_id="late")
    assert posted.outcome is UserMessageInjectionOutcome.NOT_POSTED
    tracer.checkpoint.assert_not_awaited()


@pytest.mark.asyncio
async def test_acceptance_evidence_survives_postwrite_registry_error(live):
    from xagent.core.agent.registry import ExecutionRegistry
    from xagent.core.agent.runner import track_user_message_injection

    runner, context, tracer = live
    registry = ExecutionRegistry()
    registry.register("atomic", runner)
    token = registry.subscribe(lambda _: registry.unsubscribe(token))
    with track_user_message_injection() as attempt:
        with pytest.raises(RuntimeError, match="dictionary changed"):
            await registry.post_user_message("atomic", "new", turn_id="new")
    assert attempt.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert context.messages[-1].content == "new"


@pytest.mark.asyncio
@pytest.mark.parametrize("after_write", [False, True])
async def test_cancellation_evidence_comes_from_write_boundary(live, after_write):
    from xagent.core.agent.runner import track_user_message_injection

    runner, context, tracer = live
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    if after_write:
        tracer.checkpoint.side_effect = blocked
    else:
        tracer.load_latest_checkpoint.side_effect = blocked
    with track_user_message_injection() as attempt:
        operation = asyncio.create_task(
            runner.inject_user_message("atomic", "new", turn_id="new")
        )
        await entered.wait()
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
    assert attempt.outcome is (
        UserMessageInjectionOutcome.OUTCOME_UNKNOWN
        if after_write
        else UserMessageInjectionOutcome.NOT_POSTED
    )


@pytest.mark.asyncio
async def test_write_only_checkpoint_wrapper_cannot_claim_authoritative_absence(live):
    from xagent.core.agent.checkpoint import (
        CheckpointUnavailableError,
        TraceCheckpointStore,
    )

    runner, context, tracer = live
    backend = SimpleNamespace(checkpoint=AsyncMock())
    runner.tracer = TraceCheckpointStore(backend)
    with pytest.raises(CheckpointUnavailableError, match="no readable backend"):
        await runner.inject_user_message("atomic", "new", turn_id="new")
    backend.checkpoint.assert_not_awaited()


_O = UserMessageInjectionOutcome


@pytest.mark.parametrize(
    "attempt,posted,error,expected",
    [
        # Returned outcomes.
        (_O.POSTED_FRESH, _O.POSTED_FRESH, None, "ACCEPTED"),
        (_O.POSTED_REPLAY, _O.POSTED_REPLAY, None, "ACCEPTED"),
        (_O.OUTCOME_UNKNOWN, _O.OUTCOME_UNKNOWN, None, "UNKNOWN"),
        (_O.REJECTED_RETRYABLE, _O.REJECTED_RETRYABLE, None, "NOT_ACCEPTED_RETRYABLE"),
        (_O.NOT_POSTED, _O.NOT_POSTED, None, "DEFER"),
        # A service layer that forwards the value without recording evidence.
        (_O.NOT_POSTED, _O.POSTED_FRESH, None, "ACCEPTED"),
        (_O.NOT_POSTED, None, None, "DEFER"),
        # Recorded acceptance wins over any later error, including cancellation.
        (_O.POSTED_FRESH, None, asyncio.CancelledError(), "ACCEPTED"),
        (_O.POSTED_REPLAY, None, RuntimeError("registry event"), "ACCEPTED"),
        (_O.OUTCOME_UNKNOWN, None, asyncio.CancelledError(), "UNKNOWN"),
        (_O.OUTCOME_UNKNOWN, None, RuntimeError("late"), "UNKNOWN"),
        (
            _O.NOT_POSTED,
            None,
            UserMessageInjectionRejectedError("absent"),
            "NOT_ACCEPTED_RETRYABLE",
        ),
        (_O.REJECTED_RETRYABLE, None, RuntimeError("lease"), "NOT_ACCEPTED_RETRYABLE"),
        (_O.NOT_POSTED, None, RuntimeError("read failed"), "FAILED_BEFORE_WRITE"),
        (_O.NOT_POSTED, None, asyncio.CancelledError(), "FAILED_BEFORE_WRITE"),
        (
            _O.NOT_POSTED,
            None,
            UserMessageInjectionConflictError("conflict"),
            "FAILED_BEFORE_WRITE",
        ),
    ],
)
def test_classify_injection_rules(attempt, posted, error, expected):
    from xagent.core.agent.runner import InjectionDisposition, classify_injection

    assert classify_injection(attempt, posted=posted, error=error) is getattr(
        InjectionDisposition, expected
    )


def test_classify_injection_rejects_untyped_outcome():
    from xagent.core.agent.runner import classify_injection

    with pytest.raises(TypeError):
        classify_injection(_O.NOT_POSTED, posted=False)


@pytest.mark.asyncio
async def test_real_runner_outcomes_classify_at_the_write_boundary(live):
    """The classifier agrees with what the runner actually records."""
    from xagent.core.agent.runner import (
        InjectionDisposition,
        classify_injection,
        track_user_message_injection,
    )

    runner, context, tracer = live
    # Confirmed absence: the write failed and the read-back shows no turn.
    tracer.checkpoint.side_effect = RuntimeError("write failed")
    with track_user_message_injection() as attempt:
        with pytest.raises(UserMessageInjectionRejectedError) as rejected:
            await runner.inject_user_message("atomic", "new", turn_id="absent")
    assert (
        classify_injection(attempt.outcome, error=rejected.value)
        is InjectionDisposition.NOT_ACCEPTED_RETRYABLE
    )

    # Accepted, then cancelled while tracing the accepted turn.
    tracer.checkpoint.side_effect = None

    async def cancelled_callback(**kwargs):
        raise asyncio.CancelledError()

    runner.callbacks = [SimpleNamespace(on_user_message_posted=cancelled_callback)]
    with track_user_message_injection() as attempt:
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await runner.inject_user_message("atomic", "new", turn_id="accepted")
    assert context.messages[-1].content == "new"
    assert (
        classify_injection(attempt.outcome, error=cancelled.value)
        is InjectionDisposition.ACCEPTED
    )


def _cold_checkpoint(execution_id: str | None) -> dict:
    from xagent.core.agent.context import ExecutionContext

    context = ExecutionContext(execution_id=execution_id or "unused")
    context.add_user_message("original")
    data = context.to_dict()
    if execution_id is None:
        data.pop("execution_id")
    return {"context": data}


def _yielding_reader(checkpoint: dict) -> AsyncMock:
    async def read(_execution_id):
        # Yield so a regression back into the lookup loop times out cleanly.
        await asyncio.sleep(0)
        return checkpoint

    return AsyncMock(side_effect=read)


@pytest.mark.asyncio
async def test_cold_start_adopts_the_lookup_id_when_the_checkpoint_has_none():
    manager = ContextManager()
    tracer = SimpleNamespace(
        load_latest_checkpoint=_yielding_reader(_cold_checkpoint(None)),
        checkpoint=AsyncMock(),
    )
    runner = AgentRunner(
        SimpleNamespace(llm=None), tracer=tracer, context_manager=manager
    )
    result = await asyncio.wait_for(
        runner.inject_user_message(
            "cold", "new", turn_id="turn", request_interrupt=False
        ),
        5,
    )
    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert result.context.execution_id == "cold"
    assert manager.get_context("cold") is None


@pytest.mark.asyncio
async def test_cold_start_without_an_id_still_drops_a_derived_output_language():
    from xagent.core.agent.language import (
        OUTPUT_LANGUAGE_METADATA_KEY,
        OUTPUT_LANGUAGE_SOURCE_METADATA_KEY,
    )

    checkpoint = _cold_checkpoint(None)
    checkpoint["context"]["metadata"] = {
        OUTPUT_LANGUAGE_METADATA_KEY: "fr",
        OUTPUT_LANGUAGE_SOURCE_METADATA_KEY: "detected",
    }
    manager = ContextManager()
    tracer = SimpleNamespace(
        load_latest_checkpoint=_yielding_reader(checkpoint),
        checkpoint=AsyncMock(),
    )
    runner = AgentRunner(
        SimpleNamespace(llm=None), tracer=tracer, context_manager=manager
    )
    result = await asyncio.wait_for(
        runner.inject_user_message(
            "cold", "new", turn_id="turn", request_interrupt=False
        ),
        5,
    )
    # The id is filled in on a copy; the migration must still reach it.
    assert result.context.execution_id == "cold"
    assert OUTPUT_LANGUAGE_METADATA_KEY not in result.context.metadata
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in result.context.metadata


@pytest.mark.asyncio
async def test_cold_start_rejects_a_checkpoint_of_another_execution():
    from xagent.core.agent.checkpoint import CheckpointCorruptError

    manager = ContextManager()
    tracer = SimpleNamespace(
        load_latest_checkpoint=_yielding_reader(_cold_checkpoint("other")),
        checkpoint=AsyncMock(),
    )
    runner = AgentRunner(
        SimpleNamespace(llm=None), tracer=tracer, context_manager=manager
    )
    with pytest.raises(CheckpointCorruptError, match="different execution"):
        await asyncio.wait_for(
            runner.inject_user_message("cold", "new", turn_id="turn"), 5
        )
    tracer.checkpoint.assert_not_awaited()


@pytest.mark.asyncio
async def test_write_only_tracer_cannot_prove_a_failed_write_absent():
    from xagent.core.agent.checkpoint import TraceCheckpointStore
    from xagent.core.agent.trace import Tracer

    manager = ContextManager()
    context = manager.create_context("write-only")
    context.add_user_message("original")

    class FailingWriter:
        async def handle_event(self, event):
            raise RuntimeError("write outcome unknown")

    tracer = Tracer()
    tracer.add_handler(FailingWriter())
    runner = AgentRunner(
        SimpleNamespace(llm=None),
        tracer=TraceCheckpointStore(tracer),
        context_manager=manager,
    )
    # Tracer exposes load_latest_checkpoint but no handler can read, so its
    # empty answer must not become "confirmed absent".
    result = await runner.inject_user_message("write-only", "new", turn_id="t")
    assert result.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    assert [m.content for m in context.messages] == ["original"]


@pytest.mark.asyncio
async def test_lost_baseline_on_readback_is_uncertain_not_absent(live):
    runner, context, tracer = live
    tracer.load_latest_checkpoint.return_value = {"context": context.to_dict()}

    async def write(**payload):
        # The store answers, but the checkpoint read before the write is gone.
        tracer.load_latest_checkpoint.return_value = None
        raise RuntimeError("write outcome unknown")

    tracer.checkpoint.side_effect = write
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    assert [m.content for m in context.messages] == ["original"]


# --- Context cache lifetime -------------------------------------------------


class _DurableStore:
    """Checkpoint store whose reads return copies of the latest write."""

    def __init__(self) -> None:
        self.latest: dict | None = None
        self.writes: list[str] = []
        self.reads = 0
        self.read_hook = None
        self.fail_labels: set[str] = set()
        self.ambiguous_labels: set[str] = set()
        self.reads_fail = False

    async def load_latest_checkpoint(self, execution_id):
        self.reads += 1
        if self.reads_fail:
            raise RuntimeError("read unavailable")
        snapshot = copy.deepcopy(self.latest)
        if self.read_hook is not None:
            hook, self.read_hook = self.read_hook, None
            return await hook(snapshot)
        return snapshot

    async def checkpoint(self, **payload):
        label = payload["label"]
        if label in self.fail_labels:
            raise RuntimeError(f"{label} write failed")
        if label in self.ambiguous_labels:
            # The write lands but its acknowledgement and read-back are lost.
            self.latest = copy.deepcopy(payload)
            self.reads_fail = True
            raise RuntimeError("ack lost")
        self.writes.append(label)
        self.latest = copy.deepcopy(payload)

    def seed(self, execution_id: str, *messages: str) -> None:
        from xagent.core.agent.context import ExecutionContext

        context = ExecutionContext(execution_id=execution_id)
        for message in messages:
            context.add_user_message(message)
        self.latest = {"type": "checkpoint", "context": context.to_dict()}

    def messages(self) -> list[str]:
        assert self.latest is not None
        return [m["content"] for m in self.latest["context"]["messages"]]


class _AnsweringPattern:
    """Checkpoints like a real pattern, then returns an answer the runner appends."""

    def __init__(self, answer: str = "A answer", *, before_return=None) -> None:
        self.answer = answer
        self.before_return = before_return

    def get_state(self):
        return {"marker": "state"}

    async def run(self, *, context, runtime, **kwargs):
        await runtime.checkpoint("step", context=context, pattern=self)
        if self.before_return is not None:
            await self.before_return()
        return {"success": True, "output": self.answer}


@pytest.fixture
def durable():
    store = _DurableStore()
    runner = AgentRunner(
        SimpleNamespace(llm=None),
        tracer=store,
        context_manager=ContextManager(),
        workspace_enabled=False,
    )
    yield runner, store


@pytest.mark.asyncio
async def test_follow_up_after_another_process_extended_the_checkpoint(durable):
    from xagent.core.agent.context import ExecutionContext

    runner, store = durable
    runner.agent.patterns = [_AnsweringPattern()]
    assert (await runner.run("m1", execution_id="aba"))["success"]

    # Process B takes the task over, answers another turn and checkpoints.
    other = ExecutionContext.from_dict(store.latest["context"])
    other.add_user_message("m2 from B")
    other.add_assistant_message("B answer")
    store.latest = {**store.latest, "context": other.to_dict()}

    posted = await runner.inject_user_message(
        "aba", "m3", turn_id="m3", request_interrupt=False
    )
    assert posted.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert store.messages() == ["m1", "A answer", "m2 from B", "B answer", "m3"]
    assert runner.context_manager.get_context("aba") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["memory_session", "resume_workspace"])
async def test_setup_failure_before_the_run_leaves_no_cached_context(durable, stage):
    from xagent.core.agent.context import ExecutionContext

    runner, store = durable
    runner.agent.patterns = [_AnsweringPattern()]
    kwargs = {}
    if stage == "memory_session":
        runner.memory_manager = SimpleNamespace(
            get_or_create_session=AsyncMock(side_effect=RuntimeError("setup failed"))
        )
    else:
        runner.workspace_enabled = True
        runner.workspace_manager = SimpleNamespace(
            get_or_create_workspace=AsyncMock(side_effect=RuntimeError("setup failed"))
        )
        seed = ExecutionContext(execution_id="setup")
        seed.add_user_message("m1")
        kwargs["checkpoint"] = {"context": seed.to_dict()}
    with pytest.raises(RuntimeError, match="setup failed"):
        await runner.run("m1", execution_id="setup", **kwargs)
    assert runner.context_manager.get_context("setup") is None


@pytest.mark.asyncio
async def test_live_input_without_a_local_run_defers_and_writes_nothing(durable):
    runner, store = durable
    store.seed("cold", "m1")
    posted = await runner.inject_user_message("cold", "live", turn_id="live")
    assert posted.outcome is UserMessageInjectionOutcome.NOT_POSTED
    assert store.writes == []
    assert runner.context_manager.get_context("cold") is None


@pytest.mark.asyncio
async def test_deferred_cold_start_leaves_the_cache_empty(durable):
    runner, store = durable
    store.seed("cold", "m1")
    posted = await runner.inject_user_message(
        "cold", "later", turn_id="later", request_interrupt=False
    )
    assert posted.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert store.messages() == ["m1", "later"]
    assert runner.context_manager.get_context("cold") is None


@pytest.mark.asyncio
async def test_fenced_run_keeps_its_context_until_deferred_input_reloads(durable):
    runner, store = durable
    started, release = asyncio.Event(), asyncio.Event()

    async def wait():
        started.set()
        await release.wait()

    runner.agent.patterns = [_AnsweringPattern(before_return=wait)]
    operation = asyncio.create_task(runner.run("m1", execution_id="fence"))
    try:
        await started.wait()
        store.ambiguous_labels.add("user_message_injected")
        first = await runner.inject_user_message("fence", "uncertain", turn_id="u")
        assert first.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
        store.ambiguous_labels.clear()
        store.reads_fail = False
        release.set()
        result = await operation
    finally:
        release.set()
        await asyncio.gather(operation, return_exceptions=True)
    assert result["injection_outcome_unknown"] is True
    # The fenced context never writes again, not even the tail checkpoint.
    assert "run_end_tail" not in store.writes
    fenced = runner.context_manager.get_context("fence")
    assert fenced is result["context"]

    live = await runner.inject_user_message("fence", "live", turn_id="live")
    assert live.outcome is UserMessageInjectionOutcome.REJECTED_RETRYABLE
    assert runner.context_manager.get_context("fence") is fenced

    deferred = await runner.inject_user_message(
        "fence", "deferred", turn_id="deferred", request_interrupt=False
    )
    assert deferred.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert deferred.context is not fenced
    assert store.messages() == ["m1", "uncertain", "deferred"]
    assert runner.context_manager.get_context("fence") is None


@pytest.mark.asyncio
async def test_context_stays_cached_while_an_injection_outlives_the_run(durable):
    runner, store = durable
    started, release_run = asyncio.Event(), asyncio.Event()
    in_callback, release_callback = asyncio.Event(), asyncio.Event()

    async def wait():
        started.set()
        await release_run.wait()

    async def on_user_message_posted(*, message, **_):
        if message.content == "I":
            in_callback.set()
            await release_callback.wait()

    runner.agent.patterns = [_AnsweringPattern(before_return=wait)]
    runner.callbacks = [SimpleNamespace(on_user_message_posted=on_user_message_posted)]
    operation = asyncio.create_task(runner.run("m1", execution_id="held"))
    injection = None
    try:
        await started.wait()
        injection = asyncio.create_task(
            runner.inject_user_message("held", "I", turn_id="I")
        )
        await in_callback.wait()
        release_run.set()
        result = await operation
        held = runner.context_manager.get_context("held")
        assert held is result["context"]

        later = await runner.inject_user_message(
            "held", "J", turn_id="J", request_interrupt=False
        )
        assert later.outcome is UserMessageInjectionOutcome.POSTED_FRESH
        assert later.context is held
        release_callback.set()
        assert (await injection).outcome is UserMessageInjectionOutcome.POSTED_FRESH
    finally:
        release_run.set()
        release_callback.set()
        await asyncio.gather(
            operation, *([injection] if injection else []), return_exceptions=True
        )
    assert store.messages() == ["m1", "I", "A answer", "J"]
    assert runner.context_manager.get_context("held") is None


@pytest.mark.asyncio
async def test_cold_start_reader_racing_an_eviction_reads_again(durable):
    runner, store = durable
    store.seed("race", "m1")
    reading, release = asyncio.Event(), asyncio.Event()

    async def stale_read(snapshot):
        reading.set()
        await release.wait()
        return snapshot

    store.read_hook = stale_read
    reader = asyncio.create_task(
        runner.inject_user_message("race", "R", turn_id="R", request_interrupt=False)
    )
    try:
        await reading.wait()
        other = await runner.inject_user_message(
            "race", "J", turn_id="J", request_interrupt=False
        )
        assert other.outcome is UserMessageInjectionOutcome.POSTED_FRESH
        assert runner.context_manager.get_context("race") is None
        release.set()
        assert (await reader).outcome is UserMessageInjectionOutcome.POSTED_FRESH
    finally:
        release.set()
        await asyncio.gather(reader, return_exceptions=True)
    # R discarded the snapshot it read before J's write and read again.
    assert store.reads == 3
    assert store.messages() == ["m1", "J", "R"]
    assert runner.context_manager.get_context("race") is None
    assert runner.context_manager._cold_starts == {}


@pytest.mark.asyncio
async def test_completed_run_persists_the_answer_appended_after_its_checkpoint(
    durable,
):
    runner, store = durable
    runner.agent.patterns = [_AnsweringPattern()]
    result = await runner.run("m1", execution_id="tail")
    assert result["success"]
    assert store.writes == ["step", "run_end_tail"]
    assert store.messages() == ["m1", "A answer"]
    # The pattern state of its last checkpoint is carried over unchanged.
    assert store.latest["pattern"] == "_AnsweringPattern"
    assert store.latest["pattern_state"] == {"marker": "state"}

    posted = await runner.inject_user_message(
        "tail", "next", turn_id="next", request_interrupt=False
    )
    assert posted.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert store.messages() == ["m1", "A answer", "next"]


class _SelfRecordingPattern:
    def __init__(self, *, checkpoint: bool) -> None:
        self.checkpoint = checkpoint

    async def run(self, *, context, runtime, **kwargs):
        if self.checkpoint:
            context.add_assistant_message("recorded")
            await runtime.checkpoint("final", context=context, pattern=self)
        return {"success": True, "output": "recorded"}


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpointed", [True, False])
async def test_tail_checkpoint_is_skipped_without_new_state(durable, checkpointed):
    runner, store = durable
    runner.agent.patterns = [_SelfRecordingPattern(checkpoint=checkpointed)]
    assert (await runner.run("m1", execution_id="same"))["success"]
    # Unchanged after the pattern's checkpoint, or no checkpoint in this run.
    assert store.writes == (["final"] if checkpointed else [])


@pytest.mark.asyncio
async def test_tail_checkpoint_failure_keeps_the_result(durable, caplog):
    runner, store = durable
    runner.agent.patterns = [_AnsweringPattern()]
    store.fail_labels.add("run_end_tail")
    with caplog.at_level("WARNING", logger="xagent.core.agent.runner"):
        result = await runner.run("m1", execution_id="tail-fails")
    assert result["success"] is True
    assert result["output"] == "A answer"
    assert store.writes == ["step"]
    assert "Final context checkpoint failed for tail-fails" in caplog.text
    assert runner.context_manager.get_context("tail-fails") is None


@pytest.mark.asyncio
async def test_tail_fact_failure_aborts_completion(durable, monkeypatch):
    from xagent.core.agent.checkpoint import ExecutionEventPersistenceError

    runner, store = durable
    runner.agent.patterns = [_AnsweringPattern()]
    on_run_end = AsyncMock()
    runner.callbacks = [SimpleNamespace(on_run_end=on_run_end)]
    checkpoint = store.checkpoint

    async def fail_tail(**payload):
        if payload["label"] == "run_end_tail":
            raise ExecutionEventPersistenceError("fact write failed")
        await checkpoint(**payload)

    monkeypatch.setattr(store, "checkpoint", fail_tail)
    with pytest.raises(ExecutionEventPersistenceError, match="fact write failed"):
        await runner.run("m1", execution_id="tail-fact-fails")
    on_run_end.assert_not_awaited()
    assert store.writes == ["step"]
    assert runner.context_manager.get_context("tail-fact-fails") is None


@pytest.mark.asyncio
async def test_waiting_result_is_not_tail_persisted_so_resume_keeps_waiting(durable):
    from xagent.core.agent import ReActPattern

    runner, store = durable

    class FakeLLM:
        def __init__(self, responses):
            self.responses = responses
            self.calls = []

        async def chat(self, **kwargs):
            self.calls.append(kwargs)
            return self.responses.pop(0)

    llm = FakeLLM(
        [
            {
                "tool_calls": [
                    {
                        "id": "call-send",
                        "function": {
                            "name": "send_message",
                            "arguments": (
                                '{"message":"Choose A or B",'
                                '"message_type":"question","expect_response":true}'
                            ),
                        },
                    }
                ]
            }
        ]
    )
    runner.agent = SimpleNamespace(llm=llm, patterns=[ReActPattern(max_iterations=2)])
    first = await runner.run("Ask", execution_id="waiting")
    assert first["status"] == "waiting_for_user"
    # The runner appended the question in memory only; ReAct's checkpoint
    # records the message count a reply must exceed.
    assert "run_end_tail" not in store.writes

    resumed = await runner.resume("waiting")
    assert resumed["status"] == "waiting_for_user"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_watermark_repersist_skips_a_context_replaced_meanwhile(durable):
    from xagent.core.agent.context import ExecutionContext

    runner, store = durable
    store.seed("replaced", "m1")
    in_callback, release = asyncio.Event(), asyncio.Event()

    async def on_user_message_posted(*, context, **_):
        in_callback.set()
        await release.wait()
        context.metadata["_user_message_trace_watermark"] = "traced"

    runner.callbacks = [SimpleNamespace(on_user_message_posted=on_user_message_posted)]
    injection = asyncio.create_task(
        runner.inject_user_message(
            "replaced", "I", turn_id="I", request_interrupt=False
        )
    )
    try:
        await in_callback.wait()
        # A resumed run caches a newer context and checkpoints past I.
        newer = ExecutionContext.from_dict(store.latest["context"])
        newer.add_assistant_message("resumed answer")
        runner.context_manager.set_context(newer)
        store.latest = {**store.latest, "context": newer.to_dict()}
        release.set()
        result = await injection
    finally:
        release.set()
        await asyncio.gather(injection, return_exceptions=True)
    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert store.writes == ["user_message_injected"]
    assert store.messages() == ["m1", "I", "resumed answer"]
    assert runner.context_manager.get_context("replaced") is newer
    runner.context_manager.remove_context("replaced")
