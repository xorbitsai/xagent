"""Checkpoint coordination preserves ordinary parallel snapshot writes."""

import asyncio
import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from xagent.core.agent.checkpoint import CheckpointPersistenceError
from xagent.core.agent.context import ExecutionContext
from xagent.core.agent.context.execution import context_checkpoint_gate
from xagent.core.agent.runtime import PatternRuntime


@pytest.mark.asyncio
async def test_regular_checkpoints_overlap_for_one_context() -> None:
    entered = set()
    both_entered = asyncio.Event()
    release = asyncio.Event()

    async def persist(**payload):
        entered.add(payload["label"])
        if len(entered) == 2:
            both_entered.set()
        await release.wait()

    context = ExecutionContext(execution_id="parallel")
    runtime = PatternRuntime(tracer=SimpleNamespace(checkpoint=persist))
    writes = [
        asyncio.create_task(runtime.checkpoint(label, context=context, pattern=None))
        for label in ("first", "second")
    ]
    try:
        await asyncio.wait_for(both_entered.wait(), 2)
        assert entered == {"first", "second"}
        release.set()
        assert len(await asyncio.gather(*writes)) == 2
    finally:
        for task in writes:
            task.cancel()
        await asyncio.gather(*writes, return_exceptions=True)


@pytest.mark.asyncio
async def test_exclusive_waits_for_writes_and_blocks_new_snapshots() -> None:
    context = ExecutionContext(execution_id="ordered")
    gate = context_checkpoint_gate(context)
    runtime = PatternRuntime()
    order = []
    writer_started = asyncio.Event()
    writer_entered = asyncio.Event()
    release_writer = asyncio.Event()

    class Pattern:
        def get_state(self):
            order.append("snapshot")
            return {}

    async def write():
        writer_started.set()
        async with gate.exclusive():
            order.append("exclusive")
            writer_entered.set()
            await release_writer.wait()

    async with gate.shared():
        writer = asyncio.create_task(write())
        await writer_started.wait()
        later = asyncio.create_task(
            runtime.checkpoint("later", context=context, pattern=Pattern())
        )
        await asyncio.sleep(0)
        assert not writer_entered.is_set()
        assert not order
    try:
        await asyncio.wait_for(writer_entered.wait(), 2)
        assert order == ["exclusive"]
        context.add_user_message("committed while exclusive")
        release_writer.set()
        await writer
        payload = await asyncio.wait_for(later, 2)
        assert order == ["exclusive", "snapshot"]
        assert (
            payload["context"]["messages"][0]["content"] == "committed while exclusive"
        )
    finally:
        writer.cancel()
        later.cancel()
        await asyncio.gather(writer, later, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_waiting_writer_unblocks_shared_without_draining_readers() -> (
    None
):
    gate = context_checkpoint_gate(ExecutionContext())
    started = asyncio.Event()

    async def writer():
        started.set()
        async with gate.exclusive():
            pytest.fail("writer must remain blocked")

    async def reader():
        async with gate.shared():
            return "read"

    async with gate.shared():
        waiting = asyncio.create_task(writer())
        await started.wait()
        incoming = asyncio.create_task(reader())
        await asyncio.sleep(0)
        assert not incoming.done()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert await asyncio.wait_for(incoming, 2) == "read"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["shared", "exclusive"])
async def test_cancelled_notified_waiter_does_not_lose_wakeup(mode: str) -> None:
    gate = context_checkpoint_gate(ExecutionContext())
    started = [asyncio.Event(), asyncio.Event()]
    entered = []

    async def enter(index):
        started[index].set()
        async with getattr(gate, mode)():
            entered.append(index)

    async with gate.exclusive():
        waiting = [asyncio.create_task(enter(i)) for i in range(2)]
        await asyncio.gather(*(event.wait() for event in started))
        assert not entered
    # Release wakes both, but neither has run yet. Cancel one notified waiter.
    waiting[0].cancel()
    results = await asyncio.wait_for(
        asyncio.gather(*waiting, return_exceptions=True), 2
    )
    assert isinstance(results[0], asyncio.CancelledError)
    assert entered == [1]
    async with gate.exclusive():
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["shared", "exclusive"])
async def test_cancelled_holder_releases_gate(mode: str) -> None:
    gate = context_checkpoint_gate(ExecutionContext())
    entered = asyncio.Event()

    async def hold():
        async with getattr(gate, mode)():
            entered.set()
            await asyncio.Event().wait()

    async def next_writer():
        async with gate.exclusive():
            return "done"

    holder = asyncio.create_task(hold())
    await entered.wait()
    waiting = asyncio.create_task(next_writer())
    await asyncio.sleep(0)
    assert not waiting.done()
    holder.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holder
    assert await asyncio.wait_for(waiting, 2) == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_checkpoint_failure_releases_shared_gate(cancel: bool) -> None:
    entered = asyncio.Event()

    async def persist(**payload):
        entered.set()
        if cancel:
            await asyncio.Event().wait()
        raise RuntimeError("write failed")

    context = ExecutionContext()
    runtime = PatternRuntime(tracer=SimpleNamespace(checkpoint=persist))
    write = asyncio.create_task(
        runtime.checkpoint("failure", context=context, pattern=None)
    )
    await entered.wait()
    if cancel:
        write.cancel()
    with pytest.raises(
        asyncio.CancelledError if cancel else CheckpointPersistenceError
    ):
        await write
    # Preserve existing checkpoint cache behavior even on writer failure.
    assert runtime.last_checkpoint is runtime.checkpoints[0]
    async with context_checkpoint_gate(context).exclusive():
        pass


@pytest.mark.asyncio
async def test_context_copies_and_serialization_do_not_share_active_gate() -> None:
    context = ExecutionContext(metadata={"nested": [1]})
    payload, fields, representation = context.to_dict(), asdict(context), repr(context)
    gate = context_checkpoint_gate(context)
    async with gate.exclusive():
        shallow = copy.copy(context)
        deep = copy.deepcopy(context)
        for copied in (shallow, deep):
            assert copied == context
            copied_gate = context_checkpoint_gate(copied)
            assert copied_gate is not gate
            async with asyncio.timeout(2):
                async with copied_gate.exclusive():
                    pass
        assert shallow.metadata is context.metadata
        assert deep.metadata is not context.metadata
        assert deep.metadata["nested"] is not context.metadata["nested"]
    assert context.to_dict() == payload
    assert asdict(context) == fields
    assert repr(context) == representation
    assert context_checkpoint_gate(context) is gate
    context.metadata["self"] = context
    recursive = copy.deepcopy(context)
    assert recursive.metadata["self"] is recursive


def test_idle_context_gate_can_be_reused_in_a_new_loop() -> None:
    context = ExecutionContext()

    async def checkpoint():
        gate = context_checkpoint_gate(context)
        async with gate.shared():
            pass
        async with gate.exclusive():
            pass

    asyncio.run(checkpoint())
    asyncio.run(checkpoint())


@pytest.mark.asyncio
async def test_checkpoint_without_execution_context_keeps_existing_payload() -> None:
    runtime = PatternRuntime(execution_id="generic")
    payload = await runtime.checkpoint("empty", context=None, pattern=None)
    assert payload["execution_id"] == "generic"
    assert payload["context"] is None


@pytest.mark.asyncio
async def test_failing_stall_timer_still_releases_the_gate() -> None:
    gate = context_checkpoint_gate(ExecutionContext(execution_id="timer-fails"))
    loop = asyncio.get_running_loop()

    def refuse(*_args, **_kwargs):
        raise RuntimeError("timer unavailable")

    loop.call_later = refuse  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="timer unavailable"):
            async with gate.exclusive("timer-fails"):
                pytest.fail("the section must not run without its stall timer")
    finally:
        del loop.call_later

    async def acquire_both() -> None:
        async with gate.shared():
            pass
        async with gate.exclusive():
            pass

    # Released: both kinds of acquisition proceed instead of waiting forever.
    await asyncio.wait_for(acquire_both(), 1)


@pytest.mark.asyncio
async def test_long_exclusive_hold_is_reported_but_never_interrupted(
    monkeypatch, caplog
) -> None:
    from xagent.core.agent.context import execution

    stalls: list[str] = []
    durations: list[tuple[str, float]] = []
    monkeypatch.setenv("XAGENT_CHECKPOINT_GATE_STALL_WARNING_SECONDS", "0.01")
    monkeypatch.setattr(execution, "increment_counter", stalls.append)
    monkeypatch.setattr(
        execution,
        "observe_value",
        lambda metric, value, **kwargs: durations.append((metric, value)),
    )
    gate = context_checkpoint_gate(ExecutionContext(execution_id="stalled"))

    async def reported_twice() -> None:
        while len(stalls) < 2:
            await asyncio.sleep(0.005)

    with caplog.at_level("WARNING", logger=execution.__name__):
        async with gate.exclusive("stalled"):
            await asyncio.wait_for(reported_twice(), 5)
        reported = len(stalls)
        await asyncio.sleep(0.05)

    # Released: the repeating report is cancelled, and the hold is measured.
    assert len(stalls) == reported
    assert set(stalls) == {"xagent.agent.checkpoint_gate.exclusive.stalls"}
    assert "Exclusive checkpoint section for stalled still held" in caplog.text
    assert [metric for metric, _ in durations] == [
        "xagent.agent.checkpoint_gate.exclusive.duration"
    ]
    assert durations[0][1] > 0
