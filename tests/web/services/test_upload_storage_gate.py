"""Concurrency and cancellation contracts for durable-upload admission."""

import asyncio

import pytest

from xagent.config import get_file_upload_max_concurrency
from xagent.web.services.upload_storage_gate import (
    UploadStorageCapacityError,
    UploadStorageGate,
    get_upload_storage_gate,
)


def test_accessor_survives_sequential_contended_event_loops() -> None:
    capacity = get_file_upload_max_concurrency()
    gates: list[UploadStorageGate] = []

    async def contend_for_gate() -> None:
        gate = get_upload_storage_gate()
        gates.append(gate)
        release = asyncio.Event()
        capacity_reached = asyncio.Event()
        active = 0

        async def hold_capacity() -> None:
            nonlocal active
            async with gate.lease():
                active += 1
                if active == capacity:
                    capacity_reached.set()
                await release.wait()
                active -= 1

        holders = [asyncio.create_task(hold_capacity()) for _ in range(capacity)]
        await asyncio.wait_for(capacity_reached.wait(), timeout=1)

        async def wait_for_capacity() -> None:
            async with gate.lease():
                pass

        waiter = asyncio.create_task(wait_for_capacity())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert gate.active == capacity
        assert not waiter.done()
        release.set()
        await asyncio.gather(*holders, waiter)
        assert gate.active == 0

    asyncio.run(contend_for_gate())
    asyncio.run(contend_for_gate())

    assert gates[0] is gates[1]


@pytest.mark.asyncio
async def test_accessor_rejects_a_second_live_event_loop() -> None:
    gate = get_upload_storage_gate()

    async def access_from_another_loop() -> None:
        with pytest.raises(
            RuntimeError,
            match="does not support concurrent event loops",
        ):
            get_upload_storage_gate()

    await asyncio.to_thread(asyncio.run, access_from_another_loop())

    assert get_upload_storage_gate() is gate


@pytest.mark.asyncio
async def test_gate_bounds_concurrent_leases() -> None:
    gate = UploadStorageGate(max_concurrency=2, queue_timeout_seconds=1)
    active = 0
    maximum_active = 0
    release = asyncio.Event()
    first_pair_active = asyncio.Event()

    async def use_gate() -> None:
        nonlocal active, maximum_active
        async with gate.lease():
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                first_pair_active.set()
            await release.wait()
            active -= 1

    tasks = [asyncio.create_task(use_gate()) for _ in range(8)]
    await asyncio.wait_for(first_pair_active.wait(), timeout=1)

    assert maximum_active == 2
    assert gate.active == 2

    release.set()
    await asyncio.gather(*tasks)
    assert gate.active == 0


@pytest.mark.asyncio
async def test_gate_times_out_without_consuming_capacity() -> None:
    gate = UploadStorageGate(max_concurrency=1, queue_timeout_seconds=0.01)

    async with gate.lease():
        with pytest.raises(UploadStorageCapacityError):
            async with gate.lease():
                raise AssertionError("timed-out lease must not enter")
        assert gate.active == 1

    async with gate.lease():
        assert gate.active == 1


@pytest.mark.asyncio
async def test_gate_releases_capacity_when_lease_owner_is_cancelled() -> None:
    gate = UploadStorageGate(max_concurrency=1, queue_timeout_seconds=1)
    acquired = asyncio.Event()
    block = asyncio.Event()

    async def hold_lease() -> None:
        async with gate.lease():
            acquired.set()
            await block.wait()

    task = asyncio.create_task(hold_lease())
    await asyncio.wait_for(acquired.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert gate.active == 0
    async with gate.lease():
        assert gate.active == 1
