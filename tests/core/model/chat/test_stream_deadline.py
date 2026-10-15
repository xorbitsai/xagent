"""Tests for the wall-clock stream deadline helper (#3056)."""

import asyncio

import pytest

from xagent.core.model.chat.exceptions import LLMTimeoutError
from xagent.core.model.chat.stream_deadline import await_before_deadline


@pytest.mark.asyncio
async def test_returns_result_completed_before_deadline():
    loop = asyncio.get_running_loop()

    async def quick():
        await asyncio.sleep(0.01)
        return "chunk"

    result = await await_before_deadline(
        quick(), deadline=loop.time() + 1.0, timeout_message="unused"
    )

    assert result == "chunk"


@pytest.mark.asyncio
async def test_interrupts_awaitable_that_never_completes_at_the_deadline():
    loop = asyncio.get_running_loop()
    cancelled = False

    async def silent():
        nonlocal cancelled
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled = True
            raise

    started = loop.time()
    with pytest.raises(LLMTimeoutError, match="no chunk"):
        await await_before_deadline(
            silent(), deadline=started + 0.2, timeout_message="no chunk"
        )
    elapsed = loop.time() - started

    # Upper bound leaves room for a loaded CI runner.
    assert 0.15 <= elapsed < 0.8
    # The wait is cancelled, so the SDK iterator runs its own cleanup.
    assert cancelled


@pytest.mark.asyncio
async def test_expired_deadline_raises_without_starting_the_awaitable():
    loop = asyncio.get_running_loop()
    started = False

    async def request():
        nonlocal started
        started = True
        return "late"

    with pytest.raises(LLMTimeoutError, match="expired"):
        await await_before_deadline(
            request(), deadline=loop.time() - 0.01, timeout_message="expired"
        )

    assert not started


@pytest.mark.asyncio
async def test_timeout_raised_by_the_awaitable_itself_is_not_relabelled():
    loop = asyncio.get_running_loop()

    async def transport_timeout():
        raise TimeoutError("transport read timed out")

    with pytest.raises(TimeoutError, match="transport read timed out") as excinfo:
        await await_before_deadline(
            transport_timeout(), deadline=loop.time() + 1.0, timeout_message="ours"
        )

    assert not isinstance(excinfo.value, LLMTimeoutError)
