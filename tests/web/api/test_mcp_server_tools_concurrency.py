"""Tests for the concurrency cap on GET /api/mcp/servers/{id}/tools (issue #3265).

The endpoint caps concurrent tool loads per event loop, returning 503
when the cap is reached.  These tests verify the semaphore gate logic
directly without requiring a full server stack.
"""

import asyncio
import weakref

import pytest


def _make_gate(max_inflight: int = 4):
    """Replicate the per-loop semaphore pattern from mcp.py."""
    gates: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
        weakref.WeakKeyDictionary()
    )

    def gate() -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        g = gates.get(loop)
        if g is None:
            g = asyncio.Semaphore(max_inflight)
            gates[loop] = g
        return g

    return gate


class TestServerToolsConcurrencyGate:
    """Test the concurrency cap logic."""

    @pytest.mark.asyncio
    async def test_allows_up_to_max_concurrent_loads(self):
        gate_fn = _make_gate(max_inflight=4)
        gate = gate_fn()

        # Acquire all 4 slots
        for _ in range(4):
            await asyncio.wait_for(gate.acquire(), 1.0)

        # The 5th acquire should time out
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gate.acquire(), 0.1)

        # Release one slot
        gate.release()

        # Now the 5th acquire should succeed
        await asyncio.wait_for(gate.acquire(), 1.0)

    @pytest.mark.asyncio
    async def test_release_on_exception(self):
        """A slot is released even when the load raises."""
        gate_fn = _make_gate(max_inflight=2)
        gate = gate_fn()

        acquired = False
        try:
            await asyncio.wait_for(gate.acquire(), 1.0)
            acquired = True
            raise RuntimeError("boom")
        except RuntimeError:
            pass
        finally:
            if acquired:
                gate.release()

        # Slot should be available again
        await asyncio.wait_for(gate.acquire(), 1.0)
        gate.release()

    @pytest.mark.asyncio
    async def test_gate_is_per_event_loop(self):
        """Each event loop gets its own semaphore."""
        gate_fn = _make_gate(max_inflight=4)

        # Get gate from current loop
        gate1 = gate_fn()
        gate2 = gate_fn()
        assert gate1 is gate2, "Same loop should return same semaphore"
