"""Wall-clock budgets for tests that assert on one-slot QueuePool contention.

These tests are timing-shaped by nature: they hold the only slot of a one-slot
``QueuePool`` and assert on what the runtime does while it waits. The web test
leg runs ``pytest -n 4`` on a shared CI runner, where a descheduled worker
thread can eat several hundred milliseconds on its own — enough to blow any
budget picked to make the test feel snappy. Both failure modes have been
observed in CI as spurious reds.

The budgets are therefore split by what each one actually guards:

``EXHAUSTION_POOL_TIMEOUT``
    Semantic. Tests that assert on pool-*exhaustion* behaviour need the checkout
    to give up quickly, and a slow runner only makes it fire sooner — which is
    the outcome those tests already expect.

``CONTENTION_POOL_TIMEOUT``
    The opposite. Tests that assert work *waits its turn* must never see the
    checkout give up. With ``gated_pool_checkout`` holding the operation in
    place, the real checkout only runs once the slot is free again, so this is
    pure headroom rather than part of any assertion.

``LOOP_LIVENESS_TIMEOUT`` / ``LOOP_LIVENESS_TICKS``
    How long to wait to observe that the event loop is still turning *while the
    contended operation is in flight*. A loop blocked by a synchronous checkout
    never ticks at all, so this only has to out-wait a descheduled runner — it
    is not measuring throughput.

``CONTENTION_GATE_TIMEOUT``
    Both halves of ``gated_pool_checkout``: how long the test waits for the
    operation to reach the checkout, and how long the operation waits to be let
    through. It is also what bounds the on-loop regression these tests exist to
    catch — a checkout that blocks the loop also blocks the coroutine that would
    release the gate, so neither side can make progress and the failure only
    surfaces when one of these waits expires. Keep it well above
    ``LOOP_LIVENESS_TIMEOUT`` (the window that runs between them) and well below
    anything a human would call a hang.

``GUARD_TIMEOUT``
    A hang detector, not an assertion. Once the gate opens and the pool slot is
    released the awaited work should finish immediately; the only failure worth
    reporting here is "it never finished at all", so the ceiling is deliberately
    generous.

Ordering matters as much as the budgets. Waiting for an event that the operation
sets *before* it reaches the checkout proves nothing: the worker can be
descheduled in between, the liveness window can complete, the held connection can
be released, and the eventual checkout then finds a free slot — a test that
passes having never contended for anything. ``gated_pool_checkout`` closes that
by construction: it signals from inside the checkout and holds the operation
there until the test says otherwise.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

EXHAUSTION_POOL_TIMEOUT = 0.4
CONTENTION_POOL_TIMEOUT = 15.0
LOOP_LIVENESS_TIMEOUT = 2.0
LOOP_LIVENESS_TICKS = 3
CONTENTION_GATE_TIMEOUT = 10.0
GUARD_TIMEOUT = 30.0


@dataclass(frozen=True)
class ContentionGate:
    """Handle onto a checkout held open by ``gated_pool_checkout``."""

    entered: threading.Event
    _release: threading.Event

    async def wait_until_contending(self) -> None:
        """Block until the operation is parked inside the pool checkout.

        Runs the wait in a worker thread so that a checkout which wrongly blocks
        the event loop cannot satisfy it: the loop would be stuck and unable to
        resume this coroutine, and the wait expires instead.
        """

        reached = await asyncio.to_thread(self.entered.wait, CONTENTION_GATE_TIMEOUT)
        assert reached, "the operation never reached the pool checkout"

    def let_through(self) -> None:
        """Release the parked checkout. Call once the slot is free again."""

        self._release.set()


@contextmanager
def gated_pool_checkout(engine) -> Iterator[ContentionGate]:
    """Park the next pool checkout until the test releases it.

    Patches the checkout entry point itself, so ``entered`` cannot be observed
    before the operation is genuinely committed to acquiring a connection, and
    the operation cannot slip past while the test is looking elsewhere.

    The gate — not SQLAlchemy's queue — is what holds the operation still. That
    is deliberate: a thread parked in the pool's own wait is not observable from
    outside without reaching into pool and ``threading.Condition`` internals, and
    handing the blocking back to SQLAlchemy reintroduces the race this exists to
    remove (release the gate before freeing the slot and the caller frees it
    before the woken thread is rescheduled, so the pool never blocks at all —
    measured, not assumed). Nothing under test is lost: the property is that the
    checkout blocks a worker thread rather than the event loop, and the gate
    blocks at the same call site the pool would, on the same thread. The
    exhausted one-slot pool still supplies the realistic setup.

    Acquire the connection that exhausts the pool *before* entering this context
    so the parked checkout is the contended one. Only the first checkout is
    parked; later ones pass straight through.
    """

    pool = engine.pool
    # Deliberately unguarded: if SQLAlchemy renames this, the test must break
    # loudly rather than quietly stop synchronising anything.
    original_do_get = pool._do_get
    entered = threading.Event()
    release = threading.Event()

    def gated_do_get():
        if not entered.is_set():
            entered.set()
            if not release.wait(CONTENTION_GATE_TIMEOUT):
                raise AssertionError(
                    "pool checkout was never released; the event loop was "
                    "probably blocked by this checkout"
                )
        return original_do_get()

    pool._do_get = gated_do_get
    try:
        yield ContentionGate(entered, release)
    finally:
        release.set()
        pool._do_get = original_do_get


async def wait_for_ticks(
    read_ticks: Callable[[], int],
    *,
    minimum: int = LOOP_LIVENESS_TICKS,
    timeout: float = LOOP_LIVENESS_TIMEOUT,
) -> int:
    """Give the event loop up to ``timeout`` to turn ``minimum`` more times.

    Counts from a baseline taken on entry, so only progress made *after* the
    call is credited. Callers must therefore have the contended operation parked
    inside its checkout first (see ``gated_pool_checkout``) — otherwise ticks
    banked while nothing was contending could satisfy the assertion, and the
    test would release the held connection without ever observing the loop
    during contention.

    Returns the number of ticks observed since entry, so the caller keeps the
    assertion and its message. A loop blocked by a synchronous checkout stops
    ticking altogether, so a regression still fails here no matter how long we
    are willing to wait — waiting longer only buys tolerance for a runner that
    is merely slow. Note that such a regression cannot be *detected* until the
    blocking checkout returns, so it is bounded by ``CONTENTION_GATE_TIMEOUT``,
    not by ``timeout``.
    """

    loop = asyncio.get_running_loop()
    baseline = read_ticks()
    deadline = loop.time() + timeout
    while read_ticks() - baseline < minimum and loop.time() < deadline:
        await asyncio.sleep(0.01)
    return read_ticks() - baseline


@contextmanager
def assert_pool_checkout_off_loop(engine) -> Iterator[None]:
    """Check loop progress at each real checkout, preserving pool exhaustion.

    Unlike the contention gate, this does not release the held connection.
    The worker waits for one loop acknowledgement and then performs the real
    checkout, so timeout/error assertions remain about SQLAlchemy's pool.
    """
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    original_do_get = engine.pool._do_get
    checkout_threads: list[int] = []
    acknowledged = 0

    def checked_do_get():
        nonlocal acknowledged
        thread_id = threading.get_ident()
        checkout_threads.append(thread_id)
        assert thread_id != loop_thread, "pool checkout ran on the event loop"
        progress = threading.Event()
        loop.call_soon_threadsafe(progress.set)
        assert progress.wait(GUARD_TIMEOUT), "loop never acknowledged pool checkout"
        acknowledged += 1
        return original_do_get()

    engine.pool._do_get = checked_do_get
    try:
        yield
        # Workflows may catch checkout failures; don't let that swallow a failed
        # thread/progress assertion inside the probe.
        assert checkout_threads, "the operation never attempted a pool checkout"
        assert loop_thread not in checkout_threads
        assert acknowledged == len(checkout_threads)
    finally:
        engine.pool._do_get = original_do_get
