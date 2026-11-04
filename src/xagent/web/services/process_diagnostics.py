"""One runtime diagnostics log line per minute from every app and worker process.

Every app process (``combined`` or ``web`` role) and every ``worker`` logs one
INFO line per window, whether or not anything happened::

    runtime diagnostics role=worker pid=4242 window_s=60.0 inflight=1
    resume_inflight=0 started_1m=2 finished_1m=1 ok=1 failed=0 cancelled=0
    gate_held_s=0.0 slot_tenants=1 slot_top=1 loop_lag_max_ms=12.3
    loop_lag_samples=0 degradations=- fair_misses=0 fair_fallbacks=0
    fair_all_missed=0

The line is unconditional and has no switch: a process whose pid stops
appearing for two windows is gone or has a blocked event loop, and an idle
process is told apart from a dead one only because idle windows are logged
too. It needs no metrics backend; the OpenTelemetry gauges in
``xagent.web.app`` are separate and unchanged.

In shared mode the ``web`` role only accepts work: it does not consume task
commands (``consumes_task_commands()`` is False), so its line reports
liveness and event-loop lag only, with ``inflight=0`` and zero execution
counts.

Field meanings:

- ``inflight``: execution tasks of this process that have not finished,
  counted by task object. Unlike the ``xagent.agent_tasks.running`` gauge
  (``len(running_tasks)``) it excludes finished tasks still registered and
  includes an older execution still finishing after its task id was
  registered again.
- ``resume_inflight``: resume coordinators still waiting to become the
  execution; disjoint from ``inflight``.
- ``started_1m`` / ``finished_1m``: executions started and finished in the
  window. ``failed`` counts runs that raised or whose runner handled a setup
  or run exception; ``ok`` is every other non-cancelled finish, which
  includes paused runs, runs that lost their lease and runs whose agent
  reported an unsuccessful result without raising. A resume run that handles
  its own failure also counts as ``ok``: the resume path returns normally and
  has no failure flag. These buckets describe how the execution ended in this
  process, not the task's stored status.
- ``window_s``: actual seconds covered; longer than 60 when the loop was
  blocked at flush time.
- ``gate_held_s``: seconds in the window this process was at its cap
  (``XAGENT_DISPATCHER_MAX_INFLIGHT_PER_PROCESS``): its active task ids
  numbered at least the limit, so its dispatcher did not claim commands that
  would start a new execution here. It is time at the cap, not time work waited:
  a process running exactly the limit with nothing queued reports the whole
  window. Counts only the time at the full limit: with an interactive reserve
  (``XAGENT_DISPATCHER_INTERACTIVE_RESERVE_PER_PROCESS``) batch commands are held
  from ``limit - reserve`` active tasks, and that time is not counted. Checked
  at each dispatch attempt and at the flush, so each change
  is late by at most the gap between attempts (0.5 s while the dispatcher is
  idle). Always ``0.0`` in a process that does not dispatch task commands.
- ``slot_tenants``: how many tenant keys the tasks hold that this process
  started through its dispatcher and that are still active. Executions
  started outside the dispatcher are not counted.
- ``slot_top``: how many of those tasks the largest tenant holds. Both values
  are read at the flush, like ``inflight``, not summed over the window. They
  are ``0`` in a process that does not dispatch task commands, and without
  shared execution, where dispatch takes no slot reservation.
- ``loop_lag_max_ms`` / ``loop_lag_samples``: largest event-loop scheduling
  delay seen by a 1-second sampler, and the number of samples delayed by at
  least 500 ms.
- ``degradations``: names of this process's active ``ops_signals``
  degradations, sorted and comma-separated, or ``-`` when none is active.
  ``/health`` reports them only for web processes; a worker has no HTTP
  surface, so this line is where its degradations show up.
- ``fair_misses``: how many times in the window a query for the commands of one
  tenant key and dispatch class found nothing this process could claim, so that
  the pair was skipped for a while; a tenant key ranked alone is not skipped, so
  while its windows keep missing every scan adds to the count. Counted per window
  like ``started_1m``. It counts lookups, not tenants: dispatch attempts woken
  together can each read the skip table before any of them records a miss, so one
  pair can be looked up and counted once per concurrent attempt. It is ``0`` when
  ``XAGENT_DISPATCHER_FAIR_SHARE`` is off and in a process that does not dispatch
  task commands. Read ``fair_share=`` on the dispatch policy line to tell
  whether the switch is on.
- ``fair_fallbacks``: how many dispatch attempts in the window were fallback
  rounds, which read the queue oldest first instead of ranking tenants. It
  counts the decision, so a fallback round that dispatched nothing counts too:
  once the attempt count has forced fallback rounds, every attempt is one until
  a fallback round dispatches a command holding a slot token.
- ``fair_all_missed``: how many scans in the window ranked at least two tenant
  keys and then fell back to the oldest claimable command, if any, because no
  tenant asked had a command this process could claim, or every tenant was
  skipped or held back so that none was asked. It counts scans, so an attempt
  that scans again after setting a task aside can count twice. Both are ``0`` when
  ``XAGENT_DISPATCHER_FAIR_SHARE`` is off and in a process that does not
  dispatch task commands.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FLUSH_INTERVAL_SECONDS = 60.0
SAMPLE_INTERVAL_SECONDS = 1.0
SLOW_LOOP_LAG_MS = 500.0
_UNKNOWN_ROLE = "unknown"


@dataclass(frozen=True)
class ExecutionCounters:
    role: str
    pid: int
    window_s: float
    inflight: int
    resume_inflight: int
    started: int
    finished_ok: int
    finished_failed: int
    finished_cancelled: int
    gate_held_s: float
    slot_tenants: int
    slot_top: int
    loop_lag_max_ms: float
    loop_lag_samples: int
    fair_misses: int
    fair_fallbacks: int
    fair_all_missed: int
    degradations: tuple[str, ...] = ()

    @property
    def finished(self) -> int:
        return self.finished_ok + self.finished_failed + self.finished_cancelled

    def format_line(self) -> str:
        return (
            f"runtime diagnostics role={self.role} pid={self.pid} "
            f"window_s={self.window_s:.1f} inflight={self.inflight} "
            f"resume_inflight={self.resume_inflight} "
            f"started_1m={self.started} finished_1m={self.finished} "
            f"ok={self.finished_ok} failed={self.finished_failed} "
            f"cancelled={self.finished_cancelled} "
            f"gate_held_s={self.gate_held_s:.1f} "
            f"slot_tenants={self.slot_tenants} slot_top={self.slot_top} "
            f"loop_lag_max_ms={self.loop_lag_max_ms:.1f} "
            f"loop_lag_samples={self.loop_lag_samples} "
            f"degradations={','.join(self.degradations) or '-'} "
            f"fair_misses={self.fair_misses} "
            f"fair_fallbacks={self.fair_fallbacks} "
            f"fair_all_missed={self.fair_all_missed}"
        )


class _Window:
    """Process-local window state outside the task manager."""

    def __init__(self) -> None:
        self.started_at: float | None = None
        self.lag_max_ms = 0.0
        self.lag_samples = 0

    def observe_lag(self, lag_ms: float) -> None:
        if lag_ms > self.lag_max_ms:
            self.lag_max_ms = lag_ms
        if lag_ms >= SLOW_LOOP_LAG_MS:
            self.lag_samples += 1


_window = _Window()


def _resolve_role() -> str:
    from ...config import get_task_execution_role

    try:
        return get_task_execution_role()
    except ValueError:
        return _UNKNOWN_ROLE


def drain_execution_counters(now: float | None = None) -> ExecutionCounters:
    """Close the current window and return its counters.

    Must be called on the process's event-loop thread. It is synchronous, so
    a task finishing concurrently lands either in this window or the next,
    never both and never neither. The periodic diagnostics loop calls it once
    per window; any other caller in the same process splits the window with
    that loop, and each event is still reported exactly once across the two.
    ``now`` is the event-loop clock reading used for ``window_s``.
    """

    from .ops_signals import active_degradations
    from .task_command_transport import drain_fair_bypasses, drain_fair_misses
    from .task_execution import background_task_manager

    if now is None:
        now = asyncio.get_running_loop().time()
    started_at = _window.started_at
    window_s = 0.0 if started_at is None else max(0.0, now - started_at)
    inflight, resume_inflight = background_task_manager.inflight_counts()
    started, ok, failed, cancelled = background_task_manager.drain_window_counts()
    gate_held_s = min(background_task_manager.drain_gate_held_seconds(now), window_s)
    slot_tenants, slot_top = background_task_manager.slot_tenant_counts()
    fair_misses = drain_fair_misses()
    fair_fallbacks, fair_all_missed = drain_fair_bypasses()
    counters = ExecutionCounters(
        role=_resolve_role(),
        pid=os.getpid(),
        window_s=window_s,
        inflight=inflight,
        resume_inflight=resume_inflight,
        started=started,
        finished_ok=ok,
        finished_failed=failed,
        finished_cancelled=cancelled,
        gate_held_s=gate_held_s,
        slot_tenants=slot_tenants,
        slot_top=slot_top,
        loop_lag_max_ms=_window.lag_max_ms,
        loop_lag_samples=_window.lag_samples,
        degradations=tuple(sorted(active_degradations())),
        fair_misses=fair_misses,
        fair_fallbacks=fair_fallbacks,
        fair_all_missed=fair_all_missed,
    )
    _window.started_at = now
    _window.lag_max_ms = 0.0
    _window.lag_samples = 0
    return counters


async def run_process_diagnostics(
    *,
    flush_interval_s: float = FLUSH_INTERVAL_SECONDS,
    sample_interval_s: float = SAMPLE_INTERVAL_SECONDS,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> None:
    """Sample loop lag every ``sample_interval_s`` and log a line per window."""

    if flush_interval_s <= 0 or sample_interval_s <= 0:
        raise ValueError("diagnostics intervals must be positive")
    if clock is None:
        clock = asyncio.get_running_loop().time
    # The first window starts here. Execution counts are not reset: the task
    # manager zeroes them when it opens admission, which happens earlier, so
    # work admitted before this loop started is reported in the first window.
    _window.started_at = clock()
    _window.lag_max_ms = 0.0
    _window.lag_samples = 0
    expected_at = clock() + sample_interval_s
    next_flush = clock() + flush_interval_s
    while True:
        await sleep(sample_interval_s)
        observed_at = clock()
        _window.observe_lag(max(0.0, observed_at - expected_at) * 1_000.0)
        expected_at = observed_at + sample_interval_s
        if observed_at < next_flush:
            continue
        next_flush = observed_at + flush_interval_s
        try:
            logger.info(drain_execution_counters(observed_at).format_line())
        except Exception:
            # The line is the liveness signal; one failed flush must not
            # silence every later window.
            logger.exception("Runtime diagnostics flush failed")


def start_process_diagnostics() -> asyncio.Task[None] | None:
    """Start the diagnostics loop for this process; never raises."""

    try:
        # Also set by the loop itself; set here too so a loop stopped before
        # its first step still reports a window measured from this start.
        _window.started_at = asyncio.get_running_loop().time()
        return asyncio.create_task(
            run_process_diagnostics(), name="process-diagnostics"
        )
    except Exception:
        logger.warning("Could not start runtime diagnostics", exc_info=True)
        return None


async def stop_process_diagnostics(task: asyncio.Task[None] | None) -> None:
    """Stop the loop and log one final line for the partial last window.

    Callers stop it after execution has drained, so the final line reports
    the executions that shutdown cancelled. Callers also stop it when a
    shutdown step from the drain onward raised; the final line then reports
    what had finished by that point.
    """

    if task is None:
        return
    task.cancel()
    # gather keeps a cancellation of the caller itself propagating.
    (outcome,) = await asyncio.gather(task, return_exceptions=True)
    if isinstance(outcome, Exception):
        logger.error(
            "Runtime diagnostics loop had stopped with an error",
            exc_info=(type(outcome), outcome, outcome.__traceback__),
        )
    try:
        logger.info(drain_execution_counters().format_line())
    except Exception:
        logger.exception("Runtime diagnostics flush failed")
