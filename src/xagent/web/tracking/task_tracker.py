"""Task token usage tracker.

This module provides utilities for tracking token usage during task execution,
with support for periodic updates to the database.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...core.model.chat.token_context import (
    TokenUsage,
    get_token_usage,
    set_token_usage,
)
from ..services.db_runtime import run_db_io_cancellation_safe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TaskTrackingSeed:
    """Detached primitive state used to seed one tracking turn."""

    user_id: int | None
    usage: TokenUsage


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return _safe_int(value)
    return None


def _copy_details(raw_details: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_details, list):
        return []
    return [dict(item) for item in raw_details if isinstance(item, dict)]


def _copy_usage(usage: TokenUsage) -> TokenUsage:
    """Detach a stable snapshot before yielding to a database worker."""
    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        llm_calls=usage.llm_calls,
        tool_calls=usage.tool_calls,
        details=_copy_details(usage.details),
    )


def _task_for_run(
    db_session: Any,
    task_id: int,
    expected_run_id: str | None,
    expected_runner_id: str | None = None,
) -> Any | None:
    from ..models.task import Task

    query = db_session.query(Task).filter(Task.id == task_id)
    if expected_run_id is not None:
        query = query.filter(Task.run_id == expected_run_id)
    if expected_runner_id is not None:
        query = query.filter(Task.runner_id == expected_runner_id)
    return query.first()


def _task_seed_from_session(
    db_session: Any,
    task_id: int,
    expected_run_id: str | None = None,
    expected_runner_id: str | None = None,
) -> _TaskTrackingSeed:
    task = _task_for_run(
        db_session,
        task_id,
        expected_run_id,
        expected_runner_id,
    )
    if task is None:
        run_suffix = (
            f" for run {expected_run_id}" if expected_run_id is not None else ""
        )
        raise ValueError(f"Task {task_id}{run_suffix} not found")
    return _TaskTrackingSeed(
        user_id=_optional_int(getattr(task, "user_id", None)),
        usage=TokenUsage(
            input_tokens=_safe_int(getattr(task, "input_tokens", 0)),
            output_tokens=_safe_int(getattr(task, "output_tokens", 0)),
            llm_calls=_safe_int(getattr(task, "llm_calls", 0)),
            details=_copy_details(getattr(task, "token_usage_details", None)),
        ),
    )


def _new_short_session() -> Any:
    from ..models.database import get_session_local

    return get_session_local()()


def _load_task_seed_sync(
    task_id: int,
    expected_run_id: str | None = None,
    expected_runner_id: str | None = None,
) -> _TaskTrackingSeed:
    db_session = _new_short_session()
    try:
        return _task_seed_from_session(
            db_session,
            task_id,
            expected_run_id,
            expected_runner_id,
        )
    finally:
        db_session.close()


def _commit_task_usage_if_owned(
    db_session: Any,
    task_id: int,
    usage: TokenUsage,
    expected_run_id: str | None = None,
    expected_runner_id: str | None = None,
) -> bool:
    """Persist counters only while the durable run owner still matches.

    Ownership predicates and counter values intentionally share one SQL
    ``UPDATE``. A prior ``SELECT`` followed by ORM mutation leaves a race where
    a replacement runner can take over between the read and flush.
    """
    from ..models.task import Task

    query = db_session.query(Task).filter(Task.id == task_id)
    if expected_run_id is not None:
        query = query.filter(Task.run_id == expected_run_id)
    if expected_runner_id is not None:
        query = query.filter(Task.runner_id == expected_runner_id)
    updated = query.update(
        {
            Task.input_tokens: usage.input_tokens,
            Task.output_tokens: usage.output_tokens,
            Task.total_tokens: usage.total_tokens,
            Task.llm_calls: usage.llm_calls,
            Task.token_usage_details: _copy_details(usage.details),
        },
        synchronize_session=False,
    )
    if int(updated or 0) != 1:
        db_session.rollback()
        return False
    db_session.commit()
    return True


def _write_task_usage_sync(
    task_id: int,
    usage: TokenUsage,
    expected_run_id: str | None = None,
    expected_runner_id: str | None = None,
) -> bool:
    db_session = _new_short_session()
    try:
        return _commit_task_usage_if_owned(
            db_session,
            task_id,
            usage,
            expected_run_id,
            expected_runner_id,
        )
    except Exception:
        try:
            db_session.rollback()
        except Exception as rollback_error:  # noqa: BLE001
            logger.warning(
                "Failed to rollback DB session for task %s: %s",
                task_id,
                rollback_error,
            )
        raise
    finally:
        db_session.close()


def _check_quota_on_event_loop(
    user_id: int | None,
    delta_details: list[dict[str, Any]],
    delta_actions: int,
) -> str | None:
    """Invoke the legacy progress hook on its documented event-loop thread."""
    from ..services.quota_hooks import check_run_progress_gate

    db_session = _new_short_session()
    try:
        return check_run_progress_gate(
            db_session,
            user_id,
            delta_details,
            delta_actions,
        )
    finally:
        db_session.close()


def _record_usage_on_event_loop(
    user_id: int | None,
    delta_details: list[dict[str, Any]],
    delta_actions: int,
) -> None:
    """Invoke the legacy completion hook on its established event-loop thread."""
    from ..services.quota_hooks import record_usage

    db_session = _new_short_session()
    try:
        record_usage(
            db_session,
            user_id,
            delta_details,
            delta_actions,
        )
    finally:
        # The callback owns separate durability and must not leave work pending
        # on this compatibility Session.
        if db_session.in_transaction():
            db_session.rollback()
        db_session.close()


def _complete_task_usage_sync(
    task_id: int,
    usage: TokenUsage,
    expected_run_id: str | None = None,
    expected_runner_id: str | None = None,
) -> bool:
    """Persist one run while its durable ownership fence still wins."""
    from ..services.db_runtime import is_database_pool_timeout

    db_session = _new_short_session()
    try:
        try:
            owned = _commit_task_usage_if_owned(
                db_session,
                task_id,
                usage,
                expected_run_id,
                expected_runner_id,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "Failed to commit final token usage for task %s: %s",
                task_id,
                error,
            )
            try:
                db_session.rollback()
            except Exception as rollback_error:  # noqa: BLE001
                logger.warning(
                    "Failed to rollback DB session for task %s: %s",
                    task_id,
                    rollback_error,
                )
            if is_database_pool_timeout(error):
                raise
            return False
        if not owned:
            logger.info(
                "Skipping final usage for task %s run %s; ownership changed",
                task_id,
                expected_run_id,
            )
            return False
        return True
    finally:
        db_session.close()


class TaskTracker:
    """Track token usage for a task execution.

    This class manages token tracking for a task, including:
    - Initializing token context at start
    - Periodically updating the database
    - Finalizing statistics on completion

    Usage:
        tracker = TaskTracker(task_id=123)

        # Start tracking
        await tracker.start_tracking()

        # During task execution, LLM calls will be automatically tracked
        # via the token_context

        # Periodic updates (optional)
        asyncio.create_task(tracker.periodic_update(interval=30))

        # Complete and save final stats
        await tracker.complete_tracking()
    """

    def __init__(
        self,
        task_id: int,
        db_session: Any | None = None,
        update_interval_seconds: int = 15,
        expected_run_id: str | None = None,
        expected_runner_id: str | None = None,
    ) -> None:
        """Initialize the task tracker.

        Args:
            task_id: The task ID in the database
            db_session: Deprecated compatibility input. When provided, it is
                read once to detach the task's primitive seed state; the
                session and ORM row are never retained by this tracker.
            update_interval_seconds: Interval for periodic updates (default: 15s)
            expected_run_id: Optional durable run fence. When provided, writes
                are ignored after a replacement run changes the task's run id.
            expected_runner_id: Optional durable runner fence. When provided,
                seed reads and writes require both the expected run and runner.
        """
        self.task_id = task_id
        self.update_interval_seconds = update_interval_seconds
        self.expected_run_id = expected_run_id
        self.expected_runner_id = expected_runner_id
        self._is_tracking = False
        self._update_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._last_reported_usage: Optional[TokenUsage] = None
        # Per-turn baselines captured in start_tracking; used to meter deltas.
        self._initial_details_len = 0
        self._initial_tool_calls = 0

        # Compatibility callers may still hand us their request session. Read
        # and detach the primitive seed now, but never retain the Session or ORM
        # row across awaits. New callers omit this argument and start_tracking
        # loads the same seed in a worker-owned short session.
        self._seed = (
            _task_seed_from_session(
                db_session,
                task_id,
                expected_run_id,
                expected_runner_id,
            )
            if db_session is not None
            else None
        )
        self._user_id = self._seed.user_id if self._seed is not None else None
        # Fail-open logging in the per-step gate must not flood: log once per run.
        self._quota_gate_warned = False
        # Set to the gate reason if the mid-run quota gate trips, so the run's
        # caller can surface why the run stopped instead of a silent PAUSE.
        self.quota_interrupt_reason: str | None = None

    async def start_tracking(self) -> None:
        """Start tracking token usage for this task."""
        if self._is_tracking:
            logger.warning(f"Task {self.task_id} is already being tracked")
            return

        seed = self._seed
        if seed is None:
            seed = await run_db_io_cancellation_safe(
                lambda: _load_task_seed_sync(
                    self.task_id,
                    self.expected_run_id,
                    self.expected_runner_id,
                )
            )
            self._seed = seed
        self._user_id = seed.user_id
        initial_usage = _copy_usage(seed.usage)
        set_token_usage(initial_usage)

        # Snapshot the seeded baselines so complete_tracking can meter only this
        # turn's delta (start_tracking seeds from prior turns for multi-turn
        # tasks). The new per-model detail entries appended during this turn are
        # everything past _initial_details_len.
        self._initial_details_len = len(initial_usage.details)
        self._initial_tool_calls = initial_usage.tool_calls

        logger.info(f"Started token tracking for task {self.task_id}")

        # Automatically start periodic updates (this will set _is_tracking)
        await self.start_periodic_updates()

    async def periodic_update(self) -> None:
        """Periodically update token usage to the database.

        This method should be called periodically during task execution.
        It updates the token usage in the database without stopping the tracking.

        Can be run as a background task:
            asyncio.create_task(tracker.periodic_update())
        """
        logger.debug(
            f"periodic_update called for task {self.task_id}, _is_tracking={self._is_tracking}"
        )

        if not self._is_tracking:
            logger.warning(f"Task {self.task_id} is not being tracked")
            return

        try:
            usage = _copy_usage(get_token_usage())
            logger.debug(
                f"Got token usage for task {self.task_id}: input={usage.input_tokens}, output={usage.output_tokens}"
            )

            task_exists = await run_db_io_cancellation_safe(
                lambda: _write_task_usage_sync(
                    self.task_id,
                    usage,
                    self.expected_run_id,
                    self.expected_runner_id,
                )
            )
            if not task_exists:
                self._is_tracking = False
                logger.info(
                    f"Stopping token tracking for task {self.task_id}: task no longer exists"
                )
                return

            # Only log if values have changed
            if (
                self._last_reported_usage is None
                or usage.input_tokens != self._last_reported_usage.input_tokens
                or usage.output_tokens != self._last_reported_usage.output_tokens
                or usage.total_tokens != self._last_reported_usage.total_tokens
                or usage.llm_calls != self._last_reported_usage.llm_calls
            ):
                logger.info(
                    f"Token usage updated for task {self.task_id}: "
                    f"input={usage.input_tokens}, output={usage.output_tokens}, "
                    f"total={usage.total_tokens}, calls={usage.llm_calls}"
                )
                self._last_reported_usage = _copy_usage(usage)
        except Exception as e:
            logger.error(f"Failed to update token usage for task {self.task_id}: {e}")
            # The write helper already reports a missing row with False. Any
            # exception here (including QueuePool timeout) is transient from the
            # tracker's perspective: probing the row would perform a second pool
            # checkout and can turn one timeout into a self-amplifying retry.

    async def start_periodic_updates(self) -> None:
        """Start periodic background updates to the database.

        This creates an asyncio background task that will periodically
        update the token usage in the database.
        """
        if self._is_tracking:
            logger.warning(f"Periodic updates already active for task {self.task_id}")
            return

        self._is_tracking = True
        self._stop_event.clear()

        async def update_loop() -> None:
            logger.debug(f"[update_loop] Starting update loop for task {self.task_id}")
            iteration = 0
            while self._is_tracking:
                iteration += 1
                logger.debug(
                    f"[update_loop] Iteration {iteration} for task {self.task_id}"
                )
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.update_interval_seconds,
                    )
                except TimeoutError:
                    pass
                if self._is_tracking and not self._stop_event.is_set():
                    logger.debug(
                        f"[update_loop] Calling periodic_update for task {self.task_id}"
                    )
                    await self.periodic_update()

            logger.debug(f"[update_loop] Update loop ended for task {self.task_id}")

        self._update_task = asyncio.create_task(update_loop())
        logger.debug(
            f"Created background task for task {self.task_id}, task={self._update_task}, done={self._update_task.done()}"
        )

        logger.info(
            f"Started periodic token updates for task {self.task_id} "
            f"(interval: {self.update_interval_seconds}s)"
        )

    async def stop_periodic_updates(self) -> None:
        """Stop periodic background updates."""
        self._is_tracking = False
        self._stop_event.set()
        update_task = self._update_task
        if (
            update_task
            and not update_task.done()
            and update_task is not asyncio.current_task()
        ):
            try:
                # Do not cancel a to_thread wait: cancellation cannot stop its
                # database worker and would let a stale periodic write race the
                # final completion write.
                await update_task
            except asyncio.CancelledError:
                pass

        self._update_task = None
        logger.info(f"Stopped periodic token updates for task {self.task_id}")

    def _turn_delta(self, usage: TokenUsage | None = None) -> tuple[list, int]:
        """This turn's (detail entries, tool-call count) over the baseline seeded
        in start_tracking. Single source for the completion meter and the mid-run
        gate so they can't disagree on what 'this run's usage' means."""
        if usage is None:
            usage = get_token_usage()
        return (
            usage.details[self._initial_details_len :],
            max(0, usage.tool_calls - self._initial_tool_calls),
        )

    async def interrupt_reason_for_quota(self) -> str | None:
        """Per-step interrupt-checker: return a reason when this run's live-so-far
        usage would push the team over a run-gated quota, so a single long or
        expensive run is stopped mid-flight instead of only being metered at
        completion (``complete_tracking`` still meters the partial usage on exit).

        Wired as the run's ``interrupt_checker`` and polled at every safe point
        (each LLM reply / tool call). Best-effort: it reuses the exact turn-delta
        the metering path computes, and swallows errors so a quota-infra hiccup
        fails open rather than wedging a run.

        When it trips it records the reason in ``quota_interrupt_reason`` so the
        run's caller can surface *why* the run stopped (the pattern interrupt
        path itself would otherwise flip silently to PAUSED).
        """
        if not self._is_tracking:
            return None
        try:
            delta_details, delta_actions = self._turn_delta()
            reason = _check_quota_on_event_loop(
                self._user_id,
                _copy_details(delta_details),
                delta_actions,
            )
            if reason is not None:
                self.quota_interrupt_reason = reason
            return reason
        except Exception as e:  # noqa: BLE001
            # Runs per step; log once per run so a persistent failure can't flood.
            if not self._quota_gate_warned:
                self._quota_gate_warned = True
                logger.warning(
                    f"Quota progress gate failed open for task {self.task_id}: {e}"
                )
            return None

    async def _complete_tracking_once(self, usage: TokenUsage) -> TokenUsage:
        """Drain periodic persistence, meter, and write one final snapshot."""

        await self.stop_periodic_updates()

        logger.info(f"Force updating token usage for task {self.task_id}")
        delta_details, delta_actions = self._turn_delta(usage)

        try:
            persisted = await run_db_io_cancellation_safe(
                lambda: _complete_task_usage_sync(
                    self.task_id,
                    usage,
                    self.expected_run_id,
                    self.expected_runner_id,
                )
            )
        except Exception as e:  # noqa: BLE001
            from ..services.db_runtime import is_database_pool_timeout

            logger.warning(
                f"Failed to commit final token usage for task {self.task_id}: {e}"
            )
            if is_database_pool_timeout(e):
                # The lease owner must see the exhausted checkout. It can then
                # retain the current run's lease for TTL recovery instead of
                # immediately performing a second blocking DB checkout.
                raise
            return usage
        if not persisted:
            return usage

        # The committed conditional UPDATE is the ownership linearization
        # point. Preserve the established event-loop affinity of application
        # quota callbacks, and invoke metering only after this runner still owns
        # the task. The remaining blocking risk is tracked separately from the
        # database-lifecycle changes in this PR.
        try:
            _record_usage_on_event_loop(
                self._user_id,
                _copy_details(delta_details),
                delta_actions,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Quota usage recording failed for task {self.task_id}: {e}")

        # Only log if values have changed from last report
        if (
            self._last_reported_usage is None
            or usage.input_tokens != self._last_reported_usage.input_tokens
            or usage.output_tokens != self._last_reported_usage.output_tokens
            or usage.total_tokens != self._last_reported_usage.total_tokens
            or usage.llm_calls != self._last_reported_usage.llm_calls
        ):
            logger.info(
                f"Token usage updated for task {self.task_id}: "
                f"input={usage.input_tokens}, output={usage.output_tokens}, "
                f"total={usage.total_tokens}, calls={usage.llm_calls}"
            )
            self._last_reported_usage = _copy_usage(usage)

        logger.info(
            f"Completed token tracking for task {self.task_id}: "
            f"input={usage.input_tokens}, output={usage.output_tokens}, "
            f"total={usage.total_tokens}, calls={usage.llm_calls}"
        )

        return usage

    async def complete_tracking(self) -> TokenUsage:
        """Complete tracking and return final statistics.

        Stops periodic updates and saves final token usage to the database.
        If the awaiting task is cancelled, completion is allowed to finish first
        so an already-running periodic DB worker cannot overwrite the final
        snapshot. The original cancellation is then propagated.

        Returns:
            Final TokenUsage object with all statistics

        Raises:
            RuntimeError: If tracking was not started
        """
        if not self._is_tracking:
            raise RuntimeError(f"Task {self.task_id} is not being tracked")

        # Capture from the caller's ContextVar before delegating cancellation-
        # independent cleanup to a child task.
        usage = _copy_usage(get_token_usage())
        completion_task = asyncio.create_task(self._complete_tracking_once(usage))
        try:
            return await asyncio.shield(completion_task)
        except asyncio.CancelledError:
            # Repeated cancellation must not detach an uncancellable to_thread
            # worker. Wait until periodic and final writes have both settled,
            # then preserve the caller's cancellation semantics.
            while not completion_task.done():
                try:
                    await asyncio.shield(completion_task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not completion_task.cancelled():
                completion_error = completion_task.exception()
                if completion_error is not None:
                    logger.warning(
                        "Token tracking completion failed after cancellation for "
                        "task %s: %s",
                        self.task_id,
                        completion_error,
                    )
            raise

    def get_current_usage(self) -> TokenUsage:
        """Get current token usage without stopping tracking.

        Returns:
            Current TokenUsage object
        """
        return get_token_usage()

    @property
    def is_tracking(self) -> bool:
        """Check if currently tracking."""
        return self._is_tracking


class TaskTrackerManager:
    """Manager for multiple task trackers.

    This provides a centralized way to manage tracking for multiple tasks.
    """

    def __init__(self) -> None:
        self._trackers: Dict[int, TaskTracker] = {}

    def get_or_create_tracker(
        self,
        task_id: int,
        db_session: Any | None = None,
        update_interval_seconds: int = 5,
    ) -> TaskTracker:
        """Get existing tracker or create new one."""
        if task_id not in self._trackers:
            self._trackers[task_id] = TaskTracker(
                task_id=task_id,
                db_session=db_session,
                update_interval_seconds=update_interval_seconds,
            )
        return self._trackers[task_id]

    async def complete_tracker(self, task_id: int) -> Optional[TokenUsage]:
        """Complete tracking for a task and return final usage.

        Args:
            task_id: Task ID to complete

        Returns:
            Final TokenUsage if tracker existed, None otherwise
        """
        tracker = self._trackers.pop(task_id, None)
        if tracker:
            return await tracker.complete_tracking()
        return None

    def get_tracker(self, task_id: int) -> Optional[TaskTracker]:
        """Get existing tracker without creating new one.

        Args:
            task_id: Task ID

        Returns:
            TaskTracker if exists, None otherwise
        """
        return self._trackers.get(task_id)

    async def complete_all(self) -> Dict[int, TokenUsage]:
        """Complete all active trackers and return final usage.

        Returns:
            Dictionary mapping task_id to final TokenUsage
        """
        results = {}
        for task_id in list(self._trackers.keys()):
            usage = await self.complete_tracker(task_id)
            if usage:
                results[task_id] = usage
        return results
