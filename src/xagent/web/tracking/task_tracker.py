"""Task token usage tracker.

This module provides utilities for tracking token usage during task execution,
with support for periodic updates to the database.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from ...core.model.chat.token_context import (
    TokenUsage,
    get_token_usage,
    set_token_usage,
)

logger = logging.getLogger(__name__)


class TaskTracker:
    """Track token usage for a task execution.

    This class manages token tracking for a task, including:
    - Initializing token context at start
    - Periodically updating the database
    - Finalizing statistics on completion

    Usage:
        tracker = TaskTracker(task_id=123, db_session=session)

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
        db_session: Any,
        update_interval_seconds: int = 15,
    ) -> None:
        """Initialize the task tracker.

        Args:
            task_id: The task ID in the database
            db_session: SQLAlchemy database session
            update_interval_seconds: Interval for periodic updates (default: 15s)
        """
        self.task_id = task_id
        self.db_session = db_session
        self.update_interval_seconds = update_interval_seconds
        self._is_tracking = False
        self._update_task: Optional[asyncio.Task] = None
        self._last_reported_usage: Optional[TokenUsage] = None
        # Per-turn baselines captured in start_tracking; used to meter deltas.
        self._initial_details_len = 0
        self._initial_tool_calls = 0

        # Load the task model
        from ..models.task import Task as TaskModel

        self.task_model = TaskModel
        self.task = db_session.query(TaskModel).filter(TaskModel.id == task_id).first()

        if not self.task:
            raise ValueError(f"Task {task_id} not found")

        # Cache user_id at construction: the per-step quota checker reads it every
        # step, and `self.task` is expired after periodic_update commits
        # (expire_on_commit), so a fresh `self.task.user_id` would trigger an
        # implicit blocking SELECT. user_id never changes for a task.
        self._user_id = getattr(self.task, "user_id", None)
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

        details: list[dict[str, Any]] = []
        raw_details = getattr(self.task, "token_usage_details", None)
        if isinstance(raw_details, list):
            details = [item for item in raw_details if isinstance(item, dict)]

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

        initial_usage = TokenUsage(
            input_tokens=_safe_int(getattr(self.task, "input_tokens", 0)),
            output_tokens=_safe_int(getattr(self.task, "output_tokens", 0)),
            llm_calls=_safe_int(getattr(self.task, "llm_calls", 0)),
            details=details,
        )
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
            task_exists = (
                self.db_session.query(self.task_model.id)
                .filter(self.task_model.id == self.task_id)
                .first()
                is not None
            )
            if not task_exists:
                self._is_tracking = False
                logger.info(
                    f"Stopping token tracking for task {self.task_id}: task no longer exists"
                )
                return

            # Get current token usage
            usage = get_token_usage()
            logger.debug(
                f"Got token usage for task {self.task_id}: input={usage.input_tokens}, output={usage.output_tokens}"
            )

            # Update database
            self.task.input_tokens = usage.input_tokens
            self.task.output_tokens = usage.output_tokens
            self.task.total_tokens = usage.total_tokens
            self.task.llm_calls = usage.llm_calls
            self.task.token_usage_details = usage.to_dict()["details"]

            self.db_session.commit()

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
                self._last_reported_usage = usage
        except Exception as e:
            logger.error(f"Failed to update token usage for task {self.task_id}: {e}")
            try:
                self.db_session.rollback()
            except Exception as rollback_error:
                logger.warning(
                    f"Failed to rollback DB session for task {self.task_id}: {rollback_error}"
                )

            try:
                task_exists = (
                    self.db_session.query(self.task_model.id)
                    .filter(self.task_model.id == self.task_id)
                    .first()
                    is not None
                )
                if not task_exists:
                    self._is_tracking = False
                    logger.info(
                        f"Stopped periodic token tracking for deleted task {self.task_id}"
                    )
            except Exception:
                self._is_tracking = False

            import traceback

            traceback.print_exc()

    async def start_periodic_updates(self) -> None:
        """Start periodic background updates to the database.

        This creates an asyncio background task that will periodically
        update the token usage in the database.
        """
        if self._is_tracking:
            logger.warning(f"Periodic updates already active for task {self.task_id}")
            return

        self._is_tracking = True

        async def update_loop() -> None:
            logger.debug(f"[update_loop] Starting update loop for task {self.task_id}")
            iteration = 0
            while self._is_tracking:
                iteration += 1
                logger.debug(
                    f"[update_loop] Iteration {iteration} for task {self.task_id}"
                )
                await asyncio.sleep(self.update_interval_seconds)
                if self._is_tracking:
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
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        self._is_tracking = False
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

    def interrupt_reason_for_quota(self) -> str | None:
        """Per-step interrupt-checker: return a reason when this run's live-so-far
        usage would push the team over a run-gated quota, so a single long or
        expensive run is stopped mid-flight instead of only being metered at
        completion (``complete_tracking`` still meters the partial usage on exit).

        Wired as the run's ``interrupt_checker`` and polled at every safe point
        (each LLM reply / tool call). Synchronous and best-effort: it reuses the
        exact turn-delta the metering path computes, and swallows errors so a
        quota-infra hiccup fails open rather than wedging a run.

        When it trips it records the reason in ``quota_interrupt_reason`` so the
        run's caller can surface *why* the run stopped (the pattern interrupt
        path itself would otherwise flip silently to PAUSED).
        """
        if not self._is_tracking:
            return None
        try:
            from ..services.quota_hooks import check_run_progress_gate

            delta_details, delta_actions = self._turn_delta()
            reason = check_run_progress_gate(
                self.db_session,
                self._user_id,
                delta_details,
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

    async def complete_tracking(self) -> TokenUsage:
        """Complete tracking and return final statistics.

        Stops periodic updates and saves final token usage to the database.

        Returns:
            Final TokenUsage object with all statistics

        Raises:
            RuntimeError: If tracking was not started
        """
        if not self._is_tracking:
            raise RuntimeError(f"Task {self.task_id} is not being tracked")

        # Stop periodic updates first to prevent race conditions
        await self.stop_periodic_updates()

        # Get final token usage and update database
        logger.info(f"Force updating token usage for task {self.task_id}")
        usage = get_token_usage()

        # Meter quota FIRST — on the still-clean session, from in-memory
        # counters — so a transient commit failure below cannot zero-bill a
        # token/tool-heavy turn. actions = tool calls; credits derive from tokens.
        try:
            from ..services.quota_hooks import record_usage

            delta_details, delta_actions = self._turn_delta(usage)
            record_usage(
                self.db_session,
                self._user_id,
                delta_details,
                delta_actions,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Quota usage recording failed for task {self.task_id}: {e}")

        # Update database with final statistics
        self.task.input_tokens = usage.input_tokens
        self.task.output_tokens = usage.output_tokens
        self.task.total_tokens = usage.total_tokens
        self.task.llm_calls = usage.llm_calls
        self.task.token_usage_details = usage.to_dict()["details"]

        try:
            self.db_session.commit()
        except Exception as e:
            logger.warning(
                f"Failed to commit final token usage for task {self.task_id}: {e}"
            )
            self.db_session.rollback()
            return usage

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
            self._last_reported_usage = usage

        logger.info(
            f"Completed token tracking for task {self.task_id}: "
            f"input={usage.input_tokens}, output={usage.output_tokens}, "
            f"total={usage.total_tokens}, calls={usage.llm_calls}"
        )

        return usage

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
        db_session: Any,
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
