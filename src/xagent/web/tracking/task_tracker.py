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
    reset_token_usage,
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

        # Load the task model
        from ..models.task import Task as TaskModel

        self.task_model = TaskModel
        self.task = db_session.query(TaskModel).filter(TaskModel.id == task_id).first()

        if not self.task:
            raise ValueError(f"Task {task_id} not found")

    async def start_tracking(self) -> None:
        """Start tracking token usage for this task."""
        if self._is_tracking:
            logger.warning(f"Task {self.task_id} is already being tracked")
            return

        # Initialize token context
        reset_token_usage()

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
