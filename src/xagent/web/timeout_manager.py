import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class TimeoutManager:
    """
    Manages task timeouts and triggers auto-continuation.
    """

    def __init__(self) -> None:
        # task_id -> expires_at (unix timestamp)
        self.tasks: Dict[int, float] = {}
        self.callback: Optional[Callable[[int], Any]] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

    def set_callback(self, callback: Callable[[int], Any]) -> None:
        """Set the callback function to execute when a timeout occurs"""
        self.callback = callback

    def add_task(self, task_id: int, timeout_seconds: int) -> None:
        """Register a task with a timeout"""
        expires_at = datetime.now(timezone.utc).timestamp() + timeout_seconds
        self.tasks[task_id] = expires_at
        logger.info(
            f"Registered timeout for task {task_id}: {timeout_seconds}s (expires at {expires_at})"
        )

    def remove_task(self, task_id: int) -> None:
        """Remove a task from timeout monitoring"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            logger.debug(f"Removed timeout for task {task_id}")

    async def start(self) -> None:
        """Start the monitoring loop"""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("TimeoutManager monitoring loop started")

    async def stop(self) -> None:
        """Stop the monitoring loop"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("TimeoutManager monitoring loop stopped")

    async def _monitor_loop(self) -> None:
        """Periodic check for expired tasks"""
        while self._running:
            try:
                now = datetime.now(timezone.utc).timestamp()
                expired_tasks = [
                    task_id
                    for task_id, expires_at in self.tasks.items()
                    if now >= expires_at
                ]

                for task_id in expired_tasks:
                    logger.info(f"Task {task_id} timed out. Triggering auto-continue.")
                    # Remove first to avoid repeated triggering if callback fails or takes time
                    self.remove_task(task_id)

                    if self.callback:
                        try:
                            # Execute callback (async or sync)
                            if asyncio.iscoroutinefunction(self.callback):
                                await self.callback(task_id)
                            else:
                                self.callback(task_id)
                        except Exception as e:
                            logger.error(
                                f"Error in timeout callback for task {task_id}: {e}",
                                exc_info=True,
                            )

                # Check every 1 second
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in TimeoutManager loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Wait longer on error


# Global instance
timeout_manager = TimeoutManager()
