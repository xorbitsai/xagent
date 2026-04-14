"""Progress manager for coordinating progress tracking across RAG operations."""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from ..core.schemas import DocumentProcessingStatus, TaskProgress
from ..utils import validate_and_convert_user_id
from .persistence import ProgressPersistence
from .realtime import ProgressBroadcaster, progress_broadcaster

logger = logging.getLogger(__name__)


class ProgressManager:
    """Central coordinator for progress tracking across RAG operations.

    Manages multiple concurrent tasks, coordinates persistence and real-time
    broadcasting, and provides a unified interface for progress updates.
    """

    _instance: Optional[ProgressManager] = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> ProgressManager:
        """Ensure singleton pattern."""
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = super(ProgressManager, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        persistence: Optional[ProgressPersistence] = None,
        broadcaster: Optional[ProgressBroadcaster] = None,
    ):
        # Only initialize once
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.persistence = persistence or ProgressPersistence()
        self.broadcaster = broadcaster or progress_broadcaster
        self._active_tasks: Dict[str, TaskProgress] = {}
        self._lock = threading.RLock()
        self._initialized: bool = True

        # For broadcasting from sync context to async broadcaster
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    def create_task(
        self,
        task_type: str,
        task_id: Optional[str] = None,
        user_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new progress tracking task.

        Args:
            task_type: Type of task ("ingestion", "retrieval", etc.)
            task_id: Optional custom task ID, auto-generated if not provided
            user_id: Optional user identifier for isolation (must be int or convertible to int)
            metadata: Optional metadata for the task

        Returns:
            The task ID

        Raises:
            ConfigurationError: If user_id is provided but cannot be converted to int
        """
        # Validate and convert user_id to int if provided
        user_id = validate_and_convert_user_id(user_id)

        # Lazily cleanup old tasks when creating a new one
        self._cleanup_stale_tasks()

        task_id = task_id or f"{task_type}_{uuid.uuid4().hex[:8]}"

        with self._lock:
            if task_id in self._active_tasks:
                existing_task = self._active_tasks[task_id]
                if existing_task.status in (
                    DocumentProcessingStatus.SUCCESS,
                    DocumentProcessingStatus.FAILED,
                    DocumentProcessingStatus.CANCELLED,
                ):
                    logger.info(f"Task {task_id} exists in terminal state, resetting")
                    self._active_tasks.pop(task_id, None)
                else:
                    logger.warning(f"Task {task_id} already exists, reusing")
                    return task_id

            task_progress = TaskProgress(
                task_id=task_id,
                user_id=user_id,
                task_type=task_type,
                status=DocumentProcessingStatus.PENDING,
                metadata=metadata or {},
            )

            self._active_tasks[task_id] = task_progress

            # Persist initial state
            # Critical failure: if we can't save initial state, we should fail early
            self.persistence.save_task_progress(task_progress)

            logger.info(
                f"Created progress task: {task_id} ({task_type}) for user {user_id}"
            )
            return task_id

    def _cleanup_stale_tasks(self, max_age_seconds: int = 60) -> None:
        """Cleanup completed tasks that are older than max_age_seconds."""
        import time

        now = time.time()
        tasks_to_remove = []

        with self._lock:
            for task_id, task in self._active_tasks.items():
                if task.status in (
                    DocumentProcessingStatus.SUCCESS,
                    DocumentProcessingStatus.FAILED,
                    DocumentProcessingStatus.CANCELLED,
                ):
                    if task.end_time and (now - task.end_time > max_age_seconds):
                        tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                self._active_tasks.pop(task_id, None)
                logger.debug(f"Cleaned up stale task: {task_id}")

    def get_task_progress(self, task_id: str) -> Optional[TaskProgress]:
        """Get current progress for a task."""
        with self._lock:
            return self._active_tasks.get(task_id)

    def update_task_progress(
        self,
        task_id: str,
        status: Optional[DocumentProcessingStatus] = None,
        current_step: Optional[str] = None,
        overall_progress: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Update progress for a task."""
        with self._lock:
            task = self._active_tasks.get(task_id)
            if not task:
                logger.warning(f"Task {task_id} not found for update")
                return

            # Update fields
            if status is not None:
                task.status = status
            if current_step is not None:
                task.current_step = current_step
            if overall_progress is not None:
                task.overall_progress = max(0.0, min(1.0, overall_progress))
            if metadata:
                for key, value in metadata.items():
                    if (
                        key == "steps"
                        and isinstance(value, dict)
                        and isinstance(task.metadata.get("steps"), dict)
                    ):
                        task.metadata["steps"].update(value)
                    else:
                        task.metadata[key] = value

            # Update timestamps
            import time

            if status == DocumentProcessingStatus.RUNNING and not task.start_time:
                task.start_time = time.time()
            elif status in (
                DocumentProcessingStatus.SUCCESS,
                DocumentProcessingStatus.FAILED,
                DocumentProcessingStatus.CANCELLED,
            ):
                if not task.end_time:
                    task.end_time = time.time()

            # Persist
            try:
                self.persistence.save_task_progress(task)

                # Broadcast (handling async from sync)
                if self.broadcaster:
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self.broadcaster.broadcast_progress(task))
                    except RuntimeError:
                        if self._main_loop:
                            asyncio.run_coroutine_threadsafe(
                                self.broadcaster.broadcast_progress(task),
                                self._main_loop,
                            )
            except Exception as e:
                # For updates, we log as error but don't crash the pipeline
                logger.error(f"Failed to persist/broadcast task update: {e}")

    def complete_task(self, task_id: str, success: bool = True) -> None:
        """Mark a task as completed."""
        status = (
            DocumentProcessingStatus.SUCCESS
            if success
            else DocumentProcessingStatus.FAILED
        )
        overall_progress = 1.0 if success else None
        self.update_task_progress(
            task_id, status=status, overall_progress=overall_progress
        )

        try:
            self._cleanup_stale_tasks()
        except Exception:
            pass

    @contextmanager
    def track_task(
        self,
        task_type: str,
        task_id: Optional[str] = None,
        user_id: Optional[int] = None,
        **metadata: Any,
    ) -> Iterator[Any]:
        """Context manager for tracking a complete task lifecycle."""
        # Circular import prevention
        from .tracker import ProgressTracker

        task_id = self.create_task(task_type, task_id, user_id, metadata)
        tracker = ProgressTracker(self, task_id)

        try:
            self.update_task_progress(task_id, status=DocumentProcessingStatus.RUNNING)
            yield tracker
            self.complete_task(task_id, success=True)
        except Exception as e:
            logger.exception(f"Task {task_id} failed: {e}")
            self.update_task_progress(
                task_id,
                status=DocumentProcessingStatus.FAILED,
                metadata={"error": str(e)},
            )
            self.complete_task(task_id, success=False)
            raise

    def get_active_tasks(
        self, user_id: Optional[int] = None
    ) -> Dict[str, TaskProgress]:
        """Get all currently active tasks, optionally filtered by user.

        Args:
            user_id: Optional user ID to filter tasks (must be int or convertible to int)

        Returns:
            Dictionary of active tasks, optionally filtered by user_id

        Raises:
            ConfigurationError: If user_id cannot be converted to int
        """
        # Validate and convert user_id to int if provided
        user_id = validate_and_convert_user_id(user_id)

        self._cleanup_stale_tasks()

        with self._lock:
            if user_id:
                return {
                    tid: t
                    for tid, t in self._active_tasks.items()
                    if t.user_id == user_id
                }
            return self._active_tasks.copy()


# Global singleton instance
_progress_manager: Optional[ProgressManager] = None


def get_progress_manager() -> ProgressManager:
    """Get or create the global progress manager singleton."""
    global _progress_manager
    if _progress_manager is None:
        _progress_manager = ProgressManager()
    return _progress_manager
