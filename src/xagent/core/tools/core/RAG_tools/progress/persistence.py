"""Progress persistence layer for storing and retrieving progress data."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from ..core.exceptions import ProgressPersistenceError
from ..core.schemas import DocumentProcessingStatus, TaskProgress
from ..utils import validate_and_convert_user_id

logger = logging.getLogger(__name__)


class ProgressPersistence:
    """Handles persistence of progress data to various storage backends."""

    MAX_FILENAME_BYTES = 255

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize persistence layer.

        Args:
            storage_dir: Directory for storing progress data.
                        Defaults to project_root/data/progress.
        """
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            # Try to find the data directory relative to this file
            # src/xagent/core/tools/core/RAG_tools/progress/persistence.py -> 7 levels up to root
            try:
                base_dir = Path(__file__).resolve().parents[7]
                self.storage_dir = base_dir / "data" / "progress"
            except (IndexError, ValueError):
                # Fallback to current working directory if path structure is unexpected
                self.storage_dir = Path("data/progress")

        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_task_progress(self, task_progress: TaskProgress) -> None:
        """Save task progress to persistent storage.

        Args:
            task_progress: The task progress to save
        """
        file_path = None

        try:
            file_path = self._get_task_file_path(task_progress.task_id)

            # Convert to dict for JSON serialization
            data = {
                "task_id": task_progress.task_id,
                "user_id": task_progress.user_id,
                "task_type": task_progress.task_type,
                "status": task_progress.status,
                "current_step": task_progress.current_step,
                "overall_progress": task_progress.overall_progress,
                "start_time": task_progress.start_time,
                "end_time": task_progress.end_time,
                "metadata": task_progress.metadata,
            }

            # A truncated file would survive as broken JSON that
            # cleanup_old_tasks can never date, and so never remove. mkstemp
            # rather than a name derived from file_path: two writers racing on
            # one task id would share that name, and it has no byte budget left.
            fd, tmp_name = tempfile.mkstemp(dir=self.storage_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(tmp_name, file_path)
            finally:
                Path(tmp_name).unlink(missing_ok=True)

        except Exception as e:
            location = self.storage_dir if file_path is None else file_path
            # %r: a task id carrying a lone surrogate from os.fsdecode cannot be
            # encoded by a UTF-8 log handler.
            logger.error(
                "Failed to save task progress for %r to %s: %s",
                task_progress.task_id,
                location,
                e,
            )
            raise ProgressPersistenceError(
                f"Failed to save task progress to {location}: {e}",
                details={
                    "task_id": task_progress.task_id,
                    "file_path": str(location),
                    "error": str(e),
                },
            ) from e

    def load_task_progress(self, task_id: str) -> Optional[TaskProgress]:
        """Load task progress from persistent storage.

        Args:
            task_id: The task identifier

        Returns:
            TaskProgress if found and valid, None otherwise
        """
        try:
            file_path = self._get_task_file_path(task_id)

            if not file_path.exists():
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return TaskProgress(**data)

        except Exception as e:
            logger.error("Failed to load task progress for %r: %s", task_id, e)
            return None

    def delete_task_progress(self, task_id: str) -> bool:
        """Delete task progress from persistent storage.

        Args:
            task_id: The task identifier

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            file_path = self._get_task_file_path(task_id)
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error("Failed to delete task progress for %r: %s", task_id, e)
            return False

    def list_active_tasks(
        self, task_type: Optional[str] = None, user_id: Optional[int] = None
    ) -> List[TaskProgress]:
        """List all active (non-completed) tasks.

        Args:
            task_type: Optional filter by task type
            user_id: Optional filter by user ID for tenant isolation (must be int or convertible to int)

        Returns:
            List of active TaskProgress objects

        Raises:
            ConfigurationError: If user_id cannot be converted to int
        """
        # Validate and convert user_id to int if provided
        user_id = validate_and_convert_user_id(user_id)

        active_tasks = []

        try:
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    task = TaskProgress(**data)

                    # Multi-tenant isolation
                    if user_id and task.user_id != user_id:
                        continue

                    # Filter by status (active = not completed/failed/cancelled)
                    if task.status in (
                        DocumentProcessingStatus.SUCCESS,
                        DocumentProcessingStatus.FAILED,
                        DocumentProcessingStatus.CANCELLED,
                    ):
                        continue

                    # Filter by task type if specified
                    if task_type and task.task_type != task_type:
                        continue

                    active_tasks.append(task)

                except Exception as e:
                    logger.warning("Failed to load task from %s: %s", file_path, e)
                    continue

        except Exception as e:
            logger.error("Failed to list active tasks: %s", e)

        return active_tasks

    def list_all_tasks(self, user_id: Optional[int] = None) -> List[TaskProgress]:
        """List all tasks stored in persistence.

        Args:
            user_id: Optional filter by user ID (must be int or convertible to int)

        Returns:
            List of all TaskProgress objects

        Raises:
            ConfigurationError: If user_id cannot be converted to int
        """
        # Validate and convert user_id to int if provided
        user_id = validate_and_convert_user_id(user_id)

        all_tasks = []

        try:
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    task = TaskProgress(**data)

                    if user_id and task.user_id != user_id:
                        continue

                    all_tasks.append(task)
                except Exception as e:
                    logger.warning("Failed to load task from %s: %s", file_path, e)
                    continue
        except Exception as e:
            logger.error("Failed to list all tasks: %s", e)

        return all_tasks

    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Clean up old completed/failed tasks.

        Args:
            max_age_hours: Maximum age in hours for cleanup

        Returns:
            Number of tasks cleaned up
        """
        import time

        cleaned_count = 0
        cutoff_time = time.time() - (max_age_hours * 3600)

        try:
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    task = TaskProgress(**data)

                    # Only cleanup completed/failed/cancelled tasks
                    if task.status not in (
                        DocumentProcessingStatus.SUCCESS,
                        DocumentProcessingStatus.FAILED,
                        DocumentProcessingStatus.CANCELLED,
                    ):
                        continue

                    # Check if task ended long enough ago
                    if task.end_time and task.end_time < cutoff_time:
                        file_path.unlink()
                        cleaned_count += 1

                except Exception as e:
                    logger.warning("Failed to process task file %s: %s", file_path, e)
                    continue

        except Exception as e:
            logger.error("Failed during cleanup: %s", e)

        logger.info("Cleaned up %s old task progress files", cleaned_count)
        return cleaned_count

    def _get_task_file_path(self, task_id: str) -> Path:
        """Get the file path for a task's progress data.

        Args:
            task_id: The task identifier

        Returns:
            Path to the task's progress file
        """
        # Sanitize task_id for filename
        safe_task_id = "".join(c for c in task_id if c.isalnum() or c in "-_.").strip()
        if not safe_task_id:
            safe_task_id = "unknown"

        suffix = ".json"
        # task_id is caller-supplied and unbounded; filesystems cap a filename at
        # 255 *bytes*, and a CJK collection name is 3 bytes per character.
        budget = self.MAX_FILENAME_BYTES - len(suffix)
        encoded = safe_task_id.encode("utf-8")
        if len(encoded) > budget:
            # surrogatepass: an OS path with undecodable bytes must still hash.
            # 64-bit digest, so ids differing before the cap differ in practice.
            digest = hashlib.sha256(
                task_id.encode("utf-8", "surrogatepass")
            ).hexdigest()[:16]
            prefix = encoded[: max(budget - len(digest) - 1, 0)].decode(
                "utf-8", "ignore"
            )
            safe_task_id = f"{prefix}-{digest}"

        return self.storage_dir / f"{safe_task_id}{suffix}"
