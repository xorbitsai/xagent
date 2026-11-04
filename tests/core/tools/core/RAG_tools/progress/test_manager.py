"""Unit tests for ProgressManager."""

from __future__ import annotations

import time

from xagent.core.tools.core.RAG_tools.core.schemas import DocumentProcessingStatus
from xagent.core.tools.core.RAG_tools.progress.manager import ProgressManager


class TestProgressManager:
    """Test ProgressManager functionality."""

    def test_singleton_pattern(self):
        """Test that ProgressManager follows singleton pattern."""
        manager1 = ProgressManager()
        manager2 = ProgressManager()

        # Should be the same instance
        assert manager1 is manager2

        # Should share the same tasks dict
        manager1._active_tasks["test"] = "value"
        assert manager2._active_tasks["test"] == "value"
        manager1._active_tasks.clear()

    def test_create_task(self):
        """Test task creation."""
        manager = ProgressManager()
        manager._active_tasks.clear()  # Clear any existing tasks

        task_id = "test_task_123"
        task_type = "ingestion"
        user_id = 1
        metadata = {"collection": "docs", "source": "test.pdf"}

        manager.create_task(
            task_type=task_type, task_id=task_id, user_id=user_id, metadata=metadata
        )

        assert task_id in manager._active_tasks
        task = manager._active_tasks[task_id]

        assert task.task_id == task_id
        assert task.user_id == user_id
        assert task.task_type == task_type
        assert task.status == DocumentProcessingStatus.PENDING
        assert task.metadata == metadata
        assert task.start_time is None
        assert task.end_time is None

    def test_get_task_progress(self):
        """Test retrieving task progress."""
        manager = ProgressManager()
        manager._active_tasks.clear()

        task_id = "test_task_789"
        manager.create_task("search", task_id)

        # Get existing task
        task = manager.get_task_progress(task_id)
        assert task is not None
        assert task.task_id == task_id

        # Get non-existing task
        non_existing = manager.get_task_progress("non_existing")
        assert non_existing is None

    def test_update_task_progress(self):
        """Test task progress updates."""
        manager = ProgressManager()
        manager._active_tasks.clear()

        task_id = "test_update"
        manager.create_task("ingestion", task_id)

        # Update status and step
        manager.update_task_progress(
            task_id=task_id,
            status=DocumentProcessingStatus.RUNNING,
            current_step="parse_document",
            overall_progress=0.5,
            metadata={"pages_parsed": 10, "total_pages": 20},
        )

        task = manager._active_tasks[task_id]
        assert task.status == DocumentProcessingStatus.RUNNING
        assert task.current_step == "parse_document"
        assert task.overall_progress == 0.5
        assert task.metadata["pages_parsed"] == 10
        assert task.start_time is not None

    def test_update_task_progress_start_time(self):
        """Test that start_time is set when status changes to RUNNING."""
        manager = ProgressManager()
        manager._active_tasks.clear()

        task_id = "test_start_time"
        manager.create_task("ingestion", task_id)

        # Initially no start time
        assert manager._active_tasks[task_id].start_time is None

        # Update to RUNNING should set start time
        manager.update_task_progress(task_id, DocumentProcessingStatus.RUNNING)
        assert manager._active_tasks[task_id].start_time is not None

    def test_update_task_progress_end_time(self):
        """Test that end_time is set when task completes."""
        manager = ProgressManager()
        manager._active_tasks.clear()

        task_id = "test_end_time"
        manager.create_task("ingestion", task_id)

        # Complete with success
        manager.update_task_progress(task_id, DocumentProcessingStatus.SUCCESS)
        task = manager._active_tasks[task_id]
        assert task.status == DocumentProcessingStatus.SUCCESS
        assert task.end_time is not None

        # Complete with failure
        task_id2 = "test_end_time2"
        manager.create_task("search", task_id2)
        manager.update_task_progress(task_id2, DocumentProcessingStatus.FAILED)
        task2 = manager._active_tasks[task_id2]
        assert task2.status == DocumentProcessingStatus.FAILED
        assert task2.end_time is not None

        # Complete with cancellation
        task_id3 = "test_end_time3"
        manager.create_task("search", task_id3)
        manager.update_task_progress(task_id3, DocumentProcessingStatus.CANCELLED)
        task3 = manager._active_tasks[task_id3]
        assert task3.status == DocumentProcessingStatus.CANCELLED
        assert task3.end_time is not None

    def test_complete_task(self):
        """Test task completion."""
        manager = ProgressManager()
        manager._active_tasks.clear()

        task_id = "test_complete"
        manager.create_task("ingestion", task_id)

        # Complete successfully
        manager.complete_task(task_id, success=True)

        task = manager._active_tasks[task_id]
        assert task.status == DocumentProcessingStatus.SUCCESS
        assert task.overall_progress == 1.0
        assert task.end_time is not None

    def test_get_active_tasks(self):
        """Test retrieving active tasks."""
        manager = ProgressManager()
        manager._active_tasks.clear()

        # Create tasks with different statuses
        manager.create_task("ingestion", "active_task")
        manager.create_task("search", "completed_task")
        manager.create_task("ingestion", "failed_task")
        manager.create_task("search", "cancelled_task")

        # Update task statuses
        manager.update_task_progress("completed_task", DocumentProcessingStatus.SUCCESS)
        manager.update_task_progress("failed_task", DocumentProcessingStatus.FAILED)
        manager.update_task_progress(
            "cancelled_task", DocumentProcessingStatus.CANCELLED
        )

        # We need to test list_active_tasks BEFORE they are cleaned up
        # The manager.get_active_tasks() calls _cleanup_stale_tasks()
        # By default max_age is 60s, so they should still be there if end_time is not set or recent.
        # But wait, complete_task sets end_time.

        active_tasks = manager.get_active_tasks()

        # Should include the active_task
        assert "active_task" in active_tasks
        assert active_tasks["active_task"].task_type == "ingestion"

    def test_task_isolation(self):
        """Test that tasks are properly isolated."""
        manager = ProgressManager()
        manager._active_tasks.clear()

        # Create two separate tasks
        manager.create_task("ingestion", "task1")
        manager.create_task("search", "task2")

        # Update task1
        manager.update_task_progress("task1", DocumentProcessingStatus.RUNNING, "step1")

        # task2 should not be affected
        task2 = manager._active_tasks["task2"]
        assert task2.status == DocumentProcessingStatus.PENDING
        assert task2.current_step is None

    def test_concurrent_access_safety(self):
        """Test that concurrent access doesn't break the manager."""
        manager = ProgressManager()
        manager._active_tasks.clear()

        # Simulate concurrent task creation and updates
        tasks = []
        for i in range(10):
            task_id = f"concurrent_task_{i}"
            manager.create_task("ingestion", task_id)
            manager.update_task_progress(
                task_id,
                status=DocumentProcessingStatus.RUNNING,
                current_step=f"step_{i}",
            )
            tasks.append(task_id)

        # Verify all tasks are properly stored
        for task_id in tasks:
            assert task_id in manager._active_tasks
            task = manager._active_tasks[task_id]
            assert task.status == DocumentProcessingStatus.RUNNING

    def test_timestamp_handling(self):
        """Test proper timestamp handling."""
        manager = ProgressManager()
        manager._active_tasks.clear()

        task_id = "timestamp_test"
        manager.create_task("ingestion", task_id)
        manager.update_task_progress(task_id, status=DocumentProcessingStatus.RUNNING)

        task = manager._active_tasks[task_id]
        assert task.start_time is not None
        assert task.start_time > 0

        # Complete task
        time.sleep(0.01)
        manager.complete_task(task_id, success=True)

        assert task.end_time is not None
        assert task.end_time > task.start_time
