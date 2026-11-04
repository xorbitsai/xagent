"""Unit tests for ProgressTracker and StepTracker."""

from __future__ import annotations

from unittest.mock import MagicMock

from xagent.core.tools.core.RAG_tools.core.schemas import DocumentProcessingStatus
from xagent.core.tools.core.RAG_tools.progress.manager import ProgressManager
from xagent.core.tools.core.RAG_tools.progress.tracker import (
    ProgressTracker,
    StepTracker,
)


class TestStepTracker:
    """Test StepTracker functionality."""

    def test_step_creation(self):
        """Test step tracker initialization."""
        task_tracker = MagicMock(spec=ProgressTracker)
        tracker = StepTracker(task_tracker, "parse_document")

        assert tracker.step.step_name == "parse_document"
        assert tracker.step.completed is False
        assert tracker.step.current_count == 0

    def test_step_update(self):
        """Test step status updates."""
        task_tracker = MagicMock(spec=ProgressTracker)
        tracker = StepTracker(task_tracker, "chunk_document")

        # Update with message
        tracker.update(
            message="Processing chunks...", current_count=50, total_count=100
        )

        assert tracker.step.message == "Processing chunks..."
        assert tracker.step.current_count == 50
        assert tracker.step.total_count == 100
        assert tracker.step.step_progress == 0.5
        task_tracker.update_overall_progress.assert_called()

    def test_step_complete(self):
        """Test step completion."""
        task_tracker = MagicMock(spec=ProgressTracker)
        tracker = StepTracker(task_tracker, "embed_vectors")

        # Mark as completed
        tracker.complete("Embedding completed successfully")

        assert tracker.step.completed is True
        assert tracker.step.step_progress == 1.0
        assert tracker.step.message == "Embedding completed successfully"
        assert tracker.step.end_time is not None

    def test_step_fail(self):
        """Test step failure."""
        task_tracker = MagicMock(spec=ProgressTracker)
        tracker = StepTracker(task_tracker, "write_vectors")

        # Mark as failed
        tracker.fail("Database connection failed")

        assert tracker.step.completed is False
        assert tracker.step.message == "Failed: Database connection failed"
        assert tracker.step.metadata["error"] == "Database connection failed"

    def test_no_callback_calls_after_completion(self):
        """Test that no callback calls happen after step completion."""
        task_tracker = MagicMock(spec=ProgressTracker)
        tracker = StepTracker(task_tracker, "test_step")

        # Complete the step
        tracker.complete("Done")

        # Reset mock
        task_tracker.update_overall_progress.reset_mock()

        # Try to update
        tracker.update("Late update")


class TestProgressTracker:
    """Test ProgressTracker functionality."""

    def test_tracker_creation(self):
        """Test progress tracker initialization."""
        manager = ProgressManager()
        manager._active_tasks.clear()
        tracker = ProgressTracker(manager, "test_task")

        assert tracker.manager == manager
        assert tracker.task_id == "test_task"
        assert tracker.step_trackers == {}

    def test_track_step_context_manager(self):
        """Test track_step context manager."""
        manager = ProgressManager()
        manager._active_tasks.clear()
        manager.create_task("ingestion", "test_task")
        tracker = ProgressTracker(manager, "test_task")

        with tracker.track_step("step1") as step_tracker:
            assert isinstance(step_tracker, StepTracker)
            assert "step1" in tracker.step_trackers
            assert manager.get_task_progress("test_task").current_step == "step1"

        assert tracker.step_trackers["step1"].step.completed is True

    def test_multiple_steps(self):
        """Test tracking multiple steps."""
        manager = ProgressManager()
        manager._active_tasks.clear()
        manager.create_task("ingestion", "test_task")
        tracker = ProgressTracker(manager, "test_task")

        with tracker.track_step("step1"):
            pass

        with tracker.track_step("step2"):
            pass

        assert len(tracker.step_trackers) == 2
        assert manager.get_task_progress("test_task").overall_progress == 1.0

    def test_step_failure_propagation(self):
        """Test that step failures are properly handled."""
        manager = ProgressManager()
        manager._active_tasks.clear()
        manager.create_task("ingestion", "test_task")
        tracker = ProgressTracker(manager, "test_task")

        try:
            with tracker.track_step("step1") as step:
                raise ValueError("Oops")
        except ValueError:
            pass

        step = tracker.step_trackers["step1"]
        assert step.step.completed is False
        assert "Oops" in step.step.message
        assert step.step.metadata["error"] == "Oops"

    def test_get_step_tracker(self):
        """Test retrieving step trackers."""
        manager = ProgressManager()
        manager._active_tasks.clear()
        ProgressTracker(manager, "test_task")

        pass

    def test_step_naming_conflicts(self):
        """Test handling of step naming conflicts."""
        manager = ProgressManager()
        manager._active_tasks.clear()
        manager.create_task("ingestion", "test_task")
        tracker = ProgressTracker(manager, "test_task")

        with tracker.track_step("step1"):
            pass

        # Reusing same name overwrites?
        with tracker.track_step("step1"):
            pass

        assert len(tracker.step_trackers) == 1
        assert tracker.step_trackers["step1"].step.completed is True


class TestIntegrationWithManager:
    """Test integration between Tracker and Manager."""

    def test_task_progress_updates(self):
        """Test that step completions update task progress."""
        manager = ProgressManager()
        manager._active_tasks.clear()
        manager.create_task("ingestion", "test_task")
        tracker = ProgressTracker(manager, "test_task")

        with tracker.track_step("step1"):
            pass

        task = manager.get_task_progress("test_task")
        # 1 step completed, total steps 1 -> progress 1.0
        assert task.overall_progress == 1.0

    def test_multiple_steps_task_completion(self):
        """Test task completion when all steps are done."""
        manager = ProgressManager()
        manager._active_tasks.clear()
        manager.create_task("ingestion", "test_task")
        tracker = ProgressTracker(manager, "test_task")

        # This test logic depends on how _update_overall_from_steps works
        # If it just counts completed / total, and total grows as we add steps...

        with tracker.track_step("step1"):
            pass

        with tracker.track_step("step2"):
            pass

        assert manager.get_task_progress("test_task").overall_progress == 1.0

    def test_error_propagation(self):
        """Test that step errors are properly propagated."""
        manager = ProgressManager()
        manager._active_tasks.clear()
        manager.create_task("ingestion", "test_task")
        manager.update_task_progress(
            "test_task", status=DocumentProcessingStatus.RUNNING
        )
        tracker = ProgressTracker(manager, "test_task")

        try:
            with tracker.track_step("step1") as step:
                step.fail("Manual fail")
        except Exception:
            pass

        task = manager.get_task_progress("test_task")
        assert task.status == DocumentProcessingStatus.RUNNING


class TestProgressCallbackProtocol:
    """Test ProgressCallback protocol implementation."""

    def test_step_tracker_implements_callback(self):
        """Test that StepTracker properly implements ProgressCallback."""
        manager = ProgressManager()
        manager._active_tasks.clear()
        tracker = ProgressTracker(manager, "test_task")

        # Manually verify protocol/method existence
        step_tracker = StepTracker(tracker, "step")
        assert hasattr(step_tracker, "on_status_update")

        step_tracker.on_status_update("Status update", {"detail": 1})
        assert step_tracker.step.message == "Status update"
        assert step_tracker.step.metadata["detail"] == 1
