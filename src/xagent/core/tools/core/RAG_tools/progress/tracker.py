"""Progress trackers for detailed step-by-step progress monitoring."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Protocol

from ..core.schemas import DocumentProcessingStatus

if TYPE_CHECKING:
    from .manager import ProgressManager

logger = logging.getLogger(__name__)


class ProgressCallback(Protocol):
    """Abstract interface for progress status callbacks.

    This interface allows different components (parsers, processors, etc.)
    to report their current status without worrying about progress percentages.

    Example usage with parsers:
        ```python
        # For DeepDoc parser
        def parse_with_progress(parser, file_path, callback):
            adapter = DeepDocProgressAdapter(callback)
            parser.parse_into_bboxes(file_path, callback=adapter.get_callback())

        # For other parsers
        def parse_fallback(parser, file_path, callback):
            callback.on_status_update("开始解析文档")
            # ... parsing logic ...
            callback.on_status_update("文档解析完成")
        ```
    """

    def on_status_update(
        self, status: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called when status changes.

        Args:
            status: Human-readable status message (e.g., "OCR finished", "Layout analysis completed")
            details: Optional additional details about the status
        """
        ...


@dataclass
class StepProgress:
    """Represents progress of a single step within a task."""

    step_name: str
    completed: bool = False
    current_count: int = 0
    total_count: Optional[int] = None  # May be unknown initially
    step_progress: float = 0.0  # 0.0 to 1.0 for this step
    message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds if step has ended."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return None


class ProgressTracker:
    """Tracks overall task progress and coordinates step-level tracking."""

    def __init__(self, manager: "ProgressManager", task_id: str) -> None:
        self.manager = manager
        self.task_id = task_id
        self.step_trackers: Dict[str, StepTracker] = {}

    def update_overall_progress(
        self,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        """Update overall task progress.

        Args:
            progress: Overall progress (0.0 to 1.0), or None to leave unchanged
            current_step: Current active step name
            **metadata: Additional metadata
        """
        self.manager.update_task_progress(
            self.task_id,
            overall_progress=progress,
            current_step=current_step,
            metadata=metadata,
        )

    def update_status(self, status: DocumentProcessingStatus, **metadata: Any) -> None:
        """Update task status.

        Args:
            status: New status
            **metadata: Additional metadata
        """
        self.manager.update_task_progress(
            self.task_id, status=status, metadata=metadata
        )

    @contextmanager
    def track_step(
        self,
        step_name: str,
        total_count: Optional[int] = None,
        message: str = "",
        **metadata: Any,
    ) -> Iterator[StepTracker]:
        """Context manager for tracking a step's progress.

        Args:
            step_name: Name of the step
            total_count: Total items to process (if known)
            message: Initial message
            **metadata: Additional step metadata

        Yields:
            StepTracker instance
        """
        tracker = StepTracker(self, step_name, total_count, message, metadata)
        self.step_trackers[step_name] = tracker

        try:
            tracker.start()
            yield tracker
            tracker.complete()
        except Exception as e:
            tracker.fail(str(e))
            raise
        finally:
            # Step completed, update overall progress
            self._update_overall_from_steps()

    def _update_overall_from_steps(self) -> None:
        """Calculate overall progress based on completed steps."""
        if not self.step_trackers:
            return

        completed_steps = sum(
            1 for tracker in self.step_trackers.values() if tracker.step.completed
        )
        total_steps = len(self.step_trackers)

        # Simple completion-based progress
        overall_progress = completed_steps / total_steps

        # Find current active step
        current_step = None
        for tracker in self.step_trackers.values():
            if not tracker.step.completed:
                current_step = tracker.step.step_name
                break

        self.update_overall_progress(overall_progress, current_step)


class StepTracker(ProgressCallback):
    """Tracks progress of an individual step within a task.

    Also implements ProgressCallback interface for parser integration.
    """

    def __init__(
        self,
        task_tracker: ProgressTracker,
        step_name: str,
        total_count: Optional[int] = None,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.task_tracker = task_tracker
        self.step = StepProgress(
            step_name=step_name,
            total_count=total_count,
            message=message,
            metadata=metadata or {},
        )

    def start(self) -> None:
        """Mark the step as started."""
        self.step.start_time = time.time()
        self.step.completed = False
        self._broadcast_update()

    def update(
        self,
        current_count: Optional[int] = None,
        total_count: Optional[int] = None,
        message: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        """Update step progress.

        Args:
            current_count: Current items processed
            total_count: Total items to process (can be updated)
            message: Progress message
            **metadata: Additional metadata
        """
        if current_count is not None:
            self.step.current_count = current_count

        if total_count is not None:
            self.step.total_count = total_count

        if message is not None:
            self.step.message = message

        if metadata:
            self.step.metadata.update(metadata)

        # Calculate step progress
        if self.step.total_count and self.step.total_count > 0:
            self.step.step_progress = min(
                1.0, self.step.current_count / self.step.total_count
            )
        else:
            # If no total count, use completion status
            self.step.step_progress = 1.0 if self.step.completed else 0.0

        self._broadcast_update()

    def complete(self, message: str = "Completed") -> None:
        """Mark the step as completed.

        Args:
            message: Completion message
        """
        self.step.completed = True
        self.step.step_progress = 1.0
        self.step.end_time = time.time()
        self.step.message = message
        self._broadcast_update()

    def fail(self, error_message: str) -> None:
        """Mark the step as failed.

        Args:
            error_message: Error description
        """
        self.step.completed = False
        self.step.step_progress = 0.0
        self.step.end_time = time.time()
        self.step.message = f"Failed: {error_message}"
        self.step.metadata["error"] = error_message
        self._broadcast_update()

    def on_status_update(
        self, status: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Implement ProgressCallback interface for parser integration.

        Args:
            status: Status message from parser (e.g., "OCR finished", "Layout analysis")
            details: Optional additional details
        """
        # Update step message with parser status
        self.step.message = status

        # Update metadata if details provided
        if details:
            self.step.metadata.update(details)

        # Mark step as running if we get status updates
        if not self.step.start_time:
            self.step.start_time = time.time()

        self._broadcast_update()

    def _broadcast_update(self) -> None:
        """Broadcast step progress update to task tracker."""
        step_data = {
            "step_name": self.step.step_name,
            "completed": self.step.completed,
            "current_count": self.step.current_count,
            "total_count": self.step.total_count,
            "step_progress": self.step.step_progress,
            "message": self.step.message,
            "duration_seconds": self.step.duration,
            "metadata": self.step.metadata,
        }

        self.task_tracker.update_overall_progress(
            progress=None,
            current_step=self.step.step_name,
            steps={self.step.step_name: step_data},
        )
