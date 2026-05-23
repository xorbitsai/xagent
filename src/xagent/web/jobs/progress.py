from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from ...core.tools.core.RAG_tools.progress import get_progress_manager
from ..models.background_job import BackgroundJob
from ..services.background_jobs import update_job_progress

logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _status_value(status: Any) -> str | None:
    if status is None:
        return None
    value = getattr(status, "value", status)
    return str(value)


def _step_message(
    current_step: str | None,
    metadata: dict[str, Any] | None,
) -> str | None:
    if not current_step or not metadata:
        return None
    steps = metadata.get("steps")
    if not isinstance(steps, dict):
        return None
    step_data = steps.get(current_step)
    if not isinstance(step_data, dict):
        return None
    message = step_data.get("message")
    if isinstance(message, str) and message.strip():
        return message
    return None


class BackgroundJobProgressManager:
    """Mirror RAG progress updates into the durable background job row."""

    def __init__(
        self,
        db: Session,
        job: BackgroundJob,
        *,
        delegate: Any | None = None,
        throttle_seconds: float = 0.5,
    ) -> None:
        self.db = db
        self.job = job
        self.delegate = delegate if delegate is not None else get_progress_manager()
        self.throttle_seconds = throttle_seconds
        self._last_mirror_at = 0.0
        self._task_type_by_id: dict[str, str] = {}

    def create_task(
        self,
        task_type: str,
        task_id: str | None = None,
        user_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        created_task_id = self.delegate.create_task(
            task_type=task_type,
            task_id=task_id,
            user_id=user_id,
            metadata=metadata,
        )
        self._task_type_by_id[created_task_id] = task_type
        self._mirror(
            created_task_id,
            message="Queued",
            metadata=metadata,
            force=True,
        )
        return str(created_task_id)

    def update_task_progress(
        self,
        task_id: str,
        status: Any | None = None,
        current_step: str | None = None,
        overall_progress: float | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.delegate.update_task_progress(
            task_id,
            status=status,
            current_step=current_step,
            overall_progress=overall_progress,
            metadata=metadata,
            **kwargs,
        )
        status_text = _status_value(status)
        message = (
            _step_message(current_step, metadata)
            or kwargs.get("message")
            or current_step
            or status_text
            or "Ingesting document"
        )
        force = status_text in {"success", "failed", "cancelled"}
        self._mirror(
            task_id,
            message=str(message),
            status=status_text,
            current_step=current_step,
            overall_progress=overall_progress,
            metadata=metadata,
            force=force,
        )

    def complete_task(self, task_id: str, success: bool = True) -> None:
        self.delegate.complete_task(task_id, success=success)
        self._mirror(
            task_id,
            message="Completed" if success else "Failed",
            status="success" if success else "failed",
            overall_progress=1.0 if success else None,
            force=True,
        )

    def track_task(self, *args: Any, **kwargs: Any) -> Any:
        return self.delegate.track_task(*args, **kwargs)

    def get_active_tasks(self, *args: Any, **kwargs: Any) -> Any:
        return self.delegate.get_active_tasks(*args, **kwargs)

    def _mirror(
        self,
        task_id: str,
        *,
        message: str,
        status: str | None = None,
        current_step: str | None = None,
        overall_progress: float | None = None,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and now - self._last_mirror_at < self.throttle_seconds:
            return

        completed = None
        total = None
        if overall_progress is not None:
            total = 100
            completed = max(0, min(100, int(round(overall_progress * 100))))

        extra: dict[str, Any] = {
            "task_id": task_id,
            "task_type": self._task_type_by_id.get(task_id),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if status is not None:
            extra["status"] = status
        if current_step is not None:
            extra["current_step"] = current_step
        if overall_progress is not None:
            extra["overall_progress"] = overall_progress
        if metadata is not None:
            existing_metadata = {}
            current_progress: dict[str, Any] = {}
            if isinstance(self.job.progress, dict):
                current_progress = self.job.progress
            raw_existing_metadata = current_progress.get("metadata")
            if isinstance(raw_existing_metadata, dict):
                existing_metadata = dict(raw_existing_metadata)
            merged_metadata = _jsonable(metadata)
            if isinstance(existing_metadata.get("steps"), dict) and isinstance(
                merged_metadata.get("steps"), dict
            ):
                steps = dict(existing_metadata["steps"])
                steps.update(merged_metadata["steps"])
                merged_metadata["steps"] = steps
            existing_metadata.update(merged_metadata)
            extra["metadata"] = existing_metadata

        try:
            update_job_progress(
                self.db,
                self.job,
                message=message,
                completed=completed,
                total=total,
                extra=extra,
            )
            self._last_mirror_at = now
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to mirror RAG progress to background job %s: %s",
                self.job.id,
                exc,
            )
