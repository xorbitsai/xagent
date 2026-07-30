"""Shared, foreign-key-safe deletion of task-owned database rows."""

from sqlalchemy.orm import Session

from ..models.task import (
    DAGExecution,
    Task,
    TraceCheckpointBlob,
    TraceEvent,
    TraceMessageBlob,
)
from ..models.uploaded_file import UploadedFile


def purge_task_rows(
    db: Session,
    *,
    task_id: int,
    preserve_uploaded_files: bool,
) -> bool:
    """Delete one task and its non-cascading rows in a caller-owned transaction."""

    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None:
        return False

    db.query(TraceCheckpointBlob).filter(TraceCheckpointBlob.task_id == task_id).delete(
        synchronize_session=False
    )
    db.query(TraceMessageBlob).filter(TraceMessageBlob.task_id == task_id).delete(
        synchronize_session=False
    )
    db.query(TraceEvent).filter(TraceEvent.task_id == task_id).delete(
        synchronize_session=False
    )
    db.query(DAGExecution).filter(DAGExecution.task_id == task_id).delete(
        synchronize_session=False
    )
    if preserve_uploaded_files:
        db.query(UploadedFile).filter(UploadedFile.task_id == task_id).update(
            {UploadedFile.task_id: None},
            synchronize_session=False,
        )
    db.delete(task)
    return True
