"""Shared, foreign-key-safe deletion of task-owned database rows."""

from sqlalchemy.orm import Session

from ..models.task import (
    DAGExecution,
    Task,
    TraceCheckpointBlob,
    TraceEvent,
    TraceMessageBlob,
)


def purge_task_rows(
    db: Session,
    *,
    task_id: int,
) -> bool:
    """Delete one task and its non-cascading rows in a caller-owned transaction.

    ``UploadedFile`` rows are detached, not deleted: ``Task.uploaded_files`` is a
    relationship without a cascade, so the unit of work nulls
    ``UploadedFile.task_id`` before the task row is removed. The rows and their
    backing blobs therefore outlive the task and are not reclaimed here.
    """

    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None:
        return False

    # NULL the checkpoint pointer columns before the trace_events delete
    # below: last_checkpoint_trace_event_id FKs to trace_events.id, so a
    # task still pointing at a row would block (or, without DB-level
    # enforcement, orphan) that delete. A bulk statement, not an ORM
    # attribute assignment, because this session has autoflush disabled --
    # an attribute assignment would not reach the database until a later
    # flush, by which point the trace_events delete has already run.
    db.query(Task).filter(Task.id == task_id).update(
        {
            Task.last_checkpoint_event_id: None,
            Task.last_checkpoint_trace_event_id: None,
        },
        synchronize_session=False,
    )

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
    db.delete(task)
    return True
