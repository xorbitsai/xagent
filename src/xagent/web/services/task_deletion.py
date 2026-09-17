"""Shared, foreign-key-safe deletion of task-owned database rows."""

from sqlalchemy.orm import Session

from ..models.task import (
    DAGExecution,
    Task,
    TraceCheckpointBlob,
    TraceEvent,
    TraceMessageBlob,
)
from ..models.task_interaction import TaskInteractionRequest
from .task_interaction_schema import interaction_requests_table_exists


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

    ``task_interaction_requests`` rows, unlike ``UploadedFile``, are deleted
    outright: the CASCADE on ``task_id`` would remove them anyway, and the
    delete must run before the ``trace_events`` delete below (see the
    ordering comment at that call).
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

    # Delete this task's interaction rows before the trace_events delete
    # below. task_interaction_requests.resume_trace_event_id FKs to
    # trace_events.id with ON DELETE SET NULL, and
    # ck_task_interaction_requests_active_anchor requires an active row to
    # hold a non-NULL anchor -- so deleting the anchored trace row first
    # makes the database NULL a column the CHECK forbids being NULL, and the
    # whole purge fails with an IntegrityError. Terminal rows are unaffected
    # (the CHECK's status <> 'active' branch already satisfies it); the
    # asymmetry is by design, and the ordering here is what makes the active
    # case work too.
    #
    # Plain delete, not terminate-then-delete: tasks.id cascades to these
    # rows anyway, so terminating first would cost an extra write and imply
    # -- falsely -- that an interaction row can outlive the purge of its
    # task.
    #
    # Gated on table presence: a deployment upgraded to a revision before
    # this table exists must still be able to delete tasks.
    if interaction_requests_table_exists(db):
        db.query(TaskInteractionRequest).filter(
            TaskInteractionRequest.task_id == task_id
        ).delete(synchronize_session=False)

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
