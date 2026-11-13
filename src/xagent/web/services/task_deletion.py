"""Shared, foreign-key-safe deletion and redaction of task-owned database rows."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import exists, or_, select
from sqlalchemy.orm import Session

from ...config import get_uploads_dir
from ..models.chat_message import TaskChatMessage
from ..models.task import (
    DAGExecution,
    Task,
    TaskConnectorRuntimeContext,
    TraceCheckpointBlob,
    TraceEvent,
    TraceMessageBlob,
)
from ..models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent
from ..models.task_channel_delivery import TaskChannelDelivery
from ..models.task_command import TaskExecutionCommand
from ..models.task_execution_event import TaskExecutionEvent
from ..models.task_interaction import TaskInteractionRequest
from ..models.task_runtime_secret import TaskRuntimeSecret
from ..models.trigger import TriggerRun, TriggerRunStatus
from ..models.uploaded_file import UploadedFile
from .task_file_lifecycle import (
    DetachmentReason,
    detach_task_files,
    lock_attachment_task,
)
from .task_interaction_schema import interaction_requests_table_exists
from .task_runtime import PROMPT_CONTEXT_AGENT_CONFIG_KEY
from .uploaded_file_store import UploadedFileStore

# TriggerRun error for a run whose task was deleted before it settled.
TASK_DELETED_TRIGGER_ERROR = "Task deleted before its run finished."


def _fail_unsettled_trigger_runs(db: Session, task_id: int) -> None:
    """Fail every PENDING/RUNNING TriggerRun of a task being deleted.

    A run can be left RUNNING while its task rests: a user pause or a wait
    for the user (#2177), or an automatic resume still scheduled. Only that
    task's later run would settle it, and deleting the task nulls
    ``trigger_runs.task_id`` (ON DELETE SET NULL), after which nothing could.
    A bulk statement: this session has autoflush disabled, and it must land
    before the task delete. Lock order holds: the task row is already locked.
    """

    db.query(TriggerRun).filter(
        TriggerRun.task_id == task_id,
        TriggerRun.status.in_(
            (TriggerRunStatus.PENDING.value, TriggerRunStatus.RUNNING.value)
        ),
    ).update(
        {
            TriggerRun.status: TriggerRunStatus.FAILED.value,
            TriggerRun.finished_at: datetime.now(timezone.utc),
            TriggerRun.error_message: TASK_DELETED_TRIGGER_ERROR,
        },
        synchronize_session=False,
    )


def _delete_task_content_rows(db: Session, task_id: int) -> None:
    """Delete the content rows ``purge_task_rows`` and ``redact_task_rows`` share.

    Covers every task-owned table whose rows carry execution content and that
    the task row's own deletion would not reliably cascade: interaction
    requests, auto-recovery state, the trace and its two blob tables, the DAG
    execution, connector runtime context and runtime secrets. Chat messages
    and execution events are left to the callers: the task purge relies on
    the ORM cascade and ``ON DELETE CASCADE`` for them, and the redaction
    deletes them itself.

    The caller holds the task row lock and owns the transaction. Statement
    order is load-bearing; the comments below say why.
    """

    # NULL the checkpoint pointer columns before the trace_events delete
    # below: last_checkpoint_trace_event_id FKs to trace_events.id, so a
    # task still pointing at a row would block (or, without DB-level
    # enforcement, orphan) that delete. A bulk statement, not an ORM
    # attribute assignment, because this session has autoflush disabled --
    # an attribute assignment would not reach the database until a later
    # flush, by which point the trace_events delete has already run.
    # ``updated_at`` is pinned: the redaction keeps this row, and its
    # ``updated_at`` doubles as the v1 ``completed_at`` and as a usage
    # window key, neither of which content removal may move.
    db.query(Task).filter(Task.id == task_id).update(
        {
            Task.last_checkpoint_event_id: None,
            Task.last_checkpoint_trace_event_id: None,
            Task.updated_at: Task.updated_at,
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

    # Explicit rather than relying on ON DELETE CASCADE: SQLite only enforces
    # foreign keys when the pragma is on. No presence gate, unlike the
    # interaction table: an Alembic upgrade can stop short of these tables,
    # but ``Base.metadata.create_all`` runs right after ``try_upgrade_db`` on
    # every startup, so they exist before the process serves anything.
    db.query(TaskRecoveryEvent).filter(TaskRecoveryEvent.task_id == task_id).delete(
        synchronize_session=False
    )
    db.query(TaskAutoRecovery).filter(TaskAutoRecovery.task_id == task_id).delete(
        synchronize_session=False
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
    db.query(TaskConnectorRuntimeContext).filter(
        TaskConnectorRuntimeContext.task_id == task_id
    ).delete(synchronize_session=False)
    db.query(TaskRuntimeSecret).filter(TaskRuntimeSecret.task_id == task_id).delete(
        synchronize_session=False
    )


def purge_task_rows(
    db: Session,
    *,
    task_id: int,
    detached_reason: DetachmentReason,
) -> bool:
    """Delete one task and its non-cascading rows in a caller-owned transaction.

    ``detached_reason`` is required: ``task_deleted`` for deletion/expiry or
    ``task_create_failed`` for failed-create compensation. Marking shares this
    transaction with deletion; retained file bytes are reclaimed later.

    ``task_interaction_requests`` rows, unlike ``UploadedFile``, are deleted
    outright: the CASCADE on ``task_id`` would remove them anyway, and the
    delete must run before the ``trace_events`` delete (see
    :func:`_delete_task_content_rows`).
    """

    task = lock_attachment_task(db, task_id, deleting=True)
    if task is None:
        return False
    detach_task_files(db, task_id=task_id, reason=detached_reason)
    _fail_unsettled_trigger_runs(db, task_id)
    _delete_task_content_rows(db, task_id)
    db.delete(task)
    return True


def _delete_task_files(db: Session, task_id: int) -> None:
    """Delete the task's uploaded files: rows, durable objects and local bytes.

    Unlike :func:`detach_task_files`, nothing is retained. The bytes go
    *inside* the caller's transaction, before it commits, through the same
    :meth:`UploadedFileStore.delete` the KB and file routes use: the object
    delete is idempotent, so a transaction that fails afterwards leaves rows
    whose re-run deletes them, while deleting after the commit would leave
    an object with no row left to find it by.

    Rows another owner has claimed (``compensating``) are skipped: that
    owner deletes the row and every resource it names.
    """

    store = UploadedFileStore(db)
    uploads_root = get_uploads_dir()
    records = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.task_id == task_id,
            UploadedFile.storage_status != "compensating",
        )
        .all()
    )
    for record in records:
        store.delete(record, local_root=uploads_root)


#: ``tasks.agent_config`` keys that carry content or personal data rather
#: than routing or authorization: the prompt reference data a run is given,
#: and the widget creator's IP. Every other key is identity the task's later
#: reads and turns still need (auth mode, guest id, runtime bindings, ...).
REDACTED_AGENT_CONFIG_KEYS = frozenset(
    {PROMPT_CONTEXT_AGENT_CONFIG_KEY, "widget_client_ip"}
)


def _trace_rows_exist(db: Session, task_id: int) -> bool:
    return bool(
        db.execute(
            select(
                or_(
                    *(
                        exists().where(model.task_id == task_id)
                        for model in (TraceCheckpointBlob, TraceMessageBlob, TraceEvent)
                    )
                )
            )
        ).scalar_one()
    )


def redact_task_rows(db: Session, task: Task, *, now: datetime) -> None:
    """Delete one task's content but keep its row, in a caller-owned transaction.

    ``task`` must already be locked in this transaction through
    :func:`lock_attachment_task` with ``deleting=True``, which also fences new
    attachments, and the caller must have established that it is not
    executing (see ``task_content_redaction``).

    What goes: every row :func:`_delete_task_content_rows` removes, the chat
    transcript and execution events, the task's uploaded files (rows *and*
    stored bytes, see :func:`_delete_task_files`), the payload, result and
    error of its execution commands, the payload snapshots of the trigger
    runs that started it, the content keys of ``agent_config``
    (:data:`REDACTED_AGENT_CONFIG_KEYS`), and every content-bearing ``Task``
    column (title, description, latest input/output, error, structured
    result, process-mode description and examples).

    What stays: the row and everything metering, identity, routing or
    authorization reads -- ids, owner, status, timestamps, token counters and
    usage details, model and agent identifiers, the remaining ``agent_config``
    keys, channel and source fields, the trigger run rows themselves, and the
    event sequence counter, so a later turn keeps numbering monotonically.
    ``updated_at`` is pinned (see :func:`_delete_task_content_rows`).
    ``traces_expired_at`` is stamped with ``now`` when there was a trace to
    remove: it is the marker trace readers already honour for "steps before
    this moment were removed", and a re-run that removes nothing keeps it.

    Command rows are blanked rather than deleted: their identity is the
    resend dedupe key, and a deleted row would let a client's retry of an
    old command execute it again. A retry now conflicts on the blank payload
    instead of replaying. Their pending channel deliveries are discarded in
    the same step: the delivery worker reads the destination channel out of
    the command payload, so a pending delivery of a blanked command could
    never be claimed and would sit at the head of its channel's queue
    forever -- and the reply it would carry belongs to the deleted
    conversation anyway.

    Idempotent: a second run finds nothing to delete and rewrites the same
    blank values.
    """

    task_id = int(task.id)
    had_trace = _trace_rows_exist(db, task_id)
    _delete_task_files(db, task_id)
    _delete_task_content_rows(db, task_id)
    db.query(TaskChatMessage).filter(TaskChatMessage.task_id == task_id).delete(
        synchronize_session=False
    )
    db.query(TaskExecutionEvent).filter(TaskExecutionEvent.task_id == task_id).delete(
        synchronize_session=False
    )
    task_commands = select(TaskExecutionCommand.id).where(
        TaskExecutionCommand.task_id == task_id
    )
    db.query(TaskChannelDelivery).filter(
        TaskChannelDelivery.command_id.in_(task_commands),
        TaskChannelDelivery.status == "pending",
    ).update(
        {
            TaskChannelDelivery.status: "discarded",
            TaskChannelDelivery.claim_token: None,
        },
        synchronize_session=False,
    )
    db.query(TaskExecutionCommand).filter(
        TaskExecutionCommand.task_id == task_id
    ).update(
        {
            TaskExecutionCommand.payload: {},
            TaskExecutionCommand.result: None,
            TaskExecutionCommand.error: None,
        },
        synchronize_session=False,
    )
    db.query(TriggerRun).filter(TriggerRun.task_id == task_id).update(
        {TriggerRun.payload_snapshot: None}, synchronize_session=False
    )
    values: dict[Any, Any] = {
        # NOT NULL; every other content column is nullable.
        Task.title: "",
        Task.description: None,
        Task.input: None,
        Task.output: None,
        Task.error_message: None,
        Task.structured_output_record: None,
        Task.process_description: None,
        Task.examples: None,
        # Names the interaction row deleted above.
        Task.interaction_protocol_version: None,
        Task.updated_at: Task.updated_at,
    }
    if isinstance(task.agent_config, dict):
        values[Task.agent_config] = {
            key: value
            for key, value in task.agent_config.items()
            if key not in REDACTED_AGENT_CONFIG_KEYS
        }
    if had_trace:
        values[Task.traces_expired_at] = now
    db.query(Task).filter(Task.id == task_id).update(values, synchronize_session=False)
