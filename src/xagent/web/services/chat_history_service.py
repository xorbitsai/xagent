"""Persistence helpers for task chat transcripts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from ...core.agent.transcript import (
    build_assistant_transcript_content,
    normalize_transcript_messages,
)
from ..models.chat_message import TaskChatMessage
from .file_reference_output_service import reconcile_assistant_file_references
from .ops_signals import (
    CLARIFICATION_LEGACY_SUPERSEDE_FAILED,
    clear_degradation,
    register_degradation,
)

logger = logging.getLogger(__name__)

DELIVERY_PENDING = "pending"
DELIVERY_DISPATCHED = "dispatched"
DELIVERY_COMPLETED = "completed"
DELIVERY_FAILED = "failed"

_SUPERSEDED_MESSAGE_TYPE = "question_superseded"


@dataclass(frozen=True)
class UserMessageDeliveryClaim:
    """Result of inspecting or atomically claiming a client turn id."""

    message: TaskChatMessage
    claimed: bool
    payload_matches: bool

    @property
    def failed(self) -> bool:
        return str(self.message.delivery_status) == DELIVERY_FAILED

    @property
    def pending(self) -> bool:
        return str(self.message.delivery_status) == DELIVERY_PENDING


@dataclass(frozen=True)
class UserMessageDeliveryTransition:
    """Conditional delivery-state transition staged in a caller transaction."""

    status: str | None
    outcome: str


def _attachment_identity(
    attachments: Optional[List[Dict[str, Any]]],
) -> tuple[str, ...]:
    identities: list[str] = []
    for attachment in attachments or []:
        file_id = str(attachment.get("file_id") or "").strip()
        fallback = "\x1f".join(
            str(attachment.get(key) or "") for key in ("name", "size", "type")
        )
        identities.append(file_id or f"legacy:{fallback}")
    return tuple(sorted(identities))


def _delivery_payload_matches(
    message: TaskChatMessage,
    *,
    content: str,
    attachments: Optional[List[Dict[str, Any]]],
) -> bool:
    stored_attachments = (
        message.attachments if isinstance(message.attachments, list) else None
    )
    return str(message.content) == content.strip() and _attachment_identity(
        stored_attachments
    ) == _attachment_identity(attachments)


def inspect_user_message_delivery(
    db: Session,
    task_id: int,
    content: str,
    *,
    attachments: Optional[List[Dict[str, Any]]],
    turn_id: str,
) -> Optional[UserMessageDeliveryClaim]:
    """Return the durable outcome for ``turn_id`` without creating a row."""

    existing = (
        db.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == task_id,
            TaskChatMessage.role == "user",
            TaskChatMessage.turn_id == turn_id,
        )
        .first()
    )
    if existing is None:
        return None
    return UserMessageDeliveryClaim(
        message=existing,
        claimed=False,
        payload_matches=_delivery_payload_matches(
            existing,
            content=content,
            attachments=attachments,
        ),
    )


def claim_user_message_delivery(
    db: Session,
    task_id: int,
    user_id: int,
    content: str,
    *,
    attachments: Optional[List[Dict[str, Any]]] = None,
    turn_id: str,
) -> UserMessageDeliveryClaim:
    """Atomically claim a live-control turn before dispatching it.

    The unique database index is the cross-worker serializer. A concurrent
    loser rolls back its insert and returns the winner's durable row, so only
    the claimant may inject the message into an active runtime.
    """

    existing = inspect_user_message_delivery(
        db,
        task_id,
        content,
        attachments=attachments,
        turn_id=turn_id,
    )
    if existing is not None:
        return existing

    message = TaskChatMessage(
        task_id=task_id,
        user_id=user_id,
        role="user",
        content=content.strip(),
        message_type="user_message",
        interactions=None,
        turn_id=turn_id,
        delivery_status=DELIVERY_PENDING,
        attachments=attachments,
    )
    db.add(message)
    try:
        db.commit()
        db.refresh(message)
        return UserMessageDeliveryClaim(
            message=message,
            claimed=True,
            payload_matches=True,
        )
    except IntegrityError:
        db.rollback()
        raced = inspect_user_message_delivery(
            db,
            task_id,
            content,
            attachments=attachments,
            turn_id=turn_id,
        )
        if raced is None:
            raise
        return raced


def claim_user_message_delivery_no_commit(
    db: Session,
    task_id: int,
    user_id: int,
    content: str,
    *,
    attachments: Optional[List[Dict[str, Any]]] = None,
    turn_id: str,
) -> UserMessageDeliveryClaim:
    """Stage a delivery claim without committing the caller's transaction."""

    existing = inspect_user_message_delivery(
        db,
        task_id,
        content,
        attachments=attachments,
        turn_id=turn_id,
    )
    if existing is not None:
        return existing

    message = TaskChatMessage(
        task_id=task_id,
        user_id=user_id,
        role="user",
        content=content.strip(),
        message_type="user_message",
        interactions=None,
        turn_id=turn_id,
        delivery_status=DELIVERY_PENDING,
        attachments=attachments,
    )
    db.add(message)
    db.flush()
    return UserMessageDeliveryClaim(
        message=message,
        claimed=True,
        payload_matches=True,
    )


def mark_user_message_delivery(
    db: Session,
    *,
    task_id: int,
    turn_id: str,
    status: str,
) -> UserMessageDeliveryTransition:
    """Stage one monotonic delivery transition without committing ``db``."""

    if status not in {
        DELIVERY_PENDING,
        DELIVERY_DISPATCHED,
        DELIVERY_COMPLETED,
        DELIVERY_FAILED,
    }:
        raise ValueError(f"Unknown delivery status: {status}")
    query = db.query(TaskChatMessage).filter(
        TaskChatMessage.task_id == task_id,
        TaskChatMessage.role == "user",
        TaskChatMessage.turn_id == turn_id,
    )
    message = query.first()
    if message is None:
        return UserMessageDeliveryTransition(status=None, outcome="missing")

    current = str(message.delivery_status)
    if current == status:
        return UserMessageDeliveryTransition(status=current, outcome="idempotent")
    allowed_targets = {
        DELIVERY_PENDING: {
            DELIVERY_DISPATCHED,
            DELIVERY_COMPLETED,
            DELIVERY_FAILED,
        },
        DELIVERY_DISPATCHED: {DELIVERY_COMPLETED},
    }
    if status not in allowed_targets.get(current, set()):
        return UserMessageDeliveryTransition(status=current, outcome="conflict")

    updated = query.filter(TaskChatMessage.delivery_status == current).update(
        {TaskChatMessage.delivery_status: status},
        synchronize_session=False,
    )
    if updated:
        return UserMessageDeliveryTransition(status=status, outcome="updated")

    # A concurrent terminal transition won after the read. Reload the durable
    # state instead of issuing an unguarded write that could regress it.
    db.expire_all()
    raced = query.first()
    if raced is None:
        return UserMessageDeliveryTransition(status=None, outcome="missing")
    raced_status = str(raced.delivery_status)
    return UserMessageDeliveryTransition(
        status=raced_status,
        outcome="idempotent" if raced_status == status else "conflict",
    )


def mark_user_message_delivery_sync(
    task_id: int,
    turn_id: str,
    status: str,
) -> UserMessageDeliveryTransition:
    """Update one delivery from synchronous or ``asyncio.to_thread`` callers."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        transition = mark_user_message_delivery(
            db,
            task_id=task_id,
            turn_id=turn_id,
            status=status,
        )
        db.commit()
        return transition


def supersede_legacy_question_rows(db: Session, *, task_id: int) -> int:
    """Mark every still-pending assistant question row on a task as superseded.

    Rewrites ``message_type`` from ``"question"`` to
    ``"question_superseded"`` for every row on ``task_id`` where
    ``role == "assistant"``. It only rewrites that one column: it never
    deletes rows, never touches ``content`` or ``interactions``, and never
    writes the reverse direction. Like ``mark_user_message_delivery``, it
    does not commit or roll back ``db`` — the caller owns the transaction.
    This function expects to run inside a finishing transaction that
    already holds a lock on the task's ``tasks`` row (see the paragraph on
    concurrency below).

    ``TaskChatMessage``'s only unique constraint is on
    ``(task_id, role, turn_id)``. This update never touches any of those
    three columns, so on the normal path it has no constraint-failure
    surface to hit — the ``except DBAPIError`` clause below exists for
    statement-level failures the database reports back (a constraint
    violation elsewhere in the schema, a statement timeout), not for a
    unique-index collision this statement cannot cause. A dropped
    connection is a different failure mode and not what motivates this
    clause: it kills the whole enclosing transaction outright, and no
    savepoint changes that.

    The WHERE predicate is exactly the three-condition filter
    ``get_latest_waiting_question`` runs before its own ``ORDER BY``:
    ``task_id``, ``role == "assistant"``, ``message_type == "question"``.
    A static guard pins the two predicate sets equal; changing one side
    without the other makes that guard fail.

    The update is deliberately unordered and untargeted — it rewrites the
    whole matching set instead of picking one row with
    ``ORDER BY id DESC LIMIT 1``. Under two simultaneously waiting
    questions, an ordered pick would only relabel the winner and leave the
    loser row stuck as ``"question"`` forever, which is worse than not
    superseding at all. Collapsing the whole set removes that failure mode
    structurally.

    Because nothing else in this codebase ever transitions
    ``message_type`` away from ``"question"``, this same whole-set sweep
    also flips any already-answered question row left over from an
    earlier turn on the task, not only the row still genuinely waiting —
    that is deliberate, not a side effect to fix; it is the same
    whole-set shape that keeps a stale row from ever outranking the live
    question. One consequence: for a historical row, the admin
    transcript export cannot tell "answered, then superseded" apart from
    "abandoned, then superseded" — both end up as
    ``"question_superseded"`` with nothing recording which happened.

    A rowcount of zero is a normal outcome, not a failure — it happens
    whenever the mid-turn write path already swallowed its own error,
    whenever a reentrant call already zeroed the set on an earlier pass, or
    whenever an empty projection meant no assistant question row was ever
    persisted. None of those cases should raise or alert.

    The update runs inside its own SAVEPOINT (``db.begin_nested()``). That
    is not defensive programming: on PostgreSQL, one failed statement
    marks the whole enclosing transaction aborted, so without a savepoint,
    swallowing a database error here would only relocate the crash to the
    caller's next flush or commit. The savepoint is what lets a failed
    supersede degrade instead of taking the caller's transaction down with
    it. The catch clause is narrowed to ``sqlalchemy.exc.DBAPIError`` —
    the database reporting that the statement itself failed, which
    includes ``ProgrammingError`` (e.g. a malformed or rejected SQL
    statement) — and nothing broader. Session and argument misuse on the
    Python side — ``InvalidRequestError``, ``PendingRollbackError``,
    ``ArgumentError`` — is not a ``DBAPIError`` and is left to propagate.

    Serializing this function against a concurrent call to ``respond()``
    on the same task is not this function's job — it relies on whatever
    already holds the task locked for the caller's finishing transaction:
    a real row lock on PostgreSQL, and the single-writer model on SQLite.
    A green unit test against SQLite does not exercise the PostgreSQL row
    lock.

    ``interaction_handoff`` (``task_interaction_staging.py``) is a
    context manager whose own with-block must be the transaction's last
    word before the caller commits — no other write may run inside it. A
    call site sitting inside that with-block must sit outside it instead,
    or stay inside it without issuing its own commit. On the path that
    follows ``persist_assistant_message_no_commit``, callers must issue an
    explicit ``db.flush()`` afterward rather than relying on autoflush to
    make the pending row visible to this update.

    That ordering is deliberate, and it means this update supersedes the
    very row the flush just made visible. That is the intended outcome,
    not a side effect of the unbounded predicate: on that path the caller
    has already staged a structured interaction row for the same turn,
    and the structured row — not the transcript row — is the question the
    reader is meant to serve, because the reader consults the task's
    protocol-version marker before it ever reaches this table. The
    transcript row keeps its ``content`` and ``interactions`` untouched
    and stays in the history; what it loses is only its standing as the
    *pending* question. A caller that stages a transcript question row
    meant to remain answerable must not call this function in the same
    transaction — no predicate here will spare that row.

    This UPDATE runs with ``synchronize_session=False``: a
    ``TaskChatMessage`` object already loaded into the caller's session
    keeps its stale ``message_type`` in memory until the caller calls
    ``refresh()`` or ``expire()`` on it, even if the caller runs a fresh
    query afterward in the same session that returns that same
    identity-mapped object — SQLAlchemy's identity map hands back the
    in-memory object as-is rather than re-reading the row, so the fresh
    query does not pick up the UPDATE either.

    Returns the number of rows updated. The return value exists only for
    logging; callers must not branch on it.
    """

    try:
        with db.begin_nested():
            updated = (
                db.query(TaskChatMessage)
                .filter(
                    TaskChatMessage.task_id == task_id,
                    TaskChatMessage.role == "assistant",
                    TaskChatMessage.message_type == "question",
                )
                .update(
                    {TaskChatMessage.message_type: _SUPERSEDED_MESSAGE_TYPE},
                    synchronize_session=False,
                )
            )
    except DBAPIError as exc:
        logger.error(
            "Failed to supersede legacy question rows for task %s",
            task_id,
            exc_info=True,
        )
        register_degradation(
            CLARIFICATION_LEGACY_SUPERSEDE_FAILED,
            f"task {task_id}: legacy question supersede failed ({type(exc).__name__})",
        )
        return 0

    clear_degradation(CLARIFICATION_LEGACY_SUPERSEDE_FAILED)
    logger.info("Superseded %s legacy question row(s) for task %s", updated, task_id)
    return updated


def persist_user_message(
    db: Session,
    task_id: int,
    user_id: int,
    content: str,
    *,
    attachments: Optional[List[Dict[str, Any]]] = None,
    turn_id: Optional[str] = None,
) -> Optional[TaskChatMessage]:
    return _persist_message(
        db=db,
        task_id=task_id,
        user_id=user_id,
        role="user",
        content=content,
        message_type="user_message",
        attachments=attachments,
        turn_id=turn_id,
    )


def persist_user_message_no_commit(
    db: Session,
    task_id: int,
    user_id: int,
    content: str,
    *,
    attachments: Optional[List[Dict[str, Any]]] = None,
    turn_id: Optional[str] = None,
    delivery_status: Optional[str] = None,
) -> Optional[TaskChatMessage]:
    """``persist_user_message`` variant that stages the row but does NOT commit.

    Used by ``TaskTurnOrchestrator.begin_turn`` so the atomic claim
    UPDATE and the message insert land in the same commit — if the
    insert fails, the status flip is rolled back too. Caller is
    responsible for calling ``db.commit()`` (or ``db.rollback()`` on
    failure).

    Returns ``None`` when content is whitespace-only AND no attachments
    are provided. A row with empty content but non-empty attachments is
    still persisted (the user uploaded files but didn't type anything).
    """
    normalized_content = content.strip()
    if not normalized_content and not attachments:
        return None
    message = TaskChatMessage(
        task_id=task_id,
        user_id=user_id,
        role="user",
        content=normalized_content,
        message_type="user_message",
        interactions=None,
        turn_id=turn_id,
        delivery_status=delivery_status,
        # Pass through ``attachments`` directly so an explicit empty list
        # round-trips as ``[]`` rather than being coerced to ``NULL`` —
        # callers may want to distinguish "no attachments specified" from
        # "attachments key was set, just empty".
        attachments=attachments,
    )
    db.add(message)
    return message


def persist_assistant_message(
    db: Session,
    task_id: int,
    user_id: int,
    content: str,
    *,
    message_type: str = "assistant_message",
    interactions: Optional[List[Dict[str, Any]]] = None,
    turn_id: Optional[str] = None,
    content_is_reconciled: bool = False,
) -> Optional[TaskChatMessage]:
    reconciled_content = (
        content
        if content_is_reconciled
        else reconcile_assistant_file_references(
            db,
            task_id=task_id,
            user_id=user_id,
            content=content,
        )
    )
    transcript_content = build_assistant_transcript_content(
        reconciled_content, interactions
    )
    return _persist_message(
        db=db,
        task_id=task_id,
        user_id=user_id,
        role="assistant",
        content=transcript_content,
        message_type=message_type,
        interactions=interactions,
        turn_id=turn_id,
    )


def persist_assistant_message_no_commit(
    db: Session,
    task_id: int,
    user_id: int,
    content: str,
    *,
    message_type: str = "assistant_message",
    interactions: Optional[List[Dict[str, Any]]] = None,
    turn_id: Optional[str] = None,
    content_is_reconciled: bool = False,
) -> Optional[TaskChatMessage]:
    """Stage an assistant transcript row for an atomic caller-owned commit."""

    reconciled_content = (
        content
        if content_is_reconciled
        else reconcile_assistant_file_references(
            db,
            task_id=task_id,
            user_id=user_id,
            content=content,
        )
    )
    transcript_content = build_assistant_transcript_content(
        reconciled_content, interactions
    )
    normalized_content = transcript_content.strip()
    if not normalized_content:
        return None
    message = TaskChatMessage(
        task_id=task_id,
        user_id=user_id,
        role="assistant",
        content=normalized_content,
        message_type=message_type,
        interactions=interactions,
        turn_id=turn_id,
        attachments=None,
    )
    db.add(message)
    return message


def load_task_transcript(
    db: Session,
    task_id: int,
    *,
    before_message_id: Optional[int] = None,
) -> List[Dict[str, str]]:
    if before_message_id is not None:
        # Check if the reference message actually exists
        exists = (
            db.query(TaskChatMessage.id)
            .filter(
                TaskChatMessage.id == before_message_id,
                TaskChatMessage.task_id == task_id,
            )
            .first()
        )
        if not exists:
            logger.warning(
                "Message id: {before_message_id} does not exit, returning empty list."
            )
            return []

    query = db.query(TaskChatMessage).filter(TaskChatMessage.task_id == task_id)
    if before_message_id is not None:
        query = query.filter(TaskChatMessage.id < before_message_id)

    messages = [
        {"role": str(message.role), "content": str(message.content)}
        for message in query.order_by(TaskChatMessage.id.asc()).all()
    ]
    return normalize_transcript_messages(messages)


def get_latest_waiting_question(
    db: Session, task_id: int
) -> tuple[Optional[str], Optional[list[dict[str, Any]]]]:
    """Return the latest persisted ask-user question for a waiting task."""

    latest_question = (
        db.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == task_id,
            TaskChatMessage.role == "assistant",
            TaskChatMessage.message_type == "question",
        )
        .order_by(TaskChatMessage.id.desc())
        .first()
    )
    if not latest_question:
        return None, None

    interactions = latest_question.interactions
    return (
        str(latest_question.content),
        interactions if isinstance(interactions, list) else None,
    )


def _persist_message(
    db: Session,
    task_id: int,
    user_id: int,
    role: str,
    content: str,
    message_type: str,
    interactions: Optional[List[Dict[str, Any]]] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
    turn_id: Optional[str] = None,
    delivery_status: Optional[str] = None,
) -> Optional[TaskChatMessage]:
    normalized_content = content.strip()
    if not normalized_content and not attachments:
        return None

    message = TaskChatMessage(
        task_id=task_id,
        user_id=user_id,
        role=role,
        content=normalized_content,
        message_type=message_type,
        interactions=interactions,
        turn_id=turn_id,
        delivery_status=delivery_status,
        # Pass through ``attachments`` directly so an explicit empty list
        # round-trips as ``[]`` rather than being coerced to ``NULL``.
        attachments=attachments,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message
