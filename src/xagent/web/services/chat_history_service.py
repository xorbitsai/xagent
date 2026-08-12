"""Persistence helpers for task chat transcripts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

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

QUESTION_MESSAGE_TYPE = "question"
SUPERSEDED_MESSAGE_TYPE = "question_superseded"


def _assistant_question_filters(task_id: int) -> tuple[ColumnElement[bool], ...]:
    """The three-leg WHERE predicate for a task's assistant question
    rows: ``task_id``, ``role == "assistant"``,
    ``message_type == QUESTION_MESSAGE_TYPE``. It matches every such
    row, whether it is still waiting for an answer or was already
    answered. Shared by the reader (``get_latest_waiting_question``)
    and the writer (``supersede_legacy_question_rows``) so the two
    conditions cannot drift apart by hand-editing one copy and not the
    other.

    Scoped to this one predicate. A second, ``allow_superseded`` pass
    over already-superseded rows may be added to the reader later; that
    pass is not this helper's concern.
    """
    return (
        TaskChatMessage.task_id == task_id,
        TaskChatMessage.role == "assistant",
        TaskChatMessage.message_type == QUESTION_MESSAGE_TYPE,
    )


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
    """Mark every assistant question row on a task as superseded, whether
    still pending or already answered.

    Rewrites ``message_type`` from ``QUESTION_MESSAGE_TYPE`` to
    ``SUPERSEDED_MESSAGE_TYPE`` for every row on ``task_id`` where
    ``role == "assistant"``, using the predicate shared with the reader
    via ``_assistant_question_filters``. It only rewrites that one column:
    no deletes, no ``content``/``interactions`` changes, no reverse
    direction. Runs with ``synchronize_session=False``, so a
    ``TaskChatMessage`` object already loaded in the caller's session
    keeps its stale ``message_type`` until the caller refreshes it. Like
    ``mark_user_message_delivery``, it does not commit or roll back
    ``db`` — the caller owns the transaction, and is expected to already
    hold a lock on the task's ``tasks`` row (see the concurrency
    paragraph below).

    Holding that no-commit promise on SQLite takes one extra statement: a
    no-op ``UPDATE`` issued before the savepoint opens, purely to get a
    transaction started, because pysqlite would otherwise let the
    savepoint's release commit the relabel. It is a self-assignment on a
    row id that cannot exist, so it writes nothing -- but it does take
    SQLite's write lock for the rest of the caller's transaction, which a
    zero-row call would not otherwise have taken. PostgreSQL needs none of
    this and does not get it.

    The update is deliberately whole-set and unordered rather than
    ``ORDER BY id DESC LIMIT 1``: under two simultaneously waiting
    questions, an ordered pick would relabel only the winner and leave
    the loser stuck as ``"question"`` forever. Collapsing the whole set
    removes that failure mode structurally — and, as a consequence, also
    flips any already-answered question row left over from an earlier
    turn, not only the row still genuinely waiting. That is deliberate,
    not a side effect to fix. One user-visible cost: for a historical
    row, the admin transcript export can no longer tell "answered, then
    superseded" apart from "abandoned, then superseded" — both end up as
    ``SUPERSEDED_MESSAGE_TYPE`` with nothing recording which happened.

    A rowcount of zero is a normal outcome — an already-empty set, a
    reentrant call, a mid-turn write that already failed on its own —
    not a failure. The return value exists only for logging; callers
    must not branch on it.

    ``TaskChatMessage``'s only unique constraint is on
    ``(task_id, role, turn_id)``; this UPDATE never touches any of those
    three columns, so a unique-index collision is not what the catch
    clause below is for.

    The UPDATE runs inside its own ``db.begin_nested()`` SAVEPOINT and
    catches only ``DBAPIError`` — a statement-level database failure,
    including ``ProgrammingError`` — logging and degrading via
    ``CLARIFICATION_LEGACY_SUPERSEDE_FAILED`` instead of taking the
    caller's transaction down with it; the savepoint is what makes that
    degrade possible on PostgreSQL, where a failed statement otherwise
    aborts the whole enclosing transaction. Python-side misuse —
    ``InvalidRequestError``, ``PendingRollbackError``, ``ArgumentError``
    — is not a ``DBAPIError`` and is left to propagate. This is the
    opposite failure policy from ``mark_user_message_delivery``, which
    lets a ``DBAPIError`` propagate: that helper guards a mandatory state
    transition, while this one is a best-effort cleanup sweep that can
    safely no-op and retry on a later call.

    ``Session.begin_nested()`` flushes the whole session before it issues
    the SAVEPOINT, and does so unconditionally: the flush is gated on
    transaction origin, not on ``autoflush``, so the production session's
    ``autoflush=False`` (``models/database.py``) does not suppress it.
    This function therefore calls ``db.flush()`` itself, outside the
    ``try``, before opening the savepoint. Without that, a constraint
    violation among the caller's *own* pending rows raises
    ``IntegrityError`` — a ``DBAPIError`` subclass — from inside
    ``begin_nested()`` before the UPDATE runs, and the catch absorbs it:
    the caller's failure gets reported as a legacy question supersede
    failure, and the caller meets an opaque ``PendingRollbackError`` at
    commit with nothing pointing at the real cause. With the flush first,
    that failure leaves this function on the line that caused it and the
    catch covers only the UPDATE and its savepoint. This function adds
    nothing to the session, so a flush failure here is always the
    caller's, never this helper's, and is left to propagate unchanged.
    ``interaction_handoff`` (``task_interaction_staging.py``) takes the
    same position on the same SQLAlchemy behavior, translating it into
    ``InteractionOwnerStateError`` rather than swallowing it.

    Serializing this against a concurrent ``respond()`` call on the same
    task is not this function's job — it relies on the caller's
    transaction already holding the task locked (a real row lock on
    PostgreSQL, the single-writer model on SQLite); a green SQLite test
    does not exercise that lock.

    This function has no caller in ``src/`` yet. The intended call site
    sits inside ``interaction_handoff`` (``task_interaction_staging.py``),
    after ``persist_assistant_message_no_commit``, and would supersede
    the very transcript row that call staged — on purpose, because the
    structured interaction row staged in the same turn is meant to become
    the question a future protocol-version-gated reader serves instead.
    Making that pending row visible to the UPDATE is the flush above, so
    it is this function's own job and not a call-site obligation. That
    reader gate does not exist yet: ``get_latest_waiting_question`` gates
    on ``task_id``/``role``/``message_type`` only, nothing about protocol
    version. This paragraph is therefore a contract on the follow-up that
    introduces both the caller and that reader gate, not a description of
    current behavior — the wiring change's acceptance criteria carry the
    remaining call-site obligations (ordering, and never calling this on
    a transcript question meant to remain answerable).
    """

    # ``Session.begin_nested()`` flushes the whole session before issuing the
    # SAVEPOINT, so that flush -- and any constraint violation among the
    # caller's own pending rows -- would otherwise happen inside the try below
    # and be caught as if this UPDATE had failed. Flushing here, outside the
    # try, keeps a caller-originated failure attributed to the caller.
    db.flush()

    # SQLite only: make sure a real transaction is open before the SAVEPOINT.
    # pysqlite in legacy mode emits its implicit BEGIN only ahead of
    # INSERT/UPDATE/DELETE, never ahead of SAVEPOINT or SELECT. With a clean
    # session (no DML yet in this transaction) the SAVEPOINT below would run in
    # autocommit, SQLite would open a transaction at that statement, and the
    # matching RELEASE would commit it -- so the relabel would survive the
    # caller's later rollback, the opposite of the contract stated above. The
    # same problem, the same fix and the reasons the engine-level recipe was
    # rejected are all recorded at task_interaction_staging.py:1474-1546.
    #
    # The flush directly above does not help: on a clean session it emits no
    # SQL at all, so it triggers no implicit BEGIN (measured). Nor is there
    # anything to make this conditional on -- with the DBAPI connection in
    # autocommit, both Session.in_transaction() and Connection.in_transaction()
    # still report True (measured), so the guard is on the dialect only.
    #
    # Raw text rather than sa.update(TaskChatMessage), for three reasons, the
    # first of which is load-bearing for this file's own tests: a Core update
    # is emitted as an Update construct, and the statement-capture helper in
    # test_supersede_legacy_questions.py requires exactly one Update in the
    # writer's window -- a Core dummy write makes it two and turns the
    # predicate-drift test red (measured). Second, "SET id = id" is a
    # self-assignment, so it cannot alter a row even if the WHERE clause ever
    # matched, whereas .values(id=-1) would write a real value into the primary
    # key. Third, the no-.values() Core form compiles to a SET clause naming
    # all 11 mapped columns; .values(id=-1) does narrow to one column on this
    # table today, but only because task_chat_messages carries no onupdate=
    # column -- add one and Core appends it, silently widening the SET clause
    # to a real data column, which is the trap the sibling site documents for
    # tasks.updated_at. Do not swap this for any Core form.
    #
    # This statement is outside the try below, deliberately. If it fails, the
    # failure is not a failed supersede: nothing has been relabelled, so it
    # propagates to the caller rather than being folded into the degrade
    # signal, exactly like the flush above it. A DBAPIError raised here
    # therefore reaches the caller unwrapped and registers nothing (measured).
    # Moving it inside the try would restore the misattribution the flush
    # placement exists to prevent.
    if db.get_bind(TaskChatMessage).dialect.name == "sqlite":
        db.execute(
            text(f"UPDATE {TaskChatMessage.__tablename__} SET id = id WHERE id = -1")
        )

    updated = 0
    statement_succeeded = False
    try:
        with db.begin_nested():
            updated = (
                db.query(TaskChatMessage)
                .filter(*_assistant_question_filters(task_id))
                .update(
                    {TaskChatMessage.message_type: SUPERSEDED_MESSAGE_TYPE},
                    synchronize_session=False,
                )
            )
            statement_succeeded = True
    except DBAPIError as exc:
        if statement_succeeded:
            logger.error(
                "Savepoint exit failed after superseding %s legacy question "
                "row(s) for task %s; degrading instead of trusting that count",
                updated,
                task_id,
                exc_info=True,
            )
        else:
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
        .filter(*_assistant_question_filters(task_id))
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
