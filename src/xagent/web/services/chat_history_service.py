"""Persistence helpers for task chat transcripts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from ...core.agent.attachments import build_image_context_references
from ...core.agent.context.execution import (
    COMPACT_CONTEXT_REFS_METADATA_KEY,
    COMPACT_SUMMARY_METADATA_KEY,
    COMPACT_WATERMARK_METADATA_KEY,
)
from ...core.agent.transcript import (
    build_assistant_transcript_content,
    normalize_transcript_messages,
)
from ...core.context_ref import CONTEXT_REFS_KEY, ContextReference
from ..models.chat_message import TaskChatMessage
from ..models.task import TraceEvent
from .assistant_history_safety import (
    LEGACY_UNTRUSTED_ASSISTANT_MESSAGE_TYPE,
    assistant_history_has_safe_ancillary_payload,
    client_safe_assistant_history_content,
    safe_str,
)
from .file_reference_output_service import reconcile_assistant_file_references
from .ops_signals import (
    CLARIFICATION_LEGACY_SUPERSEDE_FAILED,
    clear_degradation,
    register_degradation,
)
from .task_execution_event_writer import (
    stage_chat_message_no_commit,
    stage_delivery_fact_no_commit,
)

logger = logging.getLogger(__name__)

DELIVERY_PENDING = "pending"
DELIVERY_DISPATCHED = "dispatched"
DELIVERY_COMPLETED = "completed"
DELIVERY_FAILED = "failed"

QUESTION_MESSAGE_TYPE = "question"
SUPERSEDED_MESSAGE_TYPE = "question_superseded"

# Historical attachments are replayed context rather than the current upload
# batch. Keep a bounded recent window as a secondary replay limit; the shared
# materializer applies the active model's token and encoded-byte budgets while
# preserving current-turn priority.
_MAX_HISTORICAL_IMAGE_CONTEXT_REFS = 16


def _assistant_question_filters(
    task_id: int, *, message_type: str = QUESTION_MESSAGE_TYPE
) -> tuple[ColumnElement[bool], ...]:
    """The three-leg WHERE predicate for a task's assistant question rows:
    ``task_id``, ``role == "assistant"``, and one ``message_type``. It
    matches every such row, whether it is still waiting for an answer or
    was already answered. Shared by the reader
    (``get_latest_waiting_question``) and the writer
    (``supersede_legacy_question_rows``) so the two conditions cannot drift
    apart by hand-editing one copy and not the other.

    ``message_type`` defaults to ``QUESTION_MESSAGE_TYPE``: the value the
    writer wants and the value the reader's first pass wants, so both keep
    calling this with one argument and the drift-guard test keeps comparing
    the same two rendered clauses. The reader's second pass passes
    ``SUPERSEDED_MESSAGE_TYPE`` instead, because it looks for what the
    writer *produces* rather than what the writer consumes. That one leg is
    the only thing that legitimately differs between the two passes; the
    other two legs stay shared, which is the whole point of this helper.
    """
    return (
        TaskChatMessage.task_id == task_id,
        TaskChatMessage.role == "assistant",
        TaskChatMessage.message_type == message_type,
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
    message = stage_chat_message_no_commit(db, message)
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
    message = stage_chat_message_no_commit(db, message)
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
    from .task_execution_event_store import lock_task_execution_events_no_commit
    from .task_execution_event_writer import uses_execution_events

    if uses_execution_events(db, task_id):
        # Use the same task-before-message lock order as acceptance/finalization.
        lock_task_execution_events_no_commit(db, task_id)
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
        stage_delivery_fact_no_commit(
            db, task_id=task_id, turn_id=turn_id, status=status
        )
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
    ``db`` — the caller owns the transaction. What serializes this against
    a concurrent answer to the same question is not settled here; see the
    concurrency paragraph below.

    Holding that no-commit promise on SQLite takes one extra statement: a
    no-op ``UPDATE`` issued before the savepoint opens, purely to get a
    transaction started, because pysqlite would otherwise let the
    savepoint's release commit the relabel. It is a self-assignment on a
    row id that cannot exist, so it writes nothing -- but it does take
    SQLite's write lock for the rest of the caller's transaction.
    PostgreSQL needs none of this and does not get it.

    The update is deliberately whole-set and unordered rather than
    ``ORDER BY id DESC LIMIT 1``: under two simultaneously waiting
    questions, an ordered pick would relabel only the winner and leave
    the loser stuck as ``"question"`` forever. Collapsing the whole set
    removes that failure mode structurally — and, as a consequence, also
    flips any already-answered question row left over from an earlier
    turn, not only the row still genuinely waiting. That is deliberate,
    not a side effect to fix. What changes for the admin transcript export
    is this row's ``message_type`` value: it exports as
    ``SUPERSEDED_MESSAGE_TYPE`` instead of ``QUESTION_MESSAGE_TYPE``. The
    adjacent user rows are untouched — they do not match this UPDATE's
    predicate — so whatever the export's readers infer from a following
    user message is unaffected.

    A rowcount of zero is a normal outcome — an already-empty set, a
    reentrant call, a mid-turn write that already failed on its own —
    not a failure. The return value exists only for logging; callers
    must not branch on it.

    ``TaskChatMessage``'s only unique constraint is on
    ``(task_id, role, turn_id)``; this UPDATE never touches any of those
    three columns, so a unique-index collision is not what the catch
    clause below is for.

    The UPDATE runs inside its own ``db.begin_nested()`` SAVEPOINT, and
    what happens to a ``DBAPIError`` depends on whether the savepoint was
    in a position to contain it. If the ``SAVEPOINT`` statement itself
    fails, there is no savepoint yet and nothing is contained, so it
    propagates. If the savepoint opened and the UPDATE failed inside it,
    the savepoint contained the damage and the enclosing transaction
    survives, so this logs, degrades via
    ``CLARIFICATION_LEGACY_SUPERSEDE_FAILED`` and returns 0 rather than
    taking the caller's transaction down with it.

    One cell that path cannot contain: if the UPDATE failed and the
    rollback to the savepoint also failed, the enclosing transaction is
    unusable too. In practice that combination requires the connection to
    be gone — on a live connection the rollback to the savepoint clears
    the failure — and a caller committing on a dead connection fails
    loudly rather than discarding writes silently, which is why this is
    not routed like the release failure above.

    If the UPDATE ran and the savepoint failed on the way out, nothing
    contained that either: on PostgreSQL a failed ``RELEASE`` aborts the
    enclosing transaction, and a caller that then commits gets no error
    while every staged write is discarded, because PostgreSQL treats
    ``COMMIT`` on an aborted transaction as a rollback. That propagates
    too. Re-raising cannot rescue the transaction — it is already gone —
    but it is the difference between a caller that knows and one that
    silently loses its writes.

    Python-side misuse — ``InvalidRequestError``, ``PendingRollbackError``,
    ``ArgumentError`` — is not a ``DBAPIError`` and was never caught here.
    ``mark_user_message_delivery`` lets a ``DBAPIError`` propagate too,
    though by having no handler at all rather than by re-raising: it
    guards a mandatory state transition, while this one is a best-effort
    sweep whose contained failures can safely no-op.

    ``Session.begin_nested()`` flushes the whole session before it issues
    the SAVEPOINT, and does so unconditionally: the flush is gated on
    transaction origin, not on ``autoflush``, so the production session's
    ``autoflush=False`` (``models/database.py``) does not suppress it.
    This function calls ``db.flush()`` itself, outside the ``try``, before
    opening the savepoint, for two reasons. Attribution: a constraint
    violation among the caller's own pending rows then raises from that
    line rather than from inside the ``try``, so the traceback points at
    the caller's row and not at this sweep. Independence: the placement
    does not rely on where SQLAlchemy chooses to flush, so a change in
    that behaviour cannot move a caller's failure into this function's
    catch. Either way a flush happens before the savepoint opens, so the
    assistant row the caller staged earlier in this same turn is already
    in the database when the UPDATE runs and is matched by it.
    ``interaction_handoff`` (``task_interaction_staging.py``) takes the
    same position on the same SQLAlchemy behavior, translating it into
    ``InteractionOwnerStateError`` rather than swallowing it.

    The exposure is a relabel racing an answer to the same question: this
    function flips a row to superseded while another transaction is
    answering that row, and the two need to be ordered against each other.
    This function does not order them — it takes no row lock on the task,
    reads no lease, and checks no caller state. Where this repository needs
    that ordering elsewhere it uses a row lock plus an ownership filter
    (``websocket.py`` loads the task with ``with_for_update()`` alongside
    ``runner_id``/``run_id`` predicates before a lease-scoped write), and
    ``_assert_current_attempt`` in ``task_interaction_staging.py`` records
    what that costs
    on SQLite, where SQLAlchemy drops ``with_for_update()`` and only
    single-writer semantics keep the window shut. Choosing among those
    belongs to the change that introduces the first call site, together
    with the structured answering path it would race; neither exists yet,
    so nothing is arranged here. A passing SQLite test says nothing either
    way about that ordering, for the reason the staging module documents.

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
    current behavior. The two remaining call-site obligations are carried
    here, by this docstring, and a caller that lands without honouring
    them is out of contract regardless of what its own tracking issue
    says: call this only after the interaction row for the same turn is
    staged, and never call it on a transcript question that is meant to
    remain answerable.
    """

    # ``Session.begin_nested()`` flushes the whole session before issuing the
    # SAVEPOINT. Flushing here instead, outside the try, raises a constraint
    # violation among the caller's own pending rows from this line rather
    # than from inside the block, and keeps that attribution independent of
    # where SQLAlchemy chooses to flush.
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
    # This statement sits outside the try below, alongside the flush: if it
    # fails, nothing has been relabelled, so the failure is not a failed
    # supersede and the traceback should point at this line rather than into
    # the block. A DBAPIError raised here reaches the caller unwrapped and
    # registers nothing (measured).
    if db.get_bind(TaskChatMessage).dialect.name == "sqlite":
        db.execute(
            text(f"UPDATE {TaskChatMessage.__tablename__} SET id = id WHERE id = -1")
        )

    updated = 0
    statement_succeeded = False
    savepoint_entered = False
    try:
        with db.begin_nested():
            # Force the SAVEPOINT out now, while savepoint_entered is still
            # False. SQLAlchemy does not emit it inside begin_nested(); it
            # defers it to the next statement on the connection, which here
            # would be the UPDATE below -- by which point the flag is True
            # and a failure to create the savepoint would be misread as a
            # failure inside one. This adds no SQL: the SAVEPOINT was going
            # to be issued either way, just later. Remove this line and the
            # SAVEPOINT goes back to being issued at the UPDATE, where a
            # failure to create it reads as a failure inside one. If this
            # call itself fails the flag is still False, so that routes to
            # propagate as well.
            db.connection()
            savepoint_entered = True
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
        if statement_succeeded or not savepoint_entered:
            # Either the savepoint never opened, or it opened, the UPDATE
            # ran, and it failed on the way out. Neither failure is inside
            # anything this function can roll back -- on PostgreSQL the
            # second aborts the enclosing transaction, and a caller that
            # gets 0 and no exception commits into it, which succeeds while
            # discarding every staged write. Propagate; the docstring above
            # explains why this can be reported but not rescued.
            raise
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
    message = stage_chat_message_no_commit(db, message)
    return message


def persist_assistant_message(
    db: Session,
    task_id: int,
    user_id: int,
    content: str,
    *,
    message_type: str = LEGACY_UNTRUSTED_ASSISTANT_MESSAGE_TYPE,
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
    message_type: str = LEGACY_UNTRUSTED_ASSISTANT_MESSAGE_TYPE,
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
    message = stage_chat_message_no_commit(db, message)
    return message


# The stored ``event_type`` for a completed context compaction. Written by
# ``ws_trace_handlers.get_event_type_mapping`` from
# ``TraceEventType(ACTION, END, COMPACT)``.
_COMPACT_END_EVENT_TYPE = "action_end_compact"
# How far back to keep looking for a reusable summary. The SQL filter below
# already excludes every event that cannot carry one, so reaching a second
# candidate means the first had a well-formed key with unusable content --
# corruption, not a normal outcome. A handful is plenty; the point of looping
# at all is that "this one is unusable" must never be read as "there is none".
_COMPACT_SUMMARY_SCAN_LIMIT = 8


@dataclass(frozen=True)
class CompactSummary:
    """A stored summary and the transcript row it stands in for."""

    content: str
    watermark_message_id: int
    context_refs: tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class TranscriptWindow:
    """What a turn replays, plus the point a later summary stands in for.

    ``watermark`` is the largest stored row id this window covers. Hand it to
    the agent so a compaction during this turn can record what its summary
    absorbed; without it the summary is unpositionable and the next turn falls
    back to replaying everything.
    """

    messages: List[Dict[str, Any]]
    watermark: Optional[int]


def _latest_compact_summary(
    db: Session,
    task_id: int,
    *,
    before_message_id: Optional[int],
) -> Optional[CompactSummary]:
    """The newest usable summary for this task, or None to replay in full.

    Reads down the ordered candidates rather than judging only the newest
    one. The message-dropping backstop emits an ``action_end_compact`` event
    too -- same type, deliberately carrying no summary, because it stands in
    for nothing -- so the newest event is routinely not the newest *usable*
    one. Stopping at the newest would let one backstop turn mask a perfectly
    good summary and send every later turn back to replaying everything.

    Selecting on the payload, not on ``strategy``: the question here is "can
    this be replayed", and only the summary and watermark answer it.
    ``strategy == "llm_summary"`` answers "how was this produced", which is a
    different question -- a summary emitted with no watermark, or any event
    written before this feature existed, carries that strategy and still
    cannot be positioned.

    ``build_id IS NULL`` is load-bearing, not a tidiness filter. A delegated
    sub-agent compacts its own context and writes its own compact events under
    the SAME ``task_id``, carrying a ``build_id``. Those describe a different
    conversation entirely, and adopting one would silently drop this task's
    real history behind a watermark that never belonged to it.
    """
    query = (
        db.query(TraceEvent.data)
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.event_type == _COMPACT_END_EVENT_TYPE,
            TraceEvent.build_id.is_(None),
            # Keeps backstop events and pre-feature rows out of the candidate
            # set entirely, so a run of them cannot exhaust the scan limit
            # below. SQL only sees whether the keys are there; whether the
            # content is usable is settled in Python.
            #
            # The ``as_*`` coercions are required, not cosmetic: a bare
            # ``data[key].isnot(None)`` compares as JSON and matches rows
            # where the key is absent, which silently filters nothing.
            TraceEvent.data[COMPACT_SUMMARY_METADATA_KEY].as_string().isnot(None),
            TraceEvent.data[COMPACT_WATERMARK_METADATA_KEY].as_integer().isnot(None),
        )
        .order_by(TraceEvent.timestamp.desc(), TraceEvent.id.desc())
    )
    if before_message_id is not None:
        # Replaying the conversation as of an earlier point (an edit or a
        # branch) invalidates any summary that already absorbed rows at or
        # past that point: it speaks for turns this replay is meant to
        # discard. Watermarks only ever advance, so skipping those leaves the
        # newest summary that is still entirely in the replayed past.
        query = query.filter(
            TraceEvent.data[COMPACT_WATERMARK_METADATA_KEY].as_integer()
            < before_message_id
        )

    for (data,) in query.limit(_COMPACT_SUMMARY_SCAN_LIMIT).all():
        summary = _compact_summary_from_event(data)
        if summary is not None:
            return summary
    return None


def _compact_summary_from_event(data: Any) -> Optional[CompactSummary]:
    """Read one compact event payload, or None if it cannot be replayed."""
    if not isinstance(data, dict):
        return None
    content = data.get(COMPACT_SUMMARY_METADATA_KEY)
    watermark = data.get(COMPACT_WATERMARK_METADATA_KEY)
    if not isinstance(content, str) or not content.strip():
        return None
    # ``bool`` is an ``int`` subclass, so this rejects a stray ``true`` that
    # would otherwise be read as "the summary covers rows up to 1" and drop
    # the entire transcript.
    if not isinstance(watermark, int) or isinstance(watermark, bool):
        return None
    raw_refs = data.get(COMPACT_CONTEXT_REFS_METADATA_KEY)
    context_refs = (
        tuple(ref for ref in raw_refs if isinstance(ref, dict))
        if isinstance(raw_refs, list)
        else ()
    )
    return CompactSummary(
        content=content,
        watermark_message_id=watermark,
        context_refs=context_refs,
    )


def load_task_transcript(
    db: Session,
    task_id: int,
    *,
    before_message_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    return load_task_transcript_window(
        db, task_id, before_message_id=before_message_id
    ).messages


def load_task_transcript_window(
    db: Session,
    task_id: int,
    *,
    before_message_id: Optional[int] = None,
) -> TranscriptWindow:
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
            return TranscriptWindow(messages=[], watermark=None)

    summary = _latest_compact_summary(db, task_id, before_message_id=before_message_id)

    query = db.query(TaskChatMessage).filter(TaskChatMessage.task_id == task_id)
    if before_message_id is not None:
        query = query.filter(TaskChatMessage.id < before_message_id)
    if summary is not None:
        # Everything at or below the watermark is what the summary says. The
        # rows stay in the table -- they are the user-visible chat history --
        # they just stop being replayed into the model's context.
        query = query.filter(TaskChatMessage.id > summary.watermark_message_id)

    stored_messages = query.order_by(TaskChatMessage.id.asc()).all()
    retained_references: list[tuple[ContextReference, ...]] = [
        () for _ in stored_messages
    ]
    remaining_references = _MAX_HISTORICAL_IMAGE_CONTEXT_REFS
    for index in range(len(stored_messages) - 1, -1, -1):
        if remaining_references <= 0:
            break
        stored_message = stored_messages[index]
        if safe_str(stored_message.role) == "assistant" and not (
            assistant_history_has_safe_ancillary_payload(
                safe_str(stored_message.message_type)
            )
        ):
            continue
        references = build_image_context_references(stored_message.attachments)
        retained_references[index] = references[:remaining_references]
        remaining_references -= len(retained_references[index])

    messages: List[Dict[str, Any]] = []
    for message, references in zip(stored_messages, retained_references):
        role = safe_str(message.role)
        content = safe_str(message.content)
        if role == "assistant":
            content = client_safe_assistant_history_content(
                content=content,
                message_type=safe_str(message.message_type),
            )
        item: Dict[str, Any] = {
            "role": role,
            "content": content,
        }
        if references:
            item[CONTEXT_REFS_KEY] = [
                reference.durable_dict() for reference in references
            ]
        messages.append(item)
    if summary is not None:
        summary_item: Dict[str, Any] = {
            "role": "system",
            "content": summary.content,
        }
        # The summary stands for the oldest part of the conversation, so it
        # takes what the newest-first walk above left over -- one allocation
        # under one rule, not a reserved slice ahead of it. Giving it first
        # pick starved the images the user had only just sent: a compaction
        # can legitimately retain more references than this window replays
        # (its own budget counts tokens, this one counts references), so a
        # reserved slice can consume the window outright. Recency also
        # matters more than the age of the decision -- when compaction chose
        # what to keep, the newest attachment did not exist yet.
        if summary.context_refs and remaining_references > 0:
            summary_item[CONTEXT_REFS_KEY] = list(
                summary.context_refs[:remaining_references]
            )
        messages.insert(0, summary_item)

    # The largest row this window covers, for the next compaction to record.
    # Rows this turn goes on to write land above it and are replayed verbatim
    # next time; that is deliberate. Their ids do not exist yet, and claiming
    # a summary covers work whose rows were never seen would lose them.
    watermark: Optional[int] = summary.watermark_message_id if summary else None
    if stored_messages:
        watermark = max(watermark or 0, int(stored_messages[-1].id))

    return TranscriptWindow(
        messages=normalize_transcript_messages(messages),
        watermark=watermark,
    )


def get_latest_waiting_question(
    db: Session, task_id: int, *, allow_superseded: bool = False
) -> tuple[Optional[str], Optional[list[dict[str, Any]]]]:
    """Return the latest persisted ask-user question for a waiting task.

    Two sequential passes rather than one ``message_type.in_(...)``
    predicate. Today's writer cannot produce the inversion that shape
    avoids: it relabels a task's question rows in one unpartitioned
    statement, so any row still labelled ``QUESTION_MESSAGE_TYPE``
    postdates every ``SUPERSEDED_MESSAGE_TYPE`` row on the task. The
    passes are kept for the writer shapes that come next -- a relabel
    narrowed to one run or turn, or one committed alongside a fresh
    question insert -- where the two labels' ids interleave and a single
    id-ordered predicate would rank a superseded row over a live one,
    inverting the priority this reader owes its callers. Narrowing a
    task-wide write to the row it actually means is already a stated
    obligation next door, on ``close_legacy_resume_interaction``'s close
    statement. The first pass is unconditional and always runs first.

    ``allow_superseded`` opens a second pass over
    ``SUPERSEDED_MESSAGE_TYPE`` rows, and only when the first pass came
    back empty. It is off by default, and the read surface adapter
    (``task_interaction_read.get_pending_interaction_question``) is the
    only caller that turns it on; every other caller reads exactly what it
    reads with the parameter absent.

    What the second pass hands back is the newest question row on this
    task that ``supersede_legacy_question_rows`` has relabelled. That
    writer's WHERE is ``_assistant_question_filters`` -- task, assistant
    role, message type -- with no run partition, so one call relabels
    every round's question row on the task at once; taking the highest id
    back is therefore this task's most recent question even on a task that
    has asked several times.
    """

    latest_question = (
        db.query(TaskChatMessage)
        .filter(*_assistant_question_filters(task_id))
        .order_by(TaskChatMessage.id.desc())
        .first()
    )
    if not latest_question and allow_superseded:
        latest_question = (
            db.query(TaskChatMessage)
            .filter(
                *_assistant_question_filters(
                    task_id, message_type=SUPERSEDED_MESSAGE_TYPE
                )
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
    message = stage_chat_message_no_commit(db, message)
    db.commit()
    db.refresh(message)
    return message
