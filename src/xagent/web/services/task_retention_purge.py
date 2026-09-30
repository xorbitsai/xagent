"""The scheduled purge that acts on the retention predicate (#2563).

#2562 decided *whether* a task may be expired; this module deletes. It runs
disabled: with no period configured (:func:`get_conversation_retention_days`
and :func:`get_trace_retention_days` both ``None``) the loop never starts, and
the shipping default configures neither.

Two paths, both per whole task, never by age within a task:

* **conversation expiry** removes the task outright, reusing
  :func:`purge_task_rows` so the foreign-key ordering that path already got
  right is not restated here.
* **trace expiry** keeps the conversation and removes only the execution
  trace. It exists because traces are the bulk of the stored bytes and are
  debugging data rather than a customer asset, so #2567 has a shorter period
  for them on the table.

What trace expiry actually costs
--------------------------------
This module first claimed that a purged trace costs only mid-run resume
state, because a new turn rebuilds the model's conversation from
``task_chat_messages``. **That was wrong**, and the correction matters
because the trace period is a #2567 decision that would have been taken
against it.

A new turn reads traces twice, on paths that start at the same
``_load_persisted_conversation_history`` the old claim cited:

* ``load_task_transcript_window`` -> ``_latest_compact_summary``
  (``chat_history_service.py``) reads the ``action_end_compact`` row for the
  compaction summary *and its watermark*, and the watermark is what filters
  the stored messages. With the row gone there is no summary and no
  watermark, so a compacted conversation replays **uncompacted** -- the
  runner re-compacts at its threshold, so this degrades a turn rather than
  breaking it, but the model's context for that turn is not what it was.
* ``task_execution_context_service`` reads ``tool_execution_end`` and the
  latest execution-failure summary, so the recovered tool and failure hints a
  next turn would have been given are gone too.

So trace expiry changes the next turn's context, not merely the ability to
resume a run that is already over. Nothing here decides whether that trade is
acceptable -- #2567 does, and #2655 records the option of keeping the
compaction and skill rows on this path if it is not. What this module owes
that decision is an accurate description of the cost, which is this one.

Terminal-only still holds, and for its own reason: a non-terminal task has a
run whose resume state the trace *is*, and the predicate's status leg refuses
those.

PostgreSQL only
---------------
:func:`assess_task_retention` takes ``SELECT ... FOR UPDATE`` on the task row,
and on SQLite SQLAlchemy compiles that clause away entirely -- the assessment
is then a read with no fence between it and the delete. Rather than build a
second serialization mechanism for a store no deployment runs retention on,
:func:`ensure_retention_purge_supported` refuses to start the job on any other
dialect and says so once. The row-level functions below stay dialect-neutral
so their semantics can be tested on both.

What the lock actually fences, including the three unverified producers
-----------------------------------------------------------------------
#2562 verified that several command producers lock the task row before
inserting, and explicitly left three unverified -- ``api/websocket.py``,
``services/task_interaction_service.py`` and ``services/workforce_runtime.py``
-- for this issue to confirm or fence. Neither was necessary, because the
fence does not depend on the producer at all:

``task_execution_commands.task_id`` is ``NOT NULL`` with a real foreign key to
``tasks.id`` (``models/task_command.py``, created in
``20260711_add_task_execution_commands``). PostgreSQL validates that reference
by taking ``FOR KEY SHARE`` on the parent row, which conflicts with the
``FOR UPDATE`` this purge holds. So *any* insert of a command for a task being
assessed blocks until the purge's transaction ends, whichever module issues it
and whether or not it locked anything itself.
``test_task_retention_purge_postgresql.py`` proves this against a real server
with two sessions rather than leaving it as a reading of the lock matrix.

The race therefore resolves one of two ways, both clean: the producer commits
first and the purge's command leg sees the row and skips the task; or the
purge holds the lock and the producer blocks, then fails against a task that
no longer exists. An accepted command is never silently deleted.

The same lock is what makes a second replica harmless. One loop starts per
web process, so a deployment running several of them sweeps several times
over: two purges that pick the same task serialize on its row, and the loser's
assessment then finds no row and reports it as not eligible. The cost is
duplicated scanning, not duplicated deletion, which is why this ships without
an advisory lock -- ``docs/deployment.md`` says so rather than the code
pretending the case cannot arise.

There is deliberately no second re-check just before the commit. For the
trace path it would be redundant with the lock, and for the conversation path
it cannot exist at all -- the row whose legs it would re-read is the row being
deleted. A backstop that is impossible on one path and redundant on the other
is worse than the lock it pretends to supplement, because it invites reading
the lock as optional. The lock is the fence; the PostgreSQL test is what keeps
that claim honest.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import and_, delete, exists, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from ...config import (
    get_conversation_retention_days,
    get_retention_batch_pause_seconds,
    get_retention_batch_size,
    get_retention_dry_run,
    get_retention_enabled,
    get_retention_sweep_interval_seconds,
    get_trace_retention_days,
)
from ..models.task import (
    Task,
    TraceCheckpointBlob,
    TraceEvent,
    TraceMessageBlob,
)
from ..models.task_interaction import TaskInteractionRequest
from .task_deletion import purge_task_rows
from .task_interaction_schema import interaction_requests_table_exists
from .task_retention import (
    RetentionDisposition,
    assess_task_retention,
    retention_expiry_condition,
    retention_quiescent_condition,
)

logger = logging.getLogger(__name__)

#: The one dialect whose row lock makes an assessment a deletion licence.
SUPPORTED_DIALECT = "postgresql"

#: Interaction status that the trace path refuses to purge around. Spelled
#: here rather than imported because it is the value the CHECK constraint
#: ``ck_task_interaction_requests_active_anchor`` is written against, not an
#: application-side vocabulary member.
INTERACTION_STATUS_ACTIVE = "active"


class RetentionPurgeUnsupported(RuntimeError):
    """Raised when the configured store cannot fence the purge."""


class RetentionPurgeAction(enum.Enum):
    """What one task's turn through the purge actually did, or would do."""

    PURGED_CONVERSATION = "purged_conversation"
    PURGED_TRACES = "purged_traces"
    #: The predicate refused it: live status, live lease, or a command owed
    #: execution. Expected and uninteresting -- a task can become busy between
    #: the batch scan and its own assessment.
    SKIPPED_BUSY = "skipped_busy"
    #: Trace expiry only: the task still holds an ``active`` interaction row,
    #: whose anchor the trace delete would try to NULL against
    #: ``ck_task_interaction_requests_active_anchor``.
    SKIPPED_ACTIVE_INTERACTION = "skipped_active_interaction"
    #: Trace expiry only: the task is due, but its trace is already gone, so
    #: there was nothing to delete. The scan filters these out, so reaching
    #: here means the rows went between the scan and the lock -- normal with
    #: more than one replica sweeping.
    NOTHING_TO_PURGE = "nothing_to_purge"
    #: The task's own purge raised. Counted rather than propagated, because a
    #: task that fails deterministically would otherwise stop every task
    #: behind it from ever being expired.
    FAILED = "failed"


@dataclass(frozen=True)
class RetentionPurgeReport:
    """One sweep's outcome, in the shape the audit line prints."""

    #: Candidates the scan returned. Named ``scanned`` in the audit line
    #: because that is what it counts: the locked assessment can still refuse
    #: any of them, so it is not a count of tasks that were eligible.
    eligible: int = 0
    purged_conversations: int = 0
    purged_traces: int = 0
    skipped_busy: int = 0
    skipped_active_interaction: int = 0
    nothing_to_purge: int = 0
    failed: int = 0
    dry_run: bool = False
    #: Highest task id this batch actually processed, or ``None`` when it
    #: processed none. The loop resumes after it rather than from the start of
    #: the table, which is what stops an undeletable page being re-read
    #: forever. Advanced per task rather than set from the candidate list up
    #: front, so a batch that stops early -- shutdown, kill switch -- does not
    #: carry the cursor past tasks it never looked at.
    last_task_id: int | None = None

    @property
    def purged(self) -> int:
        return self.purged_conversations + self.purged_traces

    def with_action(self, action: RetentionPurgeAction) -> RetentionPurgeReport:
        """This report plus one task's outcome.

        Every member of :class:`RetentionPurgeAction` increments exactly one
        counter. Written out rather than resolved through a name lookup so the
        counters are type-checked; a new action has to add its line here, and
        ``test_every_action_increments_exactly_one_counter`` is what fails if
        it does not.
        """
        return replace(
            self,
            purged_conversations=self.purged_conversations
            + int(action is RetentionPurgeAction.PURGED_CONVERSATION),
            purged_traces=self.purged_traces
            + int(action is RetentionPurgeAction.PURGED_TRACES),
            skipped_busy=self.skipped_busy
            + int(action is RetentionPurgeAction.SKIPPED_BUSY),
            skipped_active_interaction=self.skipped_active_interaction
            + int(action is RetentionPurgeAction.SKIPPED_ACTIVE_INTERACTION),
            nothing_to_purge=self.nothing_to_purge
            + int(action is RetentionPurgeAction.NOTHING_TO_PURGE),
            failed=self.failed + int(action is RetentionPurgeAction.FAILED),
        )

    def audit_line(self) -> str:
        """The single line one *batch* logs.

        One line per batch, not per task: a backlog of tens of thousands of
        tasks must not be the reason a log budget is spent, and the per-task
        detail that matters (*why* a task was skipped) is carried by the
        counters rather than by prose. A sweep that drains a backlog walks it
        a page at a time, so it emits one of these per page.
        """
        return (
            f"retention purge {'dry-run' if self.dry_run else 'run'}: "
            f"scanned={self.eligible} "
            f"purged_conversations={self.purged_conversations} "
            f"purged_traces={self.purged_traces} "
            f"skipped_busy={self.skipped_busy} "
            f"skipped_active_interaction={self.skipped_active_interaction} "
            f"nothing_to_purge={self.nothing_to_purge} "
            f"failed={self.failed}"
        )


def ensure_retention_purge_supported(db: Session) -> None:
    """Refuse to purge on a store whose row lock does not fence.

    Raises:
        RetentionPurgeUnsupported: On any dialect but PostgreSQL.
    """
    dialect = db.get_bind().dialect.name
    if dialect != SUPPORTED_DIALECT:
        raise RetentionPurgeUnsupported(
            f"retention purge requires {SUPPORTED_DIALECT}, not {dialect!r}: "
            "SELECT ... FOR UPDATE does not fence there, so an assessment is "
            "not a deletion licence"
        )


def select_purge_candidates(
    db: Session,
    *,
    now: datetime,
    conversation_days: int | None,
    trace_days: int | None,
    limit: int,
    after_task_id: int = 0,
) -> list[int]:
    """Up to ``limit`` task ids that some path would expire, lowest id first.

    Scans without locking; every id is re-assessed under a lock before
    anything is deleted, so a candidate going busy in between costs one
    skipped assessment and nothing else.

    ``after_task_id`` is what keeps a sweep from starving. A task this purge
    declines stays eligible -- an ``active`` interaction row is not something
    the purge resolves, and nothing else clears it either -- so a scan that
    always restarted at the lowest id would hand back the same undeletable
    page forever and never reach the tasks behind it. That is not theoretical:
    before this parameter existed, three tasks holding active interaction rows
    kept a fourth, conversation-expired task alive across every sweep at
    ``limit=3``.

    The caller resumes past the page it processed and resets to ``0`` once a
    page comes back short, so the table is walked in order and then from the
    top again -- the same cross-tick cursor ``orphan_upload_gc`` carries, for
    the same reason.

    The two periods are admitted separately rather than through the shorter of
    them, because the trace leg carries a condition the conversation leg must
    not: a task whose traces are already gone has nothing left for trace
    expiry to do, and admitting it anyway is what made the sweep re-lock,
    re-UPDATE and re-count the whole trace-expired history on every pass --
    and, because a full page shortens the pause, do it at the batch cadence
    rather than the sweep interval, forever. A conversation-expired task with
    no traces still has its own row to delete, so the same condition on that
    leg would strand it.

    The ``EXISTS`` is spelled here rather than in
    :func:`retention_candidate_condition`, which is shared with the counters
    behind ``xagent retention preview``: adding it there would silently change
    what that diagnostic reports, and it cannot be applied to both legs
    anyway. ``ix_trace_events_task_id_event_type`` makes it a leading-column
    lookup.
    """
    conversation_due = retention_expiry_condition(now=now, days=conversation_days)
    trace_due = and_(
        retention_expiry_condition(now=now, days=trace_days),
        exists(select(1).where(TraceEvent.task_id == Task.id)),
    )
    if conversation_days is None and trace_days is None:
        return []
    rows = db.execute(
        select(Task.id)
        .where(
            Task.id > after_task_id,
            retention_quiescent_condition(now=now),
            or_(conversation_due, trace_due),
        )
        .order_by(Task.id.asc())
        .limit(limit)
    ).scalars()
    return [int(row) for row in rows]


def _has_active_interaction(db: Session, task_id: int) -> bool:
    """Whether this task holds an ``active`` interaction row.

    Gated on table presence for the same reason ``purge_task_rows`` gates its
    delete: a deployment upgraded to a revision before the table exists must
    still be able to expire data.
    """
    if not interaction_requests_table_exists(db):
        return False
    return bool(
        db.execute(
            select(TaskInteractionRequest.id)
            .where(
                TaskInteractionRequest.task_id == task_id,
                TaskInteractionRequest.status == INTERACTION_STATUS_ACTIVE,
            )
            .limit(1)
        ).scalar_one_or_none()
    )


def _rowcount(result: object) -> int:
    """Rows a DML statement touched, typed for mypy.

    ``Session.execute`` is annotated as returning ``Result``, which has no
    ``rowcount``; every DML execution actually returns ``CursorResult``, which
    does. The repo spells this the same way in ``services/triggers.py``.
    """
    return int(getattr(result, "rowcount", 0) or 0)


def _purge_trace_rows(db: Session, task_id: int) -> int:
    """Delete one task's trace, keeping the task and its conversation.

    Returns the number of rows it actually changed, so the caller can tell a
    real expiry from a task whose trace was already gone -- reporting the
    second as ``purged_traces`` made the counter claim work that never
    happened.

    Statement order matches ``purge_task_rows`` where it overlaps, and for the
    same reason: ``tasks.last_checkpoint_trace_event_id`` is a foreign key
    into ``trace_events``, so a task still pointing at a row blocks that row's
    delete.

    Both checkpoint pointers are cleared, not just the foreign-key one.
    ``last_checkpoint_event_id`` carries the trace event's application-level
    id, which after this delete resolves to nothing; leaving it set would hand
    a later reader a pointer that looks live and is not.

    ``dag_executions`` is deliberately kept, which is the one place this
    diverges from ``purge_task_rows`` beyond keeping the task itself: a DAG
    execution is the task's own plan and progress, not a trace of how it ran,
    and the task survives this path. Trace expiry removes ``trace_events`` and
    the two blob tables, and nothing else.

    ``updated_at`` is pinned rather than allowed to fire its ``onupdate``.
    Expiring a trace is maintenance, not execution activity, and #2557's
    side-effect review names maintenance writes that advance ``updated_at`` as
    a hazard in their own right.
    """
    removed = 0
    # Only when a pointer is actually set. An unconditional UPDATE matches the
    # row every time, and PostgreSQL does not elide a no-op update -- it writes
    # a new tuple and leaves a dead one. That is what made a re-purged task
    # cost a dead tuple per sweep.
    pointers_set = db.execute(
        select(Task.id).where(
            Task.id == task_id,
            or_(
                Task.last_checkpoint_event_id.is_not(None),
                Task.last_checkpoint_trace_event_id.is_not(None),
            ),
        )
    ).scalar_one_or_none()
    if pointers_set is not None:
        removed += _rowcount(
            db.execute(
                update(Task)
                .where(Task.id == task_id)
                .values(
                    last_checkpoint_event_id=None,
                    last_checkpoint_trace_event_id=None,
                    updated_at=Task.updated_at,
                )
                .execution_options(synchronize_session=False)
            )
        )
    for model in (TraceCheckpointBlob, TraceMessageBlob, TraceEvent):
        removed += _rowcount(
            db.execute(
                delete(model)
                .where(model.task_id == task_id)
                .execution_options(synchronize_session=False)
            )
        )
    return removed


def purge_task(
    db: Session,
    task_id: int,
    *,
    now: datetime,
    conversation_days: int | None,
    trace_days: int | None,
    dry_run: bool = False,
) -> RetentionPurgeAction:
    """Assess one task under a lock and expire what its disposition allows.

    Owns its transaction: it commits what it deleted, or rolls back. One
    transaction per task is what keeps the lock held from the assessment to
    the delete, and what keeps a task that fails from taking a whole batch
    with it.

    ``dry_run`` computes the same action against the same locked assessment
    and then rolls back, so what it reports is what a real run would do rather
    than a separately-derived estimate. It performs no external call because
    this module makes none at all: external cleanup is #2564's, and every
    statement here is a row delete.

    Every exit rolls back or commits, so the ``FOR UPDATE`` the assessment
    took is never held past this call -- a sweep that left one open per task
    inspected would hold locks across the whole batch.
    """
    committed = False
    try:
        assessment = assess_task_retention(
            db,
            task_id,
            now=now,
            conversation_days=conversation_days,
            trace_days=trace_days,
        )
        if assessment.disposition is RetentionDisposition.NOT_ELIGIBLE:
            return RetentionPurgeAction.SKIPPED_BUSY

        if assessment.disposition is RetentionDisposition.CONVERSATION_EXPIRED:
            action = RetentionPurgeAction.PURGED_CONVERSATION
        else:
            if _has_active_interaction(db, task_id):
                return RetentionPurgeAction.SKIPPED_ACTIVE_INTERACTION
            action = RetentionPurgeAction.PURGED_TRACES

        if dry_run:
            return action

        if action is RetentionPurgeAction.PURGED_CONVERSATION:
            purge_task_rows(db, task_id=task_id)
        elif _purge_trace_rows(db, task_id) == 0:
            # The scan filters these out, so arriving here means the rows went
            # between the scan and the lock. Committing an empty transaction
            # is still right -- it releases the lock -- but calling it a purge
            # is not.
            action = RetentionPurgeAction.NOTHING_TO_PURGE
        db.commit()
        committed = True
        return action
    finally:
        # Covers every exit -- the two skips, the dry run, the purge (both
        # branches converge on one return) and any exception -- because
        # each one either committed or must not.
        # Written as a flag rather than as a rollback before each ``return``
        # so a later exit added without one cannot leak the row lock.
        if not committed:
            db.rollback()


def run_retention_purge_batch(
    session_factory: sessionmaker[Session],
    *,
    now: datetime | None = None,
    limit: int | None = None,
    after_task_id: int = 0,
    should_continue: Callable[[], bool] | None = None,
    periods: tuple[int | None, int | None] | None = None,
    dry_run: bool | None = None,
) -> RetentionPurgeReport:
    """Purge one bounded batch and return what it did.

    ``periods`` is the pair ``(conversation_days, trace_days)``, already read.
    It is a tuple rather than two arguments precisely so that ``None`` keeps
    one meaning: elsewhere in this module ``days=None`` means *unlimited
    retention*, and a pair of parameters that also accepted ``None`` as "use
    the configured value" would give one value two opposite meanings -- the
    dangerous reading being a caller passing ``None`` for "expire nothing" and
    getting the configured period. Omitting the tuple entirely reads the
    configuration; passing one supplies it.

    ``dry_run`` is passed the same way and for the same reason. Every
    retention setting is read once, by the loop: they cannot change within a
    process (see the retention section of ``config.py``), and re-reading them
    per batch meant an unusable value logged its warning every few seconds
    while a backlog drained.

    Configuration is read per batch rather than captured at import. That is
    not the same as being changeable at run time, and this module used to
    claim it was: ``.env`` is read once at process start and nothing mutates
    the environment afterwards, so within one process these values never
    change. Reading them per batch is about not caching a value across a
    restart boundary, not about live reconfiguration.

    The kill switch is *not* re-read per task. It cannot change within a
    process, so such a read could only ever fire under a monkeypatch -- which
    made the "stops mid-batch safely" criterion pass synthetically. It is read
    once per loop, with the periods and the dry-run flag; what actually stops
    a batch mid-flight is shutdown, through ``should_continue``.

    Between tasks is the only place stopping is free: each task's purge owns
    one transaction, so a batch that stops there leaves committed work behind
    it and untouched tasks in front of it, with nothing in between. That is
    also what keeps shutdown quick, since this runs in a worker thread that
    cancelling the calling task would detach rather than end.

    Raises:
        RetentionPurgeUnsupported: If the store is not PostgreSQL.
    """
    now = now or datetime.now(timezone.utc)
    conversation_days, trace_days = (
        periods
        if periods is not None
        else (get_conversation_retention_days(), get_trace_retention_days())
    )
    limit = limit if limit is not None else get_retention_batch_size()
    dry_run = dry_run if dry_run is not None else get_retention_dry_run()

    with session_factory() as db:
        ensure_retention_purge_supported(db)
        candidates = select_purge_candidates(
            db,
            now=now,
            conversation_days=conversation_days,
            trace_days=trace_days,
            limit=limit,
            after_task_id=after_task_id,
        )

    report = RetentionPurgeReport(eligible=len(candidates), dry_run=dry_run)
    processed = 0
    try:
        for task_id in candidates:
            if should_continue is not None and not should_continue():
                logger.info(
                    "retention purge stopping after %d task(s) processed", processed
                )
                break
            try:
                with session_factory() as db:
                    action = purge_task(
                        db,
                        task_id,
                        now=now,
                        conversation_days=conversation_days,
                        trace_days=trace_days,
                        dry_run=dry_run,
                    )
            except Exception:  # noqa: BLE001
                # Counted and carried, not propagated. A task that fails the
                # same way every time -- a foreign key this census missed, a
                # constraint a deployment adds -- would otherwise abort the
                # batch before the cursor advanced, so no task at or after its
                # id would ever be expired again, with a daily warning as the
                # only symptom.
                logger.warning(
                    "retention purge failed for task %s; continuing",
                    task_id,
                    exc_info=True,
                )
                action = RetentionPurgeAction.FAILED
            report = replace(report, last_task_id=task_id).with_action(action)
            processed += 1
    finally:
        # In ``finally`` so that work already committed is still reported. Each
        # task commits its own transaction, so an exception escaping this loop
        # would otherwise discard the record of deletions that did happen.
        logger.info(report.audit_line())
    return report


def retention_purge_configured() -> bool:
    """Whether any period is configured and the kill switch is up."""
    if not get_retention_enabled():
        return False
    return (
        get_conversation_retention_days() is not None
        or get_trace_retention_days() is not None
    )


async def run_retention_purge_loop(
    session_factory: sessionmaker[Session],
    *,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Sweep until stopped, walking the table with a cursor.

    A full page advances the cursor past it and pauses only briefly, so a
    large initial backlog is worked down without waiting a sweep interval per
    batch. A short page means the backlog behind the cursor is drained: the
    cursor resets and the next sweep starts from the top of the table after a
    full interval. Same cross-tick cursor ``orphan_upload_gc`` carries, for the
    same reason -- without it a page the purge cannot delete is re-read
    forever, and because a full page also shortens the pause, the loop would
    spin on it at the batch pause rather than merely stalling.

    A page the purge declines is therefore re-assessed once per pass over the
    table rather than continuously, which is the right cadence: what makes
    such a task purgeable is something outside this loop finishing.

    The loop never raises out: a failed batch is logged with its traceback and
    retried on the next interval. An unattended sweep has no other surface,
    and a purge that stops permanently on one bad task would look exactly like
    a purge that had nothing to do.

    ``stop_event`` reaches the batch itself, not just the sleep between
    batches. It has to: the batch runs in a worker thread, and cancelling this
    coroutine would detach that thread rather than end it -- leaving a sweep
    of a full batch still deleting while the process tried to exit.
    """
    stop = stop_event if stop_event is not None else asyncio.Event()
    after_task_id = 0
    # Read once, not per batch: these cannot change within a process, and an
    # unusable one warns on every read.
    periods = (get_conversation_retention_days(), get_trace_retention_days())
    dry_run = get_retention_dry_run()
    if not get_retention_enabled():
        logger.info("retention purge not started: disabled by the kill switch")
        return
    while not stop.is_set():
        pause = get_retention_sweep_interval_seconds()
        try:
            limit = get_retention_batch_size()
            report = await asyncio.to_thread(
                run_retention_purge_batch,
                session_factory,
                limit=limit,
                after_task_id=after_task_id,
                should_continue=lambda: not stop.is_set(),
                periods=periods,
                dry_run=dry_run,
            )
            if report.eligible >= limit and report.last_task_id is not None:
                after_task_id = report.last_task_id
                # A dry run deletes nothing, so its backlog never shrinks: the
                # pages stay full forever and the short pause would walk the
                # expired set, taking a row lock on every task in it, at the
                # batch cadence with no end. It keeps the cursor -- reporting
                # the whole backlog is the point -- but waits a full interval.
                pause = (
                    get_retention_sweep_interval_seconds()
                    if report.dry_run
                    else get_retention_batch_pause_seconds()
                )
            else:
                after_task_id = 0
        except RetentionPurgeUnsupported:
            # Configuration, not weather: retrying cannot fix the dialect.
            logger.warning("retention purge stopped", exc_info=True)
            return
        except Exception:  # noqa: BLE001
            logger.warning("retention purge batch failed", exc_info=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=pause)
        except TimeoutError:
            continue


__all__ = [
    "RetentionPurgeAction",
    "RetentionPurgeReport",
    "RetentionPurgeUnsupported",
    "ensure_retention_purge_supported",
    "purge_task",
    "retention_purge_configured",
    "run_retention_purge_batch",
    "run_retention_purge_loop",
    "select_purge_candidates",
]
