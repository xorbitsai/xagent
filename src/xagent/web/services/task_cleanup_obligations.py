"""The external cleanup a task deletion still owes, and its retry (#2587).

Every task deletion has the same shape: the rows are deleted and committed,
and the resources those rows located -- the workspace directory, the state a
runtime extension holds -- are released afterwards. The order is forced. The
rows are what the resources are found from, and a database rollback cannot
restore a directory that was already removed, so the release can only follow
the commit. That leaves a window where the rows are gone and the resource is
not, and before this module nothing recorded it: a failed release was a log
line, and a crash between the commit and the release left no trace at all.

Write-ahead, not write-on-failure
---------------------------------
An obligation is recorded **in the same transaction as the row deletion**,
before anything is attempted, and deleted once its resource is released. So
the obligation exists exactly when the rows are gone, whatever happens next:
a failed release leaves it pending, a crashed process leaves it pending, and
a rolled-back deletion takes it with it.

Two ways an obligation is worked off
------------------------------------
* **Inline**, by the deletion that recorded it. The on-demand endpoints try
  the release straight after their commit and report the outcome to the
  obligation (:func:`settle_cleanup_attempt_no_commit`). They record with the due time pushed out
  by :data:`CLAIM_LEASE`, so the retry driver does not race the attempt they
  are about to make.
* **By the retry driver** (:func:`run_cleanup_obligation_batch`), for
  everything inline did not finish and for everything the retention purge
  records -- the purge makes no external call at all, because it holds a row
  lock while it records and must not hold it across someone else's network.

Row-independence
----------------
An obligation's ``locator`` is everything its release needs. The task row is
gone by the time anything reads it, and so, on account deletion, is the
owner: the workspace locator is the already-captured
:class:`WorkspaceCleanupTarget`, and the runtime-extension locator carries the
owner id and task source the provider context is built from.

Idempotence
-----------
A resource that is already gone is a success: the workspace remover finds
nothing and returns, and ``on_task_deleted`` is required to be idempotent by
the provider contract. That is what makes it safe for the driver to retry an
attempt whose outcome it never learned.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from ...config import get_task_cleanup_max_attempts
from ...core.task_runtime import TaskRuntimeContext
from ..models.task import Task
from ..models.task_cleanup_obligation import TaskCleanupObligation
from .task_runtime import delete_task_extensions
from .task_workspace_cleanup import (
    WorkspaceCleanupTarget,
    remove_task_workspace,
    unscoped_workspace_cleanup_target,
)

logger = logging.getLogger(__name__)

#: How long an obligation is left alone once someone has taken it on -- the
#: inline attempt after a deletion, or a driver that claimed it. Long enough
#: that a slow release (a large ``rmtree``, a provider round trip) is never
#: raced; short enough that a process that died mid-attempt is retried soon.
CLAIM_LEASE = timedelta(minutes=15)

#: Delay before the second attempt; each later one doubles it, up to
#: :data:`MAX_RETRY_BACKOFF`. With the default budget of eight attempts the
#: delays are 5, 10, 20, ..., 320 minutes -- roughly ten and a half hours of
#: retrying in total, never reaching :data:`MAX_RETRY_BACKOFF` itself within
#: the default budget -- which rides out a provider outage without retrying a
#: permanently broken one forever.
BASE_RETRY_BACKOFF = timedelta(minutes=5)
MAX_RETRY_BACKOFF = timedelta(hours=6)

#: How long a runtime-extension obligation whose provider is not registered
#: here is left before this driver looks at it again. ``delete_task_extensions``
#: logs an ``ERROR`` for every unregistered binding it is asked about, so a
#: short interval would flood the log on every batch until some process
#: registers the provider; a long one would instead stall a replica that
#: *does* have it registered, since that replica also waits out this interval
#: before its first attempt. An hour is a compromise between the two, not a
#: measured value.
PROVIDER_RECHECK_INTERVAL = timedelta(hours=1)

#: Obligations one driver batch looks at. Each is claimed only when its turn
#: comes, so a slow release early in the batch cannot run the later ones'
#: leases out.
DEFAULT_RETRY_BATCH_SIZE = 100

#: ``last_error`` is for a human reading the reconciliation list, not a log
#: sink; a provider that raises a megabyte of text must not put it in a row.
MAX_ERROR_LENGTH = 2000

#: Why a workspace obligation whose scope never resolved is abandoned rather
#: than discharged. See :func:`workspace_obligation`.
SCOPE_UNRESOLVED_REASON = (
    "the task's execution scope could not be resolved before its rows were "
    "deleted; the unscoped candidates were cleared, but a workspace under a "
    "scope segment cannot be located and needs manual reconciliation"
)


class CleanupResourceKind(str, enum.Enum):
    """What an obligation releases."""

    WORKSPACE = "workspace"
    RUNTIME_EXTENSION = "runtime_extension"


class CleanupObligationStatus(str, enum.Enum):
    """Where an obligation stands. There is no ``completed``: completing one
    deletes it, so what remains in the table is exactly what is still owed or
    what an operator has to reconcile."""

    PENDING = "pending"
    #: The attempt budget ran out. Terminal: left for an operator.
    EXHAUSTED = "exhausted"
    #: Deliberately not retried -- an admin force-deleted past a failing
    #: provider, or the release cannot be made safely at all. Terminal.
    ABANDONED = "abandoned"


#: What retrying will not resolve: the operator's reconciliation list.
TERMINAL_STATUSES = (
    CleanupObligationStatus.EXHAUSTED,
    CleanupObligationStatus.ABANDONED,
)


@dataclass(frozen=True)
class CleanupObligation:
    """One resource to release, before it is recorded. Plain values,
    row-independent."""

    task_id: int
    owner_id: int | None
    kind: CleanupResourceKind
    key: str
    locator: Mapping[str, Any]
    status: CleanupObligationStatus = CleanupObligationStatus.PENDING
    last_error: str | None = None


@dataclass(frozen=True)
class RecordedCleanupObligation:
    """One obligation as it stands in the table.

    ``id`` and ``attempts`` together are the fence every outcome is written
    against: an outcome for an attempt the row has since moved past is
    dropped rather than overwriting a newer one.
    """

    id: int
    task_id: int
    owner_id: int | None
    kind: CleanupResourceKind
    key: str
    locator: Mapping[str, Any]
    status: CleanupObligationStatus
    attempts: int
    last_error: str | None
    next_attempt_at: datetime
    created_at: datetime


def workspace_obligation(
    target: WorkspaceCleanupTarget,
    *,
    scope_resolved: bool = True,
) -> CleanupObligation:
    """The obligation to remove the workspace ``target`` names.

    ``scope_resolved=False`` marks a target captured without the task's
    execution scope -- the unscoped fallback candidates. Removing those is
    still worth doing, but it cannot prove the task's workspace is gone: one
    written under a scope segment is not among the candidates. Such an
    obligation is therefore never simply discharged; once its candidates are
    cleared it moves to ``abandoned`` so an operator sees it.
    """

    return CleanupObligation(
        task_id=int(target.task_id),
        owner_id=target.owner_id,
        kind=CleanupResourceKind.WORKSPACE,
        key="",
        locator={
            "base_dirs": list(target.base_dirs),
            "scope_resolved": bool(scope_resolved),
        },
    )


def captured_workspace_obligation(
    task_id: int,
    owner_id: int,
    captured: WorkspaceCleanupTarget | None,
) -> CleanupObligation:
    """The workspace obligation for a best-effort capture's result.

    ``captured`` is ``None`` when the scope would not resolve; the obligation
    then names the unscoped candidates the post-deletion fallback probes, and
    is marked so that clearing them is not taken for a full cleanup.
    """

    return workspace_obligation(
        captured or unscoped_workspace_cleanup_target(task_id, owner_id),
        scope_resolved=captured is not None,
    )


def extension_obligation(
    *,
    task_id: int,
    user_id: int,
    source: Any,
    extension: str,
    status: CleanupObligationStatus = CleanupObligationStatus.PENDING,
    reason: str | None = None,
) -> CleanupObligation:
    """The obligation to release ``extension``'s state for one task.

    ``user_id`` and ``source`` are what the provider context is built from;
    they are captured now because neither can be read once the task row is
    gone.
    """

    return CleanupObligation(
        task_id=int(task_id),
        owner_id=int(user_id),
        kind=CleanupResourceKind.RUNTIME_EXTENSION,
        key=extension,
        locator={
            "user_id": int(user_id),
            "source": str(source) if source is not None else None,
        },
        status=status,
        last_error=reason,
    )


class CleanupProviderNotRegistered(LookupError):
    """Raised by :func:`_release` when a runtime extension is not registered
    in this process.

    This is not this obligation's failure: nothing here is broken, and there
    is nothing wrong to retry away -- the provider simply is not loaded in
    *this* process, and may be in another replica, or in this one after a
    future deploy. :func:`_attempt_claimed` catches it specially and refunds
    the attempt the claim spent, so the obligation stays pending without
    counting against the budget a genuine release failure would spend. See
    module docstring and the callers in ``chat.py``/``admin_users.py`` for the
    design this preserves: a binding whose provider is not registered stays
    owed until one that can release it is.
    """


def describe_cleanup_failure(exc: BaseException) -> str:
    """The ``last_error`` text for a release that raised."""

    return f"{type(exc).__name__}: {exc}"


def _workspace_target(obligation: RecordedCleanupObligation) -> WorkspaceCleanupTarget:
    return WorkspaceCleanupTarget(
        task_id=obligation.task_id,
        owner_id=obligation.owner_id,
        base_dirs=tuple(str(base) for base in obligation.locator.get("base_dirs", ())),
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def record_cleanup_obligations_no_commit(
    db: Session,
    obligations: Iterable[CleanupObligation],
    *,
    now: datetime | None = None,
    inline_attempt: bool = False,
) -> list[RecordedCleanupObligation]:
    """Add ``obligations`` to the caller's transaction and return them.

    The caller commits it together with the row deletion the obligations
    belong to -- that is the whole point. The returned values carry the ids a
    later :func:`settle_cleanup_attempt_no_commit` settles against.

    Every call records new rows; nothing is deduplicated against what the
    table already holds. A task's rows can only be deleted once, so the only
    thing an existing obligation with the same task id can be is one left by
    an *earlier* task that held the id -- SQLite reuses the highest deleted
    rowid -- and folding the new deletion into it would lose the new
    obligation whenever the old one was terminal.

    ``inline_attempt`` says the caller will try the release itself straight
    after committing. The obligations are then not due for the retry driver
    until :data:`CLAIM_LEASE` has passed, so the two do not race; otherwise
    they are due at once.
    """

    now = now or _utcnow()
    due_at = now + CLAIM_LEASE if inline_attempt else now
    rows = [
        TaskCleanupObligation(
            task_id=obligation.task_id,
            owner_id=obligation.owner_id,
            resource_kind=obligation.kind.value,
            resource_key=obligation.key,
            locator=dict(obligation.locator),
            status=obligation.status.value,
            attempts=0,
            last_error=obligation.last_error,
            next_attempt_at=due_at,
            created_at=now,
            updated_at=now,
        )
        for obligation in obligations
    ]
    if not rows:
        return []
    db.add_all(rows)
    # Flushed for the ids; the session is ``autoflush=False``.
    db.flush()
    return [_as_recorded(row) for row in rows]


def _truncate(error: str) -> str:
    return error if len(error) <= MAX_ERROR_LENGTH else error[:MAX_ERROR_LENGTH]


def _retry_backoff(attempts: int) -> timedelta:
    """Delay after the ``attempts``-th failure."""

    exponent = max(attempts - 1, 0)
    # Capped before multiplying so a large attempt count cannot overflow.
    if exponent >= 16:
        return MAX_RETRY_BACKOFF
    backoff: timedelta = BASE_RETRY_BACKOFF * (1 << exponent)
    return min(backoff, MAX_RETRY_BACKOFF)


def _outcome_values(
    *,
    locator: Mapping[str, Any],
    attempts: int,
    error: str | None,
    now: datetime,
    max_attempts: int,
) -> dict[str, Any] | None:
    """What one attempt's outcome writes, or ``None`` to discharge it.

    ``attempts`` already counts this attempt.
    """

    if error is None:
        if locator.get("scope_resolved", True) is False:
            return {
                "status": CleanupObligationStatus.ABANDONED.value,
                "last_error": SCOPE_UNRESOLVED_REASON,
                "updated_at": now,
            }
        return None
    values: dict[str, Any] = {"last_error": _truncate(error), "updated_at": now}
    if attempts >= max_attempts:
        values["status"] = CleanupObligationStatus.EXHAUSTED.value
    else:
        values["next_attempt_at"] = now + _retry_backoff(attempts)
    return values


def _write_outcome(
    db: Session,
    obligation: RecordedCleanupObligation,
    values: dict[str, Any] | None,
) -> CleanupObligationStatus | None:
    """Write ``values`` (``None`` discharges) fenced on id and ``attempts``.

    Returns the obligation's status afterwards, or ``None`` when it is gone.
    When the fence misses -- the row moved past this attempt, or was never
    pending -- nothing is written and the row's current status is returned,
    so a caller reporting "pending" never reports a newer claim as done.

    The "gave up" line is logged here, once the fence confirms an EXHAUSTED
    write landed, so it is never printed for an outcome a newer claim beat.
    """

    fence = (
        TaskCleanupObligation.id == obligation.id,
        TaskCleanupObligation.attempts == obligation.attempts,
        TaskCleanupObligation.status == CleanupObligationStatus.PENDING.value,
    )
    if values is None:
        result = db.execute(delete(TaskCleanupObligation).where(*fence))
    else:
        result = db.execute(
            update(TaskCleanupObligation).where(*fence).values(**values)
        )
    if getattr(result, "rowcount", 0) == 1:
        if values is None:
            return None
        written_status = CleanupObligationStatus(
            values.get("status", CleanupObligationStatus.PENDING.value)
        )
        if written_status is CleanupObligationStatus.EXHAUSTED:
            logger.error(
                "Task cleanup obligation %s (task %s, %s) gave up after %d "
                "attempt(s) and needs manual reconciliation: %s",
                obligation.id,
                obligation.task_id,
                obligation.kind.value,
                values.get("attempts", obligation.attempts),
                values.get("last_error", ""),
            )
        return written_status
    current = db.execute(
        select(TaskCleanupObligation.status).where(
            TaskCleanupObligation.id == obligation.id
        )
    ).scalar_one_or_none()
    return CleanupObligationStatus(current) if current is not None else None


def settle_cleanup_attempt_no_commit(
    db: Session,
    obligation: RecordedCleanupObligation,
    *,
    error: str | None,
    now: datetime | None = None,
    max_attempts: int | None = None,
) -> CleanupObligationStatus | None:
    """Report the outcome of an inline release the caller just attempted.

    ``obligation`` is what :func:`record_cleanup_obligations_no_commit`
    returned; ``error`` is ``None`` when the resource was released. Returns
    the obligation's status afterwards, or ``None`` when it was discharged --
    so ``is not None`` is exactly "the caller should report cleanup as
    pending".

    The attempt counts against the same budget the driver's attempts do. If
    the inline release outlasted :data:`CLAIM_LEASE` and a driver has since
    claimed the row, the driver owns it and this outcome is dropped. An
    obligation recorded ``abandoned`` stays that way whatever was attempted.
    """

    now = now or _utcnow()
    max_attempts = max_attempts or get_task_cleanup_max_attempts()
    attempted = replace(obligation, attempts=obligation.attempts + 1)
    values = _outcome_values(
        locator=obligation.locator,
        attempts=attempted.attempts,
        error=error,
        now=now,
        max_attempts=max_attempts,
    )
    if values is not None:
        values["attempts"] = attempted.attempts
    return _write_outcome(db, obligation, values)


def settle_cleanup_attempts_sync(
    session_factory: Callable[[], Session],
    outcomes: Sequence[tuple[RecordedCleanupObligation, str | None]],
) -> list[CleanupObligationStatus | None]:
    """:func:`settle_cleanup_attempt_no_commit` for a batch, in its own session.

    For the deletion endpoints, which settle after their commit from a worker
    thread. A failure to settle is logged, not raised: the rows are already
    deleted, and the obligations are still pending, so the driver will get to
    them -- reported here as pending, which is what they still are.
    """

    try:
        with session_factory() as db:
            statuses = [
                settle_cleanup_attempt_no_commit(db, obligation, error=error)
                for obligation, error in outcomes
            ]
            db.commit()
            return statuses
    except Exception:
        logger.error(
            "Could not settle %d task cleanup attempt(s); the retry driver "
            "will re-attempt them",
            len(outcomes),
            exc_info=True,
        )
        return [CleanupObligationStatus.PENDING for _ in outcomes]


def _as_recorded(row: TaskCleanupObligation) -> RecordedCleanupObligation:
    return RecordedCleanupObligation(
        id=int(row.id),
        task_id=int(row.task_id),
        owner_id=int(row.owner_id) if row.owner_id is not None else None,
        kind=CleanupResourceKind(row.resource_kind),
        key=str(row.resource_key),
        locator=dict(row.locator or {}),
        status=CleanupObligationStatus(row.status),
        attempts=int(row.attempts),
        last_error=row.last_error,
        next_attempt_at=row.next_attempt_at,
        created_at=row.created_at,
    )


def list_cleanup_obligations(
    db: Session,
    *,
    statuses: Sequence[CleanupObligationStatus] | None = None,
    limit: int | None = None,
) -> list[RecordedCleanupObligation]:
    """Obligations still in the table, oldest first.

    With ``statuses=TERMINAL_STATUSES`` this is the operator's reconciliation
    list.
    """

    query = select(TaskCleanupObligation).order_by(TaskCleanupObligation.id)
    if statuses is not None:
        query = query.where(
            TaskCleanupObligation.status.in_([status.value for status in statuses])
        )
    if limit is not None:
        query = query.limit(limit)
    return [_as_recorded(row) for row in db.execute(query).scalars()]


def count_cleanup_obligations(db: Session) -> dict[CleanupObligationStatus, int]:
    """How many obligations stand in each status; every status is present."""

    counts = {status: 0 for status in CleanupObligationStatus}
    for status, count in db.execute(
        select(TaskCleanupObligation.status, func.count()).group_by(
            TaskCleanupObligation.status
        )
    ):
        counts[CleanupObligationStatus(status)] = int(count)
    return counts


@dataclass(frozen=True)
class CleanupRetryReport:
    """One driver batch's outcome, in the shape its log line prints."""

    #: How many due candidates the batch fetched, before any claim was
    #: attempted. This is what "is there a backlog" has to be measured
    #: against -- see :func:`run_cleanup_obligation_loop`. ``claimed`` is not
    #: a substitute: several replicas can split one due page between them, so
    #: each replica's ``claimed`` can stay well under ``batch_size`` even
    #: while the page it read was full and another page is waiting behind it.
    due: int = 0
    claimed: int = 0
    completed: int = 0
    retrying: int = 0
    exhausted: int = 0
    abandoned: int = 0

    def with_status(self, status: CleanupObligationStatus | None) -> CleanupRetryReport:
        """This report plus one claimed obligation's outcome (``None``: done)."""
        return replace(
            self,
            claimed=self.claimed + 1,
            completed=self.completed + int(status is None),
            retrying=self.retrying + int(status is CleanupObligationStatus.PENDING),
            exhausted=self.exhausted + int(status is CleanupObligationStatus.EXHAUSTED),
            abandoned=self.abandoned + int(status is CleanupObligationStatus.ABANDONED),
        )

    def log_line(self) -> str:
        return (
            "task cleanup retry: "
            f"due={self.due} claimed={self.claimed} completed={self.completed} "
            f"retrying={self.retrying} exhausted={self.exhausted} "
            f"abandoned={self.abandoned}"
        )


def _due_candidates_sync(
    session_factory: Callable[[], Session], *, now: datetime, limit: int
) -> list[RecordedCleanupObligation]:
    with session_factory() as db:
        return [
            _as_recorded(row)
            for row in db.execute(
                select(TaskCleanupObligation)
                .where(
                    TaskCleanupObligation.status
                    == CleanupObligationStatus.PENDING.value,
                    TaskCleanupObligation.next_attempt_at <= now,
                )
                .order_by(
                    TaskCleanupObligation.next_attempt_at, TaskCleanupObligation.id
                )
                .limit(limit)
            ).scalars()
        ]


def _claim_sync(
    session_factory: Callable[[], Session],
    candidate: RecordedCleanupObligation,
    *,
    now: datetime,
    max_attempts: int,
) -> RecordedCleanupObligation | CleanupObligationStatus | None:
    """Take one due obligation for this driver, refuse it, or report why not.

    Three outcomes:

    * The candidate's budget is already spent (``attempts >= max_attempts``):
      it is moved straight to EXHAUSTED, under the same fence, and never
      handed to a release. This has to happen here rather than only at the
      outcome (:func:`_outcome_values`), because that check only runs once an
      attempt *finishes*. A release that hangs forever, or a process that
      dies on every attempt before it can write an outcome, would otherwise
      have this row reclaimed every :data:`CLAIM_LEASE` forever -- attempts
      keeps incrementing on each claim, but nothing ever reads it against the
      budget, so it never stops. Checking here closes that gap: a lease that
      lapsed one too many times ends the row rather than reclaiming it again.
    * A normal claim: ``attempts`` increments and the due time is pushed out
      by :data:`CLAIM_LEASE`, conditional on the ``attempts`` value the
      candidate was read with -- the same compare-and-set shape as the upload
      GC's claim. So two drivers (two web replicas) cannot both claim one
      row, and a driver that died holding a claim loses it once the lease
      expires. The attempt is counted at the claim rather than at the
      outcome, so a process that dies every time it tries still spends its
      budget.
    * ``None``: the compare-and-set missed -- someone else (another replica,
      or this same check from a previous call) already moved the row past the
      ``attempts``/status/due snapshot this candidate was read with.
    """

    with session_factory() as db:
        if candidate.attempts >= max_attempts:
            error_message = (
                f"claimed {candidate.attempts} time(s) without a recorded "
                "outcome; the release may hang or crash the process"
            )
            result = db.execute(
                update(TaskCleanupObligation)
                .where(
                    TaskCleanupObligation.id == candidate.id,
                    TaskCleanupObligation.status
                    == CleanupObligationStatus.PENDING.value,
                    TaskCleanupObligation.attempts == candidate.attempts,
                    TaskCleanupObligation.next_attempt_at <= now,
                )
                .values(
                    status=CleanupObligationStatus.EXHAUSTED.value,
                    last_error=_truncate(error_message),
                    updated_at=now,
                )
            )
            db.commit()
            if getattr(result, "rowcount", 0) != 1:
                return None
            logger.error(
                "Task cleanup obligation %s (task %s, %s) %s",
                candidate.id,
                candidate.task_id,
                candidate.kind.value,
                error_message,
            )
            return CleanupObligationStatus.EXHAUSTED
        result = db.execute(
            update(TaskCleanupObligation)
            .where(
                TaskCleanupObligation.id == candidate.id,
                TaskCleanupObligation.status == CleanupObligationStatus.PENDING.value,
                TaskCleanupObligation.attempts == candidate.attempts,
                TaskCleanupObligation.next_attempt_at <= now,
            )
            .values(
                attempts=candidate.attempts + 1,
                next_attempt_at=now + CLAIM_LEASE,
                updated_at=now,
            )
        )
        db.commit()
    if getattr(result, "rowcount", 0) != 1:
        return None
    return replace(candidate, attempts=candidate.attempts + 1)


def _task_id_is_live_sync(session_factory: Callable[[], Session], task_id: int) -> bool:
    with session_factory() as db:
        return db.execute(select(Task.id).where(Task.id == task_id)).first() is not None


async def _release(
    obligation: RecordedCleanupObligation,
    session_factory: Callable[[], Session],
) -> None:
    """Release one obligation's resource; raise if it was not released."""

    if obligation.kind is CleanupResourceKind.WORKSPACE:
        # ``rmtree`` of a possibly large tree: off the event loop.
        await asyncio.to_thread(remove_task_workspace, _workspace_target(obligation))
        return
    if obligation.kind is CleanupResourceKind.RUNTIME_EXTENSION:
        context = TaskRuntimeContext(
            task_id=obligation.task_id,
            user_id=int(obligation.locator["user_id"]),
            source=obligation.locator.get("source"),
            session_factory=session_factory,
        )
        # ``force=False``: a provider failure raises, which is the retry
        # signal. An unregistered provider does not raise -- the registry
        # returns it as unreleased instead -- and is kept owed rather than
        # dropped, so it is still on the list when the provider comes back.
        unreleased = await delete_task_extensions(
            context, bound_extensions=(obligation.key,), force=False
        )
        if unreleased:
            raise CleanupProviderNotRegistered(
                f"runtime extension {obligation.key!r} is not registered"
            )
        return
    raise ValueError(f"unknown cleanup resource kind {obligation.kind!r}")


def _settle_claim_sync(
    session_factory: Callable[[], Session],
    claimed: RecordedCleanupObligation,
    values: dict[str, Any] | None,
) -> CleanupObligationStatus | None:
    with session_factory() as db:
        status = _write_outcome(db, claimed, values)
        db.commit()
        return status


async def _warn_if_task_revived_during_release(
    session_factory: Callable[[], Session],
    claimed: RecordedCleanupObligation,
) -> None:
    """Log, but do not act, if ``claimed.task_id`` went live while its release
    was in flight.

    ``_attempt_claimed`` already checks liveness once, before the release
    starts, and that check is a real fence: finding a live task there
    aborts the release outright. This second check cannot be a fence the same
    way -- the release (an ``rmtree``, a provider round trip) has already run
    by the time this is called, so there is nothing left to abort. What it
    can do is tell an operator that the window was not empty: a task was
    created with this id while this obligation's release was still running,
    so that release may have touched resources that, by the time it finished,
    belonged to the new task rather than to the one this obligation was
    recorded for. The recorded outcome is unaffected either way -- this is
    purely an observability backstop for a window this module does not hold
    a lock across.
    """

    if await asyncio.to_thread(_task_id_is_live_sync, session_factory, claimed.task_id):
        logger.error(
            "Task cleanup obligation %s (task %s, %s) was released while its "
            "task id became live again; the release may have touched the new "
            "task's resources",
            claimed.id,
            claimed.task_id,
            claimed.kind.value,
        )


async def _attempt_claimed(
    session_factory: Callable[[], Session],
    claimed: RecordedCleanupObligation,
    *,
    now: datetime,
    max_attempts: int,
) -> CleanupObligationStatus | None:
    """Release one claimed obligation and write its outcome."""

    if await asyncio.to_thread(_task_id_is_live_sync, session_factory, claimed.task_id):
        # A live task holds this id again -- SQLite reuses the highest rowid
        # once it is deleted. Whatever sits at this locator now belongs to that
        # task, so releasing it would destroy live data. (A task created in
        # the window between this check and the release returning is not
        # fenced by it -- that window is the whole release, not "one query":
        # see the re-check after the release below, which is what actually
        # covers it, as a log rather than a fence.)
        values: dict[str, Any] | None = {
            "status": CleanupObligationStatus.ABANDONED.value,
            "last_error": (
                f"task id {claimed.task_id} belongs to a live task again; "
                "releasing this obligation would destroy that task's resources"
            ),
            "updated_at": now,
        }
    else:
        error: str | None = None
        try:
            await _release(claimed, session_factory)
        except asyncio.CancelledError:
            raise
        except CleanupProviderNotRegistered as exc:
            # Not this obligation's failure: nothing was actually attempted
            # against the resource, so it must not spend from the attempt
            # budget or it could go EXHAUSTED purely because this process
            # never had the provider loaded -- contradicting the documented
            # design that such a binding stays owed until a process that can
            # release it claims it. The claim already incremented ``attempts``
            # (see :func:`_claim_sync`); this refunds exactly that increment
            # rather than leaving the fence's ``attempts`` column ahead of
            # what was genuinely attempted.
            #
            # The refund is only safe because ids are unique for the whole
            # life of this table (see the ``sqlite_autoincrement`` guard on
            # the model/migration): the fence below still checks
            # ``claimed.id`` and ``claimed.attempts`` (the post-claim value),
            # so only an actor that itself claimed this exact obligation --
            # not some unrelated obligation that later reused the id -- can
            # hold that fence value and win the write.
            values = {
                "attempts": claimed.attempts - 1,
                "next_attempt_at": now + PROVIDER_RECHECK_INTERVAL,
                "last_error": _truncate(describe_cleanup_failure(exc)),
                "updated_at": now,
            }
            await _warn_if_task_revived_during_release(session_factory, claimed)
            return await asyncio.to_thread(
                _settle_claim_sync, session_factory, claimed, values
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Task cleanup obligation %s (task %s, %s) failed attempt %d",
                claimed.id,
                claimed.task_id,
                claimed.kind.value,
                claimed.attempts,
                exc_info=True,
            )
            error = describe_cleanup_failure(exc)
        await _warn_if_task_revived_during_release(session_factory, claimed)
        values = _outcome_values(
            locator=claimed.locator,
            attempts=claimed.attempts,
            error=error,
            now=now,
            max_attempts=max_attempts,
        )
    return await asyncio.to_thread(_settle_claim_sync, session_factory, claimed, values)


async def run_cleanup_obligation_batch(
    session_factory: Callable[[], Session],
    *,
    now: datetime | None = None,
    limit: int = DEFAULT_RETRY_BATCH_SIZE,
    max_attempts: int | None = None,
) -> CleanupRetryReport:
    """Try each due obligation once.

    Obligations are claimed one at a time, as each one's turn comes, and at
    the time of that turn -- so a slow release cannot run a later obligation's
    lease out before it is even started, and backoff is measured from the
    attempt that failed. One obligation whose outcome cannot be written does
    not cost the rest of the batch: its claim lapses and it is retried.

    ``now`` pins the clock for tests; production reads it per obligation.
    """

    max_attempts = max_attempts or get_task_cleanup_max_attempts()
    candidates = await asyncio.to_thread(
        _due_candidates_sync, session_factory, now=now or _utcnow(), limit=limit
    )
    report = CleanupRetryReport(due=len(candidates))
    for candidate in candidates:
        try:
            claimed = await asyncio.to_thread(
                _claim_sync,
                session_factory,
                candidate,
                now=now or _utcnow(),
                max_attempts=max_attempts,
            )
            if claimed is None:
                continue
            if isinstance(claimed, CleanupObligationStatus):
                # The candidate's budget was already spent, so the claim step
                # moved it to EXHAUSTED instead of claiming it: nothing was
                # claimed and there is no release to run.
                report = replace(report, exhausted=report.exhausted + 1)
                continue
            status = await _attempt_claimed(
                session_factory,
                claimed,
                now=now or _utcnow(),
                max_attempts=max_attempts,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning(
                "Task cleanup obligation %s could not be attempted; its claim "
                "will lapse and it will be retried",
                candidate.id,
                exc_info=True,
            )
            continue
        report = report.with_status(status)
    if report.claimed:
        logger.info(report.log_line())
    return report


async def run_cleanup_obligation_loop(
    session_factory: Callable[[], Session],
    *,
    poll_interval_seconds: float,
    batch_size: int = DEFAULT_RETRY_BATCH_SIZE,
    backlog_pause_seconds: float = 1.0,
) -> None:
    """Retry due obligations until cancelled.

    Runs in every web process, in every deployment: obligations are recorded
    by on-demand deletion on any store, not only by the retention purge, so
    the driver cannot be tied to the purge's opt-in or to its dialect. Several
    replicas running it at once is safe -- the claim is a compare-and-set.

    A full due page means more may be due right now, so the next batch follows
    after a short pause; otherwise the loop waits the poll interval. This is
    measured against ``report.due`` -- how many candidates the batch fetched
    -- rather than ``report.claimed``: several replicas can split one due page
    between them, so one replica's ``claimed`` can stay small even when the
    page it read was full and another page is waiting right behind it. The
    ``claimed > 0`` guard exists so a due page nothing could actually claim
    (a claim-step outage, say) does not spin the loop at
    ``backlog_pause_seconds`` forever with every attempt failing. A failed
    batch is logged and retried after the interval: an unattended driver that
    stopped on its first error would look exactly like one with nothing to
    do.

    Cancellation is the stop signal. Each database step and each release runs
    in a worker thread that cancelling detaches rather than ends, so a release
    in flight at shutdown may still finish; its claim then lapses after
    :data:`CLAIM_LEASE` and the next process retries it, which idempotence
    makes harmless.
    """

    while True:
        backlog = False
        try:
            report = await run_cleanup_obligation_batch(
                session_factory, limit=batch_size
            )
            backlog = report.due >= batch_size and report.claimed > 0
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("Task cleanup retry batch failed")
        await asyncio.sleep(backlog_pause_seconds if backlog else poll_interval_seconds)


__all__ = [
    "CLAIM_LEASE",
    "CleanupObligation",
    "CleanupObligationStatus",
    "CleanupProviderNotRegistered",
    "CleanupResourceKind",
    "CleanupRetryReport",
    "PROVIDER_RECHECK_INTERVAL",
    "RecordedCleanupObligation",
    "SCOPE_UNRESOLVED_REASON",
    "TERMINAL_STATUSES",
    "captured_workspace_obligation",
    "count_cleanup_obligations",
    "describe_cleanup_failure",
    "extension_obligation",
    "list_cleanup_obligations",
    "record_cleanup_obligations_no_commit",
    "run_cleanup_obligation_batch",
    "run_cleanup_obligation_loop",
    "settle_cleanup_attempt_no_commit",
    "settle_cleanup_attempts_sync",
    "workspace_obligation",
]
