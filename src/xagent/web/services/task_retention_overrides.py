"""Per-team retention periods supplied by the deployment layer (#2600).

The retention purge runs on two deployment-wide periods. A deployment that
lets a team shorten or extend them (#2567 decision 2: shortening on every
plan, extension up to :data:`RETENTION_OVERRIDE_MAX_DAYS`) registers a
resolver here with :func:`set_retention_override_resolver`. With nothing
registered -- always, in this repository -- the purge reads exactly the
configured periods and runs exactly the SQL it ran before this module existed.

Keyed by ``user_id``
--------------------
This repository has no teams. The resolver maps each user whose team
overrides the defaults to that override, and the deployment layer decides what
"the task's owning team" means. In the SaaS deployment that is the team of the
task's ``user_id``, one team per user; a user who moves team takes their
existing tasks to the new team's period, which is how usage is attributed
there too.

Where the periods are applied
-----------------------------
At both places the purge reads a period, because the two answer different
questions:

* The **locked assessment** decides what is deleted. It must use the task's
  own effective periods; this is the only guard against deleting a task whose
  team extended its period. :func:`resolve_task_retention_periods` re-reads
  the override inside the task's own transaction, so a settings change
  committed before that transaction applies even if the scan saw the old
  value.
* The **batch scan** decides what is looked at. A scan on the global periods
  alone never admits a task that is due only under a shorter team period, and
  re-admits every not-yet-due task of an extending team on every sweep.
  :meth:`RetentionOverrideSnapshot.scan_condition` partitions the scan by the
  users each distinct period pair applies to.

The two reads can disagree when a setting changes between them. That costs a
skipped task (``skipped_not_due`` or ``skipped_override_unresolved``), a task
picked up one sweep late, or a conversation expiry downgraded to trace expiry
by an extension -- never a deletion the setting read in the task's own
transaction does not allow.

Failing closed
--------------
Every failure keeps data rather than deleting it. A resolver that raises, or
returns a result that is not a mapping or carries a user id that is not an
``int``, resolves to ``None``: the scan selects nothing and the assessment
skips the task. A single user's unusable entry -- a value that is not a
:class:`RetentionOverride`, or a period that is ``None``, zero, negative, over
the cap or not an ``int`` -- refuses only that user's tasks. None
of these fall back to the global period, because for a team that extended its
period the global period is the premature-deletion direction, the one that
cannot be undone. An over-cap value is refused rather than clamped for the
same reason: clamping deletes earlier than the team configured.

``None`` already means *unlimited retention* throughout the purge
(:func:`~xagent.web.services.task_retention.retention_cutoff`). An override
that leaves a period alone says so with :data:`INHERIT`, never with ``None``,
so a resolver cannot lift a team past the cap by returning "unlimited". An
inherited trace period follows the team's conversation period when the
deployment's trace period follows its conversation period (see :func:`_apply`).

An override also never turns on a leg the deployment left off: a conversation
override on a deployment that only expires traces is ignored. Saving a team
setting must not start deleting data a deployment has not agreed to delete. An
unusable value refuses the user even on a leg that is off, so it also stops
that user's other leg: validity is judged on what the resolver returned, not
on what happens to take effect.

Read cadence
------------
The global periods are read once per loop, because they cannot change within
a process. Overrides can, so they are an explicit exception: read once per
batch for the scan and once per task for the assessment. The resolver runs
inside the purge's transaction -- for the assessment, the task's own, just
before its row is locked; the lock does not fence the settings anyway -- so it
must be a read with no side effects, including under dry run. A resolver that
fails a statement aborts that transaction on PostgreSQL; the purge then only
rolls it back, which is why a failure has to end the task's turn rather than
fall through to the assessment.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from sqlalchemy import ColumnElement, and_, or_, select
from sqlalchemy.orm import Session

from ..models.task import Task

logger = logging.getLogger(__name__)

#: The longest period a team override may set (#2567 decision 2: two years).
#: Enforced here, in the caller, not trusted from the deployment layer.
RETENTION_OVERRIDE_MAX_DAYS = 730

#: How many refused user ids one warning names, so an operator can find the
#: team without one bad import flooding the log.
_REFUSED_IDS_LOGGED = 20


class _Inherit(enum.Enum):
    INHERIT = "inherit"


#: "Use the deployment's period for this leg." Distinct from ``None``, which
#: means unlimited retention and is not a value an override may set.
INHERIT = _Inherit.INHERIT


@dataclass(frozen=True)
class RetentionPeriods:
    """The ``(conversation_days, trace_days)`` pair the purge acts on."""

    conversation_days: int | None
    trace_days: int | None


@dataclass(frozen=True)
class RetentionOverride:
    """One team's override, as the deployment layer supplies it."""

    conversation_days: int | _Inherit = INHERIT
    trace_days: int | _Inherit = INHERIT


#: Returns the override for every user whose team has one. Users absent from
#: the mapping get the deployment's periods.
RetentionOverrideResolver = Callable[[Session], Mapping[int, RetentionOverride]]

_resolver: RetentionOverrideResolver | None = None


def set_retention_override_resolver(
    resolver: RetentionOverrideResolver | None,
) -> None:
    """Install or clear the process-wide resolver owned by the deployment."""
    if resolver is not None and not callable(resolver):
        raise TypeError("retention override resolver must be callable")
    global _resolver
    _resolver = resolver


@dataclass(frozen=True)
class RetentionOverrideSnapshot:
    """Effective periods for one read of the resolver."""

    defaults: RetentionPeriods
    #: Users whose team overrides the defaults, with the periods that result.
    periods_by_user: Mapping[int, RetentionPeriods] = field(default_factory=dict)
    #: Users whose override was unusable. Their tasks are never purged.
    refused_users: frozenset[int] = frozenset()

    def periods_for(self, user_id: int) -> RetentionPeriods | None:
        """This user's effective periods, or ``None`` when refused."""
        if user_id in self.refused_users:
            return None
        return self.periods_by_user.get(user_id, self.defaults)

    def scan_condition(
        self,
        user_column: ColumnElement[int],
        due: Callable[[RetentionPeriods], ColumnElement[bool]],
    ) -> ColumnElement[bool]:
        """``due`` applied to each user under their own effective periods.

        With no override this is ``due(defaults)`` itself, so an unregistered
        deployment's scan is unchanged down to the SQL text.
        """
        if not self.periods_by_user and not self.refused_users:
            return due(self.defaults)
        by_periods: dict[RetentionPeriods, list[int]] = {}
        for user_id, periods in self.periods_by_user.items():
            by_periods.setdefault(periods, []).append(user_id)
        overridden = sorted({*self.periods_by_user, *self.refused_users})
        # Refused users are in ``overridden`` and in no partition, so the scan
        # never selects them.
        return or_(
            and_(user_column.not_in(overridden), due(self.defaults)),
            *(
                and_(user_column.in_(sorted(users)), due(periods))
                for periods, users in by_periods.items()
            ),
        )


def _usable_days(value: object) -> bool:
    # ``bool`` is an ``int``; ``True`` is not a period.
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= RETENTION_OVERRIDE_MAX_DAYS
    )


def _apply(
    defaults: RetentionPeriods, override: RetentionOverride
) -> RetentionPeriods | None:
    """The periods ``override`` yields, or ``None`` when it is unusable.

    An inherited trace period follows the team's conversation period when the
    deployment's trace period equals its conversation period. That is what an
    unset ``XAGENT_TRACE_RETENTION_DAYS`` resolves to -- traces expire with
    their conversation -- and :func:`~xagent.config.get_trace_retention_days`
    has already collapsed it into a number by the time it reaches here, so
    equality is the only trace of it left. Without this, a team extending its
    conversation to 730 days would still lose its traces at 365. An operator
    who set the two equal explicitly gets the same reading, which can only
    keep traces longer, never shorter: a team that *shortens* its
    conversation shortens the trace with it, but conversation expiry removes
    the whole task, traces included, at that same moment, so no trace goes
    earlier than it would have anyway.
    """
    legs: list[int | None] = []
    for default, value in (
        (defaults.conversation_days, override.conversation_days),
        (defaults.trace_days, override.trace_days),
    ):
        if value is INHERIT:
            legs.append(default)
        elif not _usable_days(value):
            return None
        else:
            # A leg the deployment left off stays off.
            legs.append(None if default is None else int(value))
    conversation, trace = legs
    if (
        override.trace_days is INHERIT
        and defaults.trace_days is not None
        and defaults.trace_days == defaults.conversation_days
    ):
        trace = conversation
    return RetentionPeriods(conversation_days=conversation, trace_days=trace)


def load_retention_overrides(
    db: Session, defaults: RetentionPeriods, *, per_task: bool = False
) -> RetentionOverrideSnapshot | None:
    """Read the resolver once, or ``None`` when it failed (fail closed).

    ``per_task`` marks the locked assessment's own read (via
    :func:`resolve_task_retention_periods`), as opposed to the batch scan's.
    A per-task read skips the refused-users summary, and logs a resolver
    failure at DEBUG instead of WARNING (still with ``exc_info=True``) --
    otherwise one bad team, or one resolver outage, would log once per task
    in the sweep instead of once per batch. Neither silences the failure: a
    per-task failure is counted as ``skipped_override_unresolved``, and the
    batch logs one WARNING whenever that count is non-zero -- which matters
    for a resolver that fails only on the per-task reads, since its scan read
    then never warns.
    """
    resolver = _resolver
    if resolver is None:
        return RetentionOverrideSnapshot(defaults=defaults)
    try:
        result = resolver(db)
        if not isinstance(result, Mapping):
            raise TypeError(f"resolver returned {type(result).__name__}, not a mapping")
        periods_by_user: dict[int, RetentionPeriods] = {}
        refused: set[int] = set()
        for user_id, override in result.items():
            if not isinstance(user_id, int) or isinstance(user_id, bool):
                raise TypeError(f"resolver returned a non-int user id {user_id!r}")
            periods = (
                _apply(defaults, override)
                if isinstance(override, RetentionOverride)
                else None
            )
            if periods is None:
                refused.add(user_id)
            elif periods != defaults:
                periods_by_user[user_id] = periods
    except Exception:  # noqa: BLE001
        logger.log(
            logging.DEBUG if per_task else logging.WARNING,
            "retention override resolver failed; purging nothing it governs",
            exc_info=True,
        )
        return None
    if refused and not per_task:
        logger.warning(
            "retention overrides refused for %d user(s): unusable period "
            "(must be 1..%d days or inherit); their tasks are kept; "
            "user ids (first %d): %s",
            len(refused),
            RETENTION_OVERRIDE_MAX_DAYS,
            _REFUSED_IDS_LOGGED,
            sorted(refused)[:_REFUSED_IDS_LOGGED],
        )
    return RetentionOverrideSnapshot(
        defaults=defaults,
        periods_by_user=periods_by_user,
        refused_users=frozenset(refused),
    )


def resolve_task_retention_periods(
    db: Session, task_id: int, defaults: RetentionPeriods
) -> RetentionPeriods | None:
    """One task's effective periods, or ``None`` when it must be kept.

    Unregistered, this returns ``defaults`` without touching the database.
    A task that no longer exists also gets ``defaults``: the assessment that
    follows reports it missing.

    Reads the resolver with ``per_task=True``: a failure here is counted as
    ``skipped_override_unresolved``, and the batch logs one WARNING for that
    count, so logging each one at WARNING here would only repeat it once per
    task. The traceback is kept at DEBUG.
    """
    if _resolver is None:
        return defaults
    snapshot = load_retention_overrides(db, defaults, per_task=True)
    if snapshot is None:
        return None
    user_id = db.execute(
        select(Task.user_id).where(Task.id == task_id)
    ).scalar_one_or_none()
    if user_id is None:
        return defaults
    return snapshot.periods_for(int(user_id))


__all__ = [
    "INHERIT",
    "RETENTION_OVERRIDE_MAX_DAYS",
    "RetentionOverride",
    "RetentionOverrideResolver",
    "RetentionOverrideSnapshot",
    "RetentionPeriods",
    "load_retention_overrides",
    "resolve_task_retention_periods",
    "set_retention_override_resolver",
]
