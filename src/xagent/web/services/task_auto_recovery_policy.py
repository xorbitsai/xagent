"""Pure auto-resume policy: how often, how long and whether to retry a task.

Nothing here touches the database except :func:`scheduled_trigger_superseded`
(read-only) and nothing changes behavior until an executor calls it. The
executor calls :func:`plan_auto_resume` twice: once after recording an
interruption (to decide ``scheduled`` / ``exhausted`` / ``expired`` and the
first ``next_attempt_at``) and again at dispatch time, because the clock, the
counters and the env limits may have moved while the row waited. Config is
read at call time so an env change applies without a restart.

Datetimes are normalized to aware UTC on entry: a naive value is taken as UTC
(SQLite returns naive columns even for ``DateTime(timezone=True)``) and an
aware one is converted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

from sqlalchemy.orm import Session

from ...config import (
    get_task_auto_resume_channel_window_seconds,
    get_task_auto_resume_llm_initial_backoff_seconds,
    get_task_auto_resume_llm_max_backoff_seconds,
    get_task_auto_resume_llm_max_elapsed_seconds,
    get_task_auto_resume_max_total_per_run,
    get_task_auto_resume_model_output_max_no_progress,
    get_task_auto_resume_short_initial_backoff_seconds,
    get_task_auto_resume_short_max_no_progress,
    get_task_auto_resume_window_seconds,
)
from ...core.agent.interruption import InterruptionReason
from ..models.task import Task
from ..models.task_auto_recovery import TaskAutoRecoveryState
from ..models.trigger import (
    TEST_TRIGGER_RUN_KEY_PREFIX,
    TriggerRun,
    TriggerRunStatus,
    TriggerType,
)

# Floor of the backoff ceiling for the short-backoff reasons (lease expiry,
# persistence); a larger configured initial backoff raises the ceiling.
_SHORT_MAX_BACKOFF_SECONDS = 60
_SHORT_JITTER_CAP_SECONDS = 30.0
_LLM_JITTER_CAP_SECONDS = 30.0
_MODEL_OUTPUT_JITTER_CAP_SECONDS = 5.0

# ``2 ** attempt`` is bounded past this point: the cap always wins long before.
_MAX_BACKOFF_EXPONENT = 62


@dataclass(frozen=True)
class AutoResumePolicy:
    """Retry limits for one interruption reason."""

    initial_backoff_seconds: float
    # Backoff doubles per no-progress resume up to this ceiling.
    max_backoff_seconds: float
    max_no_progress: int | None
    # Measured from the episode start, not from the latest interruption.
    max_elapsed_seconds: float | None
    jitter_cap_seconds: float


def auto_resume_policy(reason: InterruptionReason | str) -> AutoResumePolicy | None:
    """The policy for ``reason``; ``None`` means it is never auto-resumed.

    Accepts the plain string a ``task_auto_recovery.reason`` column holds; an
    unknown string has no policy.
    """

    try:
        reason = InterruptionReason(reason)
    except ValueError:
        return None
    if reason in (
        InterruptionReason.LEASE_EXPIRED,
        InterruptionReason.PERSISTENCE_FAILURE,
    ):
        initial = get_task_auto_resume_short_initial_backoff_seconds()
        return AutoResumePolicy(
            initial_backoff_seconds=initial,
            max_backoff_seconds=max(_SHORT_MAX_BACKOFF_SECONDS, initial),
            max_no_progress=get_task_auto_resume_short_max_no_progress(),
            max_elapsed_seconds=None,
            jitter_cap_seconds=_SHORT_JITTER_CAP_SECONDS,
        )
    if reason == InterruptionReason.LLM_UNAVAILABLE:
        return AutoResumePolicy(
            initial_backoff_seconds=get_task_auto_resume_llm_initial_backoff_seconds(),
            max_backoff_seconds=get_task_auto_resume_llm_max_backoff_seconds(),
            max_no_progress=None,
            max_elapsed_seconds=get_task_auto_resume_llm_max_elapsed_seconds(),
            jitter_cap_seconds=_LLM_JITTER_CAP_SECONDS,
        )
    if reason == InterruptionReason.MODEL_OUTPUT_INVALID:
        # Retried immediately (a fresh sample usually fixes it), so a short
        # run-level limit stands in for backoff.
        return AutoResumePolicy(
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            max_no_progress=get_task_auto_resume_model_output_max_no_progress(),
            max_elapsed_seconds=None,
            jitter_cap_seconds=_MODEL_OUTPUT_JITTER_CAP_SECONDS,
        )
    # ``shutdown`` is reserved for a future graceful-shutdown hand-off and is
    # not auto-resumed yet. user_pause, input_outcome_unknown,
    # unknown_tool_effect and not_recoverable need a human by definition.
    return None


def backoff_seconds(policy: AutoResumePolicy, attempt: int) -> float:
    """Doubling backoff, capped; ``attempt`` counts no-progress resumes so far."""

    exponent = min(max(attempt, 0), _MAX_BACKOFF_EXPONENT)
    return min(
        policy.initial_backoff_seconds * 2.0**exponent, policy.max_backoff_seconds
    )


def jitter_seconds(
    policy: AutoResumePolicy, backoff: float, rng: random.Random
) -> float:
    """Random spread so a burst of interruptions does not retry in lockstep.

    A zero-backoff policy is retried "immediately" and spreads over the whole
    jitter cap; otherwise jitter is at most half the (capped) backoff.
    """

    if policy.initial_backoff_seconds == 0:
        return rng.uniform(0, policy.jitter_cap_seconds)
    return rng.uniform(0, min(backoff, policy.jitter_cap_seconds) * 0.5)


def staleness_window_seconds(kind: str | None) -> int:
    """How long after an interruption a task of ``kind`` may still be resumed."""

    if kind == "channel":
        return get_task_auto_resume_channel_window_seconds()
    return get_task_auto_resume_window_seconds()


class AutoResumeOutcome(str, Enum):
    """What the policy decided; values match :class:`TaskAutoRecoveryState`."""

    SCHEDULED = "scheduled"
    EXHAUSTED = "exhausted"
    EXPIRED = "expired"
    # Never auto-resumed; the row rests in ``manual``.
    NEVER = "never"


def auto_resume_recovery_state(outcome: AutoResumeOutcome) -> TaskAutoRecoveryState:
    """The ``task_auto_recovery.state`` a policy outcome is recorded as."""

    if outcome is AutoResumeOutcome.NEVER:
        return TaskAutoRecoveryState.MANUAL
    return TaskAutoRecoveryState(outcome.value)


@dataclass(frozen=True)
class AutoResumeVerdict:
    """Outcome of :func:`plan_auto_resume`, plus the time when it is SCHEDULED."""

    outcome: AutoResumeOutcome
    # Set only for SCHEDULED.
    next_attempt_at: datetime | None = None


def _utc(value: datetime) -> datetime:
    """Aware UTC; a naive value is read as UTC."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def auto_resume_limit(
    *,
    reason: InterruptionReason | str,
    kind: str | None,
    interrupted_at: datetime,
    episode_started_at: datetime,
    no_progress_resumes: int,
    total_resumes: int,
    now: datetime,
) -> AutoResumeOutcome | None:
    """The limit that stops an auto-resume, or ``None`` if it may proceed.

    Checks apply in order:

    1. no policy for ``reason`` -> NEVER;
    2. interrupted longer ago than the staleness window -> EXPIRED;
    3. total resume cap reached -> EXHAUSTED;
    4. no-progress limit reached -> EXHAUSTED;
    5. episode elapsed limit reached -> EXHAUSTED.

    This is the whole dispatch-time check; it computes no schedule. It trusts
    the caller's counters: ``no_progress_resumes`` counts dispatched resumes in
    the current no-progress episode, ``total_resumes`` counts this run's
    dispatches and ``episode_started_at`` is that episode's start. ``kind``
    must come from ``auto_recovery_eligibility`` evaluated at call time.
    """

    policy = auto_resume_policy(reason)
    if policy is None:
        return AutoResumeOutcome.NEVER
    now = _utc(now)
    if now - _utc(interrupted_at) > timedelta(seconds=staleness_window_seconds(kind)):
        return AutoResumeOutcome.EXPIRED
    if total_resumes >= get_task_auto_resume_max_total_per_run():
        return AutoResumeOutcome.EXHAUSTED
    if (
        policy.max_no_progress is not None
        and no_progress_resumes >= policy.max_no_progress
    ):
        return AutoResumeOutcome.EXHAUSTED
    if policy.max_elapsed_seconds is not None and now - _utc(
        episode_started_at
    ) >= timedelta(seconds=policy.max_elapsed_seconds):
        return AutoResumeOutcome.EXHAUSTED
    return None


def next_auto_resume_at(
    policy_or_reason: AutoResumePolicy | InterruptionReason | str,
    no_progress_resumes: int,
    now: datetime,
    rng: random.Random,
) -> datetime:
    """``now + backoff + jitter`` for the next attempt.

    The result may land past the staleness window or the elapsed limit; it is
    still scheduled and the dispatch-time :func:`auto_resume_limit` settles it.
    """

    policy = (
        policy_or_reason
        if isinstance(policy_or_reason, AutoResumePolicy)
        else auto_resume_policy(policy_or_reason)
    )
    if policy is None:
        raise ValueError(f"reason {policy_or_reason!r} is never auto-resumed")
    backoff = backoff_seconds(policy, no_progress_resumes)
    delay = backoff + jitter_seconds(policy, backoff, rng)
    return _utc(now) + timedelta(seconds=delay)


def plan_auto_resume(
    *,
    reason: InterruptionReason | str,
    kind: str | None,
    interrupted_at: datetime,
    episode_started_at: datetime,
    no_progress_resumes: int,
    total_resumes: int,
    now: datetime,
    rng: random.Random,
) -> AutoResumeVerdict:
    """Record-time decision: :func:`auto_resume_limit`, else a schedule.

    Record time calls this (it needs the next attempt time); dispatch time
    calls only :func:`auto_resume_limit`, so it does not draw an unused
    jitter.
    """

    limit = auto_resume_limit(
        reason=reason,
        kind=kind,
        interrupted_at=interrupted_at,
        episode_started_at=episode_started_at,
        no_progress_resumes=no_progress_resumes,
        total_resumes=total_resumes,
        now=now,
    )
    if limit is not None:
        return AutoResumeVerdict(limit)
    return AutoResumeVerdict(
        AutoResumeOutcome.SCHEDULED,
        next_attempt_at=next_auto_resume_at(reason, no_progress_resumes, now, rng),
    )


# A later run supersedes only if it actually ran: a failed one produced
# nothing newer to prefer over the paused tick.
_SUPERSEDING_RUN_STATUSES = (
    TriggerRunStatus.RUNNING.value,
    TriggerRunStatus.COMPLETED.value,
)


def _is_test_run(run: TriggerRun, agent_config: dict[object, object]) -> bool:
    key = str(run.idempotency_key or "")
    return key.startswith(TEST_TRIGGER_RUN_KEY_PREFIX) or (
        agent_config.get("trigger_test") is True
    )


def scheduled_trigger_superseded(db: Session, task: Task) -> bool:
    """Whether a newer run of the task's scheduled trigger has already started.

    A scheduled run is the latest tick of a recurring job; resuming an old
    tick once the next one is running would duplicate its work. Webhook and
    gmail runs each carry a distinct event, so they are never superseded.
    A later run supersedes once it has started and has not failed, even if
    its own task is now paused by its user or waiting for an answer: only the
    latest tick matters. Manual test fires and failed runs neither supersede
    nor are superseded.
    ``TriggerRun.task_id`` is the link; a missing run row is not superseded.
    """

    if task.source != "trigger":
        return False
    agent_config = task.agent_config
    if not isinstance(agent_config, dict):
        return False
    if agent_config.get("trigger_type") != TriggerType.SCHEDULED.value:
        return False

    run = (
        db.query(TriggerRun)
        .filter(TriggerRun.task_id == task.id)
        .order_by(TriggerRun.id.desc())
        .first()
    )
    if run is None or _is_test_run(run, agent_config):
        return False
    later = (
        db.query(TriggerRun.id)
        .filter(
            TriggerRun.trigger_id == run.trigger_id,
            TriggerRun.id > run.id,
            TriggerRun.started_at.isnot(None),
            TriggerRun.status.in_(_SUPERSEDING_RUN_STATUSES),
            ~TriggerRun.idempotency_key.startswith(TEST_TRIGGER_RUN_KEY_PREFIX),
        )
        .first()
    )
    return later is not None
