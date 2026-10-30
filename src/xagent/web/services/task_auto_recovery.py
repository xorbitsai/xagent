"""Interruption bookkeeping for automatic task recovery.

When a run is interrupted rather than finished, the settling transaction
records why in ``task_auto_recovery`` (one row per task) and appends a
``task_recovery_events`` row. Recording them changes no task status,
control state or error message; the one projection it decides is the
TriggerRun of a PAUSED task, which stays RUNNING while an automatic resume
is scheduled (see :func:`apply_interruption_outcome_no_commit`).

Which outcome a run-settling path (a run that raised, or returned an
unsuccessful result) writes for an interruption is decided by
:func:`decide_interruption`: an eligible run with a recoverable checkpoint
rests PAUSED instead of FAILED, so its user can resume it. Lease-expiry
recovery keeps its own verdict mapping.

A PAUSED interruption that automatic recovery may act on is ``scheduled``
with its first ``next_attempt_at``, for the executor (``task_auto_resume``) to
dispatch; the policy may instead stop it at once (``exhausted``,
``expired``), or a later run of its scheduled trigger may have replaced it
(``superseded``). Otherwise the state says that nothing automatic will act on
the run: ``manual`` (only the user can act -- resume a PAUSED task; a FAILED
one is terminal and the row just records why; ``state_detail`` names a
switched-off executor or a failed supersede check),
``ineligible`` (a task kind automatic recovery will never touch) or
``disabled`` (the interruption would have paused the run but
``XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED`` is off, see
:func:`settlement_pause_enabled`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from ...config import (
    get_shared_task_execution_enabled,
    get_task_auto_resume_enabled,
    get_task_infra_failure_pause_enabled,
)
from ...core.agent.checkpoint import (
    checkpoint_progress_marker,
    checkpoint_tool_call_in_flight,
)
from ...core.agent.interruption import (
    InterruptionReason,
    classify_run_failure,
    classify_run_result,
    is_database_unavailable,
)
from ...core.runtime_performance import increment_counter
from ..models.task import Task, TaskStatus
from ..models.task_auto_recovery import (
    TASK_AUTO_RECOVERY_LAST_ERROR_MAX_CHARS,
    TaskAutoRecovery,
    TaskAutoRecoveryState,
    TaskRecoveryEvent,
    TaskRecoveryEventType,
)
from ..models.trigger import TriggerType
from ..models.workforce import WorkforceRun
from ..utils.db_timezone import format_datetime_for_api
from .task_auto_recovery_policy import (
    auto_resume_policy,
    auto_resume_recovery_state,
    plan_auto_resume,
    scheduled_trigger_superseded,
)
from .task_command_transport import AUTO_RESUME_COMMAND_PREFIX
from .task_execution_controller import TaskControlState
from .task_lease_service import (
    TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR,
    CheckpointRecoveryResolution,
    CheckpointRecoveryVerdict,
    TaskLease,
    TaskLeaseRecoveryCandidate,
    lock_task_lease_for_settlement_no_commit,
    resolve_checkpoint_recovery_with_data,
    task_lease_attempt_predicate,
    utc_now,
)
from .task_settlement_report import RecordedInterruption
from .workforce_runtime import extract_workforce_run_id

logger = logging.getLogger(__name__)

# ``state_detail`` of a PAUSED row that automatic resume leaves to its user:
# the executor is switched off (XAGENT_TASK_AUTO_RESUME_ENABLED), or the
# check for a later run of its scheduled trigger failed.
AUTO_RESUME_DISABLED_DETAIL = "auto_resume_disabled"
SUPERSEDE_CHECK_FAILED_DETAIL = "supersede_check_failed"
# ``state_detail`` of a ``manual`` row whose pending automatic resume a person
# cancelled (``task_auto_resume.cancel_pending_auto_resume_no_commit``); only
# that cancel writes it, at the fence it moves the row to.
AUTO_RESUME_USER_CANCELLED_DETAIL = "user_cancelled"
# ``state_detail`` of a ``manual`` row whose run stopped with a tool call that
# may have started and has no recorded outcome (a legacy, version-1 run; see
# :func:`resolution_tool_call_in_flight`). Resuming would run that call again,
# so only a person may decide to.
TOOL_IN_FLIGHT_DETAIL = "tool_in_flight"

# The recorded states automatic resume has a plan for: each is logged as an
# extra event of its own name and announced as ``task_recovery_update``
# (``task_auto_resume.recorded_notice``).
PLANNED_RECOVERY_EVENTS = {
    TaskAutoRecoveryState.SCHEDULED: TaskRecoveryEventType.SCHEDULED,
    TaskAutoRecoveryState.EXHAUSTED: TaskRecoveryEventType.EXHAUSTED,
    TaskAutoRecoveryState.EXPIRED: TaskRecoveryEventType.EXPIRED,
    TaskAutoRecoveryState.SUPERSEDED: TaskRecoveryEventType.SUPERSEDED,
}


def _as_utc(value: datetime) -> datetime:
    """Aware UTC; SQLite returns naive values for timezone-aware columns."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def interruption_pause_result() -> dict[str, Any]:
    """The ``execution_settled`` result of a run paused by an interruption.

    The same shape lease recovery stages, so the next turn's context does not
    read the pause as a failed execution. A fresh dict per call: the fact
    writer stores it by reference in the row payload.
    """
    return {"error": None}


# Callers that own their own protocol for a stopped run: A2A clients, external
# cancellation, and anonymous visitors of a widget or shared link. SDK tasks
# are eligible: an SDK ``wait()`` keeps waiting on a PAUSED task, so a client
# still waiting on it receives the resumed run's result. Once automatic resume
# stops, the ``/v1`` read of such a task reports ``failed`` (see
# ``task_auto_resume.status_only_terminal_view``), so that wait returns.
_INELIGIBLE_SOURCES = frozenset({"a2a", "external", "widget", "shared_link"})

# Callers that only ever read a task's status. A settlement interruption of
# one of their runs is paused only while automatic resume is on: pausing is
# worth it only when something will resume the run, and when automatic resume
# stops, the ``/v1`` read of the task reports ``failed``
# (``task_auto_resume.status_only_terminal_view``). When recording such a
# pause fails, a minimal ``manual`` row is written in its place so that read
# still reports ``failed``. With automatic resume off the run ends FAILED with
# the provider's error. Lease expiry is not a settlement interruption and is
# paused (and resumed) for them as for any other task.
_STATUS_ONLY_SOURCES = frozenset({"sdk"})


def is_status_only_source(source: Any) -> bool:
    """Whether a task of ``source`` has a caller that only reads its status."""

    return str(source or "internal") in _STATUS_ONLY_SOURCES


_TRIGGER_TYPES = frozenset(member.value for member in TriggerType)

# TriggerRun error for a run a settling path paused after an interruption
# (lease expiry has its own, TASK_LEASE_PAUSED_TRIGGER_ERROR).
TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR = (
    "Task paused after a system interruption; manual resume is required."
)


@dataclass(frozen=True)
class AutoRecoveryEligibility:
    """Whether a task may ever be resumed without its user.

    ``kind`` names the eligible task family (later policy picks windows by
    it); ``detail`` is the ``ineligible:<why>`` label stored on the row.
    """

    eligible: bool
    kind: str | None = None
    detail: str | None = None


def _ineligible(why: str) -> AutoRecoveryEligibility:
    return AutoRecoveryEligibility(eligible=False, detail=f"ineligible:{why}")


def auto_recovery_eligibility(db: Session, task: Task) -> AutoRecoveryEligibility:
    """Classify ``task`` for automatic recovery; the single source of truth.

    Rules apply in order and the first match wins:

    1. A2A, external, widget and shared-link tasks are ineligible.
    2. Workforce tasks are eligible unless their run is a preview (or the
       run row is gone, so it cannot be shown not to be one).
    3. Trigger tasks are eligible.
    4. Channel tasks are eligible only under shared execution; a combined
       host runs them inline in the bot, which has no durable way to deliver
       an answer produced by a later resume.
    5. Hidden internal tasks are previews (agent builder, preview chat).
    6. Everything else is an eligible normal task.
    """

    source = str(task.source or "internal")
    if source in _INELIGIBLE_SOURCES:
        return _ineligible(f"source_{source}")

    workforce_run_id = extract_workforce_run_id(task)
    if workforce_run_id is not None:
        run = db.get(WorkforceRun, workforce_run_id)
        if run is None:
            return _ineligible("workforce_run_missing")
        if run.is_preview:
            return _ineligible("preview")
        return AutoRecoveryEligibility(eligible=True, kind="workforce")

    if source == "trigger":
        agent_config = task.agent_config
        trigger_type = (
            agent_config.get("trigger_type") if isinstance(agent_config, dict) else None
        )
        return AutoRecoveryEligibility(
            eligible=True,
            kind=(
                f"trigger_{trigger_type}"
                if isinstance(trigger_type, str) and trigger_type in _TRIGGER_TYPES
                else "trigger"
            ),
        )

    if task.channel_id is not None:
        if not get_shared_task_execution_enabled():
            return _ineligible("channel_inline")
        return AutoRecoveryEligibility(eligible=True, kind="channel")

    if source == "internal" and task.is_visible is False:
        return _ineligible("preview")

    return AutoRecoveryEligibility(eligible=True, kind="normal")


def shutdown_handoff_eligible(db: Session, task_id: int) -> bool:
    """Whether a run a host shutdown cancelled may be left to lease recovery.

    Such a run keeps its lease instead of failing, so lease recovery pauses
    it and automatic resume picks it up, as after a crash. That only helps a
    task automatic recovery would resume (:func:`auto_recovery_eligibility`);
    the others must still fail now so their callers see a terminal state.
    Tasks of a status-only source (:func:`is_status_only_source`) still fail
    now as well, as their settlement interruptions do: their caller reads
    nothing but the status, and a deploy should not keep it waiting on a
    lease TTL. Channel tasks are excluded too until their resumed reply is
    proven, and nothing is handed off while automatic resume is off.
    """

    if not get_task_auto_resume_enabled():
        return False
    task = db.get(Task, task_id)
    if task is None or task.status != TaskStatus.RUNNING:
        return False
    if is_status_only_source(task.source) or task.channel_id is not None:
        return False
    return auto_recovery_eligibility(db, task).eligible


def record_interruption_no_commit(
    db: Session,
    *,
    task: Task,
    reason: InterruptionReason,
    task_status: TaskStatus,
    interrupted_at: datetime,
    progress_marker: str | None,
    gated_by_settlement_switches: bool,
    last_error: str | None = None,
    now: datetime | None = None,
    tool_call_in_flight: bool = False,
) -> TaskAutoRecovery | None:
    """Upsert ``task``'s recovery row for one interruption and log the event.

    ``task`` must be the row as the settling write left it, in the same
    transaction: its ``run_id`` and ``state_version`` become the row's run
    and fence. A run without an id is not recorded -- it cannot be fenced.
    ``last_error`` is an operator-only diagnostic, truncated to fit the row.

    ``gated_by_settlement_switches`` says whether the caller's PAUSED outcome
    depends on :func:`settlement_pause_enabled`. Settlement paths that pause
    instead of failing are; lease-expiry recovery is not (it paused before the
    switch existed), so it never records ``disabled``. An ineligible task
    records ``ineligible`` whatever the switch says.

    ``paused_state_version`` is the task's ``state_version`` after the
    settling write, PAUSED or FAILED; only a PAUSED row's fence is ever read.

    Episodes: only an interruption of a run that automatic recovery resumed
    can continue anything -- the row still describes the same run, rests in
    ``dispatched``, and no user command reached the task after that dispatch.
    Such an interruption keeps ``total_resumes``, and keeps the no-progress
    episode (its start and ``no_progress_resumes``) when the reason and the
    progress marker are unchanged; a new reason or any progress starts a new
    episode. Anything else -- a different run, or a person having acted since
    (a manual resume, a message, a row automatic recovery had already stopped
    on) -- starts a new episode and restarts ``total_resumes`` too, so a run
    its user resumed by hand is not held to the attempts spent before. A
    missing marker (an undecodable or absent checkpoint) is unknown, not
    progress: it keeps the last known marker.

    Scheduling: an eligible PAUSED interruption whose reason has a policy,
    of a kind automatic resume schedules (:func:`auto_resume_schedulable`), is
    planned with :func:`plan_auto_resume` at ``now`` (default
    ``interrupted_at``) -- ``scheduled`` with its first ``next_attempt_at``,
    or stopped at once: ``exhausted``, or ``expired`` when the run stopped
    longer ago than the window (lease recovery after a long outage passes
    the time the lease expired as ``interrupted_at``) -- unless a later run
    of its scheduled trigger already started (``superseded``).
    It rests in ``manual`` instead, with ``state_detail`` saying why, when
    automatic resume is switched off (``auto_resume_disabled``) and when the
    supersede check fails (``supersede_check_failed``; the row is still
    written). A row that would be ``scheduled`` rests in ``manual`` with
    ``tool_in_flight`` when ``tool_call_in_flight`` says resuming would repeat
    a tool call that may already have run; the other verdicts stand. A reason
    without a policy (a user pause, an unknown input outcome) stays plain
    ``manual``.
    """

    run_id = task.run_id
    if run_id is None:
        return None

    eligibility = auto_recovery_eligibility(db, task)
    plannable = (
        eligibility.eligible
        and task_status is TaskStatus.PAUSED
        and auto_resume_policy(reason) is not None
        and auto_resume_schedulable(eligibility.kind)
    )
    enabled = get_task_auto_resume_enabled()
    # Before any change to the session: the check's own SAVEPOINT rolls back
    # nothing of the caller's on failure. ``None`` means it failed, and the
    # safe direction is not to schedule.
    superseded: bool | None = (
        _superseded_or_unknown(db, task) if plannable and enabled else False
    )
    # Ineligibility is permanent and the switch is not, so it wins: a
    # ``disabled`` row always means "turning the infra switch on would
    # help". An interruption that switch would not pause either (see
    # ``_pausable_with_infra_switch``) is not disabled, so it records
    # ``manual``, like any FAILED run.
    if not eligibility.eligible:
        state, state_detail = TaskAutoRecoveryState.INELIGIBLE, eligibility.detail
    elif (
        gated_by_settlement_switches
        and _pausable_with_infra_switch(reason, eligibility.kind, task.source)
        and not get_task_infra_failure_pause_enabled()
    ):
        state, state_detail = TaskAutoRecoveryState.DISABLED, None
    else:
        state, state_detail = TaskAutoRecoveryState.MANUAL, None

    row = db.get(TaskAutoRecovery, task.id)
    same_run = row is not None and row.run_id == run_id
    # Unattended since the last automatic dispatch: the absolute cap keeps
    # counting even if the reason changes, or alternating reasons could loop.
    continuation = (
        same_run
        and row is not None
        and row.state == TaskAutoRecoveryState.DISPATCHED.value
        and not _user_command_since_dispatch(db, int(task.id), row.last_command_id)
    )
    if same_run and row is not None and progress_marker is None:
        progress_marker = row.progress_marker
    same_episode = (
        continuation
        and row is not None
        and row.reason == reason.value
        and (row.progress_marker is None or row.progress_marker == progress_marker)
    )
    if row is None:
        row = TaskAutoRecovery(task_id=task.id)
        db.add(row)
    if not same_episode:
        row.episode_started_at = interrupted_at
        row.no_progress_resumes = 0
    if not continuation:
        row.total_resumes = 0
        row.last_command_id = None
    row.run_id = run_id
    row.reason = reason.value
    row.state = state.value
    row.state_detail = state_detail
    row.paused_state_version = int(task.state_version or 0)
    row.interrupted_at = interrupted_at
    row.progress_marker = progress_marker
    row.next_attempt_at = None
    row.last_error = (
        last_error[:TASK_AUTO_RECOVERY_LAST_ERROR_MAX_CHARS]
        if last_error is not None
        else None
    )
    if plannable and state is TaskAutoRecoveryState.MANUAL:
        if not enabled:
            state_detail = AUTO_RESUME_DISABLED_DETAIL
        elif superseded is None:
            state_detail = SUPERSEDE_CHECK_FAILED_DETAIL
        elif superseded:
            state = TaskAutoRecoveryState.SUPERSEDED
        else:
            from . import task_auto_resume

            verdict = plan_auto_resume(
                reason=reason,
                kind=eligibility.kind,
                interrupted_at=interrupted_at,
                episode_started_at=row.episode_started_at,
                no_progress_resumes=int(row.no_progress_resumes or 0),
                total_resumes=int(row.total_resumes or 0),
                now=now or interrupted_at,
                # One RNG for every automatic-resume jitter; tests seed it.
                rng=task_auto_resume._AUTO_RESUME_RNG,
            )
            state = auto_resume_recovery_state(verdict.outcome)
            if state is TaskAutoRecoveryState.SCHEDULED and tool_call_in_flight:
                # Only a resume that would actually be dispatched is held;
                # every other verdict already stops automatic resume.
                state, state_detail = (
                    TaskAutoRecoveryState.MANUAL,
                    TOOL_IN_FLIGHT_DETAIL,
                )
            else:
                row.next_attempt_at = verdict.next_attempt_at
        row.state = state.value
        row.state_detail = state_detail

    detail: dict[str, Any] = {"task_status": task_status.value, "state": state.value}
    if state_detail is not None:
        detail["state_detail"] = state_detail
    if eligibility.kind is not None:
        detail["kind"] = eligibility.kind
    db.add(
        TaskRecoveryEvent(
            task_id=task.id,
            run_id=run_id,
            event=(
                TaskRecoveryEventType.INELIGIBLE
                if state is TaskAutoRecoveryState.INELIGIBLE
                else TaskRecoveryEventType.INTERRUPTED
            ).value,
            reason=reason.value,
            detail=detail,
        )
    )
    planned_event = PLANNED_RECOVERY_EVENTS.get(state)
    if planned_event is not None:
        db.add(
            TaskRecoveryEvent(
                task_id=task.id,
                run_id=run_id,
                event=planned_event.value,
                reason=reason.value,
                attempt=RecordedInterruption.from_row(row).attempt,
                next_attempt_at=row.next_attempt_at,
                detail={"at": "record"},
            )
        )
    db.flush()
    increment_counter(
        "xagent.task.auto_resume.recorded",
        attributes={"outcome": state.value, "operation": reason.value},
    )
    log = (
        logger.warning
        if state in (TaskAutoRecoveryState.EXHAUSTED, TaskAutoRecoveryState.EXPIRED)
        or state_detail == SUPERSEDE_CHECK_FAILED_DETAIL
        else logger.info
    )
    log(
        "Recorded task interruption: task_id=%s run_id=%s reason=%s state=%s "
        "state_detail=%s next_attempt_at=%s component=auto-recovery",
        task.id,
        run_id,
        reason.value,
        state.value,
        state_detail,
        row.next_attempt_at,
    )
    return row


def _superseded_or_unknown(db: Session, task: Task) -> bool | None:
    """Whether a later run of ``task``'s scheduled trigger already started.

    Reads in its own SAVEPOINT; a failed read is rolled back to it, logged
    and reported as ``None``. A SAVEPOINT that cannot be opened or rolled back
    propagates to the caller's (the whole row is then skipped).
    """

    savepoint = db.begin_nested()
    try:
        superseded = scheduled_trigger_superseded(db, task)
    except Exception:
        savepoint.rollback()
        logger.warning(
            "component=auto-resume task_id=%s checking for a later scheduled "
            "trigger run failed; leaving the resume to its user",
            task.id,
            exc_info=True,
        )
        return None
    savepoint.commit()
    return superseded


def is_auto_resume_command_id(command_id: object) -> bool:
    """Whether ``command_id`` is in the namespace reserved for the sweeper.

    ``stage_task_command`` refuses the prefix unless the caller passes
    ``reserved=True``, which only the auto-resume sweeper will do, so a staged
    command with it is the sweeper's own.
    """

    return isinstance(command_id, str) and command_id.startswith(
        AUTO_RESUME_COMMAND_PREFIX
    )


def auto_resume_schedulable(kind: str | None) -> bool:
    """Whether an eligible task of ``kind`` may be resumed automatically now.

    Every eligible kind is: a channel task's reply waits for the resumed run
    (``channel_reply_waits_for_auto_resume``). Kept as the one place a kind
    could be held back again. It is read when an interruption is recorded
    (and by :func:`_pausable_with_infra_switch`), not at dispatch: rows
    already ``scheduled`` when it is narrowed still dispatch.
    """

    return kind is not None


def _user_command_since_dispatch(
    db: Session, task_id: int, last_command_id: str | None
) -> bool:
    """Whether a command other than the sweeper's reached the task after it.

    Commands are ordered by id per task. A dispatch command that is gone
    (deleted with nothing after it, or never staged) leaves the comparison
    NULL, which reads as no intervention.
    """

    if last_command_id is None:
        return False
    from sqlalchemy import and_, func, select

    from ..models.task_command import TaskExecutionCommand

    dispatched_id = (
        select(TaskExecutionCommand.id)
        .where(
            TaskExecutionCommand.task_id == task_id,
            TaskExecutionCommand.command_id == last_command_id,
        )
        .scalar_subquery()
    )
    later = db.execute(
        select(TaskExecutionCommand.id)
        .where(
            TaskExecutionCommand.task_id == task_id,
            TaskExecutionCommand.id > dispatched_id,
            ~and_(
                TaskExecutionCommand.kind == "resume",
                # Exact, case-sensitive prefix, as ``_normalize_command_id``
                # checks it: ``startswith`` compiles to LIKE, which ignores
                # ASCII case on SQLite, so a person's ``AUTO-RESUME:...``
                # RESUME would pass for the sweeper's.
                func.substr(
                    TaskExecutionCommand.command_id,
                    1,
                    len(AUTO_RESUME_COMMAND_PREFIX),
                )
                == AUTO_RESUME_COMMAND_PREFIX,
            ),
        )
        .limit(1)
    ).first()
    return later is not None


def current_auto_recovery_view(db: Session, task: Task) -> dict[str, Any] | None:
    """Why a PAUSED ``task``'s run stopped, for clients, or ``None``.

    The row is not cleared when its run is resumed; it goes stale as the
    task's ``run_id`` or ``state_version`` moves on. Only a PAUSED task's row
    that still describes the task as it stands -- same run, fenced at the
    current ``state_version`` -- is shown; a FAILED run's row (``manual`` or
    ``disabled``) is bookkeeping, not something its user can act on.

    ``reason`` and ``interrupted_at`` describe the interruption; ``run_id``
    and ``state_version`` name the fence, as a ``task_recovery_update``
    notice's envelope does; ``auto_resume`` is that notice's payload for the
    row as it stands (:func:`~.task_auto_resume.current_auto_resume_view`),
    ``None`` when no automatic resume is to be shown. The operator-only
    ``state``, ``state_detail``, ``last_error``, counters and command ids are
    left out.

    ``auto_resume`` changes without a task write (the sweeper's transitions
    write only this row), so the view is computed per request, outside any
    response cached on the task row; the legacy history replay keys its
    cache on the view itself.

    Informational only: a failed read is logged and reads as ``None``, inside
    a SAVEPOINT so the caller's transaction stays usable (on PostgreSQL a
    failed statement would otherwise abort it for the rest of the request).
    """

    if task.status != TaskStatus.PAUSED:
        return None
    try:
        with db.begin_nested():
            row = db.get(TaskAutoRecovery, task.id)
    except Exception:
        logger.warning(
            "Reading the auto-recovery row of task %s failed; reporting none",
            task.id,
            exc_info=True,
        )
        return None
    if (
        row is None
        or row.run_id != task.run_id
        or row.paused_state_version != int(task.state_version or 0)
    ):
        return None
    from .task_auto_resume import current_auto_resume_view  # module cycle

    return {
        "reason": row.reason,
        "interrupted_at": format_datetime_for_api(row.interrupted_at),
        "run_id": row.run_id,
        "state_version": int(row.paused_state_version),
        "auto_resume": current_auto_resume_view(task, row),
    }


def lease_expiry_interruption_reason(
    candidate: TaskLeaseRecoveryCandidate,
    verdict: CheckpointRecoveryVerdict,
) -> InterruptionReason | None:
    """Why lease recovery stopped ``candidate``'s run, by its verdict.

    A recoverable run becomes PAUSED: a user pause when the run died while
    PAUSE_REQUESTED (the user already decided), else ``lease_expired``. The
    FAILED verdicts keep their own terminal reasons. ``INDETERMINATE`` writes
    nothing, so it has no reason.
    """

    return verdict_interruption_reason(
        verdict,
        control_state=candidate.control_state,
        recoverable_reason=InterruptionReason.LEASE_EXPIRED,
    )


def verdict_interruption_reason(
    verdict: CheckpointRecoveryVerdict,
    *,
    control_state: str | None,
    recoverable_reason: InterruptionReason,
) -> InterruptionReason | None:
    """The reason a checkpoint verdict records, shared by every writer.

    ``RECOVERABLE`` records ``recoverable_reason``, or a user pause when the
    run was PAUSE_REQUESTED; the FAILED verdicts record their own terminal
    reasons; ``INDETERMINATE`` writes nothing, so it has none.
    """

    if verdict is CheckpointRecoveryVerdict.RECOVERABLE:
        if control_state == TaskControlState.PAUSE_REQUESTED.value:
            return InterruptionReason.USER_PAUSE
        return recoverable_reason
    if verdict is CheckpointRecoveryVerdict.UNKNOWN_TOOL_EFFECT:
        return InterruptionReason.UNKNOWN_TOOL_EFFECT
    if verdict is CheckpointRecoveryVerdict.NOT_RECOVERABLE:
        return InterruptionReason.NOT_RECOVERABLE
    return None


def resolution_progress_marker(
    db: Session, task_id: int, resolution: CheckpointRecoveryResolution
) -> str | None:
    """Progress fingerprint of a recoverable verdict's checkpoint, if any.

    A legacy payload is decoded first, because its message list and tool
    ledger may be stored as refs; only counts are needed, so blob hashes are
    not verified. Refs that fail to decode (``CheckpointMessageDecodeError``)
    still record the interruption, just without a marker; any other lookup
    error propagates to the caller's SAVEPOINT and skips the row.
    """

    from .trace_message_storage import (
        CheckpointMessageDecodeError,
        decode_trace_event_data,
    )

    if (
        resolution.verdict is not CheckpointRecoveryVerdict.RECOVERABLE
        or resolution.checkpoint is None
    ):
        return None
    data: Any = resolution.checkpoint
    if resolution.encoded:
        try:
            data = decode_trace_event_data(
                db,
                task_id=task_id,
                data=data,
                strict=True,
                verify_blob_hashes=False,
            )
        except CheckpointMessageDecodeError:
            logger.warning(
                "Task %s checkpoint refs are undecodable; recording its "
                "interruption without a progress marker",
                task_id,
            )
            return None
    snapshot = data.get("snapshot") if isinstance(data, dict) else None
    return checkpoint_progress_marker(snapshot if isinstance(snapshot, dict) else None)


def resolution_tool_call_in_flight(resolution: CheckpointRecoveryResolution) -> bool:
    """Whether resuming a recoverable legacy run would repeat a started tool call.

    A version-two run records every tool attempt, and recovery already fails
    one that started without a committed outcome (``UNKNOWN_TOOL_EFFECT``).
    A legacy (``trace_events``) checkpoint has no such ledger: resuming from
    a checkpoint written just before a tool call runs that call again, so a
    message it sent or a write it made would happen twice
    (:func:`checkpoint_tool_call_in_flight`). The fields read here are never
    stored as refs, so the payload is read without decoding.
    """

    if (
        resolution.verdict is not CheckpointRecoveryVerdict.RECOVERABLE
        or not resolution.encoded
        or not isinstance(resolution.checkpoint, dict)
    ):
        return False
    snapshot = resolution.checkpoint.get("snapshot")
    return checkpoint_tool_call_in_flight(
        snapshot if isinstance(snapshot, dict) else None
    )


def record_lease_expiry_interruption_no_commit(
    db: Session,
    *,
    task: Task,
    candidate: TaskLeaseRecoveryCandidate,
    resolution: CheckpointRecoveryResolution,
    task_status: TaskStatus,
    recovered_at: datetime,
) -> RecordedInterruption | None:
    """Record a lease recovery's interruption inside its open transaction.

    Returns what was recorded, or ``None`` when nothing was (no reason for
    the verdict, or the recording failed and was skipped).

    Runs in a SAVEPOINT and does not raise past it for a failure inside the
    recording itself:
    the fenced status write and its projections must commit exactly as they
    would without this metadata. Rolling the whole recovery back instead
    would retry it next tick, but a failure that repeats deterministically
    (a schema the migration has not reached, a value the row cannot hold)
    would then leave the task RUNNING with an expired lease forever. A
    skipped row only means the task takes no part in automatic recovery,
    which is the safe direction. If the SAVEPOINT itself cannot be rolled
    back, or opening or releasing it fails, the transaction itself is broken
    and the commit would fail anyway; that error propagates and the whole
    recovery retries next tick.

    Preconditions, both met by the lease-recovery transaction today:

    - The fenced status write has already run in this transaction: on
      SQLite that UPDATE is what opens it, so the SAVEPOINT is nested rather
      than autocommitted.
    - Nothing staged earlier in the transaction is still pending in the
      session. ``flush`` has no scope, so pending objects would be emitted
      inside the SAVEPOINT and discarded with it on failure. Sessions are
      built with ``autoflush=False`` and every earlier staging step (the
      orphaned-delivery reconciliation) flushes itself; a new step added
      before this call must do the same.
    """

    reason = lease_expiry_interruption_reason(candidate, resolution.verdict)
    if reason is None:
        return None
    savepoint = db.begin_nested()
    try:
        row = record_interruption_no_commit(
            db,
            task=task,
            reason=reason,
            task_status=task_status,
            # The run stopped when its lease lapsed, not when a host noticed:
            # after an outage longer than the window it is not resumed.
            interrupted_at=min(recovered_at, _as_utc(candidate.lease_expires_at)),
            now=recovered_at,
            progress_marker=resolution_progress_marker(db, int(task.id), resolution),
            gated_by_settlement_switches=False,
            tool_call_in_flight=resolution_tool_call_in_flight(resolution),
        )
        recorded = RecordedInterruption.from_row(row) if row is not None else None
    except Exception:
        savepoint.rollback()
        logger.exception(
            "Recording the lease-expiry interruption of task %s failed; "
            "recovering it without auto-recovery metadata",
            task.id,
        )
        return None
    savepoint.commit()
    return recorded


# Interruption reasons the run-settling paths act on. Settlement keeps its
# terminal FAILED outcome for every other reason, ``None`` included.
#
# Where they come from: an LLM call that exhausted its retries or kept
# returning unusable output raises inside a pattern, and the runner turns
# that into an unsuccessful result carrying ``interruption_reason``; a ReAct
# run that gave up on the tool protocol returns ``invalid_tool_protocol``. So
# both reach the result paths, not the exception path, which sees the
# persistence failures the runner lets escape. A DAG run swallows its LLM
# failures into an ordinary failed result without a reason, so it never
# pauses for either (it still pauses for a persistence failure).
#
# Every one of them is recorded; ``_pausable_with_infra_switch`` says which
# of them the infra switch decides (``model_output_invalid`` also needs an
# automatic resume to resample it).
SETTLEMENT_INTERRUPTION_REASONS = frozenset(
    {
        InterruptionReason.PERSISTENCE_FAILURE,
        InterruptionReason.LLM_UNAVAILABLE,
        InterruptionReason.MODEL_OUTPUT_INVALID,
    }
)


# Recorded, but never paused for: a reason whose pause nothing could act on
# yet goes here until something can. Empty for now; kept so deferring a
# reason again is a one-line change.
SETTLEMENT_PAUSE_DEFERRED_REASONS: frozenset[InterruptionReason] = frozenset()


def _pausable_with_infra_switch(
    reason: InterruptionReason,
    kind: str | None = None,
    source: Any = None,
) -> bool:
    """Whether ``XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED`` decides ``reason``.

    False for a deferred reason and for a status-only ``source``
    (:data:`_STATUS_ONLY_SOURCES`) while automatic resume is off: nothing
    would resume their paused run, so the switch does not decide it.
    ``model_output_invalid`` is not an
    infrastructure failure: a run paused for it is worth stopping only when
    an automatic resume will resample it, so it also needs automatic resume
    on and, when ``kind`` is known, a task kind it schedules
    (:func:`auto_resume_schedulable`; every eligible kind today, so that case
    is unreachable and kept as the seam). Otherwise it fails as before.
    """

    if reason in SETTLEMENT_PAUSE_DEFERRED_REASONS:
        return False
    if is_status_only_source(source) and not get_task_auto_resume_enabled():
        return False
    if reason is InterruptionReason.MODEL_OUTPUT_INVALID:
        return get_task_auto_resume_enabled() and (
            kind is None or auto_resume_schedulable(kind)
        )
    return True


def settlement_pause_enabled(reason: InterruptionReason) -> bool:
    """Whether settlement may pause an eligible run interrupted for ``reason``.

    ``XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED`` (unset, it follows automatic
    resume), for a reason that switch decides (``_pausable_with_infra_switch``
    without a task kind; :func:`decide_interruption` checks the kind once it
    has read eligibility).
    """

    return (
        _pausable_with_infra_switch(reason) and get_task_infra_failure_pause_enabled()
    )


def settlement_interruption_for_failure(
    exc: BaseException,
) -> InterruptionReason | None:
    """The interruption a settling path acts on for a run that raised."""

    reason = classify_run_failure(exc)
    return reason if reason in SETTLEMENT_INTERRUPTION_REASONS else None


def settlement_interruption_for_result(result: Any) -> InterruptionReason | None:
    """The interruption a settling path acts on for an unsuccessful result.

    Reads only the result's top-level fields (the ``AgentService`` contract),
    never ``agent_result``. A quota stop wins over any interruption it
    carries: the web layer rewrites ``status`` to ``quota_exceeded`` but keeps
    the runner's ``interruption_reason``, and a quota refusal is final.
    """

    if not isinstance(result, Mapping) or result.get("success"):
        return None
    if result.get("status") == "quota_exceeded":
        return None
    reason = classify_run_result(result)
    return reason if reason in SETTLEMENT_INTERRUPTION_REASONS else None


class InterruptionSettlementDeferred(RuntimeError):
    """The interrupted run cannot be decided right now.

    Raised inside the settling transaction -- for an unresolvable checkpoint
    or for any error while reading eligibility or the checkpoint, chained as
    its cause -- so the caller rolls it back and keeps the exact lease; TTL
    recovery settles the run once the database answers.
    """


class InterruptionOutcome(str, Enum):
    """What a settling path writes for an interrupted run."""

    # Automatic recovery does not apply (a switch off, an ineligible task): the
    # caller settles exactly as it always has, its own classification
    # included.
    LEGACY = "legacy"
    # PAUSED, manually resumable from the decision's checkpoint.
    PAUSE = "pause"
    # FAILED with TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR.
    FAIL_UNKNOWN_TOOL_EFFECT = "fail_unknown_tool_effect"
    # FAILED with the caller's own error.
    FAIL_NOT_RECOVERABLE = "fail_not_recoverable"


@dataclass(frozen=True)
class InterruptionDecision:
    """How a settling path ends a run interrupted for ``reason``.

    ``reason`` is what the recovery row records; ``resolution`` is the
    checkpoint a pause resumes from (``None`` otherwise).
    ``unknown_tool_attempt_id`` is, for ``FAIL_UNKNOWN_TOOL_EFFECT``, the
    attempt the verdict was reached on when known: the settling caller
    reports it (``log_unknown_tool_effect_settlement``) once its
    transaction commits.
    """

    outcome: InterruptionOutcome
    reason: InterruptionReason
    resolution: CheckpointRecoveryResolution | None = None
    unknown_tool_attempt_id: str | None = None

    @property
    def pause(self) -> bool:
        return self.outcome is InterruptionOutcome.PAUSE


_VERDICT_OUTCOMES = {
    CheckpointRecoveryVerdict.RECOVERABLE: InterruptionOutcome.PAUSE,
    CheckpointRecoveryVerdict.UNKNOWN_TOOL_EFFECT: (
        InterruptionOutcome.FAIL_UNKNOWN_TOOL_EFFECT
    ),
    CheckpointRecoveryVerdict.NOT_RECOVERABLE: InterruptionOutcome.FAIL_NOT_RECOVERABLE,
}


def legacy_interruption_decision(reason: InterruptionReason) -> InterruptionDecision:
    """The decision when automatic recovery does not apply."""

    return InterruptionDecision(InterruptionOutcome.LEGACY, reason)


def _settlement_candidate(task: Task) -> TaskLeaseRecoveryCandidate:
    """The locked row as the candidate checkpoint resolution reads."""

    row: Any = task
    return TaskLeaseRecoveryCandidate(
        task_id=int(row.id),
        runner_id=row.runner_id,
        run_id=row.run_id,
        lease_expires_at=row.lease_expires_at or utc_now(),
        state_version=int(row.state_version or 0),
        last_checkpoint_event_id=row.last_checkpoint_event_id,
        last_checkpoint_trace_event_id=row.last_checkpoint_trace_event_id,
        attempt_id=row.lease_attempt_id,
        control_state=row.control_state,
    )


def decide_interruption(
    db: Session, *, task: Task, reason: InterruptionReason
) -> InterruptionDecision:
    """Decide how to settle ``task``'s RUNNING run interrupted for ``reason``.

    ``task`` must be the run's row, locked by the settling transaction. In
    order:

    1. Settlement does not pause for ``reason`` (:func:`settlement_pause_enabled`:
       the switch is off, the reason is deferred, or it is
       ``model_output_invalid`` with automatic resume off), a task of a
       status-only source (:data:`_STATUS_ONLY_SOURCES`) while automatic
       resume is off (decided from the row alone before anything is read; with
       it on, such a task is judged like any other), an ineligible task, or
       ``model_output_invalid`` on a task kind automatic resume does not
       schedule (:func:`auto_resume_schedulable`; nothing would resample it;
       unreachable today, every eligible kind is scheduled): ``LEGACY``.
    2. The run's checkpoint, resolved as lease recovery resolves it:
       ``UNKNOWN_TOOL_EFFECT`` and ``NOT_RECOVERABLE`` fail, ``RECOVERABLE``
       pauses -- as a user pause when the run was PAUSE_REQUESTED (the user
       already decided), else for ``reason``. Nothing is reported here: the
       settling caller reports an unknown tool effect once its settlement
       commits, so a settlement that rolls back reports nothing.
    3. ``INDETERMINATE``, or a database connectivity failure while reading
       eligibility or the checkpoint, raises
       :class:`InterruptionSettlementDeferred` so every settling path keeps
       the lease for TTL recovery alike.
    4. Any other error reading them is ``LEGACY``: it would reproduce on
       every attempt, and TTL recovery reads the checkpoint the same way, so
       deferring would leave the run RUNNING forever. Settling as before
       (FAILED) keeps it terminal and visible.
    """

    if not settlement_pause_enabled(reason):
        return legacy_interruption_decision(reason)
    # Attribute-only, before any database read: a connectivity failure
    # reading eligibility must not defer a run that was never going to pause.
    # A status-only caller's paused run is worth having only when automatic
    # resume will pick it up.
    if is_status_only_source(task.source) and not get_task_auto_resume_enabled():
        return legacy_interruption_decision(reason)
    try:
        eligibility = auto_recovery_eligibility(db, task)
        if not eligibility.eligible or not _pausable_with_infra_switch(
            reason, eligibility.kind
        ):
            return legacy_interruption_decision(reason)
        candidate = _settlement_candidate(task)
        resolution = resolve_checkpoint_recovery_with_data(db, candidate)
    except Exception as exc:
        if not is_database_unavailable(exc):
            logger.exception(
                "task_id=%s component=auto-recovery deciding interrupted run "
                "%s failed; settling it as before",
                task.id,
                task.run_id,
            )
            return legacy_interruption_decision(reason)
        raise InterruptionSettlementDeferred(
            f"task {task.id}: interrupted run {task.run_id} is not decidable "
            f"now ({type(exc).__name__})"
        ) from exc
    outcome = _VERDICT_OUTCOMES.get(resolution.verdict)
    recorded = verdict_interruption_reason(
        resolution.verdict,
        control_state=candidate.control_state,
        recoverable_reason=reason,
    )
    if outcome is None or recorded is None:
        raise InterruptionSettlementDeferred(
            f"task {task.id}: checkpoint of interrupted run {task.run_id} "
            "is not resolvable now"
        )
    return InterruptionDecision(
        outcome,
        recorded,
        resolution if outcome is InterruptionOutcome.PAUSE else None,
        unknown_tool_attempt_id=(
            resolution.tool_attempt_id
            if outcome is InterruptionOutcome.FAIL_UNKNOWN_TOOL_EFFECT
            else None
        ),
    )


def decide_owned_run_interruption(
    db: Session, lease: TaskLease, reason: InterruptionReason
) -> InterruptionDecision | None:
    """Decide the interruption of ``lease``'s run, if it is still RUNNING.

    For a settling path that has not loaded the run's row. When settlement
    does not pause for ``reason`` (:func:`settlement_pause_enabled`), the
    decision is ``LEGACY`` without touching the row, so the settlement issues
    the same statements as without an interruption (plus the metadata row).
    Otherwise the run's row is locked and read first; ``None`` means the lease
    no longer owns a RUNNING row, and the caller settles exactly as it always
    has (its own fenced write decides whether anything changes).
    """

    if not settlement_pause_enabled(reason):
        return legacy_interruption_decision(reason)
    if not lock_task_lease_for_settlement_no_commit(db, lease):
        return None
    task = (
        db.query(Task)
        .filter(
            Task.id == lease.task_id,
            Task.runner_id == lease.runner_id,
            task_lease_attempt_predicate(lease),
            Task.run_id == lease.run_id,
        )
        .first()
    )
    if task is None or task.status != TaskStatus.RUNNING:
        return None
    return decide_interruption(db, task=task, reason=reason)


def apply_interruption_outcome_no_commit(
    db: Session,
    *,
    task: Task,
    decision: InterruptionDecision,
    error: str | None,
) -> RecordedInterruption | None:
    """The tail every settling path shares once it has written the outcome.

    ``task`` must be the row as the caller's settling write left it (PAUSED
    or FAILED, new ``state_version``), in the same transaction. This sets
    the unknown-tool-effect message on a run failed for it, records the
    interruption, and for a pause projects the workforce run and TriggerRun
    as lease recovery does. The caller keeps its own write, lease release,
    FAILED projections, transcript, result fact and commit.

    The recording runs in a SAVEPOINT with the semantics of
    ``record_lease_expiry_interruption_no_commit``: a failure inside it is
    logged and skipped, so the settlement commits exactly as it would without
    the row; a SAVEPOINT that cannot be opened, released or rolled back
    propagates. For a status-only source that settles PAUSED, a failed
    recording is followed by a minimal ``manual`` row for the run in a second
    SAVEPOINT, so its status-only reader does not wait on a run nothing will
    resume; if that also fails, it settles without the row. Its preconditions
    hold here: the settling write has opened the transaction, and everything
    staged so far is flushed first.
    ``error`` becomes the row's operator-only ``last_error``. The row is
    gated by the settlement switch (``disabled`` when it is off for an eligible
    task and a reason it would pause for). Returns what was recorded, for the
    caller to announce after its commit (``SettlementReport.recovery``);
    ``None`` when the recording was skipped.
    """

    if decision.outcome is InterruptionOutcome.FAIL_UNKNOWN_TOOL_EFFECT:
        setattr(task, "error_message", TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR)
    db.flush()
    recorded: RecordedInterruption | None = None
    savepoint = db.begin_nested()
    try:
        row = record_interruption_no_commit(
            db,
            task=task,
            reason=decision.reason,
            task_status=TaskStatus(task.status),
            interrupted_at=utc_now(),
            progress_marker=(
                resolution_progress_marker(db, int(task.id), decision.resolution)
                if decision.resolution is not None
                else None
            ),
            gated_by_settlement_switches=True,
            last_error=error,
            tool_call_in_flight=(
                decision.resolution is not None
                and resolution_tool_call_in_flight(decision.resolution)
            ),
        )
        snapshot = RecordedInterruption.from_row(row) if row is not None else None
    except Exception:
        savepoint.rollback()
        fallback = decision.pause and is_status_only_source(task.source)
        logger.exception(
            "Recording the settlement interruption of task %s failed; %s",
            task.id,
            "writing a minimal manual row in its place"
            if fallback
            else "settling it without auto-recovery metadata",
        )
        if fallback:
            recorded = _record_status_only_fallback_no_commit(
                db, task=task, reason=decision.reason, error=error
            )
    else:
        savepoint.commit()
        recorded = snapshot
    if decision.pause:
        _project_interruption_pause_no_commit(db, task, recorded)
    return recorded


def _record_status_only_fallback_no_commit(
    db: Session,
    *,
    task: Task,
    reason: InterruptionReason,
    error: str | None,
) -> RecordedInterruption | None:
    """Write the minimal ``manual`` row of a status-only task's PAUSED run.

    Used when :func:`record_interruption_no_commit` failed: the row carries
    no schedule, so nothing resumes the run, and its fence is the task's
    current ``state_version``, so the status-only reader reports the task as
    stopped instead of waiting on it. It runs in its own SAVEPOINT; a failure
    is logged and returns ``None``, leaving the settlement as if no row had
    been written. A SAVEPOINT that cannot be opened, released or rolled back
    propagates.
    """

    if task.run_id is None:
        return None
    savepoint = db.begin_nested()
    try:
        now = utc_now()
        row = db.get(TaskAutoRecovery, task.id)
        if row is None:
            row = TaskAutoRecovery(task_id=task.id)
            db.add(row)
        row.run_id = task.run_id
        row.reason = reason.value
        row.state = TaskAutoRecoveryState.MANUAL.value
        row.state_detail = None
        row.paused_state_version = int(task.state_version or 0)
        row.interrupted_at = now
        row.episode_started_at = now
        row.no_progress_resumes = 0
        row.total_resumes = 0
        row.progress_marker = None
        row.next_attempt_at = None
        row.last_command_id = None
        row.last_error = (
            error[:TASK_AUTO_RECOVERY_LAST_ERROR_MAX_CHARS]
            if error is not None
            else None
        )
        db.flush()
        snapshot = RecordedInterruption.from_row(row)
    except Exception:
        savepoint.rollback()
        logger.exception(
            "Writing the fallback recovery row of task %s failed; "
            "settling it without auto-recovery metadata",
            task.id,
        )
        return None
    savepoint.commit()
    return snapshot


def _project_interruption_pause_no_commit(
    db: Session, task: Task, recorded: RecordedInterruption | None
) -> None:
    """Project an interruption's PAUSED outcome as lease recovery does.

    The workforce run follows the task to ``paused``. A PENDING or RUNNING
    TriggerRun stays RUNNING while automatic resume is scheduled -- the
    resumed run settles it -- and otherwise ends FAILED with the message for
    what was recorded (:func:`trigger_error_for_recorded`); the default names
    a system interruption, not a lease expiry, which keeps its own message.
    """

    from .task_auto_resume import trigger_error_for_recorded
    from .task_orchestrator import sync_trigger_run_status
    from .workforce_runtime import sync_workforce_run_status

    sync_workforce_run_status(db, task, TaskStatus.PAUSED)
    trigger_error = trigger_error_for_recorded(
        recorded, default=TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR
    )
    if trigger_error is not None:
        sync_trigger_run_status(
            db, task, TaskStatus.PAUSED, error_message=trigger_error
        )
