"""Interruption bookkeeping for automatic task recovery.

When a run is interrupted rather than finished, the settling transaction
records why in ``task_auto_recovery`` (one row per task) and appends a
``task_recovery_events`` row. Those rows are metadata only: recording them
changes no task status, control state, error message or lifecycle
projection.

Which outcome a run-settling path (a run that raised, or returned an
unsuccessful result) writes for an interruption is decided by
:func:`decide_interruption`: an eligible run with a recoverable checkpoint
rests PAUSED instead of FAILED, so its user can resume it. Lease-expiry
recovery keeps its own verdict mapping.

The executor (``task_auto_resume``) dispatches only ``scheduled`` rows, and
recording does not schedule yet, so a recorded interruption is never
``scheduled``. Its state says only that nothing automatic will act on the
run: ``manual``
(only the user can act -- resume a PAUSED task; a FAILED one is terminal and
the row just records why), ``ineligible`` (a task kind automatic recovery will
never touch) or ``disabled`` (the interruption would have paused the run but
``XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED`` is off, see
:func:`settlement_pause_enabled`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from ...config import (
    get_shared_task_execution_enabled,
    get_task_infra_failure_pause_enabled,
)
from ...core.agent.checkpoint import checkpoint_progress_marker
from ...core.agent.interruption import (
    InterruptionReason,
    classify_run_failure,
    classify_run_result,
    is_database_unavailable,
)
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
from .task_command_transport import AUTO_RESUME_COMMAND_PREFIX
from .task_execution_controller import TaskControlState
from .task_lease_service import (
    TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR,
    CheckpointRecoveryResolution,
    CheckpointRecoveryVerdict,
    TaskLease,
    TaskLeaseRecoveryCandidate,
    lock_task_lease_for_settlement_no_commit,
    log_unknown_tool_effect_settlement,
    resolve_checkpoint_recovery_with_data,
    task_lease_attempt_predicate,
    utc_now,
)
from .workforce_runtime import extract_workforce_run_id

logger = logging.getLogger(__name__)


def interruption_pause_result() -> dict[str, Any]:
    """The ``execution_settled`` result of a run paused by an interruption.

    The same shape lease recovery stages, so the next turn's context does not
    read the pause as a failed execution. A fresh dict per call: the fact
    writer stores it by reference in the row payload.
    """
    return {"error": None}


# Callers that own their own protocol for a stopped run: SDK and A2A clients,
# external cancellation, and anonymous visitors of a widget or shared link.
_INELIGIBLE_SOURCES = frozenset({"sdk", "a2a", "external", "widget", "shared_link"})
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

    1. SDK, A2A, external, widget and shared-link tasks are ineligible.
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
    """

    run_id = task.run_id
    if run_id is None:
        return None

    eligibility = auto_recovery_eligibility(db, task)
    # Ineligibility is permanent and the switch is not, so it wins: a
    # ``disabled`` row always means "turning the switch on would help". A
    # reason settlement does not pause for yet is not disabled -- no switch
    # helps it -- so it records ``manual``, like any FAILED run.
    if not eligibility.eligible:
        state, state_detail = TaskAutoRecoveryState.INELIGIBLE, eligibility.detail
    elif (
        gated_by_settlement_switches
        # The invariant, local to this line: ``disabled`` means turning the
        # infra switch on would pause it -- false for a deferred reason, which
        # no switch pauses yet.
        and reason not in SETTLEMENT_PAUSE_DEFERRED_REASONS
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
    db.flush()
    logger.info(
        "Recorded task interruption: task_id=%s run_id=%s reason=%s state=%s "
        "component=auto-recovery",
        task.id,
        run_id,
        reason.value,
        state.value,
    )
    return row


def is_auto_resume_command_id(command_id: object) -> bool:
    """Whether ``command_id`` is in the namespace reserved for the sweeper.

    ``stage_task_command`` refuses the prefix to every caller but the
    sweeper, so a staged command with it is the sweeper's own.
    """

    return isinstance(command_id, str) and command_id.startswith(
        AUTO_RESUME_COMMAND_PREFIX
    )


def auto_resume_schedulable(kind: str | None) -> bool:
    """Whether an eligible task of ``kind`` may be resumed automatically now.

    Channel tasks are recorded but not scheduled until a resumed channel run
    can hold the channel and deliver its answer back (PR9): resuming one
    today would refuse new channel messages as busy while the answer never
    reached the channel.
    """

    return kind is not None and kind != "channel"


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
    from sqlalchemy import and_, select

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
                TaskExecutionCommand.command_id.startswith(AUTO_RESUME_COMMAND_PREFIX),
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

    Deliberately minimal: ``reason`` and ``interrupted_at`` are written by the
    same transaction as the task's own PAUSED write, so a cache keyed on the
    task's ``updated_at`` (the legacy history replay's) cannot serve a stale
    view. ``state`` is left out on purpose -- it changes without a task write
    and is not a client contract yet -- as are the operator-only
    ``last_error``, ``state_detail``, counters and command ids.

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
    return {
        "reason": row.reason,
        "interrupted_at": format_datetime_for_api(row.interrupted_at),
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


def record_lease_expiry_interruption_no_commit(
    db: Session,
    *,
    task: Task,
    candidate: TaskLeaseRecoveryCandidate,
    resolution: CheckpointRecoveryResolution,
    task_status: TaskStatus,
    recovered_at: datetime,
) -> None:
    """Record a lease recovery's interruption inside its open transaction.

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
        return
    savepoint = db.begin_nested()
    try:
        record_interruption_no_commit(
            db,
            task=task,
            reason=reason,
            task_status=task_status,
            interrupted_at=recovered_at,
            progress_marker=resolution_progress_marker(db, int(task.id), resolution),
            gated_by_settlement_switches=False,
        )
    except Exception:
        savepoint.rollback()
        logger.exception(
            "Recording the lease-expiry interruption of task %s failed; "
            "recovering it without auto-recovery metadata",
            task.id,
        )
        return
    savepoint.commit()


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
# Every one of them is recorded; ``SETTLEMENT_PAUSE_DEFERRED_REASONS`` says
# which of them settlement does not pause for yet.
SETTLEMENT_INTERRUPTION_REASONS = frozenset(
    {
        InterruptionReason.PERSISTENCE_FAILURE,
        InterruptionReason.LLM_UNAVAILABLE,
        InterruptionReason.MODEL_OUTPUT_INVALID,
    }
)


# Recorded, but never paused for until the automatic-resume executor ships;
# that change removes them from this set. ``model_output_invalid`` is not an
# infrastructure failure: a run paused for it is worth stopping only when
# something will resample it, and until then it fails exactly as before.
SETTLEMENT_PAUSE_DEFERRED_REASONS = frozenset({InterruptionReason.MODEL_OUTPUT_INVALID})


def settlement_pause_enabled(reason: InterruptionReason) -> bool:
    """Whether settlement may pause an eligible run interrupted for ``reason``.

    Never for a reason in ``SETTLEMENT_PAUSE_DEFERRED_REASONS``. Otherwise it
    is ``XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED``, which is off by default
    until automatic resume ships, so by default every reason settles as
    before.
    """

    if reason in SETTLEMENT_PAUSE_DEFERRED_REASONS:
        return False
    return get_task_infra_failure_pause_enabled()


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
    """

    outcome: InterruptionOutcome
    reason: InterruptionReason
    resolution: CheckpointRecoveryResolution | None = None

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
       the switch is off, or the reason is deferred), or an ineligible task:
       ``LEGACY``.
    2. The run's checkpoint, resolved as lease recovery resolves it:
       ``UNKNOWN_TOOL_EFFECT`` and ``NOT_RECOVERABLE`` fail, ``RECOVERABLE``
       pauses -- as a user pause when the run was PAUSE_REQUESTED (the user
       already decided), else for ``reason``.
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
    try:
        if not auto_recovery_eligibility(db, task).eligible:
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
    if outcome is InterruptionOutcome.FAIL_UNKNOWN_TOOL_EFFECT:
        log_unknown_tool_effect_settlement(candidate.task_id, candidate.run_id)
    return InterruptionDecision(
        outcome,
        recorded,
        resolution if outcome is InterruptionOutcome.PAUSE else None,
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
) -> None:
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
    propagates. Its preconditions hold here: the settling write has opened
    the transaction, and everything staged so far is flushed first.
    ``error`` becomes the row's operator-only ``last_error``. The row is
    gated by the settlement switch (``disabled`` when it is off for an eligible
    task and a reason it would pause for).
    """

    if decision.outcome is InterruptionOutcome.FAIL_UNKNOWN_TOOL_EFFECT:
        setattr(task, "error_message", TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR)
    db.flush()
    savepoint = db.begin_nested()
    try:
        record_interruption_no_commit(
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
        )
    except Exception:
        savepoint.rollback()
        logger.exception(
            "Recording the settlement interruption of task %s failed; "
            "settling it without auto-recovery metadata",
            task.id,
        )
    else:
        savepoint.commit()
    if decision.pause:
        _project_interruption_pause_no_commit(db, task)


def _project_interruption_pause_no_commit(db: Session, task: Task) -> None:
    """Project an interruption's PAUSED outcome as lease recovery does.

    The workforce run follows the task to ``paused``; a PENDING or RUNNING
    TriggerRun ends FAILED with the manual-resume message, since nothing
    resumes the task without its user yet. The message names a system
    interruption, not a lease expiry, which keeps its own message.
    """

    from .task_orchestrator import sync_trigger_run_status
    from .workforce_runtime import sync_workforce_run_status

    sync_workforce_run_status(db, task, TaskStatus.PAUSED)
    sync_trigger_run_status(
        db,
        task,
        TaskStatus.PAUSED,
        error_message=TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR,
    )
