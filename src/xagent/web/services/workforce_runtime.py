from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import case, func
from sqlalchemy.orm import Session, contains_eager, selectinload

from xagent.web.models.task import Task, TaskStatus

from ..models.workforce import Workforce, WorkforceAgent, WorkforceRun
from .task_lease_service import (
    TaskLease,
    release_current_runner_task_lease,
    release_task_lease,
)
from .workforce_snapshot import (
    WORKFORCE_CONFIG_FINGERPRINT_VERSION,
    build_agent_tool_overrides,
    compute_live_workforce_config_fingerprint,
)

logger = logging.getLogger(__name__)

# Run statuses that still hold (or can reclaim) execution resources.
ACTIVE_WORKFORCE_RUN_STATUSES = frozenset({"pending", "running", "paused"})


class WorkforceTurnRejectedError(Exception):
    """A new turn on a workforce task must not start.

    Raised by :func:`ensure_workforce_turn_allowed` when the owning workforce
    was archived or its live config no longer matches the run's pinned
    fingerprint. The turn orchestrator maps this onto its transport-facing
    ``TaskTurnError`` with the same reason string.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class WorkforceTaskRuntime:
    workforce_run_id: int
    workforce_id: int
    snapshot: dict[str, Any]
    allowed_agent_ids: list[int]
    agent_tool_overrides: dict[int, dict[str, Any]]
    worker_tool_names: set[str]
    manager_system_prompt: str | None
    manager_agent_id: int | None
    enable_global_agent_tools: bool = False
    allow_cross_user_agent_ids: bool = True

    @property
    def agent_call_stack(self) -> list[int]:
        return [self.manager_agent_id] if self.manager_agent_id is not None else []


def extract_workforce_run_id(task: Any) -> int | None:
    agent_config = getattr(task, "agent_config", None)
    if not isinstance(agent_config, dict):
        return None
    workforce_run_id = agent_config.get("workforce_run_id")
    return workforce_run_id if isinstance(workforce_run_id, int) else None


def is_workforce_task(task: Any) -> bool:
    agent_config = getattr(task, "agent_config", None)
    return isinstance(agent_config, dict) and isinstance(
        agent_config.get("workforce_run_id"), int
    )


def resolve_workforce_task_runtime(
    db: Session,
    task: Any,
) -> WorkforceTaskRuntime | None:
    workforce_run_id = extract_workforce_run_id(task)
    if workforce_run_id is None:
        return None

    task_id = getattr(task, "id", None)
    user_id = getattr(task, "user_id", None)
    if task_id is None or user_id is None:
        return None

    run = (
        db.query(WorkforceRun)
        .filter(
            WorkforceRun.id == workforce_run_id,
            WorkforceRun.task_id == int(task_id),
            WorkforceRun.user_id == int(user_id),
        )
        .first()
    )
    if run is None or not isinstance(run.snapshot, dict):
        return None

    snapshot = run.snapshot
    workforce_data = snapshot.get("workforce")
    manager_data = snapshot.get("manager")
    workers_data = snapshot.get("workers")
    if not isinstance(workforce_data, dict) or not isinstance(manager_data, dict):
        return None
    if not isinstance(workers_data, list):
        return None

    allowed_agent_ids: list[int] = []
    for worker in workers_data:
        if not isinstance(worker, dict) or worker.get("enabled") is False:
            continue
        agent_id = worker.get("agent_id")
        if isinstance(agent_id, int):
            allowed_agent_ids.append(agent_id)

    if not allowed_agent_ids:
        return None

    allowed_agent_id_set = set(allowed_agent_ids)
    overrides = {
        agent_id: override
        for agent_id, override in build_agent_tool_overrides(
            snapshot, workforce_run_id=workforce_run_id
        ).items()
        if agent_id in allowed_agent_id_set
    }
    worker_tool_names = {
        str(override["tool_name"])
        for override in overrides.values()
        if isinstance(override.get("tool_name"), str)
    }
    workforce_id = workforce_data.get("id")
    manager_agent_id = manager_data.get("agent_id")
    manager_system_prompt = manager_data.get("runtime_prompt")

    return WorkforceTaskRuntime(
        workforce_run_id=workforce_run_id,
        workforce_id=int(workforce_id) if isinstance(workforce_id, int) else 0,
        snapshot=snapshot,
        allowed_agent_ids=allowed_agent_ids,
        agent_tool_overrides=overrides,
        worker_tool_names=worker_tool_names,
        manager_system_prompt=manager_system_prompt
        if isinstance(manager_system_prompt, str)
        else None,
        manager_agent_id=manager_agent_id
        if isinstance(manager_agent_id, int)
        else None,
    )


def _load_workforce_for_fingerprint(db: Session, workforce_id: int) -> Workforce | None:
    return (
        db.query(Workforce)
        .options(
            selectinload(Workforce.manager_agent),
            selectinload(Workforce.workers).selectinload(WorkforceAgent.agent),
        )
        .filter(Workforce.id == int(workforce_id))
        .first()
    )


def ensure_workforce_turn_allowed(
    db: Session,
    *,
    task_id: int,
    task_owner_user_id: int,
    agent_config: dict[str, Any] | None = None,
) -> None:
    """Gate a new turn on a workforce task against the live workforce state.

    Called at the shared turn-entry point for every turn kind. CREATE turns
    are already validated upstream by ``validate_workforce_for_run``, but an
    archive can commit between ``create_workforce_run``'s commit and the
    claim — its cancellation sweep marks the run cancelled yet cannot stop a
    turn that never started — so this check is the last line before
    execution. Rejects with :class:`WorkforceTurnRejectedError` when:

    - ``workforce_archived``: the owning workforce was archived (or its row
      is gone). Archive terminates external exposure; long-lived sessions
      must not keep executing past it.
    - ``workforce_config_changed``: the live config no longer matches the
      fingerprint pinned in the run snapshot. The snapshot only freezes
      prompt-building data while worker execution re-reads live Agent rows,
      so a drifted config silently changes behavior mid-session; reject and
      require a fresh session instead.
    - ``workforce_run_not_found``: ``agent_config`` names a run that does
      not exist (or belongs to another task/user). The runtime resolver
      would silently degrade such a task to a bare manager agent with zero
      delegation ability; fail loudly instead of executing it.
    - ``workforce_run_not_active``: the run was cancelled by the archive path
      or the stale-preview-run reaper. Flipping the status column to
      ``cancelled`` alone enforces nothing without this check -- a RESUME (or
      any other new turn) on an already-cancelled run would otherwise
      proceed. Checked as ``status == "cancelled"`` specifically, not
      "not active": this guard runs before the run is re-synced to
      ``running`` for the turn being claimed, so ``completed``/``failed`` (a
      normal in-progress conversation's resting state between turns) must
      stay allowed.

    No-op for non-workforce tasks and for runs whose snapshot predates the
    fingerprint (backwards compatibility). Preview runs skip the fingerprint
    check: the builder edits config while previewing by design.

    ``agent_config`` may be passed by callers that already read the task row
    (the turn orchestrator reads it in its post-claim snapshot SELECT) to
    save a redundant round-trip; when omitted it is read here, scoped to the
    owner.
    """
    if agent_config is None:
        row = (
            db.query(Task.agent_config)
            .filter(Task.id == int(task_id), Task.user_id == int(task_owner_user_id))
            .first()
        )
        if row is None or not isinstance(row[0], dict):
            return
        agent_config = row[0]
    workforce_run_id = agent_config.get("workforce_run_id")
    if not isinstance(workforce_run_id, int):
        return

    run = (
        db.query(WorkforceRun)
        .filter(
            WorkforceRun.id == workforce_run_id,
            WorkforceRun.task_id == int(task_id),
            WorkforceRun.user_id == int(task_owner_user_id),
        )
        .first()
    )
    if run is None:
        raise WorkforceTurnRejectedError("workforce_run_not_found")

    # A run cancelled by the archive path or the stale-preview-run reaper
    # (see cancel_active_workforce_runs / reap_stale_preview_workforce_runs)
    # flips WorkforceRun.status to "cancelled" but has no other enforcement:
    # neither is a PAUSE dispatch guaranteed to land, nor does any other
    # turn-entry check inspect this status. Without this, a RESUME (or any
    # new turn) on an already-"cancelled" run would proceed indefinitely.
    #
    # This must check for "cancelled" specifically, NOT "not an active
    # status": this guard runs BEFORE sync_workforce_run_status projects the
    # claimed Task's RUNNING status onto the run (see the caller,
    # _begin_turn_atomic's ensure_workforce_turn_allowed call, which precedes
    # its own sync_workforce_run_status(..., RUNNING) a few lines later) --
    # so at this point a perfectly normal in-progress conversation's run sits
    # at "completed" (or "failed") from the PREVIOUS turn, exactly the
    # statuses TaskTurnOrchestrator's _APPENDABLE_STATUSES allows a new turn
    # to resume from. Rejecting "not active" here would reject every
    # second-and-later message in every workforce conversation.
    #
    # Ephemeral preview runs (test-before-save in the builder) have no
    # persisted Workforce to check archive/drift against -- the snapshot on
    # the run row is the only source of truth for their whole lifetime, so
    # the cancelled-check is their only enforcement and must apply here,
    # before the early return.
    if run.workforce_id is None:
        if run.status == "cancelled":
            raise WorkforceTurnRejectedError("workforce_run_not_active")
        return

    workforce = _load_workforce_for_fingerprint(db, int(run.workforce_id))
    if workforce is None or workforce.status == "archived":
        raise WorkforceTurnRejectedError("workforce_archived")

    # Checked after the archive check (not before): the archive endpoint
    # cancels every in-flight run in the SAME transaction it flips
    # Workforce.status to "archived", so a run cancelled that way already
    # got the more specific, established "workforce_archived" reason above.
    # This catches the other way a real (non-preview) run ends up
    # "cancelled" while its workforce stays active: the stale-preview-run
    # reaper also reaps edit-mode preview runs of an already-saved,
    # still-active workforce (see reap_stale_preview_workforce_runs).
    if run.status == "cancelled":
        raise WorkforceTurnRejectedError("workforce_run_not_active")

    if bool(run.is_preview):
        return
    snapshot: dict[str, Any] = run.snapshot if isinstance(run.snapshot, dict) else {}
    pinned = snapshot.get("config_fingerprint")
    if not isinstance(pinned, str) or not pinned:
        return
    # Only compare fingerprints pinned by the CURRENT algorithm: a version
    # bump (algorithm change) would otherwise make every pre-existing run's
    # pinned value mismatch the freshly computed live one and spuriously
    # reject those sessions on deploy — the exact failure the fingerprint
    # exists to prevent. Runs pinned under an older version are exempt.
    if snapshot.get("config_fingerprint_version") != (
        WORKFORCE_CONFIG_FINGERPRINT_VERSION
    ):
        return
    live = compute_live_workforce_config_fingerprint(workforce)
    if live != pinned:
        raise WorkforceTurnRejectedError("workforce_config_changed")


@dataclass(frozen=True)
class WorkforceRunPauseTarget:
    """A previously RUNNING task that still needs a PAUSE after archive."""

    run_id: int
    task_id: int
    # WorkforceRun.user_id is nullable=False, so the owner is always
    # available here -- carried per-target (rather than a single shared
    # value on the dispatch call) since a reap sweep's targets can span
    # multiple owners, unlike a single archive's batch.
    actor_user_id: int


def _cancel_workforce_run_rows(
    runs: list[WorkforceRun],
) -> list[WorkforceRunPauseTarget]:
    """Flip a batch of already-loaded runs to terminal ``cancelled`` in-place.

    Shared by the workforce-archive cancel path and the stale-preview-run
    reaper below; callers differ only in how ``runs`` was selected and
    whether they commit inline or hand the transaction to their own caller.
    """
    pause_targets: list[WorkforceRunPauseTarget] = []
    for run in runs:
        task = run.task
        if task is not None and task.status == TaskStatus.RUNNING:
            pause_targets.append(
                WorkforceRunPauseTarget(
                    run_id=int(run.id),
                    task_id=int(task.id),
                    actor_user_id=int(run.user_id),
                )
            )
        setattr(run, "status", "cancelled")
        if run.completed_at is None:
            setattr(run, "completed_at", datetime.now(timezone.utc))
    return pause_targets


def cancel_active_workforce_runs(
    db: Session,
    workforce_id: int,
) -> list[WorkforceRunPauseTarget]:
    """Mark every in-flight run of a workforce terminal ``cancelled``.

    Archiving only flips ``Workforce.status``; without this, in-flight runs
    keep executing because turn resolution never re-checks live workforce
    state. ``cancelled`` is non-overwritable in ``sync_workforce_run_status``,
    so a PAUSE landing later cannot flip the run back to ``paused``. New
    turns are rejected separately by ``ensure_workforce_turn_allowed``.

    Deliberately does NOT commit and does NOT touch the command transport:
    the caller commits the archive flip and these cancellations in one
    atomic transaction, then dispatches PAUSE to the returned targets via
    :func:`pause_workforce_tasks_after_archive`. (The durable enqueue commits
    internally, so calling it on this session mid-loop would leak a partial
    archive state.)
    """
    runs = (
        db.query(WorkforceRun)
        .options(selectinload(WorkforceRun.task))
        .filter(
            WorkforceRun.workforce_id == int(workforce_id),
            WorkforceRun.status.in_(ACTIVE_WORKFORCE_RUN_STATUSES),
        )
        .all()
    )
    return _cancel_workforce_run_rows(runs)


def reap_stale_preview_workforce_runs(
    db: Session,
    *,
    stale_after_seconds: int | None = None,
    limit: int = 100,
) -> list[WorkforceRunPauseTarget]:
    """Cancel abandoned workforce-builder preview runs.

    Preview runs (``is_preview IS TRUE``, the builder's "test before save")
    are invisible to archiving and :func:`cancel_active_workforce_runs`
    either way: a create-mode draft has no saved Workforce at all
    (``workforce_id IS NULL``), and an edit-mode preview against an
    already-saved workforce has a real ``workforce_id`` but is still not a
    normal run the archive-cancel query selects -- filtering on
    ``workforce_id IS NULL`` alone would only reap the former and leave the
    latter (arguably the more common case, since editing an existing
    workforce is the primary post-launch workflow) with no cleanup path at
    all. The frontend clears its own reference whenever the draft changes,
    but that is a client-side signal only -- a closed tab, crashed browser,
    or network drop leaves the run (and its hidden Task) active server-side
    with no owner left to invalidate it. This is the server-side backstop: a
    scheduled sweep, mirroring
    :func:`~.background_jobs.requeue_stale_background_jobs`, that reaps rows
    still non-terminal past ``stale_after_seconds``.

    Unlike :func:`cancel_active_workforce_runs`, this commits its own
    cancellations -- it is the only state change in its transaction (no
    sibling archive flip to stay atomic with) -- and returns pause targets
    for the caller to dispatch PAUSE to, same as the archive path.
    """
    from ...config import get_workforce_preview_run_stale_seconds

    stale_seconds = (
        stale_after_seconds
        if stale_after_seconds is not None
        else get_workforce_preview_run_stale_seconds()
    )
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)
    # last_activity_at is bumped by sync_workforce_run_status on every turn
    # (PR review round 8, F-NEW-1); created_at alone stays fixed for the run's
    # whole lifetime, so keying staleness off it could reap a conversation
    # that's genuinely still going. Falls back to created_at for the (should
    # be unreachable given the column's server_default) case a row's
    # last_activity_at is somehow NULL, rather than never reaping it at all.
    run_activity_marker = func.coalesce(
        WorkforceRun.last_activity_at, WorkforceRun.created_at
    )
    # sync_workforce_run_status only bumps last_activity_at at turn
    # boundaries (start/end), so a single turn that runs longer than the
    # stale window on its own -- e.g. one long tool-heavy execution -- would
    # otherwise still look stale mid-execution. Task.last_heartbeat_at is
    # refreshed roughly every ~20s for the whole duration of an active
    # execution (task_lease_service's heartbeat loop) and is the more direct
    # "is this actually still running right now" signal, so the newer of the
    # two is what staleness is keyed off (self-review finding after round 8).
    # A CASE expression, not func.max()/func.greatest(): SQLite's multi-arg
    # max() returns NULL if *any* argument is NULL (unlike Postgres'
    # GREATEST, which ignores NULLs), and Postgres has no scalar 2-arg
    # max() at all -- this codebase supports both, so neither is portable
    # here.
    activity_marker = case(
        (run_activity_marker.is_(None), Task.last_heartbeat_at),
        (Task.last_heartbeat_at.is_(None), run_activity_marker),
        (run_activity_marker >= Task.last_heartbeat_at, run_activity_marker),
        else_=Task.last_heartbeat_at,
    )

    runs = (
        db.query(WorkforceRun)
        .outerjoin(Task, WorkforceRun.task_id == Task.id)
        .options(contains_eager(WorkforceRun.task))
        .filter(
            WorkforceRun.is_preview.is_(True),
            WorkforceRun.status.in_(ACTIVE_WORKFORCE_RUN_STATUSES),
            # NULL-safe on its own: coalesce(NULL, NULL) <= cutoff is UNKNOWN,
            # which the WHERE clause already excludes -- an explicit not-null
            # guard on created_at specifically would be checking the wrong
            # column now that the comparison runs against activity_marker,
            # not created_at directly (self-review finding after round 8).
            activity_marker <= cutoff,
        )
        .order_by(activity_marker.asc())
        .limit(max(1, min(limit, 500)))
        .all()
    )
    if not runs:
        return []

    pause_targets = _cancel_workforce_run_rows(runs)
    db.commit()
    return pause_targets


async def pause_workforce_tasks_after_archive(
    pause_targets: list[WorkforceRunPauseTarget],
    *,
    reason: str = "archive",
) -> None:
    """Best-effort PAUSE dispatch for tasks left running by a cancellation.

    Runs AFTER the caller committed its cancel transaction (archive flip,
    or the stale-preview-run reaper's own commit), on its own short-lived
    sessions, so the durable enqueue's internal commit can never leak a
    partial state from the caller's transaction. A failed pause is logged
    and skipped: the run is already terminal and the turn-entry guard blocks
    new turns, so the orphaned execution can only run its current turn to
    completion.

    Each target's own ``actor_user_id`` (the run's owner) is used for the
    dispatched command, not a single shared actor for the whole batch -- a
    reap sweep's targets can span multiple owners, unlike a single archive.

    ``reason`` is a short label (e.g. ``"archive"``, ``"preview-reap"``)
    distinguishing why the cancellation happened, used only for the log
    message -- deliberately NOT part of the command id (see below), so it
    has no effect on dispatch behavior.
    """
    if not pause_targets:
        return

    from ..models.database import get_session_local
    from .task_command_transport import (
        TaskCommandKind,
        dispatch_task_command_promptly,
        enqueue_task_command,
    )

    SessionLocal = get_session_local()
    for target in pause_targets:
        try:
            with SessionLocal() as command_db:
                enqueued = enqueue_task_command(
                    command_db,
                    task_id=target.task_id,
                    actor_user_id=target.actor_user_id,
                    # Keyed on run_id alone (not reason): the archive path
                    # (cancel_active_workforce_runs) and the reap sweep
                    # (reap_stale_preview_workforce_runs, which can now also
                    # select an edit-mode preview run of an already-saved
                    # workforce) can race and both select the SAME run for
                    # cancellation. enqueue_task_command's (task_id,
                    # command_id) dedup only catches that if both call sites
                    # produce the identical id -- embedding `reason` would
                    # defeat the exact idempotency protection this is for.
                    command_id=f"workforce-pause-{target.run_id}",
                    kind=TaskCommandKind.PAUSE,
                    payload={},
                )
            from ..api.websocket import execute_durable_task_command

            await dispatch_task_command_promptly(
                execute_durable_task_command,
                command_db_id=enqueued.command_id,
            )
        except Exception:
            logger.warning(
                "Failed to pause running task %s (workforce run %s, %s)",
                target.task_id,
                target.run_id,
                reason,
                exc_info=True,
            )


def _map_task_status(status: Any) -> str | None:
    if isinstance(status, str):
        try:
            status = TaskStatus(status)
        except ValueError:
            return None
    if status == TaskStatus.PENDING:
        return "pending"
    if status == TaskStatus.RUNNING:
        return "running"
    if status in {TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER}:
        return "paused"
    if status == TaskStatus.COMPLETED:
        return "completed"
    if status == TaskStatus.FAILED:
        return "failed"
    return None


def sync_workforce_run_status(
    db: Session, task: Any, status: Any | None = None
) -> bool:
    workforce_run_id = extract_workforce_run_id(task)
    mapped_status = _map_task_status(status if status is not None else task.status)
    if workforce_run_id is None or mapped_status is None:
        return False

    task_id = getattr(task, "id", None)
    user_id = getattr(task, "user_id", None)
    if task_id is None or user_id is None:
        return False

    run = (
        db.query(WorkforceRun)
        .filter(
            WorkforceRun.id == workforce_run_id,
            WorkforceRun.task_id == int(task_id),
            WorkforceRun.user_id == int(user_id),
        )
        .first()
    )
    if run is None:
        return False

    # "cancelled" is terminal and only ever set explicitly (workforce
    # archive). A late task-status projection (e.g. the PAUSE issued during
    # archive landing as "paused") must not resurrect the run.
    if run.status == "cancelled":
        return False

    changed = False
    if run.status != mapped_status:
        setattr(run, "status", mapped_status)
        changed = True

    if mapped_status in {"completed", "failed", "cancelled"}:
        if run.completed_at is None:
            setattr(run, "completed_at", datetime.now(timezone.utc))
            changed = True
    elif run.completed_at is not None:
        setattr(run, "completed_at", None)
        changed = True

    if changed:
        # Explicit, not just relying on the column's own onupdate=func.now():
        # this is the one signal the preview-run reaper trusts to tell a
        # conversation that's genuinely still going from one that's simply
        # been open a long time (PR review round 8, F-NEW-1). Only bumped
        # alongside a real transition, so a no-op sync call (status/
        # completed_at already correct) doesn't manufacture activity that
        # didn't happen.
        setattr(run, "last_activity_at", datetime.now(timezone.utc))

    return changed


def mark_workforce_task_status(
    db: Session,
    task: Task,
    status: TaskStatus,
    *,
    error_message: str | None = None,
    clear_output: bool = False,
) -> bool:
    """Update the task lifecycle source of truth and project it to WorkforceRun."""
    from .task_execution_controller import (
        apply_task_control_transition,
        control_state_for_status,
    )

    changed = False
    expected_control_state = control_state_for_status(status)
    if task.status != status or task.control_state != expected_control_state.value:
        apply_task_control_transition(
            task,
            expected_control_state,
            status=status,
        )
        changed = True
    if error_message is not None and task.error_message != error_message:
        setattr(task, "error_message", error_message)
        changed = True
    if clear_output and task.output is not None:
        setattr(task, "output", None)
        changed = True

    return sync_workforce_run_status(db, task, status) or changed


def _sync_workforce_run_status_for_task_id(
    db: Session,
    task_id: int,
    status: TaskStatus,
    *,
    task_lease: TaskLease | None = None,
) -> bool:
    task_query = db.query(Task).filter(Task.id == int(task_id))
    if task_lease is not None:
        if task_lease.task_id != int(task_id) or task_lease.run_id is None:
            return False
        task_query = task_query.filter(
            Task.runner_id == task_lease.runner_id,
            Task.run_id == task_lease.run_id,
        )
    task = task_query.with_for_update().first()
    # Project the locked source-of-truth status, never the caller's stale
    # desired value. The requested status is only an eligibility assertion.
    if task is None or task.status != status:
        return False
    changed = sync_workforce_run_status(db, task, task.status)
    if changed:
        db.commit()
    return changed


def sync_workforce_run_status_for_task_id_isolated(
    task_id: int,
    status: TaskStatus,
    *,
    task_lease: TaskLease | None = None,
) -> bool:
    """Project the locked current task status in a worker-owned Session."""
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        return _sync_workforce_run_status_for_task_id(
            db,
            task_id,
            status,
            task_lease=task_lease,
        )


def release_task_lease_with_workforce_sync(
    db: Session,
    lease: TaskLease | None,
    *,
    status: TaskStatus,
) -> bool:
    released = release_task_lease(db, lease, status=status)
    if not released or lease is None:
        return released
    try:
        _sync_workforce_run_status_for_task_id(db, lease.task_id, status)
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning(
            "Failed to sync workforce run status after task lease release",
            exc_info=True,
        )
    return released


def release_current_runner_task_lease_with_workforce_sync(
    db: Session,
    task_id: int,
    *,
    status: TaskStatus,
    runner_id: str | None = None,
    expected_run_id: str | None = None,
) -> bool:
    released = release_current_runner_task_lease(
        db,
        task_id,
        status=status,
        runner_id=runner_id,
        expected_run_id=expected_run_id,
    )
    if not released:
        return released
    try:
        _sync_workforce_run_status_for_task_id(db, task_id, status)
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning(
            "Failed to sync workforce run status after current runner lease release",
            exc_info=True,
        )
    return released
