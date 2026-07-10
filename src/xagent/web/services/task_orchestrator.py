"""Single source of truth for task turn lifecycle.

Both the WebSocket UI path (``websocket.py:handle_chat_message``) and the
``/v1`` SDK endpoints (``v1/tasks.py``) route through this module. It owns
the parts of the lifecycle that *must* behave identically across both
transports so the same race / state-machine bugs don't grow back on
either side:

  - atomic state transitions (claim a task as RUNNING)
  - user message persistence (``task_chat_messages``)
  - background execution scheduling with a single-flight guard
  - assistant ``task.output`` / ``error_message`` sync after the bg
    coroutine returns

Things this module deliberately does **not** own (each transport keeps
its own adapter):

  - response shapes / error envelopes
    (``{"detail": ...}`` for ``/api/*`` vs ``{"error": {"code", "message"}}``
    for ``/v1/*``)
  - live broadcast events (WS sends ``task_started`` / ``task_completed``;
    SDK doesn't)

Background context — why we replaced the older ``task_execution.py``
helper with this orchestrator:

  - The atomic claim in ``v1/tasks.py`` previously filtered on
    ``status != RUNNING``, which let a brand-new PENDING task be
    claimed by an immediate follow-up ``POST /messages`` before the bg
    coroutine ever ran. Two bg coroutines could end up racing the same
    transcript and task.output.
  - ``background_task_manager.register_task`` overwrites the previous
    handle for a given ``task_id``. Combined with
    ``wait_for_previous``'s ``is current_task`` short-circuit, two
    concurrent kickoffs would each register themselves as "previous"
    and skip waiting. The orchestrator's ``_refuse_if_bg_inflight``
    closes this from the caller side.

Both races are prevented by funneling the WebSocket and /v1 transports
through this single turn-lifecycle chokepoint -- the atomic claim
filter and ``_refuse_if_bg_inflight`` guard close them at the
orchestrator boundary.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..models.task import Task, TaskStatus
from .chat_history_service import mark_user_message_delivery_sync
from .hot_path_cache import invalidate_task_cache
from .task_lease_service import (
    acquire_task_lease_isolated,
    get_runner_id,
    run_task_lease_heartbeat,
)
from .task_setup_snapshot import load_task_setup_snapshot_sync
from .workforce_runtime import release_current_runner_task_lease_with_workforce_sync

logger = logging.getLogger(__name__)


# Statuses for the "can a user message start the next turn?" check. A
# task in any of these is eligible for ``TurnKind.APPEND``. PENDING is
# claimed by ``CREATE``; RUNNING is still busy; WAITING_FOR_USER is an
# answer to an explicit pending agent question and resumes that execution.
_APPENDABLE_STATUSES = (
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.PAUSED,
)


@dataclass(frozen=True)
class TaskTurnPayload:
    """Both message representations a single turn carries.

    A turn has two distinct message channels, and collapsing them into
    a single string loses the WS file-context input on its way to the
    LLM:

    - ``transcript_message`` — what gets persisted to
      ``task_chat_messages`` and shown back to the user / GET endpoint
    - ``execution_message`` — what the agent / LLM actually consumes;
      may be file-enriched, system-prefix-augmented, etc.

    When ``execution_message`` is ``None``, ``for_agent`` falls back to
    ``transcript_message`` (typical for SDK callers which only have one
    representation). WS callers pass both because the file-context
    append for the LLM input is intentionally not shown verbatim in the
    transcript.
    """

    transcript_message: str
    execution_message: Optional[str] = None
    # Per-turn uploaded-file metadata persisted alongside the transcript
    # row so historical replay can render the same clickable chips the
    # user saw live. Each entry is the minimal chip shape (file_id,
    # name, size, type) — already path-stripped by the websocket layer
    # before reaching here.
    attachments: Optional[List[Dict[str, Any]]] = None
    # Stable identity shared by the transcript row and the user_message trace
    # event for this user turn. Historical replay uses it to merge persisted
    # transcript rows with trace rows without collapsing repeated text.
    turn_id: str = field(default_factory=lambda: str(uuid4()))

    @property
    def for_agent(self) -> str:
        return self.execution_message or self.transcript_message


class TurnKind(str, enum.Enum):
    """Which transition the turn represents.

    ``kind`` answers "which status filter does the atomic claim use".
    Orthogonal to ``force_fresh`` (passed alongside to ``begin_turn``),
    which answers "does the agent reconstruct prior execution state or
    start fresh". The two cover four logical combinations; only three
    are reachable in practice (CREATE + force_fresh has no meaning
    because a brand-new task has no prior state to discard — see the
    assert in ``begin_turn``).

    Continuation paths (PAUSED / WAITING_FOR_USER resumed onto the same
    turn) are deliberately not modeled here: they go through
    ``dag_pattern.request_continuation`` instead, because continuation
    is the *same* turn picking up where it paused — terminal-field reset
    would be wrong.
    """

    CREATE = "create"  # PENDING → RUNNING; new task's first turn
    APPEND = "append"  # APPENDABLE → RUNNING; new turn on an existing task


class TaskTurnError(Exception):
    """Raised when a turn cannot be started because the task is busy.

    Each transport adapter catches this and maps it to its own error
    shape:

      - ``/v1`` SDK endpoints → ``V1ApiError(TASK_BUSY, 409)``
      - WebSocket handler → broadcast an ``agent_error`` event
    """

    def __init__(self, reason: str = "busy"):
        super().__init__(reason)
        self.reason = reason


class TaskTurnNotFoundError(Exception):
    """Raised when the turn's atomic claim finds no row that both exists
    and is owned by ``task_owner_user_id``.

    Deliberately NOT a subclass of :class:`TaskTurnError`: callers map
    ``TaskTurnError`` to 409 (busy / bg_inflight), but a missing or
    not-owned task is a 404. Keeping it a separate type means a
    ``except TaskTurnError`` clause never silently turns a not-found into
    a "busy". Part of ``begin_turn``'s public error contract.
    """

    def __init__(self, task_id: int):
        super().__init__(f"task {task_id} not found or not owned by the task owner")
        self.task_id = task_id


@dataclass(frozen=True)
class TurnStarted:
    """Result of a started turn.

    Internal orchestration result (not a serialized public DTO): it
    carries the committed-row snapshot the caller needs to build its
    response WITHOUT re-reading the ORM (``begin_turn`` no longer touches
    the caller's session), plus the live bg handle.

    The snapshot fields (``status`` / ``updated_at`` / ``before_message_id``)
    are read inside the isolated worker-thread transaction, so callers do
    not pay an on-loop ``db.refresh`` to learn the post-claim state.
    ``task_source`` is internal (passed to ``_schedule_bg``); ``background_task``
    is the scheduler handle (workforce / tests await it).
    """

    task_id: int
    status: TaskStatus
    updated_at: Optional[datetime]
    before_message_id: Optional[int]
    task_source: Optional[str]
    background_task: "asyncio.Task[None]"


class TaskTurnOrchestrator:
    """Drive one task-turn lifecycle.

    All methods are static; the class is a namespace, not stateful.
    State lives in the database and in the global
    ``background_task_manager``.
    """

    @staticmethod
    async def begin_turn(
        *,
        task_id: int,
        task_owner_user_id: int,
        payload: TaskTurnPayload,
        kind: TurnKind,
        force_fresh: bool = False,
        context: Optional[Dict[str, Any]] = None,
        actor_user_id: Optional[int] = None,
    ) -> TurnStarted:
        """Single entry for any new-turn transition (CREATE / APPEND).

        Owns the full turn-start contract. The atomic write transaction
        (claim + user-message persist + commit) runs on an isolated
        worker-thread session via ``asyncio.to_thread`` so the ~5s write
        RTT does not block the asyncio event loop (issue #427). The caller
        passes primitives only; ``begin_turn`` never touches the caller's
        session, and returns a :class:`TurnStarted` snapshot the caller
        reads instead of re-fetching the ORM.

        Sequence:

          1. ``_refuse_if_bg_inflight`` — reject if a bg coroutine is still
             running for this task (``TaskTurnError("bg_inflight")``). Pure
             in-memory dict check. NOTE: it now sits before the
             ``await asyncio.to_thread`` below, so two concurrent new turns
             can both pass it; the authoritative serializer is the DB atomic
             claim (the status-filter rowcount), which lets exactly one win.
          2. ``_begin_turn_atomic_sync`` (off-loop): atomic claim
             (``id == task_id AND user_id == task_owner_user_id AND
             <status_filter>``) + persist + snapshot SELECT + single commit.
             By construction it only raises BEFORE commit, so a successful
             return means the row is committed RUNNING.
          3. ``_schedule_bg`` (sync, no await) — schedule the lease-aware bg
             coroutine. Once step 2 committed, this must succeed or the task
             is forced FAILED (no zombie RUNNING).

        ``task_owner_user_id`` is the task OWNER's id — the runtime identity
        the turn executes as. Callers derive it from the already-authorized
        task row (``task.user_id``) or the SDK agent's owner
        (``agent.user_id``), NOT from the acting principal. They differ when
        an admin operates on another user's task: authorization happens at
        the entry (e.g. the WS admin bypass), and the turn must still run as
        the owner, not the admin. The claim predicate keeps ``Task.user_id ==
        task_owner_user_id`` as defense-in-depth.

        ``actor_user_id`` is the acting principal that initiated the turn —
        the same as the owner for normal / SDK / workforce flows, but the
        admin's id when an admin acts on another user's task. It is recorded
        for audit/logging only and deliberately does NOT enter the claim,
        snapshot resolution, ``UserContext``, or tool config; the runtime
        always runs as ``task_owner_user_id``.

        Args:
            task_id: The committed task's id.
            task_owner_user_id: The task owner's id (runtime identity). Used
                for the claim predicate, the persisted user message, and the
                whole bg execution context.
            payload: Two-channel message (transcript + execution).
            kind: Which status filter the atomic claim uses.
            force_fresh: When True, the bg coroutine starts a fresh agent
                run (WS terminal re-engage); invalid with ``kind=CREATE``.
            context: Optional execution-context dict.

        Returns:
            :class:`TurnStarted` — committed-row snapshot
            (``status``/``updated_at``/``before_message_id``) plus the bg
            ``background_task`` handle.

        Raises:
            ValueError: ``kind == CREATE and force_fresh``.
            TaskTurnError("bg_inflight"): a previous bg coroutine is running.
            TaskTurnError("busy"): the row exists and is owned but its status
                did not match the claim filter.
            TaskTurnNotFoundError: no row matched id + owner.
        """
        if kind == TurnKind.CREATE and force_fresh:
            raise ValueError(
                "force_fresh has no meaning for kind=CREATE — a new task "
                "has no prior execution state to discard"
            )

        # bg-inflight guard before any DB write (see note 1 in docstring).
        _refuse_if_bg_inflight(task_id)

        # The claim and the schedule must be atomic with respect to
        # cancellation: once the claim commits, the row is RUNNING, so the bg
        # run MUST be scheduled (or the task forced FAILED) -- otherwise a
        # CancelledError landing at the ``to_thread`` resume, after the commit,
        # would strand the row as RUNNING with no worker. Running both inside
        # ``asyncio.shield`` lets them finish even when ``begin_turn``'s caller
        # is cancelled.
        async def _claim_and_schedule() -> tuple[_ClaimedTurn, "asyncio.Task[None]"]:
            # Off-loop atomic claim + persist + commit. Only raises pre-commit
            # (busy / not-found), so a normal exception here means nothing was
            # committed; reaching the schedule means the row is RUNNING.
            res = await asyncio.to_thread(
                _begin_turn_atomic_sync,
                task_id,
                task_owner_user_id,
                payload=payload,
                kind=kind,
            )
            try:
                handle = _schedule_bg(
                    task_id=task_id,
                    task_owner_user_id=task_owner_user_id,
                    task_source=res.task_source,
                    payload=payload,
                    force_fresh=force_fresh,
                    context=context,
                    before_message_id=res.before_message_id,
                )
            except BaseException:
                # Schedule failed after the claim committed -> force FAILED so
                # the row isn't left RUNNING. Off-loop, so the error path also
                # keeps the goal of no synchronous DB on the event loop.
                await asyncio.to_thread(
                    _mark_task_failed_if_running,
                    task_id,
                    "turn scheduling failed after claim commit",
                )
                try:
                    await asyncio.to_thread(
                        mark_user_message_delivery_sync,
                        task_id,
                        payload.turn_id,
                        "failed",
                    )
                except Exception:
                    logger.exception(
                        "Could not mark failed delivery for task=%s turn=%s",
                        task_id,
                        payload.turn_id,
                    )
                raise
            await asyncio.to_thread(
                mark_user_message_delivery_sync,
                task_id,
                payload.turn_id,
                "dispatched",
            )
            return res, handle

        claimed, bg_task = await asyncio.shield(_claim_and_schedule())

        # Audit who initiated the committed turn. The runtime always runs as
        # the owner; this only records the acting principal (an admin when
        # acting on another user's task) and is intentionally not used for any
        # runtime resolution.
        logger.info(
            "turn started: task=%s kind=%s owner=%s actor=%s",
            task_id,
            kind,
            task_owner_user_id,
            actor_user_id if actor_user_id is not None else task_owner_user_id,
        )

        return TurnStarted(
            task_id=task_id,
            status=claimed.status,
            updated_at=claimed.updated_at,
            before_message_id=claimed.before_message_id,
            task_source=claimed.task_source,
            background_task=bg_task,
        )


# ===== internal helpers =====


@dataclass(frozen=True)
class _ClaimedTurn:
    """Snapshot returned by ``_begin_turn_atomic_sync`` after the claim
    commits, so ``begin_turn`` can build :class:`TurnStarted` without the
    caller re-reading the ORM."""

    status: TaskStatus
    updated_at: Optional[datetime]
    before_message_id: Optional[int]
    task_source: Optional[str]


def _begin_turn_atomic_sync(
    task_id: int,
    task_owner_user_id: int,
    *,
    payload: TaskTurnPayload,
    kind: TurnKind,
) -> _ClaimedTurn:
    """Atomic claim + user-message persist + commit on its OWN session.

    Designed to run under ``asyncio.to_thread`` so the synchronous write
    transaction (~5s RTT measured in issue #427) stays off the event loop.
    Opens / commits / closes its own ``SessionLocal`` — never touches the
    caller's session.

    Owner is folded into the claim predicate (``id`` AND ``user_id`` AND
    status filter) so the UPDATE is atomic w.r.t. the owner. On ``rowcount
    0`` a diagnostic SELECT distinguishes:

      - row missing / not owned by ``task_owner_user_id`` →
        :class:`TaskTurnNotFoundError`
      - row exists + owned but wrong status → ``TaskTurnError("busy")``

    Invariant relied on by ``begin_turn``: this function only raises BEFORE
    ``commit``. The committed-row snapshot is SELECTed pre-commit
    (read-your-writes within the transaction; a bulk
    ``.update(synchronize_session=False)`` leaves no ORM object to refresh),
    and the only post-commit work — ``invalidate_task_cache`` — is
    best-effort. So a successful return always means the row is committed
    RUNNING, and any exception means it is not.
    """
    from ..models.database import get_session_local
    from .chat_history_service import DELIVERY_PENDING, persist_user_message_no_commit

    if kind == TurnKind.CREATE:
        status_filter = Task.status == TaskStatus.PENDING
    else:  # APPEND
        status_filter = Task.status.in_(_APPENDABLE_STATUSES)

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        claimed = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.user_id == task_owner_user_id,
                status_filter,
            )
            .update(
                {
                    Task.status: TaskStatus.RUNNING,
                    Task.input: payload.transcript_message,
                    Task.output: None,
                    Task.error_message: None,
                    # A new turn owns the lifecycle from this claim forward.
                    # Stale runner_id / lease columns left by a crashed worker
                    # or a release that failed to clear would otherwise make
                    # acquire_task_lease deny this worker even though no live
                    # runner is executing -- leaving the SDK row stuck RUNNING
                    # (GET /v1/chat/tasks/{id} never reaches completed; append
                    # returns task_busy). Reset here atomically with the claim.
                    Task.runner_id: None,
                    Task.lease_expires_at: None,
                    Task.last_heartbeat_at: None,
                },
                synchronize_session=False,
            )
        )
        if claimed == 0:
            db.rollback()
            owned = (
                db.query(Task.id)
                .filter(Task.id == task_id, Task.user_id == task_owner_user_id)
                .first()
            )
            if owned is None:
                raise TaskTurnNotFoundError(task_id)
            raise TaskTurnError("busy")

        persisted_message = persist_user_message_no_commit(
            db=db,
            task_id=task_id,
            user_id=task_owner_user_id,
            content=payload.transcript_message,
            attachments=payload.attachments,
            turn_id=payload.turn_id,
            delivery_status=DELIVERY_PENDING,
        )
        if persisted_message is not None:
            db.flush()
            before_message_id: Optional[int] = int(persisted_message.id)
        else:
            before_message_id = None

        # Snapshot the committed row's columns BEFORE commit (read-your-writes
        # in the same transaction). Keeps commit as the last fallible DB op,
        # so there is no post-commit window where the row is RUNNING but this
        # helper still raises.
        status, updated_at, source = (
            db.query(Task.status, Task.updated_at, Task.source)
            .filter(Task.id == task_id)
            .one()
        )

        db.commit()
    except (TaskTurnError, TaskTurnNotFoundError):
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    # Best-effort: a stale cache entry self-heals on the next write / TTL,
    # and must never strand a committed RUNNING task by raising here.
    try:
        invalidate_task_cache(task_id)
    except Exception:
        logger.warning(
            "invalidate_task_cache failed for task %s (non-fatal)",
            task_id,
            exc_info=True,
        )

    return _ClaimedTurn(
        status=status,
        updated_at=updated_at,
        before_message_id=before_message_id,
        task_source=source,
    )


def _refuse_if_bg_inflight(task_id: int) -> None:
    """Raise ``TaskTurnError`` if the manager already has a non-done
    bg coroutine registered for this task_id.

    Why this exists: ``background_task_manager.register_task`` is a plain
    dict assignment that overwrites any previous handle. Without this
    guard, two scheduling calls in quick succession both register
    themselves; the second one's bg coroutine then calls
    ``wait_for_previous(task_id)``, which sees its own handle in the
    map and returns immediately (the ``is current_task`` short-circuit
    treats "I'm the only one registered" as "I'm previous, no wait"),
    so both bg coroutines race.

    Checking from the orchestrator side before register_task closes the
    window without touching the manager's semantics (the manager still
    works fine for the legitimate "previous task naturally completed"
    case).
    """
    from ..api.websocket import background_task_manager

    existing = background_task_manager.running_tasks.get(task_id)
    if existing is not None and not existing.done():
        raise TaskTurnError("bg_inflight")


def _get_agent_manager() -> Any:
    """Resolve the global ``AgentServiceManager`` singleton.

    Local import keeps the services -> api boundary one-way at module
    load time.
    """
    from ..api.chat import get_agent_manager

    return get_agent_manager()


def _mark_task_failed_if_running(task_id: int, error_message: str) -> None:
    """Setup/run-error sentinel for ``_schedule_bg._runner``.

    ``acquire_task_lease_isolated`` sets ``task.status = RUNNING`` as
    part of taking the lease. If a later step in ``_runner`` raises
    (snapshot load, ``execute_task_background``) and no downstream
    handler moves the task to a terminal status, the release block
    would see ``status=RUNNING`` and write it back -- leaving the row
    visible as running but with no active worker (zombie state). This
    helper closes that window: ``_runner`` calls it from an outer
    ``except`` so the task is forced to ``FAILED`` before release.

    Guarded by ``status == RUNNING`` -- never overwrites a terminal /
    control status (``PAUSED`` / ``WAITING_FOR_USER`` / ``FAILED`` /
    ``COMPLETED``) that ``execute_task_background`` may have set
    inside its own inner ``try/except``. Opens / commits / closes
    its own session so the caller doesn't have to thread a session
    through the exception path.
    """
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    try:
        with SessionLocal() as db:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task is None or task.status != TaskStatus.RUNNING:
                return
            task.status = TaskStatus.FAILED
            task.error_message = error_message  # type: ignore[assignment]
            db.commit()
    except Exception as e:
        # Defensive: do not let this helper raise out of the ``except``
        # path that's already handling an error. Log loudly so the
        # zombie state, if it survives, is traceable.
        logger.error(
            "Failed to mark task %s as FAILED during setup/run error: %s",
            task_id,
            e,
            exc_info=True,
        )


# ===== finish_turn / _schedule_bg (new lifecycle API) =====


def finish_turn(bg_db: Any, task_id: int) -> None:
    """Symmetric terminal-field writer with lease ownership guard.

    Called from ``_schedule_bg._runner`` after ``execute_task_background``
    returns. Two key properties:

      - latest-turn snapshot invariant: COMPLETED, FAILED, and the
        RUNNING-fallback branch all leave the row in a state where the
        terminal field that *doesn't* apply to the current turn is
        cleared (COMPLETED clears ``error_message``; FAILED clears
        stale ``output``). SDK consumers reading ``/v1/chat/tasks/{id}``
        therefore never see a contradictory snapshot like
        ``status='failed' + output='prior successful answer'``.
      - lease ownership guard: the RUNNING-fallback branch refuses to
        flip the row to FAILED while another worker still holds a live
        lease, so a slow scheduler in this process can't overwrite the
        in-flight execution result of a different process.

    Uses :func:`get_runner_id` internally rather than accepting
    runner_id as a parameter so the comparison always reads the
    canonical process runner id and a separately-captured
    ``lease.runner_id`` can't drift from it.

    Branches:

      - ``status == COMPLETED``: set ``output`` from latest assistant
        message, clear ``error_message``
      - ``status == FAILED``: set ``error_message`` placeholder if
        absent, clear stale ``output``
      - ``status == RUNNING`` + other worker holds live lease: skip
        entirely (ownership guard)
      - ``status == RUNNING`` + we own lease or it's expired: flip to
        FAILED, set placeholder ``error_message``, clear stale
        ``output``
      - other statuses (PAUSED / WAITING_FOR_USER): leave alone
    """
    from ..models.chat_message import TaskChatMessage
    from .workforce_runtime import sync_workforce_run_status

    bg_db.expire_all()

    fresh = bg_db.query(Task).filter(Task.id == task_id).first()
    if fresh is None:
        logger.warning("finish_turn: task %s vanished after bg run", task_id)
        return

    status = fresh.status

    if status == TaskStatus.COMPLETED:
        latest_assistant = (
            bg_db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "assistant",
            )
            .order_by(TaskChatMessage.id.desc())
            .first()
        )
        if latest_assistant is not None:
            fresh.output = latest_assistant.content
            fresh.error_message = None
            sync_workforce_run_status(bg_db, fresh, TaskStatus.COMPLETED)
            _sync_trigger_run_status(bg_db, fresh, TaskStatus.COMPLETED)
            bg_db.commit()
            invalidate_task_cache(task_id)
            logger.info(
                "finish_turn: task %s output written (%d chars)",
                task_id,
                len(latest_assistant.content),
            )
        else:
            logger.warning(
                "finish_turn: task %s completed but no assistant message found",
                task_id,
            )
            run_changed = sync_workforce_run_status(bg_db, fresh, TaskStatus.COMPLETED)
            trigger_run_changed = _sync_trigger_run_status(
                bg_db, fresh, TaskStatus.COMPLETED
            )
            if run_changed or trigger_run_changed:
                bg_db.commit()
                invalidate_task_cache(task_id)
        return

    if status == TaskStatus.FAILED:
        changed = False
        if not fresh.error_message:
            fresh.error_message = "Task execution failed (see /steps for details)"
            changed = True
        if fresh.output is not None:
            # Latest-turn snapshot invariant: a failed turn must not
            # carry forward prior
            # successful output. SDK consumers reading the row otherwise
            # see a contradiction (status=failed + output populated).
            fresh.output = None
            changed = True
        run_changed = sync_workforce_run_status(bg_db, fresh, TaskStatus.FAILED)
        trigger_run_changed = _sync_trigger_run_status(bg_db, fresh, TaskStatus.FAILED)
        if changed or run_changed or trigger_run_changed:
            bg_db.commit()
            invalidate_task_cache(task_id)
            logger.info(
                "finish_turn: task %s marked failed (cleared stale output)",
                task_id,
            )
        return

    if status == TaskStatus.RUNNING:
        # Lease ownership guard: a live lease held by another worker
        # means that worker is actively executing this task; we must
        # not overwrite its in-flight result with a FAILED snapshot.
        # ``lease_expires_at`` comes back tz-naive from SQLite (the column is
        # DateTime(timezone=True) but SQLite stores only the naked timestamp);
        # normalize to UTC so the comparison stays dialect-agnostic.
        expires_at = fresh.lease_expires_at
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        live_other_owner = (
            fresh.runner_id is not None
            and fresh.runner_id != get_runner_id()
            and expires_at is not None
            and expires_at > datetime.now(timezone.utc)
        )
        if live_other_owner:
            logger.info(
                "finish_turn: task %s owned by runner %s, lease alive "
                "until %s; skipping RUNNING fallback",
                task_id,
                fresh.runner_id,
                fresh.lease_expires_at,
            )
            return
        # Genuinely stuck: our bg coroutine returned, no live lease elsewhere.
        fresh.status = TaskStatus.FAILED
        fresh.error_message = "Task execution failed without status update; see /steps."
        fresh.output = None  # latest-turn snapshot invariant
        sync_workforce_run_status(bg_db, fresh, TaskStatus.FAILED)
        _sync_trigger_run_status(bg_db, fresh, TaskStatus.FAILED)
        bg_db.commit()
        invalidate_task_cache(task_id)
        logger.warning(
            "finish_turn: task %s bg coroutine returned with status=RUNNING; "
            "flipping to FAILED",
            task_id,
        )
        return

    # PAUSED / WAITING_FOR_USER / other: leave alone.


def finish_turn_isolated(task_id: int) -> None:
    """Run finish_turn with a short-lived session owned by this thread."""
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as finalize_db:
        finish_turn(finalize_db, task_id)


def _sync_trigger_run_status(bg_db: Any, task: Task, status: TaskStatus) -> bool:
    """Best-effort mirror from task terminal state to trigger run history."""
    from ..models.trigger import TriggerRun, TriggerRunStatus

    rows = (
        bg_db.query(TriggerRun)
        .filter(
            TriggerRun.task_id == int(task.id),
            TriggerRun.status.in_(
                [TriggerRunStatus.PENDING.value, TriggerRunStatus.RUNNING.value]
            ),
        )
        .all()
    )
    if not rows:
        return False

    now = datetime.now(timezone.utc)
    run_status = (
        TriggerRunStatus.COMPLETED.value
        if status == TaskStatus.COMPLETED
        else TriggerRunStatus.FAILED.value
    )
    for run in rows:
        run.status = run_status
        run.finished_at = now
        if status == TaskStatus.FAILED:
            run.error_message = task.error_message
        else:
            run.error_message = None
        bg_db.add(run)
    return True


def _schedule_bg(
    *,
    task_id: int,
    task_owner_user_id: int,
    task_source: Optional[str],
    payload: TaskTurnPayload,
    force_fresh: bool,
    context: Optional[Dict[str, Any]],
    before_message_id: Optional[int] = None,
) -> "asyncio.Task[None]":
    """Lease-aware bg scheduler.

    Synchronous: it defines ``_runner``, schedules it via
    ``asyncio.create_task``, registers the handle and returns it — there is
    no ``await`` at this level (every await lives inside ``_runner``, which
    runs later as its own task). Being sync removes a misleading
    suspension/cancellation point right after ``begin_turn``'s claim commit.

    Owns the full lease lifecycle for the bg run:

      - acquire at ``_runner`` entry. If another worker already holds
        the lease the scheduler returns immediately without invoking
        ``execute_task_background`` or ``finish_turn`` — the
        running-elsewhere short-circuit. ``finish_turn``'s ownership
        guard would catch the same situation a level deeper, but
        skipping at the entry means we never even attempt local work
        on a task another worker is executing.
      - heartbeat alongside the run.
      - release in ``finally`` as the single owner of the release
        call, regardless of whether ``execute_task_background``
        returned normally or raised. ``execute_task_background`` only
        writes ``task.status`` and never touches the lease columns;
        the scheduler is responsible for the whole lease lifecycle.

    Takes primitives only (``task_id`` / ``task_owner_user_id`` /
    ``task_source``); the
    bg run loads its own snapshot and opens its own sessions, so no
    caller-bound ORM object crosses into the coroutine.
    """
    from ..api.websocket import background_task_manager, execute_task_background

    async def _runner() -> None:
        # ``bg_db`` is opened lazily inside the post-run finalize block
        # only. We no longer keep a SessionLocal open across the entire
        # agent run -- that previously held an idle connection-pool
        # slot for tens of seconds to minutes (long-running agents)
        # without doing any work. The lease acquire / heartbeat /
        # snapshot load all open their own short-lived sessions, and
        # ``finish_turn`` + release run inside a single ``with`` block
        # below.
        from ..models.database import get_session_local

        lease = None
        try:
            # Running-elsewhere short-circuit: acquire lease before
            # doing anything else. If another worker owns it, skip
            # execution entirely so finish_turn never touches the row.
            #
            # The acquire is a conditional UPDATE + commit that
            # measured 3.75s of synchronous DB write on the main
            # event loop (issue #427). ``acquire_task_lease_isolated``
            # wraps the existing helper with its own SessionLocal so
            # the work runs on a worker thread.
            lease = await asyncio.to_thread(acquire_task_lease_isolated, task_id)
            if lease is None:
                logger.info(
                    "task %s acquired by another worker; skipping "
                    "execution and finish_turn",
                    task_id,
                )
                return

            # INVARIANT: ``asyncio.create_task(run_task_lease_heartbeat(...))``
            # MUST be scheduled before any ``await`` that yields the
            # loop (snapshot to_thread, agent setup, execute_task_background).
            # The lease has a bounded TTL; nothing downstream of acquire
            # may ride bare past this point. If a future refactor moves
            # the heartbeat creation below the snapshot load, a
            # contended worker could drop the lease while snapshot is
            # in-flight, hand the task to another runner mid-setup, and
            # double-run the same turn. Do not reorder.
            stop_event = asyncio.Event()
            hb_task = asyncio.create_task(run_task_lease_heartbeat(lease, stop_event))
            try:
                # Outer ``try/except`` is the lease-acquire-to-terminal
                # safety net: ``acquire_task_lease_isolated`` already
                # set ``status=RUNNING`` for this row, so any unhandled
                # exception from snapshot load / execute_task_background
                # would leave the task in a zombie state (visible as
                # running, no active worker) once the release block
                # below clears ``runner_id``. ``_mark_task_failed_if_running``
                # closes the window. We swallow the exception so
                # ``finish_turn`` + lease release still run cleanly with
                # the now-terminal status.
                try:
                    # Load the synchronous DB block on a worker thread
                    # so the main loop stays responsive. The loader
                    # opens / closes its own SessionLocal (no ORM
                    # leak), and the snapshot is passed straight
                    # through to execute_task_background →
                    # get_agent_for_task. That turns the previous
                    # chain of three redundant Task queries into a
                    # single off-loop read.
                    snapshot = await asyncio.to_thread(
                        load_task_setup_snapshot_sync, task_id, task_owner_user_id
                    )
                    if snapshot is None:
                        logger.warning(
                            "bg task %s aborted: task vanished before snapshot load",
                            task_id,
                        )
                        _mark_task_failed_if_running(
                            task_id, "task vanished before snapshot load"
                        )
                        return

                    await execute_task_background(
                        task_id=task_id,
                        user_message=payload.transcript_message,
                        context=_execution_context_with_turn_id(
                            context, payload.turn_id
                        ),
                        agent_manager=_get_agent_manager(),
                        task_owner_user_id=task_owner_user_id,
                        before_message_id=before_message_id,
                        llm_user_message=payload.execution_message,
                        task_setup_snapshot=snapshot,
                    )
                except Exception as setup_or_run_err:
                    logger.error(
                        "bg task %s setup/run failed: %s",
                        task_id,
                        setup_or_run_err,
                        exc_info=True,
                    )
                    _mark_task_failed_if_running(
                        task_id,
                        f"setup/run error: "
                        f"{type(setup_or_run_err).__name__}: {setup_or_run_err}",
                    )
                    # Do not re-raise: ``finish_turn`` + release below
                    # must run so the lease is freed and the row is
                    # not stuck mid-lifecycle.

                # Short-lived finalize session. ``finish_turn`` only
                # reads / updates the task row once, but the DB work can
                # still block the event loop under load; run it in a
                # worker-thread session.
                try:
                    await asyncio.to_thread(finish_turn_isolated, task_id)
                except Exception as e:
                    logger.error(
                        "finish_turn failed for task %s: %s",
                        task_id,
                        e,
                        exc_info=True,
                    )
            finally:
                stop_event.set()
                try:
                    await hb_task
                except Exception:
                    pass
        finally:
            if lease is not None:
                # Single owner of release. Open a fresh short-lived
                # session for both the status read and the release UPDATE
                # so we don't hold a connection across the agent run.
                # Defensive: if the read raises (DB connectivity issue),
                # default to FAILED so the lease still gets released
                # instead of stuck-until-TTL.
                SessionLocal = get_session_local()
                with SessionLocal() as release_db:
                    final_status: TaskStatus = TaskStatus.FAILED
                    try:
                        fresh = (
                            release_db.query(Task).filter(Task.id == task_id).first()
                        )
                        if fresh is not None:
                            final_status = fresh.status
                    except Exception as query_err:
                        logger.warning(
                            "task %s status read failed during lease release "
                            "(%s); rolling session back and defaulting to FAILED",
                            task_id,
                            query_err,
                        )
                        try:
                            release_db.rollback()
                        except Exception:
                            pass
                    # Use the workforce-aware release helper: it wraps
                    # ``release_current_runner_task_lease`` (signature
                    # unchanged) and additionally syncs the workforce
                    # run status when the released task belongs to one.
                    # Both PR #461 (short-open/short-close release_db
                    # pattern) and PR #528 (workforce sync) compose
                    # cleanly here -- decorator-style, no perf regression.
                    try:
                        release_current_runner_task_lease_with_workforce_sync(
                            release_db, task_id, status=final_status
                        )
                    except Exception as e:
                        logger.warning(
                            "lease release failed for task %s: %s; "
                            "TTL expiry will reclaim it",
                            task_id,
                            e,
                        )
            turn_id = getattr(payload, "turn_id", None)
            try:
                from .connector_runtime import pop_ephemeral_runtime_values

                if turn_id is not None:
                    pop_ephemeral_runtime_values(turn_id)
            except Exception:
                logger.warning(
                    "connector runtime cleanup failed for task %s turn %s",
                    task_id,
                    turn_id,
                    exc_info=True,
                )

    bg_task = asyncio.create_task(_runner())
    background_task_manager.register_task(task_id, bg_task)
    logger.info(
        "task %s scheduled in background v2 (source=%s, force_fresh=%s)",
        task_id,
        task_source,
        force_fresh,
    )
    return bg_task


def _execution_context_with_turn_id(
    context: Optional[Dict[str, Any]], turn_id: str
) -> Dict[str, Any]:
    execution_context = dict(context or {})
    if turn_id:
        execution_context["turn_id"] = turn_id
    return execution_context
