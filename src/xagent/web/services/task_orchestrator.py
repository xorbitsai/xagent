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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.orm import Session

from ...core.agent.context.execution import CLOCK_TIMEZONE_METADATA_KEY
from ...core.execution_scope import resolve_execution_scope
from ...core.tools.adapters.vibe.config import RequiredMCPUnavailableError
from ..models.task import Task, TaskStatus
from .assistant_history_safety import (
    CLIENT_SAFE_FAILURE_MESSAGE_TYPE,
    TASK_FAILURE_MESSAGE_TYPE,
)
from .chat_history_service import (
    DELIVERY_COMPLETED,
    DELIVERY_DISPATCHED,
    DELIVERY_FAILED,
    DELIVERY_PENDING,
    mark_user_message_delivery_sync,
)
from .client_error_messages import (
    CLIENT_SAFE_TASK_FAILURE,
    required_mcp_unavailable_client_message,
)
from .db_runtime import (
    drain_async_task_cancellation_safe,
    is_database_pool_timeout,
    run_db_io_cancellation_safe,
)
from .external_task_cancel import (
    EXTERNAL_TASK_SOURCE,
    EXTERNAL_TURN_INTERRUPTED_MESSAGE,
)
from .file_turn import bind_turn_files_no_commit
from .hot_path_cache import invalidate_task_cache
from .mcp_runtime import (
    MCPBuiltinOAuthActorPolicy,
    MCPBuiltinOAuthActorPolicyRequiredError,
)
from .task_execution_controller import (
    TaskControlState,
    apply_task_control_transition,
    task_execution_controller,
)
from .task_lease_service import (
    TaskLease,
    TaskLeaseHeartbeatOutcome,
    TaskLeaseLostError,
    TaskLeaseRefreshState,
    acquire_task_lease_cancellation_safe,
    acquire_task_lease_isolated,
    acquire_task_lease_no_commit,
    fail_and_release_task_lease_no_commit,
    get_runner_id,
    release_task_lease,
    run_task_lease_heartbeat,
    run_while_task_lease_owned,
    stop_task_lease_heartbeat,
    validate_preacquired_task_lease_isolated,
)
from .task_runtime import mcp_runtime_authorization_policy_required
from .task_setup_snapshot import load_task_setup_snapshot_sync

logger = logging.getLogger(__name__)


# Statuses for the "can a user message start the next turn?" check. A
# task in any of these is eligible for ``TurnKind.APPEND``. PENDING is
# claimed by ``CREATE``; RUNNING is still busy; WAITING_FOR_USER is an
# answer to a pending agent question and is handled by the dedicated
# reply endpoint instead, which resumes the existing run rather than
# claiming a new turn.
_APPENDABLE_STATUSES = (
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.PAUSED,
)


def timezone_schedule_context(timezone: str | None) -> dict[str, Any] | None:
    """Build the schedule ``context`` carrying the caller's clock timezone.

    Shared by every create-first entry point (workforce widget/share, the
    workforce SDK, and ``/v1/chat/tasks``) so blank normalization and the
    metadata key stay identical across them. Whitespace-only degrades to
    ``None`` so the renderer keeps UTC.
    """
    normalized = (timezone or "").strip()
    return {CLOCK_TIMEZONE_METADATA_KEY: normalized} if normalized else None


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
    # Authorized uploads are bound by the acceptance transaction, never by
    # WebSocket preparation. Keeping only IDs here prevents a second
    # authorization path from drifting from the transcript owner.
    file_ids: tuple[str, ...] = ()
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
    turn) are deliberately not modeled here: the resume executor adopts
    the exact run-fenced checkpoint and lease instead of claiming a new
    turn, because terminal-field reset would be wrong.
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


class TaskTurnCommitOutcomeUnknown(Exception):
    """A turn COMMIT may have succeeded but cannot yet be reconciled."""

    def __init__(self, task_id: int, turn_id: str):
        super().__init__(
            f"turn {turn_id} for task {task_id} has an unknown commit outcome"
        )
        self.task_id = task_id
        self.turn_id = turn_id


class TaskTurnFileBindingError(ValueError):
    """The acceptance transaction could not bind every requested upload."""

    def __init__(self, missing_file_ids: list[str]):
        self.missing_file_ids = tuple(missing_file_ids)
        super().__init__(
            "Files are no longer bindable: " + ", ".join(self.missing_file_ids)
        )


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
    run_id: str = ""
    state_version: int = 0
    control_state: str = TaskControlState.RUNNING.value


class TaskTurnOrchestrator:
    """Drive one task-turn lifecycle.

    All methods are static; the class is a namespace, not stateful.
    State lives in the database and in the global
    ``background_task_manager``.
    """

    @staticmethod
    async def schedule_existing_task_execution(
        *,
        task_id: int,
        task_owner_user_id: int,
        task_source: Optional[str],
        payload: TaskTurnPayload,
        context: Optional[Dict[str, Any]] = None,
        actor_user_id: Optional[int] = None,
    ) -> "asyncio.Task[None]":
        """Schedule the legacy ``execute_task`` command without a caller Session.

        Unlike :meth:`begin_turn`, this compatibility entry does not persist a
        new user message or reset terminal fields.  The legacy WebSocket command
        executes the description already stored on ``Task``; changing it into a
        new chat turn would duplicate that text in conversation history.

        The shared scheduler still owns the runtime invariants: one local
        in-flight coroutine, an exact task lease, worker-owned setup snapshot,
        one off-loop execution-scope resolution, heartbeat, and fenced
        settlement.  Cross-worker duplicates are rejected by lease acquisition.
        """
        async with task_execution_controller.command(task_id):
            actor_marked = await run_db_io_cancellation_safe(
                lambda: _task_requires_actor_policy_sync(
                    task_id,
                    task_owner_user_id,
                )
            )
            if actor_marked:
                raise MCPBuiltinOAuthActorPolicyRequiredError(
                    f"Task {task_id} is actor-marked; legacy execution is unsupported"
                )
            _refuse_if_bg_inflight(task_id)
            background_task = _schedule_bg(
                task_id=task_id,
                task_owner_user_id=task_owner_user_id,
                task_source=task_source,
                payload=payload,
                force_fresh=False,
                context=context,
            )

        logger.info(
            "existing task execution scheduled: task=%s owner=%s actor=%s",
            task_id,
            task_owner_user_id,
            actor_user_id if actor_user_id is not None else task_owner_user_id,
        )
        return background_task

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
        """Serialize every transport's new-turn command for one task."""

        async with task_execution_controller.command(task_id):
            operation = asyncio.create_task(
                TaskTurnOrchestrator._begin_turn_unserialized(
                    task_id=task_id,
                    task_owner_user_id=task_owner_user_id,
                    payload=payload,
                    kind=kind,
                    force_fresh=force_fresh,
                    context=context,
                    actor_user_id=actor_user_id,
                )
            )
            try:
                return await asyncio.shield(operation)
            except asyncio.CancelledError:
                # The atomic claim/schedule contract is cancellation-safe and
                # may still be running under its own shield. Keep the command
                # gate until it settles so a replacement message cannot enter
                # the task while that claim is committing in the background.
                while not operation.done():
                    try:
                        await asyncio.shield(operation)
                    except asyncio.CancelledError:
                        # Repeated cancellation must not release the command
                        # gate while the atomic claim is still in flight.
                        continue
                    except Exception:
                        break
                if not operation.cancelled():
                    operation_error = operation.exception()
                    if operation_error is not None:
                        logger.error(
                            "Turn start failed while cancelled caller waited for task %s",
                            task_id,
                            exc_info=(
                                type(operation_error),
                                operation_error,
                                operation_error.__traceback__,
                            ),
                        )
                raise

    @staticmethod
    def ensure_no_background_turn(task_id: int) -> None:
        """Reject a new claim while this worker still owns a live runner.

        Domain-owned transactions call this before staging an APPEND claim.
        The database status predicate remains the cross-worker authority; this
        process-local guard covers the shorter tail window where a runner is
        still registered after its terminal task status has committed.
        """

        _refuse_if_bg_inflight(task_id)

    @staticmethod
    async def _begin_turn_unserialized(
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
             in-memory dict check. The process-local task controller serializes
             callers in this worker; callers in different workers can still
             both pass, so the DB atomic claim remains the authoritative
             cross-worker guard and lets exactly one win.
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
            TaskTurnError("interaction_response_required"): ``kind ==
                APPEND`` and the row's status is ``WAITING_FOR_USER`` --
                use the reply endpoint instead of append.
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
        # CancelledError landing at the worker resume, after the commit, would
        # strand the row as RUNNING with no worker. One owned child covers
        # claim through scheduling and is drained before cancellation reaches
        # the caller.
        async def _claim_and_schedule() -> tuple[_ClaimedTurn, "asyncio.Task[None]"]:
            # Off-loop atomic claim + persist + commit. Only raises pre-commit
            # (busy / not-found), so a normal exception here means nothing was
            # committed; reaching the schedule means the row is RUNNING.
            claimed = await run_db_io_cancellation_safe(
                lambda: _begin_turn_atomic_sync(
                    task_id,
                    task_owner_user_id,
                    payload=payload,
                    kind=kind,
                )
            )
            handle = await _schedule_committed_turn(
                task_id=task_id,
                task_owner_user_id=task_owner_user_id,
                payload=payload,
                claimed=claimed,
                kind=kind,
                force_fresh=force_fresh,
                context=context,
            )
            return claimed, handle

        start_task = asyncio.create_task(_claim_and_schedule())
        claimed, bg_task = await drain_async_task_cancellation_safe(start_task)
        return _turn_started_snapshot(
            task_id=task_id,
            task_owner_user_id=task_owner_user_id,
            actor_user_id=actor_user_id,
            kind=kind,
            claimed=claimed,
            background_task=bg_task,
        )

    @staticmethod
    def claim_created_turn_no_commit(
        db: Session,
        *,
        task_id: int,
        task_owner_user_id: int,
        payload: TaskTurnPayload,
    ) -> "_ClaimedTurn":
        """Stage one CREATE turn inside the caller-owned transaction.

        This is used only when the task row itself is still uncommitted, so the
        row creation, initial turn claim, first user message, and any enclosing
        domain projection can become visible in one commit. The caller owns
        commit and rollback.
        """

        return _claim_turn_no_commit(
            db,
            task_id=task_id,
            task_owner_user_id=task_owner_user_id,
            payload=payload,
            kind=TurnKind.CREATE,
        )

    @staticmethod
    def claim_append_turn_no_commit(
        db: Session,
        *,
        task_id: int,
        task_owner_user_id: int,
        payload: TaskTurnPayload,
    ) -> "_ClaimedTurn":
        """Stage one APPEND turn inside a domain-owned transaction.

        File/runtime domain mutations that must be atomic with acceptance can
        run in the same transaction before the caller commits. A busy or
        missing task raises before any staged mutation becomes visible.
        """

        return _claim_turn_no_commit(
            db,
            task_id=task_id,
            task_owner_user_id=task_owner_user_id,
            payload=payload,
            kind=TurnKind.APPEND,
        )

    @staticmethod
    async def schedule_claimed_turn(
        *,
        task_id: int,
        task_owner_user_id: int,
        actor_user_id: int | None,
        payload: TaskTurnPayload,
        claimed: "_ClaimedTurn",
        kind: TurnKind,
        force_fresh: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> TurnStarted:
        """Schedule an ordinary turn already committed by its domain owner."""

        return await TaskTurnOrchestrator._schedule_claimed_turn(
            task_id=task_id,
            task_owner_user_id=task_owner_user_id,
            actor_user_id=actor_user_id,
            payload=payload,
            claimed=claimed,
            kind=kind,
            force_fresh=force_fresh,
            context=context,
        )

    @staticmethod
    async def _schedule_claimed_turn(
        *,
        task_id: int,
        task_owner_user_id: int,
        actor_user_id: int | None,
        payload: TaskTurnPayload,
        claimed: "_ClaimedTurn",
        kind: TurnKind,
        force_fresh: bool = False,
        context: Optional[Dict[str, Any]] = None,
        mcp_runtime_authorization_policy: MCPBuiltinOAuthActorPolicy | None = None,
    ) -> TurnStarted:
        """Shared scheduler with actor context reserved for trusted CREATE."""

        async def _schedule() -> TurnStarted:
            # The no-commit claim delegates commit to its domain owner, so
            # invalidate only once that owner has returned with a committed
            # claim and before scheduling can observe cached state.
            # The configured cache backend may perform synchronous Redis I/O,
            # so keep that work off the event-loop thread.
            await asyncio.to_thread(invalidate_task_cache_best_effort, task_id)
            async with task_execution_controller.command(task_id):
                background_task = await _schedule_committed_turn(
                    task_id=task_id,
                    task_owner_user_id=task_owner_user_id,
                    payload=payload,
                    claimed=claimed,
                    kind=kind,
                    force_fresh=force_fresh,
                    context=context,
                    mcp_runtime_authorization_policy=(mcp_runtime_authorization_policy),
                )
                return _turn_started_snapshot(
                    task_id=task_id,
                    task_owner_user_id=task_owner_user_id,
                    actor_user_id=actor_user_id,
                    kind=kind,
                    claimed=claimed,
                    background_task=background_task,
                )

        # Create the owned child before any await. Once the caller's transaction
        # commits RUNNING, cancellation must still reach scheduling or the
        # existing compensated failure path before it can propagate.
        schedule_task = asyncio.create_task(_schedule())
        return await drain_async_task_cancellation_safe(schedule_task)

    @staticmethod
    async def schedule_claimed_create_turn(
        *,
        task_id: int,
        task_owner_user_id: int,
        actor_user_id: int | None,
        payload: TaskTurnPayload,
        claimed: "_ClaimedTurn",
        context: Optional[Dict[str, Any]] = None,
        mcp_runtime_authorization_policy: MCPBuiltinOAuthActorPolicy | None = None,
    ) -> TurnStarted:
        """Compatibility wrapper for an already committed CREATE claim."""

        return await TaskTurnOrchestrator._schedule_claimed_turn(
            task_id=task_id,
            task_owner_user_id=task_owner_user_id,
            actor_user_id=actor_user_id,
            payload=payload,
            claimed=claimed,
            kind=TurnKind.CREATE,
            context=context,
            mcp_runtime_authorization_policy=mcp_runtime_authorization_policy,
        )


# ===== internal helpers =====


@dataclass(frozen=True)
class _ClaimedTurn:
    """Snapshot returned by ``_begin_turn_atomic_sync`` after the claim
    commits, so ``begin_turn`` can build :class:`TurnStarted` without the
    caller re-reading the ORM."""

    task_lease: TaskLease
    status: TaskStatus
    updated_at: Optional[datetime]
    before_message_id: Optional[int]
    task_source: Optional[str]
    run_id: str = ""
    state_version: int = 0
    control_state: str = TaskControlState.RUNNING.value
    agent_config: Optional[dict[str, Any]] = None


async def _schedule_committed_turn(
    *,
    task_id: int,
    task_owner_user_id: int,
    payload: TaskTurnPayload,
    claimed: _ClaimedTurn,
    kind: TurnKind,
    force_fresh: bool,
    context: Optional[Dict[str, Any]],
    mcp_runtime_authorization_policy: MCPBuiltinOAuthActorPolicy | None = None,
) -> "asyncio.Task[None]":
    """Own scheduling and compensation after a turn claim has committed."""

    try:
        actor_marked = mcp_runtime_authorization_policy_required(claimed.agent_config)
        if actor_marked and (
            kind is not TurnKind.CREATE or mcp_runtime_authorization_policy is None
        ):
            raise MCPBuiltinOAuthActorPolicyRequiredError(
                f"Task {task_id} is actor-marked; only trusted fresh direct "
                "execution is supported"
            )
        if mcp_runtime_authorization_policy is not None and not actor_marked:
            raise MCPBuiltinOAuthActorPolicyRequiredError(
                f"Task {task_id} was not persisted as actor-marked"
            )
        _refuse_if_bg_inflight(task_id)
        handle = _schedule_bg(
            task_id=task_id,
            task_owner_user_id=task_owner_user_id,
            task_source=claimed.task_source,
            run_id=claimed.run_id,
            task_lease=claimed.task_lease,
            payload=payload,
            force_fresh=force_fresh,
            context=context,
            before_message_id=claimed.before_message_id,
            mcp_runtime_authorization_policy=mcp_runtime_authorization_policy,
        )
    except BaseException as schedule_error:
        # Schedule failed after the claim committed -> force FAILED so the row
        # is not left RUNNING. The terminal write remains off-loop.
        try:
            await run_db_io_cancellation_safe(
                lambda: settle_task_lease_isolated(
                    claimed.task_lease,
                    error_message="turn scheduling failed after claim commit",
                )
            )
        except Exception as terminal_error:
            if not is_database_pool_timeout(terminal_error):
                raise
            # This checkout already waited for the exhausted pool. Do not
            # immediately attempt the delivery update below; its committed
            # PENDING marker remains reclaimable.
            logger.error(
                "task_id=%s component=turn-schedule-terminal database pool "
                "checkout timed out; skipping delivery update: %s",
                task_id,
                terminal_error,
                exc_info=True,
            )
            raise schedule_error from terminal_error
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

    try:
        await asyncio.to_thread(
            mark_user_message_delivery_sync,
            task_id,
            payload.turn_id,
            "dispatched",
        )
    except Exception as delivery_error:
        # Claim commit + background scheduling already succeeded. A transient
        # delivery projection failure must not turn that success into an API
        # error or invite the caller to retry a turn that is already running.
        _log_delivery_projection_failure(
            task_id,
            component="turn-delivery",
            error=delivery_error,
            detail=" after scheduling",
        )
    return handle


def _log_delivery_projection_failure(
    task_id: int,
    *,
    component: str,
    error: Exception,
    detail: str = "",
) -> None:
    """Log one swallowed delivery-projection failure without re-raising.

    Pool checkout timeouts get their explicit marker — the committed row stays
    reclaimable and the caller must not issue another checkout against the
    exhausted pool. Anything else keeps its traceback via ``logger.exception``.
    """

    if is_database_pool_timeout(error):
        logger.error(
            "task_id=%s component=%s database pool checkout timed out%s; "
            "leaving delivery pending for durable recovery: %s",
            task_id,
            component,
            detail,
            error,
            exc_info=True,
        )
    else:
        logger.exception(
            "task_id=%s component=%s projection failed%s; "
            "leaving delivery pending for durable recovery",
            task_id,
            component,
            detail,
        )


def _reconcile_finalized_turn_delivery(
    *,
    task_id: int,
    turn_id: str | None,
    settlement_error: str | None,
    execution_started: bool,
) -> None:
    """Close a terminally settled turn's delivery row (xorbitsai/xagent-saas#332).

    A user-message delivery row is written ``pending`` with the turn claim and
    projected to ``dispatched`` right after scheduling. That projection is
    best-effort: a pool timeout or a transient DB error leaves the row
    ``pending`` even though the turn ran. Nothing else on the orchestrator path
    advances it, so the session WS dedup probe classifies every
    same-``client_message_id`` retry as an in-flight ``PENDING`` and rejects it
    forever.

    Reconciling at finalize closes that window for every turn that reaches a
    terminal lease settlement owned by its coroutine. The call is intentionally
    skipped — leaving the row ``pending`` — when the coroutine is not the
    authoritative closer: the lease went to another worker, settlement was
    deferred to TTL recovery or raised, the settle was fenced out / superseded
    (returned ``False``), or the pre-execution SETTLEMENT_READY short-circuit
    fired. A hard process crash skips it too. Of those leftovers, only the
    paused-task resume path has an in-repo consumer that re-posts the same
    ``turn_id``; ``task_lease_recovery`` terminalizes the *task* but never
    redrives the turn or touches delivery rows, so the deferral/crash cases
    remain a narrower instance of the stuck-``pending`` window, not a solved
    one.

    Target selection is driven by what finalize actually knows, because the
    downstream contract is asymmetric: the session WS probe answers a
    ``failed`` row with ``retry_with_new_id`` — an invitation to re-send the
    same content — so ``failed`` is only safe with positive evidence the
    message was never consumed.

    - ``settlement_error is None`` → ``completed``: the run returned without
      raising. Note this does not certify success — ``execute_task_background``
      may set ``task.status = FAILED`` internally and return normally, leaving
      ``settlement_error`` unset — but both ``completed`` and ``dispatched``
      read identically downstream ("delivered, do not retry"), so the
      distinction has no client-visible effect.
    - error and ``execution_started`` → ``dispatched``: the run may have
      consumed the message (and produced side effects) before failing, so the
      one certain fact is "no longer in flight". A retry invitation here would
      double-execute the turn. The probe treats ``dispatched`` as delivered;
      the turn's failure is surfaced through task status, not by reopening
      delivery. This also removes the timing asymmetry with the post-schedule
      projection: whichever best-effort write lands, the row converges on
      ``dispatched``.
    - error and not ``execution_started`` → ``failed``:
      ``execute_task_background`` was never invoked (setup failed, or the run
      was cancelled first), so the message was provably never consumed and a
      fresh-id retry is safe. The accepted cost is that the ``failed`` row
      remains visible in the transcript (no history-loading path filters on
      ``delivery_status``), so a fresh-id resend can show the user's message
      twice — a pre-existing property of every ``failed`` writer, tracked in
      xorbitsai/xagent-saas#417, and strictly better than wedging the client
      on a ``pending`` row forever.

    The transition is monotonic — it never regresses — but only ``completed``
    and ``failed`` are terminal: ``dispatched`` still has an outgoing edge to
    ``completed``, which is how a turn that failed its projection but later
    settles cleanly still closes. An already-terminal row, or a ``dispatched``
    row on a later failure, is left untouched. The final delivery state is NOT
    a proxy for the turn's outcome: it answers "may this client_message_id be
    retried?", never "did the turn succeed?".
    """

    if turn_id is None:
        return
    if settlement_error is None:
        target = DELIVERY_COMPLETED
    elif execution_started:
        target = DELIVERY_DISPATCHED
    else:
        target = DELIVERY_FAILED
    try:
        mark_user_message_delivery_sync(task_id, turn_id, target)
    except Exception as delivery_error:
        # Best-effort projection. Leaving the row for durable recovery is strictly
        # better than amplifying pool exhaustion or masking the turn's outcome.
        _log_delivery_projection_failure(
            task_id,
            component="turn-finalize-delivery",
            error=delivery_error,
            detail=f" (target={target})",
        )


def _turn_started_snapshot(
    *,
    task_id: int,
    task_owner_user_id: int,
    actor_user_id: int | None,
    kind: TurnKind,
    claimed: _ClaimedTurn,
    background_task: "asyncio.Task[None]",
) -> TurnStarted:
    """Build the detached turn result and emit its owner/actor audit record."""

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
        run_id=claimed.run_id,
        state_version=claimed.state_version,
        control_state=claimed.control_state,
        background_task=background_task,
    )


def _persist_claimed_turn_no_commit(
    db: Session,
    *,
    task_id: int,
    task_owner_user_id: int,
    payload: TaskTurnPayload,
    task_lease: TaskLease,
) -> _ClaimedTurn:
    """Persist the first message and snapshot one already-claimed turn."""

    from .chat_history_service import persist_user_message_no_commit

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

    (
        status,
        updated_at,
        source,
        stored_run_id,
        state_version,
        control_state,
        agent_config,
    ) = (
        db.query(
            Task.status,
            Task.updated_at,
            Task.source,
            Task.run_id,
            Task.state_version,
            Task.control_state,
            Task.agent_config,
        )
        .filter(Task.id == task_id)
        .one()
    )
    return _ClaimedTurn(
        task_lease=task_lease,
        status=status,
        updated_at=updated_at,
        before_message_id=before_message_id,
        task_source=source,
        run_id=str(stored_run_id) if stored_run_id is not None else "",
        state_version=int(state_version),
        control_state=str(control_state),
        agent_config=agent_config if isinstance(agent_config, dict) else None,
    )


def invalidate_task_cache_best_effort(task_id: int) -> None:
    """Keep cache invalidation best-effort after a committed task write."""

    try:
        invalidate_task_cache(task_id)
    except Exception:
        logger.warning(
            "invalidate_task_cache failed for task %s (non-fatal)",
            task_id,
            exc_info=True,
        )


def _task_requires_actor_policy(
    db: Session,
    task_id: int,
    task_owner_user_id: int,
) -> bool:
    row = (
        db.query(Task.agent_config)
        .filter(Task.id == task_id, Task.user_id == task_owner_user_id)
        .first()
    )
    return bool(row is not None and mcp_runtime_authorization_policy_required(row[0]))


def _task_requires_actor_policy_sync(
    task_id: int,
    task_owner_user_id: int,
) -> bool:
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        return _task_requires_actor_policy(db, task_id, task_owner_user_id)


def _claim_turn_no_commit(
    db: Session,
    task_id: int,
    task_owner_user_id: int,
    *,
    payload: TaskTurnPayload,
    kind: TurnKind,
) -> _ClaimedTurn:
    """Stage one atomic turn claim; the caller owns commit and rollback."""

    if kind == TurnKind.APPEND and _task_requires_actor_policy(
        db,
        task_id,
        task_owner_user_id,
    ):
        raise TaskTurnError("actor_task_reuse_unsupported")

    if kind == TurnKind.CREATE:
        status_filter = Task.status == TaskStatus.PENDING
    else:  # APPEND
        status_filter = Task.status.in_(_APPENDABLE_STATUSES)

    run_id = str(uuid4())
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
                Task.runner_id: None,
                Task.lease_attempt_id: None,
                Task.lease_expires_at: None,
                Task.last_heartbeat_at: None,
                Task.run_id: run_id,
                Task.last_checkpoint_event_id: None,
                Task.last_checkpoint_trace_event_id: None,
                Task.state_version: func.coalesce(Task.state_version, 0) + 1,
                Task.control_state: TaskControlState.RUNNING.value,
            },
            synchronize_session=False,
        )
    )
    if claimed == 0:
        owned = (
            db.query(Task.id, Task.status)
            .filter(Task.id == task_id, Task.user_id == task_owner_user_id)
            .first()
        )
        if owned is None:
            raise TaskTurnNotFoundError(task_id)
        if owned.status == TaskStatus.WAITING_FOR_USER:
            raise TaskTurnError("interaction_response_required")
        raise TaskTurnError("busy")

    task_lease = acquire_task_lease_no_commit(
        db,
        task_id,
        expected_run_id=run_id,
    )
    if task_lease is None:
        raise RuntimeError(
            f"task {task_id} claim could not stage its exact execution lease"
        )

    result = _persist_claimed_turn_no_commit(
        db,
        task_id=task_id,
        task_owner_user_id=task_owner_user_id,
        payload=payload,
        task_lease=task_lease,
    )
    missing_bindings = bind_turn_files_no_commit(
        file_ids=list(payload.file_ids),
        task_id=task_id,
        owner_user_id=task_owner_user_id,
        db=db,
    )
    if missing_bindings:
        raise TaskTurnFileBindingError(missing_bindings)
    agent_config = result.agent_config
    if isinstance(agent_config, dict) and isinstance(
        agent_config.get("workforce_run_id"), int
    ):
        # Workforce tasks: reject turns whose owning workforce was archived or
        # whose live config drifted from the run's pinned fingerprint. CREATE
        # turns are also gated because archive can race the upstream create.
        from .workforce_runtime import (
            WorkforceTurnRejectedError,
            ensure_workforce_turn_allowed,
        )

        try:
            ensure_workforce_turn_allowed(
                db,
                task_id=task_id,
                task_owner_user_id=task_owner_user_id,
                agent_config=agent_config,
            )
        except WorkforceTurnRejectedError as exc:
            raise TaskTurnError(exc.reason) from exc
        # Keep the WorkforceRun projection in the same transaction as the
        # Task RUNNING claim and exact prelease. A later best-effort worker can
        # otherwise arrive after completion and resurrect the projection.
        from .workforce_runtime import sync_workforce_run_status

        claimed_task = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.status == TaskStatus.RUNNING,
                Task.runner_id == task_lease.runner_id,
                Task.run_id == task_lease.run_id,
            )
            .one()
        )
        sync_workforce_run_status(db, claimed_task, TaskStatus.RUNNING)
    return result


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
      - row exists + owned but wrong status → ``TaskTurnError("busy")``,
        or ``TaskTurnError("interaction_response_required")`` when the
        row's status is ``WAITING_FOR_USER`` (use the reply endpoint
        instead of append)

    The committed-row snapshot is SELECTed pre-commit (read-your-writes within
    the transaction; a bulk ``.update(synchronize_session=False)`` leaves no
    ORM object to refresh). A lost COMMIT acknowledgement is reconciled from
    fresh Sessions after the failed Session has released its connection.
    """
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    db = SessionLocal()
    result: _ClaimedTurn | None = None
    session_retired = False
    try:
        try:
            result = _claim_turn_no_commit(
                db,
                task_id,
                task_owner_user_id,
                payload=payload,
                kind=kind,
            )
            db.flush()
        except Exception:
            db.rollback()
            raise
        try:
            db.commit()
        except Exception as commit_error:
            # A driver can lose the acknowledgement after the server applied
            # COMMIT. Do not roll that durable graph back or report a false
            # rejection; inspect it through a fresh Session instead.
            _retire_turn_session_best_effort(db, task_id=task_id)
            session_retired = True
            if _reconcile_claimed_turn_after_commit_ack_failure(
                task_id=task_id,
                task_owner_user_id=task_owner_user_id,
                payload=payload,
                claimed=result,
            ):
                invalidate_task_cache_best_effort(task_id)
                return result
            raise TaskTurnCommitOutcomeUnknown(
                task_id, payload.turn_id
            ) from commit_error
    finally:
        if not session_retired:
            _retire_turn_session_best_effort(db, task_id=task_id)

    invalidate_task_cache_best_effort(task_id)
    assert result is not None
    return result


def _reconcile_claimed_turn_after_commit_ack_failure(
    *,
    task_id: int,
    task_owner_user_id: int,
    payload: TaskTurnPayload,
    claimed: _ClaimedTurn,
) -> bool:
    """Check the complete accepted turn graph in a fresh owned Session."""

    from ..models.chat_message import TaskChatMessage
    from ..models.database import get_session_local
    from ..models.uploaded_file import UploadedFile

    SessionLocal = get_session_local()
    for attempt in range(3):
        reconcile_db: Session | None = None
        try:
            reconcile_db = SessionLocal()
            task = (
                reconcile_db.query(Task)
                .filter(
                    Task.id == task_id,
                    Task.user_id == task_owner_user_id,
                    Task.status == TaskStatus.RUNNING,
                    Task.run_id == claimed.run_id,
                    Task.runner_id == claimed.task_lease.runner_id,
                )
                .first()
            )
            if task is not None:
                message = (
                    reconcile_db.query(TaskChatMessage)
                    .filter(
                        TaskChatMessage.task_id == task_id,
                        TaskChatMessage.role == "user",
                        TaskChatMessage.turn_id == payload.turn_id,
                        TaskChatMessage.content == payload.transcript_message.strip(),
                        TaskChatMessage.delivery_status.in_(
                            (
                                DELIVERY_PENDING,
                                DELIVERY_DISPATCHED,
                                DELIVERY_COMPLETED,
                            )
                        ),
                    )
                    .first()
                )
                if message is not None:
                    if not payload.file_ids:
                        return True
                    bound_count = (
                        reconcile_db.query(UploadedFile.file_id)
                        .filter(
                            UploadedFile.file_id.in_(payload.file_ids),
                            UploadedFile.user_id == task_owner_user_id,
                            UploadedFile.task_id == task_id,
                        )
                        .count()
                    )
                    if bound_count == len(set(payload.file_ids)):
                        return True
        except Exception:
            logger.warning(
                "turn commit reconciliation attempt %s failed for task %s",
                attempt + 1,
                task_id,
                exc_info=True,
            )
        finally:
            if reconcile_db is not None:
                _retire_turn_session_best_effort(reconcile_db, task_id=task_id)
        if attempt < 2:
            time.sleep(0.01)
    return False


def _retire_turn_session_best_effort(db: Session, *, task_id: int) -> None:
    """Release an owned Session without replacing the transaction error."""

    try:
        db.close()
        return
    except Exception:
        logger.warning(
            "failed to close turn session for task %s", task_id, exc_info=True
        )
    try:
        db.invalidate()
    except Exception:
        logger.warning(
            "failed to invalidate turn session for task %s",
            task_id,
            exc_info=True,
        )


def commit_claimed_turn_or_reconcile(
    db: Session,
    *,
    task_id: int,
    task_owner_user_id: int,
    payload: TaskTurnPayload,
    claimed: _ClaimedTurn,
) -> None:
    """Commit a complete turn graph or prove an ambiguous COMMIT succeeded.

    On an acknowledgement failure the caller-owned Session is retired before
    any fresh read, which prevents a retained connection from starving a
    one-slot pool. The caller may safely schedule only after this returns.
    """

    try:
        db.flush()
    except Exception:
        db.rollback()
        raise
    try:
        db.commit()
        return
    except Exception as commit_error:
        _retire_turn_session_best_effort(db, task_id=task_id)
        if _reconcile_claimed_turn_after_commit_ack_failure(
            task_id=task_id,
            task_owner_user_id=task_owner_user_id,
            payload=payload,
            claimed=claimed,
        ):
            return
        raise TaskTurnCommitOutcomeUnknown(task_id, payload.turn_id) from commit_error


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


def settle_task_lease_isolated(
    lease: TaskLease,
    *,
    error_message: str | None = None,
    client_error_message: str = CLIENT_SAFE_TASK_FAILURE,
    client_message_type: str = TASK_FAILURE_MESSAGE_TYPE,
) -> bool:
    """Settle exactly one run/runner lease in one worker-owned Session.

    A genuine execution error is failed and released by one conditional UPDATE
    and one commit. Related workforce/trigger projections and the durable error
    transcript are staged in that same transaction. If the row is already
    terminal (for example a completion broadcast failed), ``finish_turn``
    reconciles and releases it without rewriting the outcome.

    The return value means that the requested outcome was committed. With an
    ``error_message`` it is ``True`` only when this call changed the exact owned
    run to FAILED. A pre-existing terminal/control outcome may still be
    reconciled and released, but returns ``False`` so callers do not publish a
    contradictory failure event.

    On checkout or commit failure the transaction is rolled back and the lease
    is intentionally retained for TTL recovery; this function never creates an
    ownerless RUNNING task.
    """
    from ..models.database import get_session_local
    from .chat_history_service import persist_assistant_message_no_commit
    from .workforce_runtime import sync_workforce_run_status

    SessionLocal = get_session_local()
    with SessionLocal() as settle_db:
        try:
            if error_message is not None:
                failed = fail_and_release_task_lease_no_commit(
                    settle_db,
                    lease,
                    error_message=error_message,
                )
                if failed:
                    task = settle_db.query(Task).filter(Task.id == lease.task_id).one()
                    sync_workforce_run_status(settle_db, task, TaskStatus.FAILED)
                    sync_trigger_run_status(settle_db, task, TaskStatus.FAILED)
                    if task.user_id is not None:
                        persist_assistant_message_no_commit(
                            settle_db,
                            task_id=lease.task_id,
                            user_id=int(task.user_id),
                            content=client_error_message,
                            message_type=client_message_type,
                        )
                    settle_db.commit()
                    invalidate_task_cache_best_effort(lease.task_id)
                    return True

                # The task may already have committed a terminal/control state.
                # Clear the failed conditional UPDATE transaction before the
                # fenced terminal reconciliation below.
                settle_db.rollback()

            settled = finish_turn(
                settle_db,
                lease.task_id,
                task_lease=lease,
            )
            if error_message is not None:
                return False
            return settled
        except Exception:
            settle_db.rollback()
            raise


# ===== finish_turn / _schedule_bg (new lifecycle API) =====


def finish_turn(
    bg_db: Any,
    task_id: int,
    *,
    task_lease: TaskLease | None = None,
) -> bool:
    """Reconcile terminal fields and, when supplied, release one exact lease.

    Called from ``_schedule_bg._runner`` after ``execute_task_background``
    returns. Two key properties:

      - latest-turn snapshot invariant: COMPLETED, FAILED, and the
        RUNNING-fallback branch all leave the row in a state where the
        terminal field that *doesn't* apply to the current turn is
        cleared (COMPLETED clears ``error_message``; FAILED clears
        stale ``output``). SDK consumers reading ``/v1/chat/tasks/{id}``
        therefore never see a contradictory snapshot like
        ``status='failed' + output='prior successful answer'``.
      - lease ownership guard: orchestrated callers provide the concrete
        ``TaskLease``. Every read and final write is then fenced by both
        ``runner_id`` and ``run_id``; an old coroutine cannot touch a newer
        run claimed by the same process-global runner id.

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
      - other statuses (PAUSED / WAITING_FOR_USER): control status and
        ``output`` are preserved; the lease release clears any stale
        ``error_message`` left by an earlier failed attempt
    """
    from ..models.chat_message import TaskChatMessage
    from .workforce_runtime import sync_workforce_run_status

    bg_db.expire_all()

    query = bg_db.query(Task).filter(Task.id == task_id)
    if task_lease is not None:
        # A runner id alone is not a sufficient fence: two sequential runs in
        # one process share it. Production acquisitions always return run_id;
        # refuse to mutate when a caller cannot identify the concrete run.
        if task_lease.run_id is None:
            logger.warning(
                "finish_turn: refusing unfenced lease settlement for task %s",
                task_id,
            )
            return False
        query = query.filter(
            Task.runner_id == task_lease.runner_id,
            Task.run_id == task_lease.run_id,
        )
        # PostgreSQL locks the exact owned row until release_task_lease commits;
        # SQLite serializes the write transaction. This prevents an ORM flush
        # from racing a replacement owner between the read and fenced release.
        query = query.with_for_update()

    fresh = query.first()
    if fresh is None:
        logger.info(
            "finish_turn: task %s missing or no longer owned by this lease",
            task_id,
        )
        return False

    def commit_terminal(status: TaskStatus, *, changed: bool = True) -> bool:
        if task_lease is not None:
            released = release_task_lease(bg_db, task_lease, status=status)
            if released:
                invalidate_task_cache_best_effort(task_id)
            return released
        if changed:
            bg_db.commit()
            invalidate_task_cache_best_effort(task_id)
        return changed

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
            sync_trigger_run_status(bg_db, fresh, TaskStatus.COMPLETED)
            committed = commit_terminal(TaskStatus.COMPLETED)
            logger.info(
                "finish_turn: task %s output written (%d chars)",
                task_id,
                len(latest_assistant.content),
            )
            return committed
        else:
            logger.warning(
                "finish_turn: task %s completed but no assistant message found",
                task_id,
            )
            run_changed = sync_workforce_run_status(bg_db, fresh, TaskStatus.COMPLETED)
            trigger_run_changed = sync_trigger_run_status(
                bg_db, fresh, TaskStatus.COMPLETED
            )
            return commit_terminal(
                TaskStatus.COMPLETED,
                changed=run_changed or trigger_run_changed,
            )

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
        trigger_run_changed = sync_trigger_run_status(bg_db, fresh, TaskStatus.FAILED)
        if changed or run_changed or trigger_run_changed:
            committed = commit_terminal(TaskStatus.FAILED)
            logger.info(
                "finish_turn: task %s marked failed (cleared stale output)",
                task_id,
            )
            return committed
        return commit_terminal(TaskStatus.FAILED, changed=False)

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
            task_lease is None
            and fresh.runner_id is not None
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
            return False
        # Genuinely stuck: our bg coroutine returned, no live lease elsewhere.
        apply_task_control_transition(
            fresh,
            TaskControlState.FAILED,
            status=TaskStatus.FAILED,
        )
        fresh.error_message = "Task execution failed without status update; see /steps."
        fresh.output = None  # latest-turn snapshot invariant
        sync_workforce_run_status(bg_db, fresh, TaskStatus.FAILED)
        sync_trigger_run_status(bg_db, fresh, TaskStatus.FAILED)
        committed = commit_terminal(TaskStatus.FAILED)
        logger.warning(
            "finish_turn: task %s bg coroutine returned with status=RUNNING; "
            "flipping to FAILED",
            task_id,
        )
        return committed

    # PAUSED / WAITING_FOR_USER / other: preserve the control status while
    # releasing this exact run's lease. Legacy callers still leave it alone.
    if task_lease is not None:
        return commit_terminal(status)
    return False


def sync_trigger_run_status(
    bg_db: Any,
    task: Task,
    status: TaskStatus,
    *,
    error_message: str | None = None,
) -> bool:
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
        if status in {TaskStatus.FAILED, TaskStatus.PAUSED}:
            run.error_message = error_message or task.error_message
        else:
            run.error_message = None
        bg_db.add(run)
    return True


def _schedule_bg(
    *,
    task_id: int,
    task_owner_user_id: int,
    task_source: Optional[str],
    run_id: str | None = None,
    task_lease: TaskLease | None = None,
    payload: TaskTurnPayload,
    force_fresh: bool,
    context: Optional[Dict[str, Any]],
    before_message_id: Optional[int] = None,
    mcp_runtime_authorization_policy: MCPBuiltinOAuthActorPolicy | None = None,
) -> "asyncio.Task[None]":
    """Lease-aware bg scheduler.

    Synchronous: it defines ``_runner``, schedules it via
    ``asyncio.create_task``, registers the handle and returns it — there is
    no ``await`` at this level (every await lives inside ``_runner``, which
    runs later as its own task). Being sync removes a misleading
    suspension/cancellation point right after ``begin_turn``'s claim commit.

    Owns the full lease lifecycle for the bg run:

      - primary turn claims supply the exact lease committed with the turn
        state. Delayed execution validates that same run/runner fence before
        doing local work. Legacy ``schedule_existing`` callers acquire at
        ``_runner`` entry instead. If another worker already owns the task,
        the scheduler returns without invoking ``execute_task_background`` or
        ``finish_turn``.
      - heartbeat alongside the run.
      - release in ``finally`` as the single owner of the release call when
        execution returns normally or raises a non-pool error.
        ``execute_task_background`` only writes ``task.status`` and never
        touches the lease columns; the scheduler is responsible for the whole
        lease lifecycle. A SQLAlchemy pool checkout timeout is the deliberate
        exception: heartbeat stops, the exact lease remains intact, and TTL
        recovery reconciles it without an immediate second checkout.

    Takes primitives only (``task_id`` / ``task_owner_user_id`` /
    ``task_source``); the
    bg run loads its own snapshot and opens its own sessions, so no
    caller-bound ORM object crosses into the coroutine.
    """
    from ..api.websocket import (
        background_task_manager,
        create_terminal_task_error_event,
        execute_task_background,
    )
    from ..api.websocket import manager as websocket_manager

    async def _runner() -> None:
        lease: TaskLease | None = task_lease
        stop_event: asyncio.Event | None = None
        hb_task: asyncio.Task[TaskLeaseHeartbeatOutcome] | None = None
        settlement_error: str | None = None
        client_history_error_message: str | None = None
        client_history_message_type = TASK_FAILURE_MESSAGE_TYPE
        broadcast_error_message: str | None = None
        defer_settlement_to_ttl_recovery = False
        skip_delivery_reconciliation = False
        # Positive evidence for finalize's delivery target: once
        # ``execute_task_background`` has been invoked, the run may have
        # consumed the user message, so a failure must not be projected as a
        # retryable ``failed`` delivery (it would invite a double execution).
        execution_started = False
        cleanup_cancellation: asyncio.CancelledError | None = None
        try:
            # A cancellation can land after the worker commits but before the
            # await returns. Drain that worker and settle any late lease before
            # propagating cancellation, otherwise the task remains RUNNING
            # with no coroutine that knows it owns the lease.
            preacquired_lease = lease is not None
            if lease is None:
                lease = await acquire_task_lease_cancellation_safe(
                    lambda: acquire_task_lease_isolated(
                        task_id,
                        expected_run_id=run_id,
                    ),
                    lambda acquired: settle_task_lease_isolated(
                        acquired,
                        error_message=(
                            "task execution cancelled during lease acquisition"
                        ),
                    ),
                )
                if lease is None:
                    logger.info(
                        "task %s acquired by another worker; skipping "
                        "execution and finish_turn",
                        task_id,
                    )
                    return
            assert lease is not None

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
                if preacquired_lease:
                    validation = await run_db_io_cancellation_safe(
                        lambda: validate_preacquired_task_lease_isolated(lease)
                    )
                    if validation == TaskLeaseRefreshState.LOST:
                        defer_settlement_to_ttl_recovery = True
                        logger.info(
                            "task %s pre-acquired lease is no longer current; "
                            "skipping delayed local execution",
                            task_id,
                        )
                        return
                    if validation == TaskLeaseRefreshState.SETTLEMENT_READY:
                        # The task reached a terminal/control state before this
                        # delayed run executed anything. Settlement below is
                        # still correct, but the turn body never ran, so the
                        # delivery row must NOT be closed as ``completed`` —
                        # leave it for the authoritative writer.
                        skip_delivery_reconciliation = True
                        return

                async def execute_owned_run() -> None:
                    # Snapshot and scope resolution each own a short Session in a
                    # worker. Drain either worker if cancellation arrives so final
                    # settlement never races an abandoned pool checkout.
                    snapshot = await run_db_io_cancellation_safe(
                        lambda: load_task_setup_snapshot_sync(
                            task_id,
                            task_owner_user_id,
                            before_message_id=before_message_id,
                        )
                    )
                    if snapshot is None:
                        raise RuntimeError("task vanished before snapshot load")

                    scope = await run_db_io_cancellation_safe(
                        lambda: resolve_execution_scope(task_id)
                    )
                    nonlocal execution_started
                    execution_started = True
                    await execute_task_background(
                        task_id=task_id,
                        user_message=payload.transcript_message,
                        context=_execution_context_with_turn_id(
                            context,
                            payload.turn_id,
                            files=payload.attachments,
                        ),
                        agent_manager=_get_agent_manager(),
                        task_owner_user_id=task_owner_user_id,
                        before_message_id=before_message_id,
                        llm_user_message=payload.execution_message,
                        task_setup_snapshot=snapshot,
                        expected_run_id=lease.run_id,
                        task_lease=lease,
                        resolved_execution_scope=scope,
                        mcp_runtime_authorization_policy=(
                            mcp_runtime_authorization_policy
                        ),
                    )

                await run_while_task_lease_owned(
                    execute_owned_run(),
                    hb_task,
                )
            except TaskLeaseLostError:
                defer_settlement_to_ttl_recovery = True
                logger.warning(
                    "task %s execution cancelled after lease ownership loss",
                    task_id,
                )
            except asyncio.CancelledError:
                # Settlement text lands in ``task.error_message`` and, for a
                # task with an owner, in the transcript as an assistant
                # message. An external task's transcript is read by the
                # visitor who asked the question, so that audience gets a
                # sentence written for it instead of the operator wording
                # every other source keeps.
                if task_source == EXTERNAL_TASK_SOURCE:
                    settlement_error = EXTERNAL_TURN_INTERRUPTED_MESSAGE
                    client_history_error_message = settlement_error
                    client_history_message_type = CLIENT_SAFE_FAILURE_MESSAGE_TYPE
                else:
                    settlement_error = "task execution cancelled"
                raise
            except Exception as setup_or_run_err:
                if is_database_pool_timeout(setup_or_run_err):
                    # The failed setup/run checkout already waited for the
                    # exhausted pool. An immediate settlement would perform a
                    # second checkout against the same exhausted pool. Keep
                    # the exact run/runner lease intact and let its TTL recovery
                    # path reconcile the task once capacity returns.
                    defer_settlement_to_ttl_recovery = True
                    logger.error(
                        "task_id=%s component=setup/run database pool checkout "
                        "timed out; skipping immediate settlement and retaining "
                        "lease for TTL recovery: %s",
                        task_id,
                        setup_or_run_err,
                        exc_info=True,
                    )
                else:
                    is_public_safe_mcp_error = isinstance(
                        setup_or_run_err, RequiredMCPUnavailableError
                    )
                    if is_public_safe_mcp_error:
                        # This exception's string contract is deliberately
                        # public-safe. Preserve it exactly in the durable task
                        # and TriggerRun projections; adding the exception type
                        # would replace the user-facing failure contract even
                        # though the fenced settlement remains the sole owner of
                        # the terminal write.
                        settlement_error = str(setup_or_run_err)
                        client_history_message_type = CLIENT_SAFE_FAILURE_MESSAGE_TYPE
                        broadcast_error_message = (
                            required_mcp_unavailable_client_message(
                                setup_or_run_err,
                                fallback=CLIENT_SAFE_TASK_FAILURE,
                            )
                        )
                    else:
                        settlement_error = (
                            "setup/run error: "
                            f"{type(setup_or_run_err).__name__}: {setup_or_run_err}"
                        )
                        broadcast_error_message = CLIENT_SAFE_TASK_FAILURE
                    logger.error(
                        "bg task %s setup/run failed: %s",
                        task_id,
                        setup_or_run_err,
                        exc_info=True,
                    )
        finally:
            turn_id = getattr(payload, "turn_id", None)
            if lease is not None:
                try:
                    heartbeat_outcome = await stop_task_lease_heartbeat(
                        hb_task, stop_event
                    )
                    if (
                        isinstance(heartbeat_outcome, TaskLeaseHeartbeatOutcome)
                        and heartbeat_outcome.requires_ttl_recovery
                    ):
                        defer_settlement_to_ttl_recovery = True
                        logger.error(
                            "task_id=%s component=lease-heartbeat unhealthy "
                            "at shutdown; skipping immediate settlement and "
                            "retaining lease for TTL recovery (lost=%s, "
                            "pool_timeout=%s)",
                            task_id,
                            heartbeat_outcome.lease_lost,
                            heartbeat_outcome.pool_timeout is not None,
                        )
                except asyncio.CancelledError as exc:
                    cleanup_cancellation = exc
                except Exception:
                    logger.warning(
                        "task %s heartbeat shutdown failed",
                        task_id,
                        exc_info=True,
                    )

                if not defer_settlement_to_ttl_recovery:
                    lease_settled = False
                    try:
                        settled = await run_db_io_cancellation_safe(
                            lambda: settle_task_lease_isolated(
                                lease,
                                error_message=settlement_error,
                                client_error_message=(
                                    client_history_error_message
                                    or broadcast_error_message
                                    or CLIENT_SAFE_TASK_FAILURE
                                ),
                                client_message_type=client_history_message_type,
                            )
                        )
                        # Gate on the returned value, not on "didn't raise":
                        # ``settle_task_lease_isolated`` returns ``False``
                        # without raising when the settlement was fenced out
                        # (lease row gone or owned by a newer run) or when a
                        # pre-existing terminal/control outcome was reconciled
                        # instead of this coroutine's view. In both cases this
                        # coroutine is no longer authoritative for the turn and
                        # must not close its delivery row.
                        lease_settled = bool(settled)
                        if settled and broadcast_error_message is not None:
                            try:
                                await websocket_manager.broadcast_to_task(
                                    create_terminal_task_error_event(
                                        task_id,
                                        broadcast_error_message,
                                    ),
                                    task_id,
                                )
                            except Exception:
                                logger.warning(
                                    "task %s failure was committed but its "
                                    "terminal broadcast failed",
                                    task_id,
                                    exc_info=True,
                                )
                    except asyncio.CancelledError as exc:
                        cleanup_cancellation = cleanup_cancellation or exc
                    except Exception as settle_err:
                        # Preserve the concrete lease on failure. Its TTL is the
                        # recovery path; clearing it here would create an ownerless
                        # RUNNING row and permit an overlapping execution.
                        logger.error(
                            "task %s lease settlement failed: %s; "
                            "retaining lease for TTL recovery",
                            task_id,
                            settle_err,
                            exc_info=True,
                        )

                    # Only when this coroutine's own settlement committed is it
                    # safe to close the delivery row (xorbitsai/xagent-saas#332):
                    # a projection that never reached ``dispatched`` would
                    # otherwise orphan the row at ``pending`` and reject every
                    # same-id retry forever. Skipped whenever this coroutine is
                    # not the authoritative closer: settle raised (lease retained
                    # for TTL recovery), settle was fenced out / superseded
                    # (returned ``False``), settlement was deferred, or the
                    # pre-execution SETTLEMENT_READY short-circuit fired (the
                    # turn body never ran, so ``completed`` would be a lie).
                    # KNOWN GAP: rows skipped here stay ``pending`` permanently —
                    # lease TTL recovery terminalizes the *task* only and nothing
                    # in this tree redrives delivery rows. The skip is still
                    # correct (a non-authoritative close is worse). Note the
                    # deferral triggers correlate with this bug's own trigger:
                    # post-schedule dispatch, setup/run and settle all draw from
                    # the same DB pool, so one exhaustion event can both orphan
                    # the row and defer the settlement that would have closed it.
                    # Tracked in xorbitsai/xagent-saas#409.
                    if lease_settled and not skip_delivery_reconciliation:
                        try:
                            await run_db_io_cancellation_safe(
                                lambda: _reconcile_finalized_turn_delivery(
                                    task_id=task_id,
                                    turn_id=turn_id,
                                    settlement_error=settlement_error,
                                    execution_started=execution_started,
                                )
                            )
                        except asyncio.CancelledError as exc:
                            cleanup_cancellation = cleanup_cancellation or exc
                        except Exception:
                            logger.warning(
                                "task %s finalize delivery reconciliation failed",
                                task_id,
                                exc_info=True,
                            )
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
            if mcp_runtime_authorization_policy is not None:
                try:
                    await run_db_io_cancellation_safe(
                        lambda: _get_agent_manager().remove_agent(
                            task_id,
                            task_owner_user_id,
                            expected_run_id=(
                                task_lease.run_id if task_lease is not None else run_id
                            ),
                        )
                    )
                except asyncio.CancelledError as exc:
                    cleanup_cancellation = cleanup_cancellation or exc
                except Exception:
                    logger.warning(
                        "actor runtime cleanup failed for task %s",
                        task_id,
                        exc_info=True,
                    )
            if cleanup_cancellation is not None:
                raise cleanup_cancellation

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
    context: Optional[Dict[str, Any]],
    turn_id: str,
    *,
    files: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    execution_context = dict(context or {})
    if turn_id:
        execution_context["turn_id"] = turn_id
    # A resumed/retried turn may already carry its authoritative file batch.
    if files and not execution_context.get("files"):
        execution_context["files"] = [dict(file_info) for file_info in files]
    return execution_context
