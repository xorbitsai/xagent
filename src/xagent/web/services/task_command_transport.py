"""Durable, cross-worker transport for task execution commands.

Ingress commits a command before acknowledging it. Every web worker runs the
same dispatcher; a database claim chooses one consumer while per-task ordering
prevents a later command from overtaking an earlier unfinished command.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from sqlalchemy import and_, exists, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Query, Session, aliased

from ...config import (
    get_task_lease_heartbeat_seconds,
    get_task_lease_ttl_seconds,
)
from ..models.task import Task, TaskStatus
from ..models.task_command import TaskExecutionCommand
from .task_lease_service import get_runner_id

logger = logging.getLogger(__name__)

COMMAND_PENDING = "pending"
COMMAND_PROCESSING = "processing"
COMMAND_COMPLETED = "completed"
COMMAND_FAILED = "failed"
COMMAND_TERMINAL = (COMMAND_COMPLETED, COMMAND_FAILED)

COMMAND_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,64}")
MAX_COMMAND_FAILURES = 5
MAX_COMMAND_DEFERS = 60
DISPATCHER_IDLE_SECONDS = 0.5
DISPATCHER_CONCURRENCY = 4


class TaskCommandKind(str, enum.Enum):
    MESSAGE = "message"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"


class TaskCommandDeferred(RuntimeError):
    """The command is durable but its downstream handoff is still pending."""


class TaskCommandRejected(RuntimeError):
    """The downstream handoff reached a durable failed state."""

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class EnqueuedTaskCommand:
    command_id: int
    client_command_id: str
    created: bool
    payload_matches: bool
    status: str


@dataclass(frozen=True)
class ClaimedTaskCommand:
    id: int
    task_id: int
    actor_user_id: int | None
    command_id: str
    kind: TaskCommandKind
    payload: dict[str, Any]
    target_run_id: str | None
    attempt_count: int
    failure_count: int = 0
    defer_count: int = 0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _matches_existing(
    command: TaskExecutionCommand,
    *,
    actor_user_id: int | None,
    kind: TaskCommandKind,
    payload: dict[str, Any],
) -> bool:
    stored_payload: dict[str, Any] = (
        command.payload if isinstance(command.payload, dict) else {}
    )
    stored_actor_user_id = getattr(command, "actor_user_id", None)
    return (
        stored_actor_user_id == actor_user_id
        and str(command.kind) == kind.value
        and _canonical_payload(stored_payload) == _canonical_payload(payload)
    )


def enqueue_task_command(
    db: Session,
    *,
    task_id: int,
    actor_user_id: int | None,
    command_id: str,
    kind: TaskCommandKind,
    payload: dict[str, Any],
) -> EnqueuedTaskCommand:
    """Commit an idempotent command and return only after it is durable."""

    normalized_id = command_id.strip()
    if COMMAND_ID_PATTERN.fullmatch(normalized_id) is None:
        raise ValueError("command_id must be 1-64 URL-safe characters")
    task = db.query(Task).filter(Task.id == int(task_id)).first()
    if task is None:
        raise ValueError(f"Task {task_id} not found")

    existing = (
        db.query(TaskExecutionCommand)
        .filter(
            TaskExecutionCommand.task_id == int(task_id),
            TaskExecutionCommand.command_id == normalized_id,
        )
        .first()
    )
    if existing is not None:
        matches = _matches_existing(
            existing,
            actor_user_id=actor_user_id,
            kind=kind,
            payload=payload,
        )
        return EnqueuedTaskCommand(
            command_id=int(existing.id),
            client_command_id=normalized_id,
            created=False,
            payload_matches=matches,
            status=str(existing.status),
        )

    active_runner_id = (
        task.runner_id
        if task.status == TaskStatus.RUNNING and task.runner_id is not None
        else None
    )
    command = TaskExecutionCommand(
        task_id=int(task_id),
        actor_user_id=actor_user_id,
        command_id=normalized_id,
        kind=kind.value,
        payload=payload,
        target_run_id=task.run_id,
        target_runner_id=active_runner_id,
        status=COMMAND_PENDING,
    )
    db.add(command)
    try:
        db.commit()
        db.refresh(command)
    except IntegrityError:
        db.rollback()
        raced = (
            db.query(TaskExecutionCommand)
            .filter(
                TaskExecutionCommand.task_id == int(task_id),
                TaskExecutionCommand.command_id == normalized_id,
            )
            .one()
        )
        matches = _matches_existing(
            raced,
            actor_user_id=actor_user_id,
            kind=kind,
            payload=payload,
        )
        return EnqueuedTaskCommand(
            command_id=int(raced.id),
            client_command_id=normalized_id,
            created=False,
            payload_matches=matches,
            status=str(raced.status),
        )

    notify_task_command_dispatcher()
    return EnqueuedTaskCommand(
        command_id=int(command.id),
        client_command_id=normalized_id,
        created=True,
        payload_matches=True,
        status=COMMAND_PENDING,
    )


def _claim_availability_predicate(now: datetime) -> Any:
    return or_(
        and_(
            TaskExecutionCommand.status == COMMAND_PENDING,
            or_(
                TaskExecutionCommand.claim_expires_at.is_(None),
                TaskExecutionCommand.claim_expires_at < now,
            ),
        ),
        and_(
            TaskExecutionCommand.status == COMMAND_PROCESSING,
            TaskExecutionCommand.claim_expires_at < now,
        ),
    )


def _command_routing_predicate(runner_id: str, now: datetime) -> Any:
    return or_(
        TaskExecutionCommand.target_runner_id.is_(None),
        TaskExecutionCommand.target_runner_id == runner_id,
        Task.runner_id == runner_id,
        Task.runner_id.is_(None),
        Task.lease_expires_at.is_(None),
        Task.lease_expires_at < now,
    )


def _unfinished_earlier_command() -> Any:
    earlier = aliased(TaskExecutionCommand)
    return exists(
        select(1).where(
            earlier.task_id == TaskExecutionCommand.task_id,
            earlier.id < TaskExecutionCommand.id,
            earlier.status.notin_(COMMAND_TERMINAL),
        )
    )


def _claimable_query(
    db: Session, *, runner_id: str, command_db_id: int | None
) -> Query[Any]:
    now = _utc_now()
    query = (
        db.query(TaskExecutionCommand)
        .join(Task, Task.id == TaskExecutionCommand.task_id)
        .filter(
            _claim_availability_predicate(now),
            ~_unfinished_earlier_command(),
            _command_routing_predicate(runner_id, now),
        )
    )
    if command_db_id is not None:
        query = query.filter(TaskExecutionCommand.id == command_db_id)
    return query.order_by(TaskExecutionCommand.id.asc())


def claim_task_command(
    db: Session,
    *,
    runner_id: str | None = None,
    command_db_id: int | None = None,
) -> ClaimedTaskCommand | None:
    """Atomically claim the oldest eligible command for this worker."""

    resolved_runner_id = runner_id or get_runner_id()
    candidate = _claimable_query(
        db,
        runner_id=resolved_runner_id,
        command_db_id=command_db_id,
    ).first()
    if candidate is None:
        return None

    now = _utc_now()
    expires = now + timedelta(seconds=get_task_lease_ttl_seconds())
    routable_task = exists(
        select(1).where(
            Task.id == TaskExecutionCommand.task_id,
            _command_routing_predicate(resolved_runner_id, now),
        )
    )
    claimed = (
        db.query(TaskExecutionCommand)
        .filter(
            TaskExecutionCommand.id == int(candidate.id),
            _claim_availability_predicate(now),
            ~_unfinished_earlier_command(),
            routable_task,
        )
        .update(
            {
                TaskExecutionCommand.status: COMMAND_PROCESSING,
                TaskExecutionCommand.claimed_by: resolved_runner_id,
                TaskExecutionCommand.claim_expires_at: expires,
                TaskExecutionCommand.attempt_count: (
                    TaskExecutionCommand.attempt_count + 1
                ),
                TaskExecutionCommand.error: None,
                TaskExecutionCommand.updated_at: now,
            },
            synchronize_session=False,
        )
    )
    if claimed != 1:
        db.rollback()
        return None
    db.commit()
    fresh = (
        db.query(TaskExecutionCommand)
        .filter(TaskExecutionCommand.id == int(candidate.id))
        .one()
    )
    payload: dict[str, Any] = fresh.payload if isinstance(fresh.payload, dict) else {}
    return ClaimedTaskCommand(
        id=int(fresh.id),
        task_id=int(fresh.task_id),
        actor_user_id=(
            int(fresh.actor_user_id) if fresh.actor_user_id is not None else None
        ),
        command_id=str(fresh.command_id),
        kind=TaskCommandKind(str(fresh.kind)),
        payload=payload,
        target_run_id=(str(fresh.target_run_id) if fresh.target_run_id else None),
        attempt_count=int(fresh.attempt_count or 0),
        failure_count=int(fresh.failure_count or 0),
        defer_count=int(fresh.defer_count or 0),
    )


def renew_task_command_claim(
    command_db_id: int,
    runner_id: str,
    *,
    expected_attempt_count: int | None = None,
) -> bool:
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    now = _utc_now()
    with SessionLocal() as db:
        query = db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.id == command_db_id,
            TaskExecutionCommand.status == COMMAND_PROCESSING,
            TaskExecutionCommand.claimed_by == runner_id,
        )
        if expected_attempt_count is not None:
            query = query.filter(
                TaskExecutionCommand.attempt_count == expected_attempt_count
            )
        updated = query.update(
            {
                TaskExecutionCommand.claim_expires_at: now
                + timedelta(seconds=get_task_lease_ttl_seconds()),
                TaskExecutionCommand.updated_at: now,
            },
            synchronize_session=False,
        )
        db.commit()
        return updated == 1


async def _claim_heartbeat(
    command_db_id: int,
    runner_id: str,
    attempt_count: int,
    stop_event: asyncio.Event,
) -> None:
    interval = get_task_lease_heartbeat_seconds()
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        try:
            renewed = await asyncio.to_thread(
                renew_task_command_claim,
                command_db_id,
                runner_id,
                expected_attempt_count=attempt_count,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to renew task command claim %s: %s",
                command_db_id,
                exc,
                exc_info=True,
            )
            continue
        if not renewed:
            return


def finish_task_command(
    command_db_id: int,
    runner_id: str,
    *,
    result: dict[str, Any] | None = None,
    expected_attempt_count: int | None = None,
) -> bool:
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    now = _utc_now()
    with SessionLocal() as db:
        query = db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.id == command_db_id,
            TaskExecutionCommand.status == COMMAND_PROCESSING,
            TaskExecutionCommand.claimed_by == runner_id,
        )
        if expected_attempt_count is not None:
            query = query.filter(
                TaskExecutionCommand.attempt_count == expected_attempt_count
            )
        updated = query.update(
            {
                TaskExecutionCommand.status: COMMAND_COMPLETED,
                TaskExecutionCommand.result: result,
                TaskExecutionCommand.error: None,
                TaskExecutionCommand.claimed_by: None,
                TaskExecutionCommand.claim_expires_at: None,
                TaskExecutionCommand.completed_at: now,
                TaskExecutionCommand.updated_at: now,
            },
            synchronize_session=False,
        )
        db.commit()
        return updated == 1


def fail_task_command(
    command_db_id: int,
    runner_id: str,
    error: str,
    *,
    force_terminal: bool = False,
    expected_attempt_count: int | None = None,
    result: dict[str, Any] | None = None,
) -> bool:
    """Retry a failed claim, or make it terminal after bounded attempts."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    now = _utc_now()
    with SessionLocal() as db:
        snapshot_query = db.query(
            TaskExecutionCommand.failure_count,
            TaskExecutionCommand.attempt_count,
        ).filter(
            TaskExecutionCommand.id == command_db_id,
            TaskExecutionCommand.status == COMMAND_PROCESSING,
            TaskExecutionCommand.claimed_by == runner_id,
        )
        if expected_attempt_count is not None:
            snapshot_query = snapshot_query.filter(
                TaskExecutionCommand.attempt_count == expected_attempt_count
            )
        snapshot = snapshot_query.first()
        if snapshot is None:
            return False
        observed_failure_count = int(snapshot.failure_count or 0)
        observed_attempt_count = int(snapshot.attempt_count or 0)
        failure_count = observed_failure_count + 1
        terminal = force_terminal or failure_count >= MAX_COMMAND_FAILURES
        updated = (
            db.query(TaskExecutionCommand)
            .filter(
                TaskExecutionCommand.id == command_db_id,
                TaskExecutionCommand.status == COMMAND_PROCESSING,
                TaskExecutionCommand.claimed_by == runner_id,
                TaskExecutionCommand.failure_count == observed_failure_count,
                TaskExecutionCommand.attempt_count == observed_attempt_count,
            )
            .update(
                {
                    TaskExecutionCommand.status: (
                        COMMAND_FAILED if terminal else COMMAND_PENDING
                    ),
                    TaskExecutionCommand.failure_count: failure_count,
                    TaskExecutionCommand.error: error[:4000],
                    TaskExecutionCommand.result: result,
                    TaskExecutionCommand.claimed_by: None,
                    TaskExecutionCommand.claim_expires_at: (
                        None
                        if terminal
                        else now + timedelta(seconds=min(2**failure_count, 30))
                    ),
                    TaskExecutionCommand.updated_at: now,
                    TaskExecutionCommand.completed_at: now if terminal else None,
                },
                synchronize_session=False,
            )
        )
        db.commit()
        if updated != 1:
            return False
    if not terminal:
        notify_task_command_dispatcher()
    return True


def defer_task_command(
    command_db_id: int,
    runner_id: str,
    reason: str,
    *,
    expected_attempt_count: int | None = None,
) -> bool:
    """Release a claim for retry without consuming the failure budget."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    now = _utc_now()
    with SessionLocal() as db:
        snapshot_query = db.query(
            TaskExecutionCommand.defer_count,
            TaskExecutionCommand.attempt_count,
        ).filter(
            TaskExecutionCommand.id == command_db_id,
            TaskExecutionCommand.status == COMMAND_PROCESSING,
            TaskExecutionCommand.claimed_by == runner_id,
        )
        if expected_attempt_count is not None:
            snapshot_query = snapshot_query.filter(
                TaskExecutionCommand.attempt_count == expected_attempt_count
            )
        snapshot = snapshot_query.first()
        if snapshot is None:
            return False
        observed_defer_count = int(snapshot.defer_count or 0)
        observed_attempt_count = int(snapshot.attempt_count or 0)
        defer_count = observed_defer_count + 1
        terminal = defer_count >= MAX_COMMAND_DEFERS
        updated = (
            db.query(TaskExecutionCommand)
            .filter(
                TaskExecutionCommand.id == command_db_id,
                TaskExecutionCommand.status == COMMAND_PROCESSING,
                TaskExecutionCommand.claimed_by == runner_id,
                TaskExecutionCommand.defer_count == observed_defer_count,
                TaskExecutionCommand.attempt_count == observed_attempt_count,
            )
            .update(
                {
                    TaskExecutionCommand.status: (
                        COMMAND_FAILED if terminal else COMMAND_PENDING
                    ),
                    TaskExecutionCommand.defer_count: defer_count,
                    TaskExecutionCommand.error: (
                        (
                            f"Deferred command exceeded retry budget: {reason}"
                            if terminal
                            else reason
                        )[:4000]
                    ),
                    TaskExecutionCommand.claimed_by: None,
                    TaskExecutionCommand.claim_expires_at: (
                        None if terminal else now + timedelta(seconds=1)
                    ),
                    TaskExecutionCommand.updated_at: now,
                    TaskExecutionCommand.completed_at: now if terminal else None,
                },
                synchronize_session=False,
            )
        )
        db.commit()
        if updated != 1:
            return False
    if not terminal:
        notify_task_command_dispatcher()
    return True


def retry_failed_task_command(
    db: Session,
    command_db_id: int,
    *,
    target_run_id: str | None,
    target_runner_id: str | None,
) -> bool:
    """Atomically reset a terminal command for an explicit client retry."""

    now = _utc_now()
    updated = (
        db.query(TaskExecutionCommand)
        .filter(
            TaskExecutionCommand.id == command_db_id,
            TaskExecutionCommand.status == COMMAND_FAILED,
        )
        .update(
            {
                TaskExecutionCommand.status: COMMAND_PENDING,
                TaskExecutionCommand.failure_count: 0,
                TaskExecutionCommand.defer_count: 0,
                TaskExecutionCommand.error: None,
                TaskExecutionCommand.result: None,
                TaskExecutionCommand.claimed_by: None,
                TaskExecutionCommand.claim_expires_at: None,
                TaskExecutionCommand.target_run_id: target_run_id,
                TaskExecutionCommand.target_runner_id: target_runner_id,
                TaskExecutionCommand.completed_at: None,
                TaskExecutionCommand.updated_at: now,
            },
            synchronize_session=False,
        )
    )
    db.commit()
    if updated == 1:
        notify_task_command_dispatcher()
        return True
    return False


def task_has_live_foreign_runner(
    task_id: int,
    *,
    runner_id: str | None = None,
) -> bool:
    """Return whether another process currently owns the task lease."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    resolved_runner_id = runner_id or get_runner_id()
    now = _utc_now()
    with SessionLocal() as db:
        return (
            db.query(Task.id)
            .filter(
                Task.id == task_id,
                Task.runner_id.is_not(None),
                Task.runner_id != resolved_runner_id,
                Task.lease_expires_at.is_not(None),
                Task.lease_expires_at >= now,
            )
            .first()
            is not None
        )


CommandExecutor = Callable[[ClaimedTaskCommand], Awaitable[dict[str, Any] | None]]
_dispatcher_wakeup: asyncio.Event | None = None
_dispatcher_task: asyncio.Task[Any] | None = None
_dispatcher_loop: asyncio.AbstractEventLoop | None = None
_prompt_dispatch_tasks: set[asyncio.Task[bool]] = set()


def notify_task_command_dispatcher() -> None:
    wakeup = _dispatcher_wakeup
    loop = _dispatcher_loop
    if wakeup is not None and loop is not None and not loop.is_closed():
        loop.call_soon_threadsafe(wakeup.set)


async def dispatch_one_task_command(
    executor: CommandExecutor,
    *,
    command_db_id: int | None = None,
) -> bool:
    from ..models.database import get_session_local

    runner_id = get_runner_id()
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        command = claim_task_command(
            db,
            runner_id=runner_id,
            command_db_id=command_db_id,
        )
    if command is None:
        return False

    stop_event = asyncio.Event()
    heartbeat = asyncio.get_running_loop().create_task(
        _claim_heartbeat(command.id, runner_id, command.attempt_count, stop_event)
    )
    try:
        result = await executor(command)
    except asyncio.CancelledError:
        # Leave the processing claim intact. Another worker may reclaim it only
        # after the claim expires, which avoids concurrent replay on shutdown.
        raise
    except TaskCommandDeferred as exc:
        await asyncio.to_thread(
            defer_task_command,
            command.id,
            runner_id,
            str(exc),
            expected_attempt_count=command.attempt_count,
        )
    except TaskCommandRejected as exc:
        await asyncio.to_thread(
            fail_task_command,
            command.id,
            runner_id,
            str(exc),
            force_terminal=True,
            expected_attempt_count=command.attempt_count,
            result=(
                {"rejection_reason": exc.reason} if exc.reason is not None else None
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Task command %s (%s) failed on attempt %s",
            command.command_id,
            command.kind.value,
            command.attempt_count,
        )
        await asyncio.to_thread(
            fail_task_command,
            command.id,
            runner_id,
            str(exc),
            expected_attempt_count=command.attempt_count,
        )
    else:
        if not await asyncio.to_thread(
            finish_task_command,
            command.id,
            runner_id,
            result=result,
            expected_attempt_count=command.attempt_count,
        ):
            logger.warning("Lost claim while completing task command %s", command.id)
    finally:
        stop_event.set()
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass
    return True


async def dispatch_task_command_promptly(
    executor: CommandExecutor,
    *,
    command_db_id: int,
) -> None:
    """Kick local consumption without tying WS receive to command execution.

    Fast commands usually finish during the short handoff window. Slow file or
    checkpoint work continues in its own task, allowing the socket receive loop
    to accept the next pause/message command instead of blocking behind it.
    """

    task = asyncio.get_running_loop().create_task(
        dispatch_one_task_command(executor, command_db_id=command_db_id)
    )
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
    except asyncio.TimeoutError:
        _prompt_dispatch_tasks.add(task)
        task.add_done_callback(_consume_prompt_dispatch_result)
        return


def _consume_prompt_dispatch_result(task: asyncio.Task[bool]) -> None:
    """Retain and observe timed-out prompt dispatches until they finish."""

    _prompt_dispatch_tasks.discard(task)
    if task.cancelled():
        return
    try:
        task.result()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Detached prompt task command dispatch failed: %s",
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )


async def _run_task_command_dispatcher_worker(executor: CommandExecutor) -> None:
    while True:
        wakeup = _dispatcher_wakeup
        if wakeup is None:
            return
        # Clear before checking for work. A notify that arrives during the DB
        # claim remains set, so an empty claim cannot erase that wakeup and
        # sleep while work is waiting.
        wakeup.clear()
        processed = await dispatch_one_task_command(executor)
        if processed:
            continue
        try:
            await asyncio.wait_for(wakeup.wait(), timeout=DISPATCHER_IDLE_SECONDS)
        except asyncio.TimeoutError:
            pass


async def run_task_command_dispatcher(executor: CommandExecutor) -> None:
    """Recover queued commands without serializing unrelated tasks."""

    global _dispatcher_loop, _dispatcher_wakeup
    _dispatcher_loop = asyncio.get_running_loop()
    _dispatcher_wakeup = asyncio.Event()
    workers = [
        asyncio.create_task(_run_task_command_dispatcher_worker(executor))
        for _ in range(DISPATCHER_CONCURRENCY)
    ]
    try:
        await asyncio.gather(*workers)
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


def start_task_command_dispatcher(executor: CommandExecutor) -> asyncio.Task[Any]:
    global _dispatcher_task
    if _dispatcher_task is not None and not _dispatcher_task.done():
        return _dispatcher_task
    _dispatcher_task = asyncio.create_task(run_task_command_dispatcher(executor))
    return _dispatcher_task


async def stop_task_command_dispatcher() -> None:
    global _dispatcher_loop, _dispatcher_task, _dispatcher_wakeup
    prompt_tasks = list(_prompt_dispatch_tasks)
    for prompt_task in prompt_tasks:
        prompt_task.cancel()
    if prompt_tasks:
        await asyncio.gather(*prompt_tasks, return_exceptions=True)
    _prompt_dispatch_tasks.clear()

    task = _dispatcher_task
    _dispatcher_task = None
    _dispatcher_wakeup = None
    _dispatcher_loop = None
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def load_task_command(command_db_id: int) -> TaskExecutionCommand | None:
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        command = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.id == command_db_id)
            .first()
        )
        if command is not None:
            # Make the session boundary explicit. Callers only read scalar
            # command state and must never depend on a live/lazy session.
            db.expunge(command)
        return command
