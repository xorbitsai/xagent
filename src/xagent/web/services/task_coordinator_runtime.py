"""Single ownership lifecycle for shared task commands and their executions.

The durable dispatcher selects commands; one coordinator serializes their
application and owns every registered execution through callback cleanup.
Run-scoped execution fences never acquire, renew or release task ownership.
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

from sqlalchemy import select
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import Session

from ...config import get_task_lease_heartbeat_seconds, get_task_lease_ttl_seconds
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.task_command import TaskExecutionCommand
from .db_runtime import (
    await_task_settlement,
    cancel_and_drain_async_task,
    drain_async_task_cancellation_safe,
    is_database_pool_timeout,
    propagate_deferred_cancellation,
    run_db_io_cancellation_safe,
)
from .task_coordinator_service import (
    TaskExecutionContext,
    TaskLease,
    TaskLeaseRenewalState,
    acquire_task_lease_no_commit,
    lock_task_execution_no_commit,
    lock_task_lease_no_commit,
    recover_expired_idle_task_lease_no_commit,
    release_task_lease_no_commit,
    renew_task_leases_no_commit,
)
from .task_lease_service import TaskLease as ExecutionLease
from .task_lease_service import TaskLeaseHeartbeatOutcome, get_runner_id

logger = logging.getLogger(__name__)

Admission = Callable[[Session, TaskLease], TaskExecutionContext | None]
Execution = Callable[[TaskExecutionContext], Awaitable[None]]
Settlement = Callable[[Session, TaskExecutionContext, BaseException | None], None]


class CoordinatorState(str, Enum):
    ACQUIRING = "acquiring"
    ACTIVE = "active"
    QUIESCING = "quiescing"
    LOST = "lost"
    CLOSED = "closed"


class TaskCoordinatorRegistry:
    """Share acquisition, batch renewals, and one running handle per task."""

    def __init__(self, session_factory: Callable[[], Session] | None = None):
        self.session_factory = session_factory or get_session_local()
        self.loop = asyncio.get_running_loop()
        self.runner_id = get_runner_id()
        self._coordinators: dict[int, TaskCoordinator] = {}
        self._close_task: asyncio.Task[None] | None = None
        self._heartbeats: dict[TaskLease, TaskCoordinator] = {}
        self._heartbeat_wake = asyncio.Event()
        self._heartbeat_runner: asyncio.Task[None] | None = None

    def _register_heartbeat(self, coordinator: TaskCoordinator) -> None:
        assert coordinator.lease is not None
        coordinator._heartbeat_done = self.loop.create_future()
        self._heartbeats[coordinator.lease] = coordinator
        if self._heartbeat_runner is None or self._heartbeat_runner.done():
            self._heartbeat_runner = self.loop.create_task(self._run_heartbeats())
        self._heartbeat_wake.set()

    async def _unregister_heartbeat(self, coordinator: TaskCoordinator) -> None:
        assert coordinator.lease is not None
        self._heartbeats.pop(coordinator.lease, None)
        self._heartbeat_wake.set()
        # A released owner must never have an unfinished renewal transaction.
        if coordinator._renewal_waiter is not None:
            await asyncio.shield(coordinator._renewal_waiter)
        done = coordinator._heartbeat_done
        if done is not None and not done.done():
            done.set_result(None)

    def _renew(
        self, leases: tuple[TaskLease, ...]
    ) -> dict[TaskLease, TaskLeaseRenewalState]:
        with self.session_factory() as db, db.begin():
            states = renew_task_leases_no_commit(db, leases)
        return states

    def _finish_heartbeat(
        self, coordinator: TaskCoordinator, error: BaseException | None = None
    ) -> None:
        assert coordinator.lease is not None
        assert coordinator._heartbeat_done is not None
        self._heartbeats.pop(coordinator.lease, None)
        if error is None:
            coordinator._heartbeat_done.set_result(None)
        else:
            coordinator._healthy = False
            coordinator._heartbeat_error = error
            coordinator._recovery_required = True
            coordinator._heartbeat_done.set_exception(error)
        coordinator._request_close()

    async def _run_heartbeats(self) -> None:
        next_refresh_at = self.loop.time() + get_task_lease_heartbeat_seconds()
        try:
            while self._heartbeats:
                next_attempt_at = min(
                    [next_refresh_at]
                    + [
                        c._retry_at
                        for c in self._heartbeats.values()
                        if c._retry_at is not None
                    ]
                )
                delay = max(0.0, next_attempt_at - self.loop.time())
                if delay:
                    self._heartbeat_wake.clear()
                    try:
                        await asyncio.wait_for(self._heartbeat_wake.wait(), delay)
                    except asyncio.TimeoutError:
                        pass
                    if self.loop.time() < next_attempt_at:
                        continue
                now = self.loop.time()
                regular_refresh = now >= next_refresh_at
                snapshot = tuple(
                    (lease, c)
                    for lease, c in self._heartbeats.items()
                    if regular_refresh
                    or (c._retry_at is not None and c._retry_at <= now)
                )
                if not snapshot:
                    continue
                waiter: asyncio.Future[None] = self.loop.create_future()
                for _, c in snapshot:
                    c._renewal_waiter = waiter
                    c._retry_at = None
                try:
                    try:
                        states = await run_db_io_cancellation_safe(
                            lambda: self._renew(tuple(lease for lease, _ in snapshot))
                        )
                    except Exception as error:
                        sqlstate = (
                            (
                                getattr(error.orig, "sqlstate", None)
                                or getattr(error.orig, "pgcode", None)
                            )
                            if isinstance(error, DBAPIError)
                            else None
                        )
                        # A renewal only extends the exact existing owner.
                        # Rollback or connection loss gives no new lease time,
                        # but does not invalidate the last acknowledged lease.
                        rolled_back = sqlstate in (
                            "55P03",
                            "57014",
                            "40P01",
                            "40001",
                        )
                        connection_error = isinstance(error, DBAPIError) and (
                            error.connection_invalidated
                            or (
                                isinstance(error, OperationalError)
                                and sqlstate is None
                                and (
                                    hasattr(error.orig, "pgcode")
                                    or hasattr(error.orig, "sqlstate")
                                )
                            )
                        )
                        retryable = (
                            rolled_back
                            or connection_error
                            or is_database_pool_timeout(error)
                        )
                        for _, c in snapshot:
                            # Preserve prior health/errors while there is margin;
                            # only an acknowledged renewal can restore health.
                            if (
                                rolled_back or connection_error
                            ) and c._has_renewal_margin():
                                continue
                            c._healthy = False
                            c._heartbeat_error = error
                            if not retryable or connection_error:
                                self._finish_heartbeat(c, error)
                        logger.warning(
                            "Coordinator heartbeat batch failed: tasks=%s count=%s "
                            "error=%s sqlstate=%s retryable=%s",
                            [lease.task_id for lease, _ in snapshot[:10]],
                            len(snapshot),
                            type(error).__name__,
                            sqlstate,
                            retryable,
                            exc_info=True,
                        )
                        if retryable:
                            # A shared database failure must not be retried by
                            # either an imminent regular tick or a row's timer.
                            next_refresh_at = (
                                self.loop.time() + get_task_lease_heartbeat_seconds()
                            )
                            for c in self._heartbeats.values():
                                c._retry_at = None
                    else:
                        for lease, c in snapshot:
                            state = states[lease]
                            if state == TaskLeaseRenewalState.DEFERRED:
                                # A skipped lock does not invalidate the last
                                # committed renewal. Reserve one heartbeat of
                                # margin, and never restore health from a skip.
                                if c._healthy and not c._has_renewal_margin():
                                    c._healthy = False
                                    logger.warning(
                                        "Task %s heartbeat remains deferred; "
                                        "last renewal %.3fs ago",
                                        c.task_id,
                                        self.loop.time() - c._last_renewed_at,
                                    )
                                c._retry_at = self.loop.time() + min(
                                    1.0, get_task_lease_heartbeat_seconds() / 4
                                )
                            elif state == TaskLeaseRenewalState.LOST:
                                c.state = CoordinatorState.LOST
                                self._finish_heartbeat(c)
                            else:
                                # Use dispatch time, not commit acknowledgement:
                                # database time spent cannot extend our margin.
                                c._last_renewed_at = now
                                c._healthy = True
                                c._heartbeat_error = None
                finally:
                    for _, c in snapshot:
                        c._renewal_waiter = None
                    waiter.set_result(None)
                if regular_refresh:
                    interval = get_task_lease_heartbeat_seconds()
                    while next_refresh_at <= self.loop.time():
                        next_refresh_at += interval
        except BaseException as error:
            for c in tuple(self._heartbeats.values()):
                self._finish_heartbeat(c, error)
            raise
        finally:
            self._heartbeat_runner = None

    def busy_command_tasks(self) -> tuple[int, ...]:
        """Let queue scans serve other tasks while this owner applies a command."""
        return tuple(
            task_id
            for task_id, coordinator in self._coordinators.items()
            if coordinator._command_tasks
            or not coordinator._healthy
            or coordinator.state
            not in (CoordinatorState.ACQUIRING, CoordinatorState.ACTIVE)
        )

    async def ensure(self, task_id: int) -> TaskCoordinator | None:
        while True:
            if self._close_task is not None:
                return None
            coordinator = self._coordinators.get(task_id)
            if coordinator is None:
                coordinator = TaskCoordinator(self, task_id)
                self._coordinators[task_id] = coordinator
                break
            if coordinator.state in (
                CoordinatorState.ACQUIRING,
                CoordinatorState.ACTIVE,
            ):
                break
            # Preserve this wake without replacing an owner that is still
            # draining. Cancellation of the waiter must not abort cleanup.
            assert coordinator._close_task is not None
            await asyncio.shield(coordinator._close_task)
        coordinator._waiters += 1
        try:
            await asyncio.shield(coordinator._startup)
            if (
                self._close_task is not None
                or coordinator.state != CoordinatorState.ACTIVE
            ):
                return None
            coordinator._delivered = True
            coordinator.wake()
            return coordinator
        finally:
            coordinator._waiters -= 1
            # Cancellation of one waiter must not cancel another's acquisition.
            # If nobody received it, drain even a late commit before releasing.
            if coordinator._waiters == 0 and not coordinator._delivered:
                await coordinator.close()

    def _remove(self, coordinator: TaskCoordinator) -> None:
        if self._coordinators.get(coordinator.task_id) is coordinator:
            del self._coordinators[coordinator.task_id]

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_all())
        await drain_async_task_cancellation_safe(self._close_task)

    async def _close_all(self) -> None:
        await asyncio.gather(*(c.close() for c in list(self._coordinators.values())))
        if self._heartbeat_runner is not None:
            await drain_async_task_cancellation_safe(self._heartbeat_runner)


class TaskCoordinator:
    """Own acquisition, heartbeat, admission, execution and drained shutdown."""

    def __init__(self, registry: TaskCoordinatorRegistry, task_id: int):
        self.task_id = task_id
        self.state = CoordinatorState.ACQUIRING
        self.lease: TaskLease | None = None
        self.wakeup = asyncio.Event()
        self._registry = registry
        self._waiters = 0
        self._delivered = False
        self._healthy = True
        self._last_renewed_at = registry.loop.time()
        self._heartbeat_error: BaseException | None = None
        self._recovery_required = False
        self._released = False
        self._heartbeat_done: asyncio.Future[None] | None = None
        self._renewal_waiter: asyncio.Future[None] | None = None
        self._retry_at: float | None = None
        self._execution_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._command_lock = asyncio.Lock()
        self._command_tasks: set[asyncio.Task[Any]] = set()
        self._children: set[asyncio.Task[Any]] = set()
        self._idle_task: asyncio.Task[None] | None = None
        self._startup = asyncio.create_task(self._start())

    def _has_renewal_margin(self) -> bool:
        return self._registry.loop.time() < (
            self._last_renewed_at
            + get_task_lease_ttl_seconds()
            - get_task_lease_heartbeat_seconds()
        )

    def wake(self) -> None:
        self.wakeup.set()

    def _acquire(self) -> TaskLease | None:
        with self._registry.session_factory() as db, db.begin():
            recover_expired_idle_task_lease_no_commit(db, self.task_id)
            return acquire_task_lease_no_commit(
                db, self.task_id, runner_id=self._registry.runner_id
            )

    def _release(self) -> bool:
        assert self.lease is not None
        with self._registry.session_factory() as db, db.begin():
            return release_task_lease_no_commit(db, self.lease)

    async def _start(self) -> None:
        self.lease = await run_db_io_cancellation_safe(self._acquire)
        if self.lease is None:
            return
        # Shutdown may have begun while the acquisition transaction was blocked.
        if self._close_task is None and self._registry._close_task is None:
            self._registry._register_heartbeat(self)
            self.state = CoordinatorState.ACTIVE

    def require_recovery(self) -> None:
        self._recovery_required = True

    def owns_execution(self, execution: ExecutionLease) -> bool:
        """Check a fixed execution fence against the actual in-process owner."""
        return self.lease is not None and (
            execution.task_id == self.task_id
            and execution.runner_id == self.lease.runner_id
            and execution.attempt_id == self.lease.attempt_id
        )

    async def observe_ownership(self, stop: asyncio.Event) -> TaskLeaseHeartbeatOutcome:
        """Observe the owner's heartbeat without creating another renewal loop."""
        from .task_lease_service import TaskLeaseHeartbeatOutcome

        assert self._heartbeat_done is not None
        waiter = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait(
                (waiter, self._heartbeat_done), return_when=asyncio.FIRST_COMPLETED
            )
            if self._heartbeat_done.done():
                # Propagate a failed renewal rather than declaring it healthy.
                self._heartbeat_done.result()
            return TaskLeaseHeartbeatOutcome(
                lease_lost=self.state == CoordinatorState.LOST,
                pool_timeout=self._heartbeat_error,
            )
        finally:
            await cancel_and_drain_async_task(waiter)

    def track_execution(self, handle: asyncio.Task[Any]) -> None:
        """Retain ownership until the actual outer execution handle finishes."""
        if self.state != CoordinatorState.ACTIVE:
            handle.cancel()
            raise RuntimeError("Task coordinator is closing")
        if handle in self._children:
            return
        self._children.add(handle)
        handle.add_done_callback(self._child_done)

    def _child_done(self, handle: asyncio.Task[Any]) -> None:
        self._children.discard(handle)
        self._ensure_idle_check()

    async def execute_command(
        self, command: Any, execute: Callable[[], Awaitable[_T]]
    ) -> _T:
        """Serialize command application without blocking controls on a long run."""

        async def apply() -> _T:
            async with self._command_lock:
                if self.state != CoordinatorState.ACTIVE:
                    raise _CoordinatorClosed
                # New executions may queue while a previous result is visible,
                # but cannot change its run or inputs before its finalizers exit.
                if command.kind.value in ("start", "resume_input"):
                    await asyncio.gather(
                        *(asyncio.shield(child) for child in tuple(self._children)),
                        return_exceptions=True,
                    )
                    if self.state != CoordinatorState.ACTIVE:
                        raise _CoordinatorClosed
                if not self._healthy:
                    from .task_command_transport import TaskCommandDeferred

                    raise TaskCommandDeferred(
                        "Task owner is awaiting a healthy renewal"
                    )
                token = _current_coordinator.set(self)
                try:
                    return await execute()
                finally:
                    _current_coordinator.reset(token)

        handle = asyncio.create_task(apply())
        self._command_tasks.add(handle)
        try:
            return await asyncio.shield(handle)
        except asyncio.CancelledError:
            await cancel_and_drain_async_task(handle)
            raise
        finally:
            self._command_tasks.discard(handle)
            if self._recovery_required:
                self._request_close()
            else:
                self._ensure_idle_check()

    def _ensure_idle_check(self) -> None:
        self.wake()
        if self.state == CoordinatorState.ACTIVE and (
            self._idle_task is None or self._idle_task.done()
        ):
            self._idle_task = asyncio.create_task(self._check_idle())

    def _release_if_idle(self) -> str:
        assert self.lease is not None
        with self._registry.session_factory() as db, db.begin():
            if not lock_task_lease_no_commit(db, self.lease):
                return "released"
            pending = db.execute(
                select(TaskExecutionCommand.id)
                .where(
                    TaskExecutionCommand.task_id == self.task_id,
                    TaskExecutionCommand.status.in_(("pending", "processing")),
                )
                .limit(1)
            ).first()
            if pending is not None:
                return "busy"
            task = db.get(Task, self.task_id)
            assert task is not None
            if task.status == TaskStatus.RUNNING:
                # Admission may have committed before the process could register
                # an execution. Stop renewal and preserve the token for recovery.
                return "recover"
            return (
                "released"
                if release_task_lease_no_commit(db, self.lease)
                else "recover"
            )

    async def _check_idle(self) -> None:
        try:
            while self.state == CoordinatorState.ACTIVE:
                self.wakeup.clear()
                async with self._command_lock:
                    if (
                        self._healthy
                        and not self._children
                        and not self._command_tasks
                        and self._execution_task is None
                    ):
                        outcome = await run_db_io_cancellation_safe(
                            self._release_if_idle
                        )
                        if outcome != "busy":
                            self._recovery_required = outcome == "recover"
                            self._released = outcome == "released"
                            self._request_close()
                            return
                # Handler completion precedes the transport's receipt commit.
                # Polling that durable queue also covers a lost wakeup.
                try:
                    await asyncio.wait_for(self.wakeup.wait(), timeout=1)
                except asyncio.TimeoutError:
                    pass
        except Exception:
            self._recovery_required = True
            self._request_close()
            logger.exception("Task %s idle release failed", self.task_id)

    def submit_execution(
        self, *, admit: Admission, execute: Execution, settle: Settlement
    ) -> asyncio.Task[None] | None:
        """Reserve one handle before yielding, or retain the caller's defer path.

        Admission and settlement run under task fences in short transactions.
        Admission must return the context it created in that transaction (or
        None to roll back). Settlement records the business outcome, including
        cancellation before execution starts; it never releases ownership.
        Await the returned handle through shield if only observing completion.
        """
        if (
            self.state != CoordinatorState.ACTIVE
            or self._registry._close_task is not None
            or not self._healthy
            or self._execution_task is not None
        ):
            return None
        self._execution_task = asyncio.create_task(
            self._run_execution(admit, execute, settle)
        )
        self._execution_task.add_done_callback(self._execution_done)
        return self._execution_task

    def _admit(self, admit: Admission) -> TaskExecutionContext | None:
        assert self.lease is not None
        with self._registry.session_factory() as db, db.begin():
            if not lock_task_lease_no_commit(db, self.lease):
                return None
            context = admit(db, self.lease)
            if context is None:
                db.rollback()
            elif context.lease != self.lease or not lock_task_execution_no_commit(
                db, context
            ):
                raise ValueError("Admission returned an execution outside this lease")
            return context

    def _settle(
        self,
        settle: Settlement,
        context: TaskExecutionContext,
        error: BaseException | None,
    ) -> None:
        with self._registry.session_factory() as db, db.begin():
            if lock_task_execution_no_commit(db, context):
                settle(db, context, error)

    async def _run_execution(
        self, admit: Admission, execute: Execution, settle: Settlement
    ) -> None:
        worker = asyncio.create_task(asyncio.to_thread(self._admit, admit))
        try:
            context, cancellation = await await_task_settlement(worker)
        except BaseException:
            # A failed commit can have an unknown durable outcome. Stop this
            # owner instead of leaving a RUNNING row on an idle heartbeat.
            self._recovery_required = True
            self._request_close()
            raise
        with propagate_deferred_cancellation(cancellation):
            if context is None:
                return
            error: BaseException | None = cancellation
            try:
                if cancellation is not None:
                    raise cancellation
                await execute(context)
            except BaseException as exc:
                error = exc
                raise
            finally:
                settlement = asyncio.create_task(
                    asyncio.to_thread(self._settle, settle, context, error)
                )
                try:
                    _, settlement_cancellation = await await_task_settlement(settlement)
                except BaseException:
                    self._recovery_required = True
                    self._request_close()
                    raise
                # The transaction succeeded; deferred cancellation must not
                # turn a known commit into an uncertain recovery outcome.
                if settlement_cancellation is not None:
                    raise settlement_cancellation

    def _execution_done(self, task: asyncio.Task[None]) -> None:
        if self._execution_task is task:
            self._execution_task = None
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "Task %s coordinated execution failed",
                self.task_id,
                exc_info=task.exception(),
            )
        self.wake()

    def _request_close(self) -> asyncio.Task[None]:
        if self._close_task is None:
            if self.state != CoordinatorState.LOST:
                self.state = CoordinatorState.QUIESCING
            self._close_task = asyncio.create_task(self._close())
        return self._close_task

    async def close(self) -> None:
        """Stop admission immediately and drain cleanup despite repeated cancel.

        Normal idle release checks the durable queue under the task lock.
        Shutdown drains all registered work and lets another owner take queued commands.
        """
        await drain_async_task_cancellation_safe(self._request_close())

    async def _close(self) -> None:
        try:
            await asyncio.gather(self._startup, return_exceptions=True)
            if self._idle_task is not None:
                await cancel_and_drain_async_task(self._idle_task)
            await asyncio.gather(
                *(cancel_and_drain_async_task(t) for t in tuple(self._command_tasks)),
                return_exceptions=True,
            )
            await asyncio.gather(
                *(cancel_and_drain_async_task(t) for t in tuple(self._children)),
                return_exceptions=True,
            )
            if self._execution_task is not None:
                await cancel_and_drain_async_task(self._execution_task)
            # Keep renewing while execution and its settlement are being drained.
            if self._heartbeat_done is not None:
                await self._registry._unregister_heartbeat(self)
                await asyncio.gather(self._heartbeat_done, return_exceptions=True)
            if (
                self.lease is not None
                and not self._released
                and self._healthy
                and not self._recovery_required
                and self.state != CoordinatorState.LOST
            ):
                await run_db_io_cancellation_safe(self._release)
        except Exception:
            # Leave the exact token for TTL recovery if cleanup cannot commit.
            logger.exception("Task %s coordinator cleanup failed", self.task_id)
        finally:
            self.state = CoordinatorState.CLOSED
            self._registry._remove(self)


_T = TypeVar("_T")
_current_coordinator: ContextVar[TaskCoordinator | None] = ContextVar(
    "task_coordinator", default=None
)
_registry: TaskCoordinatorRegistry | None = None


class _CoordinatorClosed(Exception):
    """An idle owner retired before this command entered its application gate."""


def current_task_coordinator(task_id: int) -> TaskCoordinator | None:
    coordinator = _current_coordinator.get()
    return (
        coordinator
        if coordinator is not None and coordinator.task_id == task_id
        else None
    )


def get_task_coordinator_registry() -> TaskCoordinatorRegistry:
    global _registry
    if _registry is None or _registry.loop is not asyncio.get_running_loop():
        _registry = TaskCoordinatorRegistry()
    return _registry


async def execute_coordinated_command(
    command: Any, execute: Callable[[], Awaitable[_T]]
) -> _T:
    from .task_command_transport import TaskCommandDeferred

    registry = get_task_coordinator_registry()
    while True:
        coordinator = await registry.ensure(command.task_id)
        if coordinator is None:
            raise TaskCommandDeferred("Waiting for the task execution owner")
        try:
            return await coordinator.execute_command(command, execute)
        except _CoordinatorClosed:
            # The transport still owns this exact claim. Re-enter through the
            # registry after the old owner has drained, without redelivering it.
            continue


async def close_task_coordinators() -> None:
    global _registry
    if _registry is not None and _registry.loop is asyncio.get_running_loop():
        await _registry.close()
        _registry = None
