"""Phase B ownership lifecycle; production consumers are not connected yet.

One registry belongs to one worker event loop. Entry adapters supply their
existing admission and settlement transactions; they must not acquire, renew,
or release leases themselves. Execution includes draining its own callbacks.
Command selection and idle queue checks will be connected with the transport.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Awaitable, Callable

from sqlalchemy.orm import Session

from ...config import get_task_lease_heartbeat_seconds
from ..models.database import get_session_local
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
    acquire_task_lease_no_commit,
    lock_task_execution_no_commit,
    lock_task_lease_no_commit,
    recover_expired_idle_task_lease_no_commit,
    release_task_lease_no_commit,
    renew_task_lease_no_commit,
)
from .task_lease_service import get_runner_id

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
    """Share acquisition and one running handle across all entry adapters."""

    def __init__(self, session_factory: Callable[[], Session] | None = None):
        self.session_factory = session_factory or get_session_local()
        self.runner_id = get_runner_id()
        self._coordinators: dict[int, TaskCoordinator] = {}
        self._close_task: asyncio.Task[None] | None = None

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
        self._recovery_required = False
        self._stop = asyncio.Event()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._execution_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._startup = asyncio.create_task(self._start())

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
            self._heartbeat_task = asyncio.create_task(self._heartbeat())
            self.state = CoordinatorState.ACTIVE

    def _renew(self) -> bool:
        assert self.lease is not None
        with self._registry.session_factory() as db, db.begin():
            return renew_task_lease_no_commit(db, self.lease)

    async def _heartbeat(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=get_task_lease_heartbeat_seconds()
                    )
                    return
                except asyncio.TimeoutError:
                    pass
                try:
                    renewed = await run_db_io_cancellation_safe(self._renew)
                except Exception as error:
                    self._healthy = False
                    if is_database_pool_timeout(error):
                        # No ownership verdict: retry only at the normal cadence.
                        logger.warning("Task %s heartbeat pool timeout", self.task_id)
                        continue
                    raise
                self._healthy = True
                if not renewed:
                    self.state = CoordinatorState.LOST
                    self._request_close()
                    return
        except BaseException:
            self._healthy = False
            self._recovery_required = True
            self._request_close()
            logger.exception("Task %s coordinator heartbeat stopped", self.task_id)
            raise

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

        This is shutdown, not idle queue release. A transport must perform its
        task-locked queue check before requesting idle release in the next phase.
        """
        await drain_async_task_cancellation_safe(self._request_close())

    async def _close(self) -> None:
        try:
            await asyncio.gather(self._startup, return_exceptions=True)
            if self._execution_task is not None:
                await cancel_and_drain_async_task(self._execution_task)
            # Keep renewing while execution and its settlement are being drained.
            self._stop.set()
            if self._heartbeat_task is not None:
                await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            if (
                self.lease is not None
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
