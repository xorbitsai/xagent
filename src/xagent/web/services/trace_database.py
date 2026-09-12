"""Bounded trace persistence, scoped to the application's event loop.

Admission precedes worker submission and connection checkout. Required events
wait; they are not dropped. The handler owns cancellation-safe draining through
session close and retention cleanup, so a permit cannot be reused while an
abandoned worker still owns a transaction. AsyncSession.run_sync bridges the
existing ORM transaction to native async I/O, not a thread pool. Its Python
encoding/ORM work still runs on the loop; this is not CPU isolation.
"""

import asyncio
from collections.abc import Callable

from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool

from ...config import (
    get_async_trace_db_enabled,
    get_db_pool_kwargs,
    get_trace_db_max_inflight,
)
from ...core.runtime_performance import observe_duration, run_in_thread_with_telemetry
from ..models.database import get_engine
from .db_runtime import drain_async_task_cancellation_safe

_LOOP_ATTRIBUTE = "_xagent_trace_database_runtime"


class TraceDatabaseRuntime:
    def __init__(self, source: Engine | None, *, use_async: bool, limit: int):
        if limit < 1:
            raise ValueError("Trace database concurrency must be positive")
        self.engine: AsyncEngine | None = None
        if use_async:
            if source is None or source.dialect.name != "postgresql":
                raise ValueError("Async trace persistence requires PostgreSQL")
            # Psycopg preserves existing libpq URL options (SSL/search_path/etc.).
            # This is a SEPARATE bounded pool; include it in the process budget.
            try:
                self.engine = create_async_engine(
                    source.url.set(drivername="postgresql+psycopg"),
                    **{**get_db_pool_kwargs(), "pool_size": limit, "max_overflow": 0},
                    hide_parameters=True,
                    execution_options=source.get_execution_options(),
                )
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Async trace persistence requires the postgresql-async extra"
                ) from exc
        elif source is not None and isinstance(source.pool, QueuePool):
            # Do not rely on overflow for API headroom. A size-one pool cannot
            # reserve a connection; deployments needing isolation must enlarge
            # it or opt into the separate async trace pool. Other writers are
            # outside this budget and can still exhaust database capacity.
            limit = min(limit, max(1, source.pool.size() - 1))
        self.limit = limit
        self._slots = asyncio.Semaphore(limit)
        self._operations: set[asyncio.Task[None]] = set()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    async def run(
        self,
        sync_write: Callable[[], None],
        transaction: Callable[[Session], None],
    ) -> None:
        """Called inside the handler's cancellation-drained owned task."""
        if self._closing:
            raise RuntimeError("Trace database runtime is closing")
        task = asyncio.current_task()
        assert task is not None
        self._operations.add(task)
        try:
            with observe_duration("xagent.trace.database.admission_wait.duration"):
                await self._slots.acquire()
            try:
                if self.engine is None:
                    await run_in_thread_with_telemetry(
                        "trace_database_write", sync_write
                    )
                else:
                    async with AsyncSession(self.engine, autoflush=False) as db:
                        await db.run_sync(transaction)
            finally:
                self._slots.release()
        finally:
            self._operations.remove(task)

    async def close(self) -> None:
        """Stop admission, drain accepted writes, then dispose pooled connections."""
        if self._close_task is None:
            self._closing = True

            async def finish() -> None:
                await asyncio.gather(*self._operations, return_exceptions=True)
                if self.engine is not None:
                    await self.engine.dispose()

            self._close_task = asyncio.create_task(finish())
        await drain_async_task_cancellation_safe(self._close_task)


def get_trace_database_runtime() -> TraceDatabaseRuntime:
    """Loop ownership avoids reusing async pools/semaphores across event loops."""
    loop = asyncio.get_running_loop()
    runtime = getattr(loop, _LOOP_ATTRIBUTE, None)
    if isinstance(runtime, TraceDatabaseRuntime):
        return runtime
    try:
        source = get_engine()
    except RuntimeError:
        # Custom hosts/tests can provide their own synchronous session factory.
        # Async mode still fails closed without a configured PostgreSQL engine.
        source = None
    runtime = TraceDatabaseRuntime(
        source,
        use_async=get_async_trace_db_enabled(),
        limit=get_trace_db_max_inflight(),
    )
    setattr(loop, _LOOP_ATTRIBUTE, runtime)
    return runtime


async def close_trace_database_runtime() -> None:
    loop = asyncio.get_running_loop()
    runtime = getattr(loop, _LOOP_ATTRIBUTE, None)
    if isinstance(runtime, TraceDatabaseRuntime):
        await runtime.close()
        if getattr(loop, _LOOP_ATTRIBUTE, None) is runtime:
            delattr(loop, _LOOP_ATTRIBUTE)
