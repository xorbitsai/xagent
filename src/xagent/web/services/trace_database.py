"""Bounded trace persistence, scoped to the application's event loop.

Admission precedes worker submission and connection checkout. Required events
wait; they are not dropped. The handler owns cancellation-safe draining through
session close and retention cleanup, so a permit cannot be reused while an
abandoned worker still owns a transaction. AsyncSession.run_sync bridges the
existing ORM transaction to async driver I/O, not the default thread pool.
SQLite's aiosqlite driver uses a dedicated thread per connection. Bulk payload
preparation and JSON encoding run in a bounded dedicated worker before Session
creation. ORM bookkeeping and retention still run on the loop; threads share
the GIL, so this does not replace process isolation for arbitrary CPU load.
"""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import AsyncAdaptedQueuePool, QueuePool

from ...config import (
    get_async_trace_db_enabled,
    get_db_pool_kwargs,
    get_trace_db_max_inflight,
)
from ...core.runtime_performance import observe_duration, run_in_thread_with_telemetry
from ...db.sqlite import apply_sqlite_concurrency_pragmas
from ..models.database import get_engine
from .db_runtime import drain_async_task_cancellation_safe
from .trace_message_storage import trace_json_dumps

_LOOP_ATTRIBUTE = "_xagent_trace_database_runtime"


class TraceDatabaseRuntime:
    def __init__(self, source: Engine | None, *, use_async: bool, limit: int):
        if limit < 1:
            raise ValueError("Trace database concurrency must be positive")
        self.engine: AsyncEngine | None = None
        # A second engine cannot see a private in-memory SQLite database.
        # Keep the supplied session factory for memory/custom hosts rather than
        # silently writing to an empty database. File databases use aiosqlite.
        sqlite = source is not None and source.dialect.name == "sqlite"
        memory = (
            source is not None
            and sqlite
            and (
                not source.url.database
                or source.url.database == ":memory:"
                or source.url.database.startswith("file:")
                and (
                    "mode=memory" in source.url.database
                    or source.url.query.get("mode") == "memory"
                    or source.url.database.startswith("file::memory:")
                )
            )
        )
        if source is None or memory:
            use_async = False
        if sqlite:
            limit = 1
        if use_async:
            if source is None or source.dialect.name not in {"postgresql", "sqlite"}:
                raise ValueError(
                    "Async trace persistence requires PostgreSQL or SQLite; "
                    "set XAGENT_ASYNC_TRACE_DB_ENABLED=false for bounded sync writes"
                )
            # Psycopg preserves existing libpq URL options (SSL/search_path/etc.).
            # This is a SEPARATE bounded pool; include it in the process budget.
            pool_kwargs = get_db_pool_kwargs()
            if sqlite:
                # Local file connections need no network liveness probes or
                # periodic recycling; retain bounded checkout configuration.
                pool_kwargs.pop("pool_pre_ping", None)
                pool_kwargs.pop("pool_recycle", None)
            try:
                self.engine = create_async_engine(
                    source.url.set(
                        drivername="sqlite+aiosqlite"
                        if sqlite
                        else "postgresql+psycopg"
                    ),
                    **{**pool_kwargs, "pool_size": limit, "max_overflow": 0},
                    poolclass=AsyncAdaptedQueuePool,
                    hide_parameters=True,
                    execution_options=source.get_execution_options(),
                    json_serializer=trace_json_dumps,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Could not configure async trace persistence. Check DATABASE_URL "
                    "and install aiosqlite or the postgresql extra, or set "
                    "XAGENT_ASYNC_TRACE_DB_ENABLED=false for bounded sync writes."
                ) from exc
            if sqlite:
                apply_sqlite_concurrency_pragmas(self.engine.sync_engine)
        elif source is not None and isinstance(source.pool, QueuePool):
            # Do not rely on overflow for API headroom. A size-one pool cannot
            # reserve a connection; deployments needing isolation must enlarge
            # it or opt into the separate async trace pool. Other writers are
            # outside this budget and can still exhaust database capacity.
            limit = min(limit, max(1, source.pool.size() - 1))
        self.limit = limit
        # One dedicated CPU preparation worker, never the API default executor.
        # Admission above bounds queued preparation as well as active writes.
        self._preparation_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="trace-prepare"
        )
        self._slots = asyncio.Semaphore(limit)
        self._operations: set[asyncio.Task[None]] = set()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    async def run(
        self,
        sync_write: Callable[[], None],
        transaction: Callable[[Session], None],
        prepare: Callable[[], Callable[[Session], None]] | None = None,
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
                    if prepare is not None:
                        context = copy_context()
                        with observe_duration(
                            "xagent.trace.database.preparation.duration"
                        ):
                            transaction = (
                                await asyncio.get_running_loop().run_in_executor(
                                    self._preparation_pool, context.run, prepare
                                )
                            )
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
                self._preparation_pool.shutdown(wait=True)

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
        # Without a shared engine, retain that factory via bounded sync writes.
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
