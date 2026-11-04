"""Worker-owned synchronous database sessions for authentication."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager, contextmanager
from typing import TypeVar

from sqlalchemy.orm import Session

from ...core.runtime_performance import run_in_thread_with_telemetry
from ..services.db_runtime import drain_async_task_cancellation_safe
from .database import get_db

SyncAuthSessionFactory = Callable[[], AbstractContextManager[Session]]
_T = TypeVar("_T")


async def run_auth_db_worker(operation: str, work: Callable[[], _T]) -> _T:
    """Drain the owning worker, including Session close, before cancellation.

    A cancelled request may have committed already. Never retry its transaction
    here, and never close a Session concurrently with its still-running worker.
    """
    return await drain_async_task_cancellation_safe(
        asyncio.create_task(run_in_thread_with_telemetry(operation, work))
    )


async def get_auth_db() -> AsyncIterator[SyncAuthSessionFactory]:
    """Yield a factory, not a Session: only the worker may enter/use/exit it.

    Tests and hosts replacing auth storage should override this dependency
    with a lazy context-manager factory. Other routes still use get_db.
    """
    yield contextmanager(get_db)
