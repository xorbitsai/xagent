"""Per-call SQLAlchemy session scope for DB-touching tools.

A tool that holds a long-lived shared ``Session`` is unsafe under in-turn tool
concurrency (one connection/cursor, a mutable identity map). This context
manager mints a fresh ``Session`` from a factory for the duration of a single
tool invocation and closes it in ``finally``. Commit/rollback stays the caller's
responsibility — the scope only guarantees the session is closed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator


@contextmanager
def tool_session_scope(session_factory: Callable[[], Any]) -> Iterator[Any]:
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
