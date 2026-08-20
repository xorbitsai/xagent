"""Usage metering for work that is not a tracked agent task.

``TaskTracker`` binds a ``TokenUsage`` for chat/agent runs and reports the
delta to the quota hook when the run completes. Everything else — KB ingestion
over HTTP or Celery, ``/speech/transcribe``, Telegram voice — records usage
with no context bound, so ``get_token_usage()`` lazily creates a throwaway
object that nothing ever reads. The calls succeed, the provider bills, and the
usage silently evaporates.

This module is the equivalent sink for those paths: bind a context around the
work, then report whatever was recorded.

Why a shared helper rather than a `TokenContextManager` at each call site: the
quota hook has a transaction contract that is easy to violate by accident. It
must not be handed a caller's request Session (it manages its own durability
and must not commit or leave writes pending on someone else's session), so the
report step opens and disposes a short-lived compatibility Session of its own.
Reproducing that at four call sites would mean four chances to get it wrong.
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from contextvars import copy_context
from typing import Any, Callable, Iterator, Optional, TypeVar

from ...core.model.chat.token_context import TokenUsage, set_token_usage

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _report(user_id: Optional[int], usage: TokenUsage) -> None:
    """Hand this unit of work's usage to the quota hook, best-effort."""
    details = [dict(item) for item in usage.details if isinstance(item, dict)]
    if not details:
        return
    try:
        from ..models.database import get_session_local
        from ..services.quota_hooks import has_usage_record_hook, record_usage

        # Check the hook before checking out a session: with no hook installed
        # (the stock configuration) record_usage is a guaranteed no-op, and a
        # pool checkout + transaction + close per ingest/transcription is pure
        # overhead.
        if not has_usage_record_hook():
            return

        db_session = get_session_local()()
        try:
            # delta_actions=0: these paths make provider calls, not agent tool
            # calls, and tool invocations are what that counter bills for.
            record_usage(db_session, user_id, details, 0)
        finally:
            # The hook owns its own durability and must not leave work pending
            # on this compatibility Session.
            if db_session.in_transaction():
                db_session.rollback()
            db_session.close()
    except Exception as e:  # noqa: BLE001
        # Metering must never break the work it is measuring.
        logger.warning("Standalone usage recording failed: %s", e)


@contextmanager
def usage_scope(user_id: Optional[int]) -> Iterator[TokenUsage]:
    """Bind a usage context for one unit of non-task work and report it after.

    Usage is reported even when the body raises: a provider call that already
    happened is billable regardless of what fails afterwards.

    Note the body must not cross a thread boundary that drops contextvars —
    see :func:`bind_usage_to_thread` for `run_in_executor`-style hops.
    """
    from ...core.model.chat.token_context import token_context

    usage = TokenUsage()
    # Restore whatever was bound before (usually None) so a nested scope cannot
    # leak its usage object into the caller's.
    previous = token_context.get(None)
    set_token_usage(usage)
    try:
        yield usage
    finally:
        try:
            # Only restore if we are still the bound context. A TaskTracker
            # started *inside* this scope also calls set_token_usage, and
            # clobbering that would silently detach it from its own run.
            # Unreachable at today's call sites, but cheap to get right before
            # a sixth one appears.
            if token_context.get(None) is usage:
                set_token_usage(previous)  # type: ignore[arg-type]
            else:
                logger.debug(
                    "usage_scope exiting with a different context bound; "
                    "leaving it in place rather than detaching its owner"
                )
        except Exception:  # noqa: BLE001
            pass
        _report(user_id, usage)


def bind_usage_to_thread(fn: Callable[..., T]) -> Callable[..., T]:
    """Wrap a callable so it records into the *calling* thread's usage.

    ``loop.run_in_executor(None, fn)`` and bare ``ThreadPoolExecutor`` do not
    propagate contextvars, so without this the worker records into a fresh
    ``TokenUsage`` that is discarded on return. ``asyncio.to_thread`` already
    copies the context and does not need this.

    Captures at wrap time, on the calling thread — wrapping must therefore
    happen before the hop, not inside the worker.

    The binding runs inside a copied context (``copy_context()`` +
    ``ctx.run``), the same idiom used at
    ``core/tools/adapters/vibe/file_ingestion_tool.py``. That confines the
    contextvar write to this one call instead of mutating the worker thread:
    ``run_in_executor(None, ...)`` uses the loop's long-lived default executor,
    so a plain ``set_token_usage`` would outlive the job and leak the caller's
    ``TokenUsage`` into the next unrelated task that reuses the thread —
    cross-tenant usage misattribution, plus a strong reference pinned for the
    thread's lifetime. A copied context also needs no ``finally`` restore:
    there is nothing in the caller's context left to restore.
    """
    from ...core.model.chat.token_context import get_token_usage

    caller_usage = get_token_usage()

    @functools.wraps(fn)
    def _bound(*args: Any, **kwargs: Any) -> T:
        def _run() -> T:
            set_token_usage(caller_usage)
            return fn(*args, **kwargs)

        # Fresh copy per invocation: a wrapped callable may be submitted more
        # than once, and each run must get its own isolated context.
        return copy_context().run(_run)

    return _bound
