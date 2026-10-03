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

import logging
from contextlib import contextmanager
from typing import Iterator, Optional

from ...core.model.chat.token_context import TokenUsage, set_token_usage

logger = logging.getLogger(__name__)


def _report(user_id: Optional[int], usage: TokenUsage) -> None:
    """Hand this unit of work's usage to the quota hook, best-effort."""
    details = [dict(item) for item in usage.details if isinstance(item, dict)]
    if not details:
        return
    try:
        from ..services.quota_hooks import has_usage_record_hook

        # Imported lazily, like the hook above: this module is deliberately
        # light so importing it costs nothing at startup, and task_tracker
        # pulls in the whole tracking chain.
        from .task_tracker import _record_usage_on_event_loop

        # Check the hook before checking out a session: with no hook installed
        # (the stock configuration) record_usage is a guaranteed no-op, and a
        # pool checkout + transaction + close per ingest/transcription is pure
        # overhead. The sibling path in task_tracker predates this check and
        # still pays it; extending the short-circuit there is a separate change
        # to a hot path this PR does not otherwise touch.
        if not has_usage_record_hook():
            return

        # Reuses task_tracker's helper rather than repeating its session
        # lifecycle: it already owns the "hand the hook a short-lived
        # compatibility Session, never leave it holding a transaction"
        # contract, and two copies of that is how they drift.
        # delta_actions=0: these paths make provider calls, not agent tool
        # calls, and tool invocations are what that counter bills for.
        _record_usage_on_event_loop(user_id, details, 0)
    except Exception as e:  # noqa: BLE001
        # Metering must never break the work it is measuring.
        logger.warning("Standalone usage recording failed: %s", e)


@contextmanager
def usage_scope(user_id: Optional[int]) -> Iterator[TokenUsage]:
    """Bind a usage context for one unit of non-task work and report it after.

    Usage is reported even when the body raises: a provider call that already
    happened is billable regardless of what fails afterwards.

    Note the body must not cross a thread boundary that drops contextvars.
    ``asyncio.to_thread`` copies the context and is safe; a bare
    ``ThreadPoolExecutor`` or ``run_in_executor`` is not, and would need the
    caller's usage bound explicitly inside the worker.
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
