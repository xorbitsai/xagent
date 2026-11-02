"""Usage metering for work that is not an agent run.

``TaskTracker`` binds a ``TokenUsage`` for agent runs and reports it through
the quota hooks when the run completes. Everything else -- a knowledge-base
search or ingestion over HTTP or a background job, the memory API, a model
connection test, ``/speech/transcribe``, Telegram voice -- has no context
bound, so the providers record into a throwaway ``TokenUsage`` that nothing
reads: the call happens, the provider bills, and the usage evaporates.

:func:`usage_scope` is the sink for those paths. It binds a fresh
``TokenUsage`` around the work and hands whatever was recorded to the hook
registered with
:func:`xagent.web.services.quota_hooks.set_standalone_usage_hook` -- once,
whether the work succeeded or raised, because a provider call that already
happened is billed regardless of what fails after it.

Use ``async with`` in coroutines and plain ``with`` in synchronous code
(background jobs, worker threads). The binding follows the context:
``asyncio.to_thread`` and ``asyncio.run`` carry it, while
``loop.run_in_executor`` and a bare ``ThreadPoolExecutor`` do not, so work
handed to those must be submitted through ``contextvars.copy_context().run``.

A scope delivers what was recorded when it exits. Blocking work handed to a
thread from a coroutine outlives the coroutine when the await is cancelled or
times out, so it runs under :func:`run_in_usage_scope`, which opens the scope
in the thread and delivers once the work itself has finished.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from types import TracebackType
from typing import Any, Callable, Literal, Optional, TypeVar

from ...core.model.chat.token_context import TokenUsage, token_context

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: The entry points a scope is opened for, passed to the hook as its third
#: argument.
UsageSource = Literal[
    "kb_search",
    "kb_ingest",
    "kb_ingest_cloud",
    "kb_ingest_web",
    "kb_ingest_job",
    "kb_ingest_web_job",
    "memory_api",
    "model_probe",
    "speech_transcribe",
    "speech_synthesize",
    "telegram_voice",
]


class UsageScope:
    """One unit of non-run work, metered for one user. See :func:`usage_scope`."""

    def __init__(self, user_id: Optional[int], usage_source: UsageSource) -> None:
        self.user_id = user_id
        self.usage_source = usage_source
        self.usage = TokenUsage()
        self._token: Optional[contextvars.Token[Optional[TokenUsage]]] = None
        self._previous: Optional[TokenUsage] = None
        self._delivered = False

    def _bind(self) -> TokenUsage:
        self._previous = token_context.get(None)
        self._token = token_context.set(self.usage)
        return self.usage

    def _unbind(self) -> None:
        token, self._token = self._token, None
        if token is None:
            return
        try:
            token_context.reset(token)
        except ValueError:
            # Exited in a different context than it was entered in. Restore by
            # hand, but only if this scope is still the one bound: a tracker
            # started inside the scope rebinds the context, and replacing its
            # usage would detach it from its own run.
            if token_context.get(None) is self.usage:
                token_context.set(self._previous)

    def _pending(self) -> Optional[list[dict[str, Any]]]:
        """The rows to deliver, or None when there is nothing to do.

        Marks the scope delivered, so a scope reports at most once.
        """
        if self._delivered:
            return None
        self._delivered = True
        details = [dict(row) for row in self.usage.details if isinstance(row, dict)]
        if not details:
            return None
        from ..services.quota_hooks import get_standalone_usage_hook

        if get_standalone_usage_hook() is None:
            return None
        if self.user_id is None:
            logger.info(
                "Delivering %d %s usage rows with no attributable user",
                len(details),
                self.usage_source,
            )
        return details

    def _deliver(self, details: list[dict[str, Any]]) -> None:
        from ..services.quota_hooks import get_standalone_usage_hook

        hook = get_standalone_usage_hook()
        if hook is None:
            return
        try:
            hook(self.user_id, details, self.usage_source)
        except Exception as e:  # noqa: BLE001
            # Metering must never fail the work it measured. exc_info: the
            # rows are lost here, so the log is all there is to go on.
            logger.warning(
                "Standalone usage hook failed for %s (%d rows): %s",
                self.usage_source,
                len(details),
                e,
                exc_info=True,
            )

    def __enter__(self) -> TokenUsage:
        return self._bind()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self._unbind()
        details = self._pending()
        if details is not None:
            self._deliver(details)

    async def __aenter__(self) -> TokenUsage:
        return self._bind()

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self._unbind()
        details = self._pending()
        if details is not None:
            # Off the event loop: the hook writes to the database. If this
            # await is cancelled the thread still runs the hook to the end, so
            # the rows are delivered exactly once either way.
            try:
                await asyncio.to_thread(self._deliver, details)
            except RuntimeError:
                # The default executor is already shut down (the loop is
                # closing): deliver here rather than lose the rows, and keep
                # the body's own outcome.
                self._deliver(details)


def usage_scope(user_id: Optional[int], *, usage_source: UsageSource) -> UsageScope:
    """Meter one unit of non-run work for ``user_id``.

    ``user_id`` is the user the work is done for; pass ``None`` when there is
    none to attribute it to, and the usage is still delivered.
    ``usage_source`` names the entry point for the hook.

    Scopes nest: an inner scope's usage is delivered on its own and never
    reaches the outer one.
    """
    return UsageScope(user_id, usage_source)


def run_in_usage_scope(
    user_id: Optional[int],
    usage_source: UsageSource,
    fn: Callable[..., _T],
    /,
    *args: Any,
    **kwargs: Any,
) -> _T:
    """Call ``fn(*args, **kwargs)`` inside a synchronous :func:`usage_scope`.

    For submitting blocking work to a thread: the scope lives in the thread,
    so its rows are delivered when the work finishes, even after the
    coroutine that awaited it was cancelled or timed out.
    """
    with usage_scope(user_id, usage_source=usage_source):
        return fn(*args, **kwargs)
