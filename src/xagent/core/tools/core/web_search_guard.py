"""Shared call step for every built-in web search provider.

Every provider core routes its ``search`` through :func:`run_guarded_search`:

1. the query is redacted (:mod:`.search_query_redaction`),
2. the provider is called with the redacted text only,
3. the optional audit hook is called with the redacted query.

The unredacted query exists only inside this function and the caller's
frame; neither the provider callable nor the audit hook ever receives it.

Audit hook
----------
Nothing is registered by default. A host (for example the SaaS layer, to write
one audit row per call) registers one callable with
:func:`set_search_audit_hook`. The callable receives a single
:class:`SearchAuditEvent` and may be sync or ``async``. An exception raised by
the hook is logged (without the query) and never fails the search.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Optional, Tuple, TypeVar, Union

from .search_query_redaction import redact_search_query

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class SearchAuditEvent:
    """What the audit hook receives after each provider call."""

    # Provider name: "google", "tavily", "exa", "zhipu" (and "brave" once it
    # routes through the same step).
    provider: str
    # The query exactly as it was sent to the provider (already redacted).
    query: str
    # Entity types that were replaced, e.g. ("email", "phone"); () if none.
    entities: Tuple[str, ...]
    # Number of results the provider returned; None on failure.
    result_count: Optional[int]
    # "success" or "failure".
    outcome: Literal["success", "failure"]
    # On failure: the class name of the underlying error (the cause when the
    # core wrapped it, e.g. "ConnectTimeout", "HTTPStatusError"); else None.
    failure_kind: Optional[str] = None


SearchAuditHook = Callable[[SearchAuditEvent], Union[None, Awaitable[None]]]

_audit_hook: Optional[SearchAuditHook] = None


def set_search_audit_hook(hook: Optional[SearchAuditHook]) -> None:
    """Register the audit hook (replacing any previous one); ``None`` clears it."""
    global _audit_hook
    _audit_hook = hook


def get_search_audit_hook() -> Optional[SearchAuditHook]:
    return _audit_hook


def _failure_kind(exc: BaseException) -> str:
    return type(exc.__cause__ or exc).__name__


async def _emit(event: SearchAuditEvent) -> None:
    hook = _audit_hook
    if hook is None:
        return
    try:
        result = hook(event)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        # Log the failure class only: neither the hook's message nor the
        # event may carry anything we have not already redacted.
        logger.warning(
            "Search audit hook failed for provider %s: %s",
            event.provider,
            type(exc).__name__,
        )


async def run_guarded_search(
    provider: str,
    query: str,
    call: Callable[[str], Awaitable[T]],
    count_results: Callable[[T], int],
) -> T:
    """Redact ``query``, run ``call`` with the redacted text, then audit.

    Args:
        provider: Provider name reported to the audit hook.
        query: The query as the model wrote it.
        call: The provider call; it receives the redacted query only.
        count_results: Maps the provider's return value to a result count.
    """
    redacted = redact_search_query(query)
    try:
        result = await call(redacted.text)
    except Exception as exc:
        await _emit(
            SearchAuditEvent(
                provider=provider,
                query=redacted.text,
                entities=redacted.entities,
                result_count=None,
                outcome="failure",
                failure_kind=_failure_kind(exc),
            )
        )
        raise
    try:
        count: Optional[int] = count_results(result)
    except Exception:
        count = None
    await _emit(
        SearchAuditEvent(
            provider=provider,
            query=redacted.text,
            entities=redacted.entities,
            result_count=count,
            outcome="success",
        )
    )
    return result
