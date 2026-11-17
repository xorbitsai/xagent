"""Host hook and per-run context for the ``show_result_card`` display tool.

The tool is off by default. A host turns it on by registering a source
resolver with :func:`set_result_card_source_resolver`; without one the tool is
never offered to the model and nothing about a run changes.

A result card is a display message: it is persisted as an assistant chat
message (so it survives a reload) and never pauses the run. Its provenance is
checked server-side against the connectors the run actually invoked, so the
model cannot attribute a card to a connector it never called.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

RESULT_CARD_TOOL_NAME = "show_result_card"
# ``message_type`` of the persisted assistant message, and the value of
# ``metadata["display"]`` on both the stream event and the history entry.
RESULT_CARD_MESSAGE_TYPE = "result_card"
RESULT_CARD_DISPLAY = "result_card"

MAX_TITLE_CHARS = 120
MAX_ITEMS = 12
MAX_ITEM_CHARS = 240
MAX_FOOTNOTE_CHARS = 200


@dataclass(frozen=True)
class ResultCardSource:
    """Display data of the connector a card is attributed to."""

    name: str
    app_id: Optional[str] = None
    logo: Optional[dict[str, Any]] = None


class ResultCardSourceResolver(Protocol):
    """Host callback mapping a normalized connector identity to display data.

    Return ``None`` for a connector the host has no display data for; the card
    then shows the connector identity the run used.
    """

    def __call__(self, connector: str) -> Optional[ResultCardSource]: ...


_source_resolver: Optional[ResultCardSourceResolver] = None


def set_result_card_source_resolver(
    resolver: Optional[ResultCardSourceResolver],
) -> None:
    """Enable (or, with ``None``, disable) the result-card tool."""
    global _source_resolver
    _source_resolver = resolver


def result_card_enabled() -> bool:
    return _source_resolver is not None


def resolve_result_card_source(connector: str) -> ResultCardSource:
    """Display data for ``connector``; never raises.

    A resolver that raises or returns something else falls back to the bare
    connector identity, so a faulty host hook cannot fail the run.
    """
    fallback = ResultCardSource(name=connector)
    resolver = _source_resolver
    if resolver is None:
        return fallback
    try:
        resolved = resolver(connector)
    except Exception:
        logger.warning("result card source resolver failed", exc_info=True)
        return fallback
    return resolved if isinstance(resolved, ResultCardSource) else fallback


@dataclass(frozen=True)
class RunDisplayContext:
    """What a display tool may use of the run that is calling it."""

    # Normalized ``source_server`` of every connector tool that completed
    # successfully earlier in this run.
    invoked_connectors: frozenset[str]
    # ``PatternRuntime.send_message``: emits one assistant-side chat message.
    send_message: Callable[..., Awaitable[dict[str, Any]]]


_run_display_context: ContextVar[Optional[RunDisplayContext]] = ContextVar(
    "run_display_context", default=None
)


@contextmanager
def bind_run_display_context(context: RunDisplayContext) -> Iterator[None]:
    token = _run_display_context.set(context)
    try:
        yield
    finally:
        _run_display_context.reset(token)


def current_run_display_context() -> Optional[RunDisplayContext]:
    return _run_display_context.get()
