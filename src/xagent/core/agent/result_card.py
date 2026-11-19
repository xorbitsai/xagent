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

import json
import logging
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
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


# Connector kinds a tool can come from. ``None`` means the origin is unknown, or
# that two kinds with the same name were both used in the run.
CONNECTOR_KIND_MCP = "mcp"
CONNECTOR_KIND_CUSTOM_API = "custom_api"


@dataclass(frozen=True)
class ResultCardConnector:
    """Which connector a card is attributed to, as the run used it.

    ``name`` is the normalized server name (``normalize_mcp_server_name``).
    ``kind`` says what kind of connector carries that name: an MCP server or a
    user-defined Custom API. Names are unique among MCP servers but not across
    kinds, so a name alone does not identify a catalog app: a user's Custom API
    can be called "Salesforce" too. A host maps ``(kind, name)`` to its catalog
    and must not apply catalog branding to ``custom_api`` or to ``None``.
    """

    name: str
    kind: Optional[str] = None


@dataclass(frozen=True)
class ResultCardSource:
    """Display data of the connector a card is attributed to."""

    name: str
    app_id: Optional[str] = None
    logo: Optional[dict[str, Any]] = None


class ResultCardSourceResolver(Protocol):
    """Host callback mapping a connector identity to display data.

    Return ``None`` for a connector the host has no display data for; the card
    then shows the connector name the run used. Exceptions and invalid results
    are ignored the same way.
    """

    def __call__(
        self, connector: ResultCardConnector
    ) -> Optional[ResultCardSource]: ...


_source_resolver: Optional[ResultCardSourceResolver] = None


def set_result_card_source_resolver(
    resolver: Optional[ResultCardSourceResolver],
) -> None:
    """Enable (or, with ``None``, disable) the result-card tool."""
    global _source_resolver
    _source_resolver = resolver


def result_card_enabled() -> bool:
    return _source_resolver is not None


def _storable_source(source: Any) -> bool:
    """Whether ``source`` can be stored and replayed as it is.

    The card is written into a JSON event fact: a non-string name, a logo that
    is not a JSON object, or a NaN would fail that write, which would fail the
    run rather than just the card.
    """
    if not isinstance(source, ResultCardSource):
        return False
    if not isinstance(source.name, str) or not source.name.strip():
        return False
    if source.app_id is not None and not isinstance(source.app_id, str):
        return False
    if source.logo is not None and not isinstance(source.logo, dict):
        return False
    try:
        json.loads(json.dumps(source.logo, allow_nan=False))
    except (TypeError, ValueError):
        return False
    return True


def resolve_result_card_source(connector: ResultCardConnector) -> ResultCardSource:
    """Display data for ``connector``; never raises.

    A resolver that raises, or returns something that cannot be stored, falls
    back to the bare connector name, so a faulty host hook cannot fail the run.
    """
    fallback = ResultCardSource(name=connector.name)
    resolver = _source_resolver
    if resolver is None:
        return fallback
    try:
        resolved = resolver(connector)
    except Exception:
        logger.warning("result card source resolver failed", exc_info=True)
        return fallback
    if resolved is None:
        return fallback
    if not _storable_source(resolved):
        logger.warning("result card source resolver returned an unusable source")
        return fallback
    return resolved


def result_card_metadata(data: Any) -> Optional[dict[str, Any]]:
    """The ``{display, result_card}`` metadata of a card message, or ``None``.

    The one place that decides whether an outbound agent message is a card.
    The live stream, the transcript write and the v2 projection all call it.
    A card is a visible, non-waiting message of type ``result_card`` whose
    metadata carries a well-formed card; nothing else qualifies.
    """
    if not isinstance(data, dict):
        return None
    if data.get("message_type") != RESULT_CARD_MESSAGE_TYPE:
        return None
    if data.get("expect_response") or data.get("visible") is False:
        return None
    message = data.get("message")
    if not isinstance(message, str) or not message:
        return None
    metadata = data.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("display") != RESULT_CARD_DISPLAY:
        return None
    card = metadata.get("result_card")
    if not isinstance(card, dict):
        return None
    title, items, source = card.get("title"), card.get("items"), card.get("source")
    if not isinstance(title, str) or not title:
        return None
    if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
        return None
    if not isinstance(source, dict) or not isinstance(source.get("name"), str):
        return None
    return {"display": RESULT_CARD_DISPLAY, "result_card": card}


@dataclass(frozen=True)
class RunDisplayContext:
    """What a display tool may use of the run that is calling it."""

    # Every connector one of whose tools completed successfully earlier in
    # this run (all steps of a plan, and runs resumed from a checkpoint).
    invoked_connectors: frozenset[ResultCardConnector]
    # ``PatternRuntime.send_message``: emits one assistant-side chat message.
    send_message: Callable[..., Awaitable[dict[str, Any]]]
    # Attribution of the tool call that is showing the card (tool attempt,
    # assistant message, turn), merged into the message metadata so the fact
    # is attributed to the call and replays idempotently.
    message_source: Mapping[str, Any] = field(default_factory=dict)


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
