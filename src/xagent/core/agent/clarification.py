"""Typed clarification draft derived from a ReAct waiting request.

This module is pure core layer: it imports nothing from ``xagent.web`` and
opens no database connection. Everything here is a plain function or a frozen
dataclass over already-in-memory data.

A :class:`ClarificationDraft` is a derived value, not stored state. Its only
source of truth is ``ReActPattern.waiting_for_user_request``, which already
travels through ``get_state()`` / ``load_state()`` on its own. The draft
itself does not join that round trip -- it is cheap to recompute from the
restored request every time a waiting turn is produced, so there is nothing
to persist or reconcile a second time.

``turn_marker`` is the only idempotency-relevant value this module produces,
and it carries no persistence-format promise of its own: it is a stable,
opaque string. Deriving a storage key (length limits, character-set
validation, or any other on-disk contract) is a web-layer concern that reads
this string; this module does not know what that contract looks like.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

logger = logging.getLogger(__name__)

CLARIFICATION_SOURCES = frozenset({"send_message", "ask_user_question", "tool_waiting"})

# Marker composition constants. ``_MARKER_KEEP`` must stay character-for-
# character identical to the whitelist in
# ``src/xagent/web/api/trace_handlers.py``'s ``clean_string`` (see
# ``_marker_clean`` below for the full cross-layer contract note).
_MARKER_KEEP = "\n\r\t"
_MARKER_SEP = "|"


@dataclass(frozen=True)
class ClarificationRequestItem:
    """One tool's pending question inside a (possibly multi-tool) draft."""

    tool_name: str
    tool_call_id: str
    interaction_id: str


@dataclass(frozen=True)
class ClarificationDraft:
    """A typed view over a waiting turn, common to all clarification sources.

    ``requests`` carries one item per pending question: the single-source
    waiting points (``send_message`` / ``ask_user_question``) contribute
    exactly one item, and the multi-tool waiting point contributes one item
    per waiting tool. A malformed or legacy waiting payload can yield an
    empty tuple; construction still succeeds and the marker stays
    deterministic.

    ``turn_marker`` is a stable string built from ``turn_message_count``,
    ``origin_step_id``, and the ordered ``(tool_call_id, interaction_id)``
    pairs from ``requests``. Four invariants hold for it:

    1. It is stable across a re-entrant resume that carries no new user
       message: recomputing it from the same restored request yields the
       same string.
    2. Two drafts that differ in ``origin_step_id`` -- the same turn resolved
       against two different DAG steps -- never produce the same marker.
    3. It survives the trace serializer's control-character cleanup
       (``_marker_clean`` shares that filter's domain) without changing
       value, and re-cleaning an already-clean marker is a no-op.
    4. Component boundaries cannot be forged by content: every component is
       written as ``<length-after-filtering>:<value>`` before being joined,
       so no combination of field values can make two different request
       lists collapse onto the same marker.
    """

    source: str
    message: str
    message_type: str
    interactions: tuple[dict[str, Any], ...]
    requests: tuple[ClarificationRequestItem, ...]
    origin_step_id: str
    origin_execution_id: str
    turn_message_count: int
    turn_marker: str

    def with_origin_step(self, step_id: str) -> "ClarificationDraft":
        """Return a copy attributed to ``step_id``, with the marker redone.

        A plain field replace would leave ``turn_marker`` computed against
        the old ``origin_step_id``, silently breaking invariant 2 above.
        """

        turn_marker = _compose_turn_marker(
            turn_message_count=self.turn_message_count,
            origin_step_id=step_id,
            requests=self.requests,
        )
        return replace(self, origin_step_id=step_id, turn_marker=turn_marker)


def _marker_clean(value: str) -> str:
    """Filter ``value`` to the character domain a marker may contain.

    This function must stay in the same domain as ``clean_string`` in
    ``src/xagent/web/api/trace_handlers.py:_serialize_data_for_json`` --
    that function is a closure the core layer cannot import, so this is a
    forced duplicate rather than a convenience one. If someone changes the
    domain of ``clean_string``, this function must change with it. The test
    that watches for drift is
    ``test_marker_survives_trace_serialization_of_a_dirty_interaction_id`` in
    ``tests/core/agent/test_react_clarification_draft.py``, which calls the
    real ``DatabaseTraceHandler._serialize_data_for_json`` rather than a copy
    of it.
    """

    cleaned = value.replace("\x00", "")  # Remove NULL character
    cleaned = cleaned.replace("\u0000", "")  # Remove Unicode NULL
    return "".join(char for char in cleaned if ord(char) >= 32 or char in _MARKER_KEEP)


def _field(value: object) -> str:
    """Encode one marker component as ``<filtered length>:<filtered value>``.

    The length prefix is what keeps component boundaries unambiguous: two
    request lists whose raw values differ can still be told apart even when
    a value happens to contain the ``|`` separator, because the separator
    carries no parsing responsibility on its own.
    """

    cleaned = _marker_clean(str(value))
    return f"{len(cleaned)}:{cleaned}"


def _compose_turn_marker(
    *,
    turn_message_count: int,
    origin_step_id: str,
    requests: tuple[ClarificationRequestItem, ...],
) -> str:
    parts = [
        _field(turn_message_count),
        _field(origin_step_id),
        _field(len(requests)),
    ]
    for item in requests:
        parts.append(_field(item.tool_call_id))
        parts.append(_field(item.interaction_id))
    return _MARKER_SEP.join(parts)


def draft_from_waiting_request(
    request: dict[str, Any] | None,
    *,
    execution_id: str | None,
    step_id: str | None,
) -> ClarificationDraft | None:
    """Build a :class:`ClarificationDraft` from a ``waiting_for_user_request``.

    ``request`` is classified into exactly one of three sources by checking
    facts already present on the dict, never by re-deriving them:

    - ``request["kind"] == "tool_waiting_for_user"`` -> ``"tool_waiting"``,
      one :class:`ClarificationRequestItem` per entry in ``request["requests"]``.
    - otherwise ``request["message_type"] == "question"`` *and* ``request``
      has an ``"interactions"`` key (key presence, not truthiness -- an
      empty-form ``ask_user_question`` still has the key) -> ``"ask_user_question"``.
    - otherwise -> ``"send_message"``.

    Returns ``None`` in exactly one case: ``request`` carries neither a
    non-empty message nor an ``"interactions"`` key nor a non-empty
    ``"requests"`` list. That case has no reachable production caller today;
    it exists so that resuming a checkpoint written by an older schema
    degrades to "no draft" instead of raising and blocking resume.
    """

    if not isinstance(request, dict):
        return None

    message = str(request.get("message") or "")
    has_interactions_key = "interactions" in request
    raw_tool_requests = request.get("requests")
    has_tool_requests = isinstance(raw_tool_requests, list) and bool(raw_tool_requests)

    if not message and not has_interactions_key and not has_tool_requests:
        logger.warning(
            "draft_from_waiting_request: waiting request has neither a "
            "message nor an interactions field; skipping clarification draft"
        )
        return None

    origin_step_id = "" if step_id is None else str(step_id)
    origin_execution_id = "" if execution_id is None else str(execution_id)
    turn_message_count = int(request.get("message_count", 0) or 0)
    interactions_raw = request.get("interactions")
    interactions = tuple(interactions_raw) if isinstance(interactions_raw, list) else ()

    if request.get("kind") == "tool_waiting_for_user":
        source = "tool_waiting"
        message_type = str(request.get("message_type") or "question")
        items: list[ClarificationRequestItem] = []
        if isinstance(raw_tool_requests, list):
            for raw_item in raw_tool_requests:
                if not isinstance(raw_item, dict):
                    continue
                tool_call_id = str(raw_item.get("tool_call_id") or "")
                items.append(
                    ClarificationRequestItem(
                        tool_name=str(raw_item.get("tool_name") or ""),
                        tool_call_id=tool_call_id,
                        interaction_id=str(
                            raw_item.get("interaction_id") or tool_call_id
                        ),
                    )
                )
        requests = tuple(items)
    elif request.get("message_type") == "question" and has_interactions_key:
        source = "ask_user_question"
        tool_call_id = str(request.get("tool_call_id") or "")
        requests = (
            ClarificationRequestItem(
                tool_name=str(request.get("tool_name") or ""),
                tool_call_id=tool_call_id,
                interaction_id=tool_call_id,
            ),
        )
        message_type = "question"
    else:
        source = "send_message"
        tool_call_id = str(request.get("tool_call_id") or "")
        requests = (
            ClarificationRequestItem(
                tool_name=str(request.get("tool_name") or ""),
                tool_call_id=tool_call_id,
                interaction_id=tool_call_id,
            ),
        )
        message_type = str(request.get("message_type") or "info")

    turn_marker = _compose_turn_marker(
        turn_message_count=turn_message_count,
        origin_step_id=origin_step_id,
        requests=requests,
    )

    return ClarificationDraft(
        source=source,
        message=message,
        message_type=message_type,
        interactions=interactions,
        requests=requests,
        origin_step_id=origin_step_id,
        origin_execution_id=origin_execution_id,
        turn_message_count=turn_message_count,
        turn_marker=turn_marker,
    )
