"""Generic tracing module for tracking events in the xagent system."""

import inspect
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

DISPLAY_MESSAGE_KEY = "display_message"
DISPLAY_USER_MESSAGE_KEY = "display_user_message"


def _display_message_from_metadata(metadata: Any) -> str | None:
    if not isinstance(metadata, dict):
        return None
    for key in (DISPLAY_MESSAGE_KEY, DISPLAY_USER_MESSAGE_KEY):
        if key in metadata:
            display_message = metadata.get(key)
            return display_message if isinstance(display_message, str) else ""
    return None


def get_display_user_message(context: Any, fallback: str) -> str:
    """Return the user-visible message for trace/UI events."""
    messages = getattr(context, "messages", None)
    if isinstance(messages, list):
        for message in reversed(messages):
            if getattr(message, "role", None) != "user":
                continue
            display_message = _display_message_from_metadata(
                getattr(message, "metadata", None)
            )
            if display_message is not None:
                return display_message
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content
            break

    candidates: list[dict[str, Any]] = []
    if isinstance(context, dict):
        candidates.append(context)

    state = getattr(context, "state", None)
    if isinstance(state, dict):
        candidates.append(state)

    metadata = getattr(context, "metadata", None)
    if isinstance(metadata, dict):
        request_context = metadata.get("request_context")
        if isinstance(request_context, dict):
            candidates.append(request_context)
        candidates.append(metadata)

    for candidate in candidates:
        display_message = _display_message_from_metadata(candidate)
        if display_message is not None:
            return display_message

    return fallback


# Fields the trace pipeline reads as identifiers, routing flags, or metrics
# (WS visibility filter, audit SQL queries, Langfuse observation naming,
# frontend rendering). Must survive normalization untouched.
_RESERVED_TRACE_FIELDS = frozenset(
    {
        # routing / visibility
        "__audit_only__",
        # call attribution
        "model_name",
        "llm_type",
        "task_type",
        "step_id",
        "step_name",
        "action",
        "attempt",
        "json_mode_failed",
        # outcomes
        "success",
        "error_type",
        "response_type",
        # metrics
        "usage",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        # cardinality counters (not the values they count)
        "messages_count",
        "candidates_count",
        "tools_count",
        "is_tool_call",
        "has_tools",
    }
)

# Fields that carry the bulky LLM I/O payload we actually want to cap.
# Each present field gets ``max_bytes // N_present`` budget; the rest pass
# through unchanged.
_TRUNCATABLE_CONTENT_FIELDS = frozenset(
    {
        "messages",
        "context_preview",
        "response",
        "content",
        "tool_calls",
        "tools",
        "error",
        "error_message",
        "traceback",
        "candidate_names",
    }
)


def truncate_for_trace(value: Any, max_bytes: Optional[int] = None) -> Any:
    """Recursive per-leaf truncation of a trace value.

    Strings longer than ``max_bytes`` are sliced on a UTF-8 byte boundary
    and suffixed with ``"...[truncated N chars]"``. Lists and dicts are
    walked with the budget split evenly across children. Scalars
    (int / bool / None) pass through unchanged.

    Unlike :func:`normalize_llm_trace_payload`, this helper does NOT
    distinguish reserved control fields from content fields and does NOT
    enforce a total-size cap on the serialized result. The serialized
    payload can exceed ``max_bytes`` once per-child suffix overhead is
    accounted for. Callers that need a tracer-boundary cap on LLM event
    payloads should use ``normalize_llm_trace_payload`` instead — this
    function is the low-level worker.

    Args:
        value: Original value.
        max_bytes: Per-subtree byte budget. When ``None``, reads
            ``XAGENT_MAX_TRACE_PAYLOAD_BYTES`` (default 50_000, 0 disables).

    Returns:
        The original value or a recursively trimmed copy.
    """
    if max_bytes is None:
        # Local import to avoid circular dependency at module load
        from ...config import get_max_trace_payload_bytes

        max_bytes = get_max_trace_payload_bytes()

    if max_bytes <= 0:
        return value

    return _trim_subtree(value, max_bytes)


# Short per-message metadata kept verbatim by ``_reduce_messages``. These
# fields (typically 4-12 bytes each) are essential for RCA grouping —
# losing ``role`` makes a trace event impossible to interpret.
_MESSAGE_METADATA_KEYS = frozenset({"role", "name", "type", "tool_call_id"})


def _serialized_size(value: Any) -> int:
    """UTF-8 byte length of ``json.dumps`` output, used for size budgeting."""
    try:
        return len(json.dumps(value, default=str, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        return len(str(value).encode("utf-8"))


def _truncate_string(value: str, budget: int) -> str:
    """Byte-boundary string truncate with multi-byte UTF-8 safety.

    Reserves a small slack (~30 bytes) for the trailing
    ``...[truncated N chars]`` marker so the post-marker length still
    fits ``budget``.
    """
    if budget <= 0:
        return ""
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= budget:
        return value
    slack = min(30, max(8, budget // 4))
    head_budget = max(1, budget - slack)
    head = encoded[:head_budget].decode("utf-8", errors="ignore")
    return f"{head}...[truncated {len(value) - len(head)} chars]"


def _placeholder(message: str) -> Dict[str, str]:
    return {"__truncated__": message}


def _reduce_text(value: Any, budget: int) -> Any:
    """Reducer for plain text fields (``content``, ``error``, ``traceback``)."""
    if not isinstance(value, str):
        value = str(value)
    return _truncate_string(value, budget)


def _reduce_text_list(value: Any, budget: int) -> Any:
    """Reducer for ``candidate_names`` and similar list-of-string fields.

    Preserves head + tail entries with a middle-omitted marker.
    """
    if not isinstance(value, list):
        return value
    if _serialized_size(value) <= budget:
        return value

    for keep_head, keep_tail in [(3, 3), (2, 2), (1, 1), (1, 0)]:
        if keep_head + keep_tail >= len(value):
            continue
        per_item = max(8, (budget - 64) // (keep_head + keep_tail))
        head = [_reduce_text(v, per_item) for v in value[:keep_head]]
        tail = (
            [_reduce_text(v, per_item) for v in value[-keep_tail:]] if keep_tail else []
        )
        omitted = len(value) - keep_head - keep_tail
        marker = f"...[truncated {omitted} middle items]"
        candidate: List[Any] = head + [marker] + tail
        if _serialized_size(candidate) <= budget:
            return candidate

    return [_placeholder(f"{len(value)} list items exceed {budget}-byte budget")]


def _shrink_message(msg: Any, msg_budget: int) -> Any:
    """Preserve short metadata keys verbatim, reduce large content keys."""
    if not isinstance(msg, dict):
        return _reduce_text(msg, msg_budget)

    out: Dict[str, Any] = {}
    for k in _MESSAGE_METADATA_KEYS:
        if k in msg:
            out[k] = msg[k]

    # Reserve ~40 bytes for JSON braces / commas / quotes around content keys.
    content_keys = [k for k in msg if k not in _MESSAGE_METADATA_KEYS]
    if not content_keys:
        return out
    remaining = max(64, msg_budget - _serialized_size(out) - 40)
    per_content = max(32, remaining // len(content_keys))

    for k in content_keys:
        v = msg[k]
        if k == "tool_calls" and isinstance(v, list):
            out[k] = _reduce_tool_calls(v, per_content)
        elif isinstance(v, str):
            out[k] = _reduce_text(v, per_content)
        elif isinstance(v, (dict, list)):
            out[k] = _reduce_text(
                json.dumps(v, default=str, ensure_ascii=False), per_content
            )
        else:
            out[k] = v
    return out


def _reduce_messages(value: Any, budget: int) -> Any:
    """Reducer for ``messages`` / ``context_preview``.

    Keeps head + tail entries (system / recent prompts have the highest
    RCA value) with their ``role`` / ``name`` / ``type`` metadata intact;
    each kept message's content is text-reduced. Middle entries become a
    single ``{"__truncated__": "K messages omitted ..."}`` placeholder.
    Iteratively shrinks the kept-count if budget still overshoots.
    """
    if not isinstance(value, list):
        return value
    if _serialized_size(value) <= budget:
        return value

    # If the list is already short, no middle to omit — just shrink each
    # entry to a per-msg budget.
    if len(value) <= 4:
        per_msg = max(128, (budget - 16) // len(value))
        shrunk = [_shrink_message(m, per_msg) for m in value]
        if _serialized_size(shrunk) <= budget:
            return shrunk
        return [_placeholder(f"{len(value)} messages exceed {budget}-byte budget")]

    for keep_head, keep_tail in [(2, 2), (1, 2), (1, 1), (1, 0)]:
        if keep_head + keep_tail >= len(value):
            continue
        omitted = len(value) - keep_head - keep_tail
        marker = _placeholder(
            f"{omitted} middle messages omitted to fit {budget}-byte budget"
        )
        marker_size = _serialized_size(marker)
        per_msg = max(128, (budget - marker_size - 16) // (keep_head + keep_tail))
        head = [_shrink_message(m, per_msg) for m in value[:keep_head]]
        tail = (
            [_shrink_message(m, per_msg) for m in value[-keep_tail:]]
            if keep_tail
            else []
        )
        candidate: List[Any] = head + [marker] + tail
        if _serialized_size(candidate) <= budget:
            return candidate

    # Even (1 head, 0 tail) overshoots — collapse the whole list.
    return [_placeholder(f"{len(value)} messages exceed {budget}-byte budget")]


def _shrink_tool(tool: Any, tool_budget: int) -> Any:
    """Per-tool: keep type/function.name/function.description; truncate parameters schema."""
    if not isinstance(tool, dict):
        return _reduce_text(tool, tool_budget)
    out: Dict[str, Any] = {"type": tool.get("type", "function")}
    func = tool.get("function")
    if isinstance(func, dict):
        shrunk_func: Dict[str, Any] = {}
        if "name" in func:
            shrunk_func["name"] = func["name"]
        if "description" in func and isinstance(func["description"], str):
            desc_budget = max(64, (tool_budget - _serialized_size(shrunk_func)) // 2)
            shrunk_func["description"] = _reduce_text(func["description"], desc_budget)
        if "parameters" in func:
            remaining = max(
                32,
                tool_budget
                - _serialized_size({"type": out["type"], "function": shrunk_func})
                - 30,
            )
            if _serialized_size(func["parameters"]) <= remaining:
                shrunk_func["parameters"] = func["parameters"]
            else:
                shrunk_func["parameters"] = _placeholder(
                    f"schema exceeds {remaining}-byte budget"
                )
        out["function"] = shrunk_func
    return out


def _reduce_tools(value: Any, budget: int) -> Any:
    """Reducer for ``tools``. Keeps tool names + descriptions, truncates schemas."""
    if not isinstance(value, list):
        return value
    if _serialized_size(value) <= budget:
        return value

    # Short list: no middle to omit — shrink each tool to per-tool budget.
    if len(value) <= 3:
        per_tool = max(128, (budget - 16) // len(value))
        shrunk = [_shrink_tool(t, per_tool) for t in value]
        if _serialized_size(shrunk) <= budget:
            return shrunk
        return [_placeholder(f"{len(value)} tools exceed {budget}-byte budget")]

    for keep_head in (3, 2, 1):
        omitted = len(value) - keep_head
        marker = _placeholder(f"{omitted} tools omitted to fit {budget}-byte budget")
        per_tool = max(128, (budget - _serialized_size(marker) - 16) // keep_head)
        head = [_shrink_tool(t, per_tool) for t in value[:keep_head]]
        candidate: List[Any] = head + [marker]
        if _serialized_size(candidate) <= budget:
            return candidate

    return [_placeholder(f"{len(value)} tools exceed {budget}-byte budget")]


def _shrink_tool_call(call: Any, call_budget: int) -> Any:
    """Per-call: keep id/type/function.name; truncate function.arguments."""
    if not isinstance(call, dict):
        return _reduce_text(call, call_budget)
    out: Dict[str, Any] = {}
    for k in ("id", "type"):
        if k in call:
            out[k] = call[k]
    func = call.get("function")
    if isinstance(func, dict):
        shrunk_func: Dict[str, Any] = {}
        if "name" in func:
            shrunk_func["name"] = func["name"]
        if "arguments" in func:
            args = func["arguments"]
            args_str = (
                args
                if isinstance(args, str)
                else json.dumps(args, default=str, ensure_ascii=False)
            )
            arg_budget = max(
                64,
                call_budget - _serialized_size({**out, "function": shrunk_func}) - 30,
            )
            shrunk_func["arguments"] = _reduce_text(args_str, arg_budget)
        out["function"] = shrunk_func
    return out


def _reduce_tool_calls(value: Any, budget: int) -> Any:
    """Reducer for ``tool_calls``. Keeps id + name per call, truncates arguments."""
    if not isinstance(value, list):
        return value
    if _serialized_size(value) <= budget:
        return value

    # Short list: no middle to omit — shrink each call to per-call budget.
    if len(value) <= 3:
        per_call = max(128, (budget - 16) // len(value))
        shrunk = [_shrink_tool_call(c, per_call) for c in value]
        if _serialized_size(shrunk) <= budget:
            return shrunk
        return [_placeholder(f"{len(value)} tool_calls exceed {budget}-byte budget")]

    for keep_head in (3, 2, 1):
        omitted = len(value) - keep_head
        marker = _placeholder(
            f"{omitted} tool_calls omitted to fit {budget}-byte budget"
        )
        per_call = max(128, (budget - _serialized_size(marker) - 16) // keep_head)
        head = [_shrink_tool_call(c, per_call) for c in value[:keep_head]]
        candidate: List[Any] = head + [marker]
        if _serialized_size(candidate) <= budget:
            return candidate

    return [_placeholder(f"{len(value)} tool_calls exceed {budget}-byte budget")]


def _reduce_response(value: Any, budget: int) -> Any:
    """Reducer for ``response``. String → text trim; dict → trim text sub-fields,
    preserve scalars verbatim (e.g. usage / token counts coming back from
    ``_short_response``).
    """
    if isinstance(value, str):
        return _reduce_text(value, budget)
    if not isinstance(value, dict):
        return _reduce_text(str(value), budget)
    if _serialized_size(value) <= budget:
        return value

    text_keys = ("content", "answer", "output", "message")
    out: Dict[str, Any] = {}
    other_keys = [k for k in value if k not in text_keys]
    for k in other_keys:
        out[k] = value[k]

    content_keys = [k for k in value if k in text_keys]
    if content_keys:
        remaining = max(64, budget - _serialized_size(out) - 30)
        per_content = max(64, remaining // len(content_keys))
        for k in content_keys:
            v = value[k]
            if k == "tool_calls" and isinstance(v, list):
                out[k] = _reduce_tool_calls(v, per_content)
            elif isinstance(v, str):
                out[k] = _reduce_text(v, per_content)
            else:
                out[k] = _reduce_text(str(v), per_content)

    if _serialized_size(out) > budget:
        return _placeholder(f"response exceeds {budget}-byte budget after reduction")
    return out


# Dispatch table: field name -> reducer. ``normalize_llm_trace_payload``
# routes each present truncatable field to its reducer. Fields not in this
# table but in ``_TRUNCATABLE_CONTENT_FIELDS`` fall back to ``_reduce_text``.
_FIELD_REDUCERS: Dict[str, Callable[[Any, int], Any]] = {
    "messages": _reduce_messages,
    "context_preview": _reduce_messages,
    "tools": _reduce_tools,
    "tool_calls": _reduce_tool_calls,
    "response": _reduce_response,
    "content": _reduce_text,
    "error": _reduce_text,
    "error_message": _reduce_text,
    "traceback": _reduce_text,
    "candidate_names": _reduce_text_list,
}


def normalize_llm_trace_payload(
    data: Any,
    max_bytes: Optional[int] = None,
) -> Any:
    """Trace-boundary normalizer for LLM-category trace event payloads.

    Three layers:

      1. **Reserved + unknown fields pass through verbatim.** Routing /
         control / metrics fields (:data:`_RESERVED_TRACE_FIELDS`) and
         fields not in :data:`_TRUNCATABLE_CONTENT_FIELDS` are copied
         as-is so downstream consumers (WS visibility filter, audit SQL,
         Langfuse) keep working under truncation. Unknown-field
         passthrough is intentional — a future emit that adds a new
         routing flag must not be silently dropped.

      2. **Per-field semantic reducers.** Each present truncatable
         field is routed to its reducer in :data:`_FIELD_REDUCERS`:

           - ``messages`` / ``context_preview``: keep head + tail with
             role/name/type metadata intact, truncate kept content
           - ``tools``: keep tool name + description, truncate
             parameters schema
           - ``tool_calls``: keep id + function.name, truncate
             function.arguments
           - ``response``: dict-shape-preserving for ``_short_response``
             output; string → byte slice with truncation marker
           - ``content`` / ``error`` / ``error_message`` / ``traceback``:
             direct text truncation
           - ``candidate_names``: head + tail list slice

         Reducers self-verify their output (post-reduce serialized size
         ≤ budget) and fall back to a single placeholder dict if even
         the smallest preservation overshoots.

      3. **Envelope-level hard cap.** After all reducers, if the
         serialized total still exceeds ``max_bytes`` (extreme case —
         e.g. reserved metadata itself oversized), collapse the
         currently-largest truncatable field to a single placeholder
         and re-check. Outer reserved metadata always survives.

    Wired into :func:`PatternRuntime._emit_trace_event` for LLM-category
    events so every v2-runtime LLM trace is capped, not just calls that
    go through :func:`trace_llm_call_start` / :func:`trace_action_end`.

    Args:
        data: Event ``data`` dict for an LLM-category trace event.
            Non-dict input is returned unchanged.
        max_bytes: Total byte budget for the serialized result. When
            ``None``, reads ``XAGENT_MAX_TRACE_PAYLOAD_BYTES`` (default
            50_000, ``0`` disables).

    Returns:
        A new dict honoring the cap with reserved metadata intact.
    """
    if not isinstance(data, dict):
        return data

    if max_bytes is None:
        from ...config import get_max_trace_payload_bytes

        max_bytes = get_max_trace_payload_bytes()

    if max_bytes <= 0:
        return data

    present_content = [k for k in data if k in _TRUNCATABLE_CONTENT_FIELDS]
    if not present_content:
        # Nothing to cap — all reserved or unknown. Even if reserved
        # itself overshoots, we don't touch it (that would defeat the
        # routing contract).
        return data

    # Reserve envelope budget for non-truncatable fields + JSON separators.
    reserved_part = {
        k: v for k, v in data.items() if k not in _TRUNCATABLE_CONTENT_FIELDS
    }
    reserved_overhead = _serialized_size(reserved_part)
    content_budget = max(256, max_bytes - reserved_overhead - 64)
    per_field_budget = max(128, content_budget // len(present_content))

    out: Dict[str, Any] = {}
    for k, v in data.items():
        if k in _TRUNCATABLE_CONTENT_FIELDS:
            reducer = _FIELD_REDUCERS.get(k, _reduce_text)
            out[k] = reducer(v, per_field_budget)
        else:
            out[k] = v

    # Final envelope hard cap. Collapse the largest remaining truncatable
    # field iteratively until under cap. Stops if all collapsed (already
    # placeholders) — at that point we can't shrink any further without
    # touching reserved metadata, which is contractually off-limits.
    while _serialized_size(out) > max_bytes:
        largest = None
        largest_size = -1
        for k in present_content:
            v = out.get(k)
            if isinstance(v, dict) and "__truncated__" in v:
                continue  # already collapsed
            size = _serialized_size(v)
            if size > largest_size:
                largest_size = size
                largest = k
        if largest is None or largest_size < 100:
            break
        out[largest] = _placeholder(
            f"{largest} exceeded total {max_bytes}-byte envelope cap"
        )

    return out


# Bound on recursion depth inside `_trim_subtree`. LLM payloads we care
# about (messages[*].content, tool_calls) nest at most 4-5 levels; 50 is
# well above that and well below Python's default 1000-frame limit, so a
# pathological payload can't blow the stack via this helper.
_MAX_TRACE_DEPTH = 50


def _trim_subtree(value: Any, max_bytes: int, _depth: int = 0) -> Any:
    """Recursive worker for :func:`truncate_for_trace`.

    Handles scalar truncation and container recursion. Does NOT enforce
    the final container size cap -- that's the outer function's job.
    Keeping the recursive case in its own function avoids re-running the
    expensive serialization-based check at every nesting level.
    """
    if _depth >= _MAX_TRACE_DEPTH:
        return f"...[truncated: depth exceeds {_MAX_TRACE_DEPTH}]"

    if isinstance(value, str):
        encoded = value.encode("utf-8", errors="replace")
        if len(encoded) <= max_bytes:
            return value
        # Slice on byte boundary, then decode with errors="ignore" so an
        # incomplete trailing multi-byte char is dropped instead of producing
        # U+FFFD replacement chars — otherwise len(head) inflates and the
        # reported truncation count becomes inaccurate (can even go negative).
        head = encoded[:max_bytes].decode("utf-8", errors="ignore")
        return f"{head}...[truncated {len(value) - len(head)} chars]"

    if isinstance(value, list):
        per_item = max(1, max_bytes // max(1, len(value)))
        return [_trim_subtree(item, per_item, _depth + 1) for item in value]

    if isinstance(value, dict):
        per_value = max(1, max_bytes // max(1, len(value)))
        return {k: _trim_subtree(v, per_value, _depth + 1) for k, v in value.items()}

    # Numbers, bools, None — pass through
    return value


class TraceScope(Enum):
    """Defines the scope of trace events for clear task/step attribution."""

    TASK = "task"  # Task-level events
    STEP = "step"  # Step-level events
    ACTION = "action"  # Action-level events (within steps)
    SYSTEM = "system"  # System-level events


class TraceAction(Enum):
    """Defines the action type of trace events."""

    START = "start"
    END = "end"
    UPDATE = "update"
    ERROR = "error"
    INFO = "info"


class TraceCategory(Enum):
    """Defines the category of trace events."""

    DAG = "dag"  # DAG execution events
    DAG_PLAN = "dag_plan"  # DAG planning events
    REACT = "react"  # ReAct pattern events
    LLM = "llm"  # LLM call events
    TOOL = "tool"  # Tool execution events
    VISUALIZATION = "visualization"  # UI update events
    MESSAGE = "message"  # User/AI message events
    MEMORY_GENERATE = "memory_generate"  # Memory generation events
    MEMORY_STORE = "memory_store"  # Memory storage events
    MEMORY_RETRIEVE = "memory_retrieve"  # Memory retrieval events
    COMPACT = "compact"  # Context compaction events
    SKILL = "skill"  # Skill selection events
    GENERAL = "general"  # General events


class TraceEventType:
    """Unified trace event type that combines scope, action, and category."""

    def __init__(self, scope: TraceScope, action: TraceAction, category: TraceCategory):
        self.scope = scope
        self.action = action
        self.category = category

    @property
    def value(self) -> str:
        return f"{self.scope.value}_{self.action.value}_{self.category.value}"

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TraceEventType):
            return False
        return (
            self.scope == other.scope
            and self.action == other.action
            and self.category == other.category
        )

    def __hash__(self) -> int:
        return hash((self.scope, self.action, self.category))


# Predefined event types for convenience
# Task-level events
TASK_START_DAG = TraceEventType(TraceScope.TASK, TraceAction.START, TraceCategory.DAG)
TASK_END_DAG = TraceEventType(TraceScope.TASK, TraceAction.END, TraceCategory.DAG)
TASK_START_REACT = TraceEventType(
    TraceScope.TASK, TraceAction.START, TraceCategory.REACT
)
TASK_END_REACT = TraceEventType(TraceScope.TASK, TraceAction.END, TraceCategory.REACT)
TASK_START_GENERAL = TraceEventType(
    TraceScope.TASK, TraceAction.START, TraceCategory.GENERAL
)
TASK_END_GENERAL = TraceEventType(
    TraceScope.TASK, TraceAction.END, TraceCategory.GENERAL
)
TASK_ERROR = TraceEventType(TraceScope.TASK, TraceAction.ERROR, TraceCategory.GENERAL)

# AI message event (for chat responses)
AI_MESSAGE = TraceEventType(TraceScope.TASK, TraceAction.END, TraceCategory.MESSAGE)

# Step-level events
STEP_START_DAG = TraceEventType(TraceScope.STEP, TraceAction.START, TraceCategory.DAG)
STEP_END_DAG = TraceEventType(TraceScope.STEP, TraceAction.END, TraceCategory.DAG)
STEP_START_REACT = TraceEventType(
    TraceScope.STEP, TraceAction.START, TraceCategory.REACT
)
STEP_END_REACT = TraceEventType(TraceScope.STEP, TraceAction.END, TraceCategory.REACT)
STEP_ERROR = TraceEventType(TraceScope.STEP, TraceAction.ERROR, TraceCategory.GENERAL)

# Memory-related events
MEMORY_GENERATE_START = TraceEventType(
    TraceScope.TASK, TraceAction.START, TraceCategory.MEMORY_GENERATE
)
MEMORY_GENERATE_END = TraceEventType(
    TraceScope.TASK, TraceAction.END, TraceCategory.MEMORY_GENERATE
)
MEMORY_STORE_START = TraceEventType(
    TraceScope.TASK, TraceAction.START, TraceCategory.MEMORY_STORE
)
MEMORY_STORE_END = TraceEventType(
    TraceScope.TASK, TraceAction.END, TraceCategory.MEMORY_STORE
)
MEMORY_RETRIEVE_START = TraceEventType(
    TraceScope.TASK, TraceAction.START, TraceCategory.MEMORY_RETRIEVE
)
MEMORY_RETRIEVE_END = TraceEventType(
    TraceScope.TASK, TraceAction.END, TraceCategory.MEMORY_RETRIEVE
)

# Compact-related events
COMPACT_START = TraceEventType(
    TraceScope.ACTION, TraceAction.START, TraceCategory.COMPACT
)
COMPACT_END = TraceEventType(TraceScope.ACTION, TraceAction.END, TraceCategory.COMPACT)

# Action-level events (consolidated)
ACTION_START_TOOL = TraceEventType(
    TraceScope.ACTION, TraceAction.START, TraceCategory.TOOL
)
ACTION_END_TOOL = TraceEventType(TraceScope.ACTION, TraceAction.END, TraceCategory.TOOL)
ACTION_START_LLM = TraceEventType(
    TraceScope.ACTION, TraceAction.START, TraceCategory.LLM
)
ACTION_END_LLM = TraceEventType(TraceScope.ACTION, TraceAction.END, TraceCategory.LLM)

# System-level events
SYSTEM_VISUALIZATION_UPDATE = TraceEventType(
    TraceScope.SYSTEM, TraceAction.UPDATE, TraceCategory.VISUALIZATION
)
SYSTEM_INFO = TraceEventType(TraceScope.SYSTEM, TraceAction.INFO, TraceCategory.GENERAL)


class TraceEvent:
    """Represents a single trace event with clear task/step attribution."""

    def __init__(
        self,
        event_type: TraceEventType,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        require_persisted: bool = False,
    ):
        self.id = str(uuid4())
        self.event_type = event_type
        self.task_id = task_id
        self.step_id = step_id
        self.timestamp = timestamp or datetime.now(timezone.utc).timestamp()
        self.data = data or {}
        self.parent_id = parent_id
        self.require_persisted = require_persisted

        # Validate scope requirements
        self._validate_scope()

    def _validate_scope(self) -> None:
        """Validate that required fields are present based on event scope."""
        if self.event_type.scope == TraceScope.TASK and not self.task_id:
            raise ValueError(
                f"Task-level event {self.event_type.value} requires task_id"
            )
        if self.event_type.scope == TraceScope.STEP and not self.step_id:
            raise ValueError(
                f"Step-level event {self.event_type.value} requires step_id"
            )
        if self.event_type.scope == TraceScope.ACTION and not self.step_id:
            raise ValueError(
                f"Action-level event {self.event_type.value} requires step_id"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace event to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "scope": self.event_type.scope.value,
            "action": self.event_type.action.value,
            "category": self.event_type.category.value,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "data": self.data,
            "parent_id": self.parent_id,
            "require_persisted": self.require_persisted,
        }


class TraceHandler(ABC):
    """Abstract base class for trace handlers."""

    @abstractmethod
    async def handle_event(self, event: TraceEvent) -> None:
        """Handle a trace event."""
        pass


class BaseTraceHandler(TraceHandler):
    """Base trace handler with common functionality."""

    def __init__(self) -> None:
        self.event_transformers = {
            TraceScope.TASK: self._handle_task_event,
            TraceScope.STEP: self._handle_step_event,
            TraceScope.ACTION: self._handle_action_event,
            TraceScope.SYSTEM: self._handle_system_event,
        }

    async def handle_event(self, event: TraceEvent) -> None:
        """Handle a trace event by delegating to scope-specific handler."""
        try:
            handler = self.event_transformers.get(event.event_type.scope)
            if handler:
                await handler(event)
            else:
                logger.warning(f"No handler for event scope: {event.event_type.scope}")
        except Exception as e:
            logger.error(f"Error handling event {event.event_type.value}: {e}")
            if event.require_persisted:
                raise
            # Don't re-raise to avoid breaking the main execution

    async def _handle_task_event(self, event: TraceEvent) -> None:
        """Handle task-level events. Override in subclasses."""
        pass

    async def _handle_step_event(self, event: TraceEvent) -> None:
        """Handle step-level events. Override in subclasses."""
        pass

    async def _handle_action_event(self, event: TraceEvent) -> None:
        """Handle action-level events. Override in subclasses."""
        pass

    async def _handle_system_event(self, event: TraceEvent) -> None:
        """Handle system-level events. Override in subclasses."""
        pass


class ConsoleTraceHandler(BaseTraceHandler):
    """Trace handler that logs events to console with clear scope information."""

    async def _handle_task_event(self, event: TraceEvent) -> None:
        """Handle task-level events."""
        logger.info(
            f"[TASK] {event.event_type.action.value.upper()} {event.event_type.category.value.upper()} - Task {event.task_id} - {event.data}"
        )

    async def _handle_step_event(self, event: TraceEvent) -> None:
        """Handle step-level events."""
        logger.info(
            f"[STEP] {event.event_type.action.value.upper()} {event.event_type.category.value.upper()} - Step {event.step_id} - {event.data}"
        )

    async def _handle_action_event(self, event: TraceEvent) -> None:
        """Handle action-level events."""
        logger.info(
            f"[ACTION] {event.event_type.action.value.upper()} {event.event_type.category.value.upper()} - Step {event.step_id} - {event.data}"
        )

    async def _handle_system_event(self, event: TraceEvent) -> None:
        """Handle system-level events."""
        logger.info(
            f"[SYSTEM] {event.event_type.action.value.upper()} {event.event_type.category.value.upper()} - {event.data}"
        )


class DatabaseTraceHandler(BaseTraceHandler):
    """Trace handler that saves events to database."""

    def __init__(self, task_id: Optional[int] = None) -> None:
        super().__init__()
        self.task_id = task_id
        # Import here to avoid circular dependencies
        # Actual database import will be handled in web-specific handler

    async def _handle_task_event(self, event: TraceEvent) -> None:
        """Handle task-level events for database storage."""
        # This will be implemented in the web-specific handler
        logger.debug(
            f"[DB] Task event: {event.event_type.value} for task {event.task_id}"
        )

    async def _handle_step_event(self, event: TraceEvent) -> None:
        """Handle step-level events for database storage."""
        # This will be implemented in the web-specific handler
        logger.debug(
            f"[DB] Step event: {event.event_type.value} for step {event.step_id}"
        )

    async def _handle_action_event(self, event: TraceEvent) -> None:
        """Handle action-level events for database storage."""
        # This will be implemented in the web-specific handler
        logger.debug(
            f"[DB] Action event: {event.event_type.value} for step {event.step_id}"
        )

    async def _handle_system_event(self, event: TraceEvent) -> None:
        """Handle system-level events for database storage."""
        # This will be implemented in the web-specific handler
        logger.debug(f"[DB] System event: {event.event_type.value}")


class Tracer:
    """Main tracing class that manages trace events and handlers."""

    def __init__(self) -> None:
        self.handlers: List[TraceHandler] = []
        # No default handlers - let users add their own

    def add_handler(self, handler: TraceHandler) -> None:
        """Add a trace handler."""
        self.handlers.append(handler)

    async def trace_event(
        self,
        event_type: TraceEventType,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        require_persisted: bool = False,
    ) -> str:
        """Record a trace event and return its ID."""
        logger.info(
            f"trace_event called: {event_type.value} for task {task_id}, step {step_id} with data keys: {list(data.keys()) if data else []}"
        )

        event = TraceEvent(
            event_type=event_type,
            task_id=task_id,
            step_id=step_id,
            data=data or {},
            parent_id=parent_id,
            require_persisted=require_persisted,
        )

        # Notify all handlers
        logger.info(
            f"Notifying {len(self.handlers)} handlers for event {event_type.value}"
        )
        handler_errors: List[Exception] = []
        for i, handler in enumerate(self.handlers):
            try:
                logger.info(f"Calling handler {i}: {type(handler).__name__}")
                await handler.handle_event(event)
                logger.info(f"Handler {i} completed successfully")
            except Exception as e:
                if require_persisted:
                    handler_errors.append(e)
                else:
                    logger.warning(f"Trace handler {i} failed: {e}")

        if require_persisted:
            if handler_errors:
                raise RuntimeError(
                    f"{len(handler_errors)} trace handler(s) failed while "
                    "persisting required trace event."
                ) from handler_errors[0]
            if not self.handlers:
                raise RuntimeError(
                    "No trace handlers are configured for required trace persistence."
                )

        logger.info(
            f"trace_event completed for {event_type.value}, event_id: {event.id}"
        )
        return event.id

    async def load_latest_checkpoint(
        self,
        execution_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint from the first handler that supports it."""
        for handler in self.handlers:
            method = getattr(handler, "load_latest_checkpoint", None)
            if not callable(method):
                continue
            payload = method(execution_id)
            if inspect.isawaitable(payload):
                payload = await payload
            if isinstance(payload, dict):
                return payload
        return None

    async def get_latest_checkpoint(
        self,
        execution_id: str,
    ) -> Optional[Dict[str, Any]]:
        return await self.load_latest_checkpoint(execution_id)

    async def latest_checkpoint(
        self,
        execution_id: str,
    ) -> Optional[Dict[str, Any]]:
        return await self.load_latest_checkpoint(execution_id)


# Simplified convenience functions for common tracing operations
async def trace_task_start(
    tracer: Tracer,
    task_id: str,
    category: TraceCategory,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace task start event."""
    event_type = TraceEventType(TraceScope.TASK, TraceAction.START, category)
    return await tracer.trace_event(event_type, task_id=task_id, data=data or {})


async def trace_task_end(
    tracer: Tracer,
    task_id: str,
    category: TraceCategory,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace task end event."""
    event_type = TraceEventType(TraceScope.TASK, TraceAction.END, category)
    return await tracer.trace_event(event_type, task_id=task_id, data=data or {})


async def trace_task_completion(
    tracer: Tracer,
    task_id: str,
    result: Any,
    success: bool = True,
) -> str:
    """Trace task completion event with result data."""
    event_type = TASK_END_GENERAL

    data = {
        "result": result,
        "success": success,
    }
    return await tracer.trace_event(event_type, task_id=task_id, data=data)


async def trace_step_start(
    tracer: Tracer,
    task_id: str,
    step_id: str,
    category: TraceCategory,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace step start event."""
    event_type = TraceEventType(TraceScope.STEP, TraceAction.START, category)
    return await tracer.trace_event(
        event_type, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_step_end(
    tracer: Tracer,
    task_id: str,
    step_id: str,
    category: TraceCategory,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace step end event."""
    event_type = TraceEventType(TraceScope.STEP, TraceAction.END, category)
    return await tracer.trace_event(
        event_type, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_action_start(
    tracer: Tracer,
    task_id: str,
    step_id: str,
    category: TraceCategory,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace action start event."""
    event_type = TraceEventType(TraceScope.ACTION, TraceAction.START, category)
    return await tracer.trace_event(
        event_type, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_action_end(
    tracer: Tracer,
    task_id: str,
    step_id: str,
    category: TraceCategory,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace action end event."""
    event_type = TraceEventType(TraceScope.ACTION, TraceAction.END, category)
    payload = data or {}
    # Bound LLM I/O audit rows (messages / response can be tens of KB each)
    # while preserving reserved control fields. Other categories pass through.
    if category == TraceCategory.LLM and payload:
        payload = normalize_llm_trace_payload(payload)
    return await tracer.trace_event(
        event_type, task_id=task_id, step_id=step_id, data=payload
    )


async def trace_error(
    tracer: Tracer,
    task_id: str,
    step_id: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    traceback_str: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace error event with clear scope attribution."""
    event_data = {}
    if error_type:
        event_data["error_type"] = error_type
    if error_message:
        event_data["error_message"] = error_message
    if traceback_str:
        event_data["traceback"] = traceback_str
    if data:
        event_data.update(data)

    # Determine scope based on whether step_id is provided
    scope = TraceScope.STEP if step_id else TraceScope.TASK
    event_type = TraceEventType(scope, TraceAction.ERROR, TraceCategory.GENERAL)

    return await tracer.trace_event(
        event_type, task_id=task_id, step_id=step_id, data=event_data
    )


async def trace_info(
    tracer: Tracer,
    task_id: str,
    step_id: Optional[str] = None,
    category: TraceCategory = TraceCategory.GENERAL,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace info event with flexible scope."""
    scope = TraceScope.STEP if step_id else TraceScope.TASK
    event_type = TraceEventType(scope, TraceAction.INFO, category)

    return await tracer.trace_event(
        event_type, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_memory_generate_start(
    tracer: Tracer,
    task_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace memory generation start."""
    return await tracer.trace_event(
        MEMORY_GENERATE_START, task_id=task_id, data=data or {}
    )


async def trace_memory_generate_end(
    tracer: Tracer,
    task_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace memory generation end."""
    return await tracer.trace_event(
        MEMORY_GENERATE_END, task_id=task_id, data=data or {}
    )


async def trace_memory_store_start(
    tracer: Tracer,
    task_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace memory storage start."""
    return await tracer.trace_event(
        MEMORY_STORE_START, task_id=task_id, data=data or {}
    )


async def trace_memory_store_end(
    tracer: Tracer,
    task_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace memory storage end."""
    return await tracer.trace_event(MEMORY_STORE_END, task_id=task_id, data=data or {})


async def trace_memory_retrieve_start(
    tracer: Tracer,
    task_id: str,
    data: Optional[Dict[str, Any]] = None,
    step_id: Optional[str] = None,
) -> str:
    """Trace memory retrieval start event."""
    return await tracer.trace_event(
        MEMORY_RETRIEVE_START, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_memory_retrieve_end(
    tracer: Tracer,
    task_id: str,
    data: Optional[Dict[str, Any]] = None,
    step_id: Optional[str] = None,
) -> str:
    """Trace memory retrieval end event."""
    return await tracer.trace_event(
        MEMORY_RETRIEVE_END, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_compact_start(
    tracer: Tracer,
    task_id: str,
    step_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace context compaction start."""
    return await tracer.trace_event(
        COMPACT_START, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_compact_end(
    tracer: Tracer,
    task_id: str,
    step_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace context compaction end."""
    return await tracer.trace_event(
        COMPACT_END, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_system_event(
    tracer: Tracer,
    action: TraceAction,
    category: TraceCategory,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace system-level event."""
    event_type = TraceEventType(TraceScope.SYSTEM, action, category)
    return await tracer.trace_event(event_type, data=data or {})


# Common convenience functions for specific use cases
async def trace_dag_plan_start(
    tracer: Tracer, task_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace DAG plan start event."""
    return await trace_task_start(tracer, task_id, TraceCategory.DAG_PLAN, data)


async def trace_dag_plan_end(
    tracer: Tracer, task_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace DAG plan end event."""
    return await trace_task_end(tracer, task_id, TraceCategory.DAG_PLAN, data)


async def trace_dag_execution(
    tracer: Tracer,
    task_id: str,
    phase: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace DAG execution status event.

    Args:
        tracer: The tracer instance
        task_id: The task ID
        phase: The execution phase ("planning", "executing", "completed", "failed")
        data: Additional data to include in the event

    Returns:
        The event ID
    """
    event_data = data or {}
    event_data["phase"] = phase
    return await tracer.trace_event(
        TraceEventType(TraceScope.TASK, TraceAction.UPDATE, TraceCategory.DAG),
        task_id=task_id,
        data=event_data,
    )


async def trace_dag_step_start(
    tracer: Tracer, task_id: str, step_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace DAG step start event."""
    return await trace_step_start(tracer, task_id, step_id, TraceCategory.DAG, data)


async def trace_llm_call_start(
    tracer: Tracer, task_id: str, step_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace LLM call start event."""
    # Bound the payload via the tracer-boundary normalizer so reserved
    # control fields survive even when content fields are truncated.
    payload = normalize_llm_trace_payload(data) if data else data
    return await trace_action_start(
        tracer, task_id, step_id, TraceCategory.LLM, payload
    )


async def trace_task_llm_call_start(
    tracer: Tracer, task_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace task-level LLM call start event (not associated with a specific step)."""
    return await trace_task_start(tracer, task_id, TraceCategory.LLM, data)


async def trace_task_llm_call_end(
    tracer: Tracer, task_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace task-level LLM call end event (not associated with a specific step)."""
    return await trace_task_end(tracer, task_id, TraceCategory.LLM, data)


async def trace_tool_execution_start(
    tracer: Tracer,
    task_id: str,
    step_id: str,
    tool_name: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace tool execution start event."""
    event_data = {"tool_name": tool_name}
    if data:
        event_data.update(data)
    return await trace_action_start(
        tracer, task_id, step_id, TraceCategory.TOOL, event_data
    )


async def trace_visualization_update(
    tracer: Tracer, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace visualization update event."""
    return await trace_system_event(
        tracer, TraceAction.UPDATE, TraceCategory.VISUALIZATION, data
    )


async def trace_user_message(
    tracer: Tracer, task_id: str, message: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace a user-visible message event.

    ``message`` is display text for transcript/UI surfaces. Runtime prompts and
    internal execution context belong in LLM/runtime trace events instead.
    """
    event_data = {"message": message}
    if data:
        event_data.update(data)
    return await trace_task_start(tracer, task_id, TraceCategory.MESSAGE, event_data)


async def trace_ai_message(
    tracer: Tracer, task_id: str, message: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace AI message event."""
    event_data = {"content": message}  # Use 'content' to match frontend expectations
    if data:
        event_data.update(data)
    # Use AI_MESSAGE event type to generate "ai_message" event_type for frontend
    return await tracer.trace_event(AI_MESSAGE, task_id=task_id, data=event_data)


async def trace_skill_select_start(
    tracer: Tracer, task_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace skill selection start event."""
    return await trace_task_start(tracer, task_id, TraceCategory.SKILL, data)


async def trace_skill_select_end(
    tracer: Tracer, task_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace skill selection end event."""
    return await trace_task_end(tracer, task_id, TraceCategory.SKILL, data)
