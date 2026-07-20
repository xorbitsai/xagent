"""Pure mapping from internal trace events to public SDK step types.

Background:

    Internally the agent runtime emits a rich tree of ~32 ``event_type``
    strings (see ``ws_trace_handlers.get_event_type_mapping``) that
    capture every phase, sub-phase, tool call, LLM call, memory op,
    visualization tick, etc. That granularity is useful for the web
    UI's live trace view and for internal debugging, but it's far too
    much surface area to commit to in a stable public SDK contract.

    This module collapses those 32 internal types into the **4 public
    step types** the SDK promises:

      - ``thinking``         (reasoning phases: planning / step / action)
      - ``tool_call``        (tool invocations or skill selections)
      - ``agent_delegation`` (a sub-agent invocation -- surfaced
                              separately because the call shape is
                              meaningfully different from a flat tool)
      - ``message``          (one user or assistant message)

    Everything else (llm_call_*, memory_*, dag_execute_*, react_task_*,
    react_step_*, visualization_update, task_completion, trace_error,
    action_*_compact) is **intentionally not exposed** so the SDK
    surface can evolve without breaking clients.

Pure-function design:

    ``map_trace_events_to_public_steps`` takes a list of DB
    ``TraceEvent`` rows and returns a list of :class:`PublicStep`
    dicts. It has no DB / FastAPI / async dependencies so it's
    independently unit-testable against synthetic events and the only
    place SDK clients can observe a behavior change is through this
    one function's output -- which makes regressions easy to gate.

Pairing rule:

    Start / end events are paired by a stable ``key``:

      - ``tool_execution_*`` events pair on ``data['tool_execution_id']``
        when present, falling back to ``step_id``. The tool execution
        id is generated per-invocation and is the only safe key when
        the same tool is called twice in the same step.
      - ``react_action_*`` events pair on ``step_id``.
      - ``dag_step_*`` events pair on ``step_id``.
      - ``dag_plan_*`` events pair on ``task_id`` (single planning
        phase per task; no per-plan identifier available).
      - ``skill_select_*`` events pair on ``task_id`` (single skill
        selection phase per task).

    Orphan ends (end with no matching start) are dropped -- they
    represent malformed data and the SDK contract is "every step has
    a start"; emitting an orphan would make ``started_at`` synthetic
    and confusing.

    Orphan starts (start with no matching end) are emitted with
    ``status='running'`` and ``completed_at=None``. This naturally
    handles the case where the SDK polls ``/steps`` mid-task.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from xagent.core.tools.adapters.vibe.agent_tool_names import (
    LEGACY_AGENT_TOOL_NAME_PREFIX,
    is_agent_tool_name,
)
from xagent.core.tools.adapters.vibe.connector_runtime import (
    redact_runtime_sensitive_payload,
)

logger = logging.getLogger(__name__)


# Agent tool names are routed to the public ``agent_delegation`` type
# instead of ``tool_call`` so SDK consumers can render nested
# timelines without pattern-matching tool names client-side.


def map_trace_events_to_public_steps(
    events: List[Any],
) -> List[Dict[str, Any]]:
    """Collapse internal trace events into the 4 public SDK step types.

    Args:
        events: List of ``TraceEvent`` ORM rows (or any objects with
            ``event_type``, ``data``, ``step_id``, ``timestamp``, and
            ``event_id`` attributes). Caller is responsible for
            filtering by ``task_id`` and ordering by ``id`` ASC; this
            function preserves the input order in its output.

    Returns:
        List of public-step dicts in the order their **start** events
        first appeared. Each dict has the shape:

            {
                "id": str,
                "type": "thinking" | "tool_call" | "agent_delegation" | "message",
                "status": "running" | "completed" | "failed",
                "started_at": datetime,
                "completed_at": Optional[datetime],
                "data": dict (type-specific fields, see below)
            }

        Type-specific ``data`` fields:

          - ``thinking``: ``{"phase": "planning" | "step" | "action"}``
          - ``tool_call``: ``{"name", "args", "result"?, "error"?}``
            (``result`` populated on success, ``error`` on failure)
          - ``agent_delegation``: ``{"sub_agent_name", "input"?, "output"?}``
            (``sub_agent_name`` is derived from the agent tool name;
            ``input`` from ``tool_args`` if available, ``output`` from
            end event's ``result``)
          - ``message``: ``{"role": "user"|"assistant", "content": str}``

    Notes:
        - This function is pure (no I/O, no global state). All test
          cases live in ``tests/web/api/v1/test_steps_mapping.py``.
        - Unknown event types (anything not listed in the mapping
          tables in this module) are silently dropped. Adding a new
          public type is a deliberate per-type opt-in.
    """
    # In-progress (start seen, end not yet seen) steps keyed by
    # (public_type, pairing_key). Order of insertion is preserved by
    # Python 3.7+ dict semantics, which is what we use to emit final
    # output in the order steps were started.
    pending: Dict[Tuple[str, str], Dict[str, Any]] = {}
    # Completed-or-emitted-immediately steps. Filled in either by an
    # end event matching a pending start, or by a one-shot event like
    # ``user_message`` / ``ai_message`` which has no separate end.
    finished: List[Dict[str, Any]] = []
    # ``dag_plan_*`` has no per-plan identifier in the event data, so
    # we synthesize one by counting starts and remembering the
    # currently-open key. Replan in a single task (rare but legal)
    # produces N >= 2 pairs; without this counter the second
    # dag_plan_start would silently overwrite the first's pending
    # entry. We assume plans don't nest (only one in flight at a
    # time); nesting would require a stack, which DAG doesn't emit.
    plan_counter = 0
    open_plan_key: Optional[str] = None

    for event in events:
        event_type = _safe_get(event, "event_type")
        if not event_type:
            continue

        # ===== messages: one event per message, no pairing =====
        if event_type == "user_message":
            finished.append(_build_message_step(event, role="user"))
            continue
        if event_type == "ai_message":
            finished.append(_build_message_step(event, role="assistant"))
            continue

        # ===== thinking: paired start/end =====
        thinking_phase = _thinking_phase_for(event_type)
        if thinking_phase == "planning":
            # Special-cased because plan events have no per-plan id;
            # we generate one from a counter and remember the open
            # key so the next dag_plan_end pairs with the latest start.
            if event_type.endswith("_start"):
                plan_counter += 1
                task_ref = (
                    _safe_get(event, "task_id")
                    or _safe_get(event, "event_id")
                    or "anon"
                )
                open_plan_key = f"plan:{task_ref}:{plan_counter}"
                pending[("thinking", open_plan_key)] = _build_thinking_start(
                    event, phase="planning", key=open_plan_key
                )
            elif event_type.endswith("_end") and open_plan_key is not None:
                _finalize_pending(
                    pending,
                    finished,
                    ("thinking", open_plan_key),
                    end_event=event,
                    status="completed",
                )
                open_plan_key = None
            # Orphan end with no open plan: drop silently (same policy
            # as orphan tool_execution_end).
            continue

        if thinking_phase is not None:
            # action / step branch -- step_id is the natural pair key.
            key = _thinking_pair_key(event, thinking_phase)
            if event_type.endswith("_start"):
                pending[("thinking", key)] = _build_thinking_start(
                    event, phase=thinking_phase, key=key
                )
            elif event_type.endswith("_end"):
                _finalize_pending(
                    pending,
                    finished,
                    ("thinking", key),
                    end_event=event,
                    status="completed",
                )
            # other actions (e.g. dag_execution UPDATE) for these
            # categories are not exposed.
            continue

        # ===== tool_call / agent_delegation: paired start/end + failure =====
        if event_type in (
            "tool_execution_start",
            "tool_execution_end",
            "tool_execution_failed",
        ):
            tool_name = _data_get(event, "tool_name")
            is_delegation = is_agent_tool_name(tool_name)
            public_type = "agent_delegation" if is_delegation else "tool_call"
            # Pair on a per-invocation id (unique even when one step
            # invokes the same tool twice). v1 emits
            # ``tool_execution_id``; v2 emits ``tool_call_id``. Either
            # is fine — the fallback chain accepts both. step_id alone
            # is unsafe because one step may invoke multiple tools.
            key = (
                _data_get(event, "tool_execution_id")
                or _data_get(event, "tool_call_id")
                or _safe_get(event, "step_id")
                or _safe_get(event, "event_id")
            )
            if not key:
                continue

            if event_type == "tool_execution_start":
                pending[(public_type, str(key))] = _build_tool_start(
                    event,
                    public_type=public_type,
                    tool_name=tool_name,
                    key=str(key),
                )
            elif event_type == "tool_execution_end":
                success = _data_get(event, "success", default=True)
                status = "completed" if success else "failed"
                # ``tool_call`` and ``agent_delegation`` use different keys
                # on the public schema: tool_call exposes ``result``
                # (generic tool return), agent_delegation exposes
                # ``output`` (mirroring ``input`` on the start side). The
                # underlying internal field is still ``data['result']`` --
                # we only rename on the public surface. ``error`` is the
                # same on both for failures.
                success_key = (
                    "output" if public_type == "agent_delegation" else "result"
                )
                _finalize_pending(
                    pending,
                    finished,
                    (public_type, str(key)),
                    end_event=event,
                    status=status,
                    extra_data_fn=lambda ev, succ=success, k=success_key: (
                        {k: _data_get(ev, "result")}
                        if succ
                        else {
                            "error": _data_get(ev, "error") or "Tool execution failed"
                        }
                    ),
                )
            else:  # tool_execution_failed
                # v2 runtime emits a dedicated failure event
                # (TraceCategory.TOOL + TraceAction.ERROR) instead of
                # tool_execution_end with success=False. Without this
                # branch the pending start was never finalized and the
                # public step stayed at status='running' indefinitely.
                _finalize_pending(
                    pending,
                    finished,
                    (public_type, str(key)),
                    end_event=event,
                    status="failed",
                    extra_data_fn=lambda ev: {
                        "error": _data_get(ev, "error")
                        or _data_get(ev, "error_message")
                        or "Tool execution failed"
                    },
                )
            continue

        # ===== skill_select_*: surface as tool_call with skill name =====
        if event_type in ("skill_select_start", "skill_select_end"):
            key = (
                _data_get(event, "skill_name")
                or _safe_get(event, "step_id")
                or str(_safe_get(event, "task_id") or "skill")
            )
            if event_type == "skill_select_start":
                pending[("tool_call", str(key))] = _build_tool_start(
                    event,
                    public_type="tool_call",
                    tool_name=_data_get(event, "skill_name") or "skill_select",
                    key=str(key),
                )
            else:
                _finalize_pending(
                    pending,
                    finished,
                    ("tool_call", str(key)),
                    end_event=event,
                    status="completed",
                    extra_data_fn=lambda ev: {"result": _data_get(ev, "result")},
                )
            continue

        # Everything else (llm_call_*, memory_*, dag_execute_*,
        # react_task_*, react_step_*, visualization_update,
        # task_completion, trace_error, action_*_compact) -- not
        # exposed in the SDK contract. Silently drop.

    # Emit pending starts as ``running`` steps. The insertion order in
    # ``pending`` plus the iteration order of ``finished`` gives us a
    # stable public ordering: ``finished`` are sorted by completion
    # (which is the order they fired), and any still-running steps
    # come after in started-at order.
    output = list(finished)
    output.extend(pending.values())
    # Final sort by ``started_at`` so output is monotonic regardless of
    # whether a step finishes before the next one starts. Stable sort
    # preserves insertion order for ties.
    output.sort(key=lambda s: s["started_at"])
    return output


# ===== thinking helpers =====


_THINKING_PHASE_BY_PREFIX: Tuple[Tuple[str, str], ...] = (
    # Order matters: longer / more-specific prefixes first so
    # ``react_action_*`` doesn't accidentally match a future
    # ``react_*`` general rule.
    ("react_action_", "action"),
    ("dag_step_", "step"),
    ("dag_plan_", "planning"),
)


def _thinking_phase_for(event_type: str) -> Optional[str]:
    """Return the public ``thinking.phase`` value for an internal event,
    or ``None`` if this event is not a thinking event.
    """
    for prefix, phase in _THINKING_PHASE_BY_PREFIX:
        if event_type.startswith(prefix):
            return phase
    return None


def _thinking_pair_key(event: Any, phase: str) -> str:
    """Pairing key for a non-planning thinking start/end event.

    ``react_action_*`` and ``dag_step_*`` always carry a step_id
    which is the natural pairing key. Planning events are handled
    inline in :func:`map_trace_events_to_public_steps` because they
    lack a per-plan identifier and need a synthesized counter.
    """
    return str(_safe_get(event, "step_id") or _safe_get(event, "event_id") or "")


def _build_thinking_start(event: Any, *, phase: str, key: str) -> Dict[str, Any]:
    return {
        "id": f"thinking:{key}",
        "type": "thinking",
        "status": "running",
        "started_at": _ts(event),
        "completed_at": None,
        "data": {"phase": phase},
    }


# ===== tool_call / agent_delegation helpers =====


def _build_tool_start(
    event: Any,
    *,
    public_type: str,
    tool_name: Optional[str],
    key: str,
) -> Dict[str, Any]:
    """Build the start side of a tool_call or agent_delegation step.

    For legacy ``call_agent_<name>`` delegation tool names we extract
    the suffix for backward-compatible display. Canonical ``agent_<id>``
    and semantic Workforce tool names are already stable public identifiers.

    The args/input value lives under different keys depending on which
    runtime emitted the event: v1 uses ``tool_args``, v2 uses
    ``tool_params``. We read whichever is present so the public step
    surface stays uniform across runtimes.
    """
    args = _data_get(event, "tool_args")
    if args is None:
        args = _data_get(event, "tool_params")
    args = redact_runtime_sensitive_payload(args)
    assistant_content = _data_get(event, "assistant_content")
    assistant_content = (
        assistant_content.strip()
        if isinstance(assistant_content, str) and assistant_content.strip()
        else None
    )
    if public_type == "agent_delegation" and isinstance(tool_name, str):
        sub_agent_name = (
            tool_name.removeprefix(LEGACY_AGENT_TOOL_NAME_PREFIX)
            if tool_name.startswith(LEGACY_AGENT_TOOL_NAME_PREFIX)
            else tool_name
        )
        data = {
            "sub_agent_name": sub_agent_name,
            "input": args,
        }
        if assistant_content:
            data["assistant_content"] = assistant_content
        return {
            "id": f"agent_delegation:{key}",
            "type": "agent_delegation",
            "status": "running",
            "started_at": _ts(event),
            "completed_at": None,
            "data": data,
        }
    data = {
        "name": tool_name,
        "args": args,
    }
    if assistant_content:
        data["assistant_content"] = assistant_content
    return {
        "id": f"tool_call:{key}",
        "type": "tool_call",
        "status": "running",
        "started_at": _ts(event),
        "completed_at": None,
        "data": data,
    }


# ===== message helpers =====


def _build_message_step(event: Any, *, role: str) -> Dict[str, Any]:
    """One-shot message step (no pairing).

    user_message stores its text in ``data['message']`` (see
    ``trace_user_message``); ai_message uses ``data['content']`` (see
    ``trace_ai_message``). We normalize both into ``content`` here so
    SDK consumers don't have to know about the asymmetry.
    """
    content = _data_get(event, "content")
    if content is None:
        content = _data_get(event, "message")
    ts = _ts(event)
    return {
        "id": f"message:{_safe_get(event, 'event_id') or _safe_get(event, 'id')}",
        "type": "message",
        "status": "completed",
        "started_at": ts,
        "completed_at": ts,
        "data": {
            "role": role,
            "content": content or "",
        },
    }


# ===== shared finalization =====


def _finalize_pending(
    pending: Dict[Tuple[str, str], Dict[str, Any]],
    finished: List[Dict[str, Any]],
    key: Tuple[str, str],
    *,
    end_event: Any,
    status: str,
    extra_data_fn: Optional[Any] = None,
) -> None:
    """Move ``pending[key]`` to ``finished`` and patch with end metadata.

    Orphan end (no matching start in ``pending``) is dropped on
    purpose -- see module docstring.
    """
    step = pending.pop(key, None)
    if step is None:
        # Orphan end event; skip.
        return
    step["status"] = status
    step["completed_at"] = _ts(end_event)
    if extra_data_fn is not None:
        try:
            extra = extra_data_fn(end_event) or {}
            step["data"].update(redact_runtime_sensitive_payload(extra))
        except Exception as exc:  # defensive; data shape is external
            logger.debug("step extra_data_fn failed: %s", exc)
    finished.append(step)


# ===== attribute / data accessors =====


def _safe_get(event: Any, name: str, default: Any = None) -> Any:
    """Read an attribute that may exist on the ORM row OR in event.data.

    Handles both real ``TraceEvent`` rows and the lightweight dict-like
    stubs used in unit tests, without forcing either to mimic the other.
    """
    if hasattr(event, name):
        return getattr(event, name)
    if isinstance(event, dict):
        return event.get(name, default)
    return default


def _data_get(event: Any, name: str, default: Any = None) -> Any:
    """Read a field from ``event.data`` regardless of whether ``data``
    is a JSON column dict or already-deserialized dict on a stub.
    """
    data = _safe_get(event, "data")
    if isinstance(data, dict):
        return data.get(name, default)
    return default


def _ts(event: Any) -> datetime:
    """Coerce the event's timestamp into a tz-aware datetime."""
    ts = _safe_get(event, "timestamp")
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return datetime.now(timezone.utc)
