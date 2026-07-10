"""Quota enforcement hook seams.

Core stays quota-agnostic. An application layer (e.g. xagent-cloud) registers
callbacks via the setters below; core calls the getters at run start, run
completion and KB ingest. When no hook is registered every gate is open and
recording is a no-op, so stock xagent is unaffected.

Follows the same setter/getter idiom as set_user_tool_overrides_hook etc.
"""

from __future__ import annotations

from typing import Any, Callable

# (db, user_id) -> reason str if the team is out of quota (block the run), else None
_run_gate_hook: Callable[[Any, Any], str | None] | None = None
# (db, user_id, delta_tokens, delta_actions) -> None; best-effort post-run metering.
# delta_actions counts tool calls (one billable action per tool invocation).
_usage_record_hook: Callable[[Any, Any, int, int], None] | None = None
# (db, user_id) -> reason str if the team is out of storage quota, else None
_storage_gate_hook: Callable[[Any, Any], str | None] | None = None
# (user_id) -> None; +1 billable action when a run is fired by a trigger
# (webhook / scheduled / API). Opens its own session on the application side.
_trigger_record_hook: Callable[[Any], None] | None = None


def set_run_gate_hook(hook: Callable[[Any, Any], str | None] | None) -> None:
    global _run_gate_hook
    _run_gate_hook = hook


def set_usage_record_hook(hook: Callable[[Any, Any, int, int], None] | None) -> None:
    global _usage_record_hook
    _usage_record_hook = hook


def set_storage_gate_hook(hook: Callable[[Any, Any], str | None] | None) -> None:
    global _storage_gate_hook
    _storage_gate_hook = hook


def set_trigger_record_hook(hook: Callable[[Any], None] | None) -> None:
    global _trigger_record_hook
    _trigger_record_hook = hook


def check_run_gate(db: Any, user_id: Any) -> str | None:
    if _run_gate_hook is None or user_id is None:
        return None
    return _run_gate_hook(db, user_id)


def record_usage(db: Any, user_id: Any, delta_tokens: int, delta_actions: int) -> None:
    if _usage_record_hook is None or user_id is None:
        return
    _usage_record_hook(db, user_id, delta_tokens, delta_actions)


def check_storage_gate(db: Any, user_id: Any) -> str | None:
    if _storage_gate_hook is None or user_id is None:
        return None
    return _storage_gate_hook(db, user_id)


def record_trigger(user_id: Any) -> None:
    if _trigger_record_hook is None or user_id is None:
        return
    _trigger_record_hook(user_id)
