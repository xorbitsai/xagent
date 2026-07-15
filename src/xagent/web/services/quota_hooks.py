"""Quota enforcement hook seams.

Core stays quota-agnostic. An application layer (e.g. xagent-cloud) registers
callbacks via the setters below; core calls the getters at run start, run
completion and KB ingest. When no hook is registered every gate is open and
recording is a no-op, so stock xagent is unaffected.

Follows the same setter/getter idiom as set_user_tool_overrides_hook etc.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

# (db, user_id) -> reason if the team is out of quota (block the run), else None.
# The reason is either a plain message str or a structured mapping the app layer
# builds (e.g. {"code","metric","limit","plan","message"}). Core does not
# interpret the structure — it forwards it to the run result so the client can
# localise / branch on it; ``message`` is the human-readable fallback.
_run_gate_hook: Callable[[Any, Any], str | Mapping[str, Any] | None] | None = None
# (db, user_id, delta_details, delta_actions) -> None; best-effort post-run
# metering. delta_details is this turn's per-model token breakdown (list of
# {"type","tokens","model"}) for cost-based credits; delta_actions counts tool
# calls (one billable action per tool invocation).
#
# TRANSACTION CONTRACT: the hook is invoked from TaskTracker.complete_tracking
# BEFORE that method commits the token-usage row on the SAME `db` session, and
# a later commit failure there rolls the session back. The hook therefore MUST
# manage its own durability — either persist through an independent
# session/transaction, or accumulate in a store that doesn't depend on this
# session's commit. It must NOT leave writes pending on `db` expecting the
# caller to commit them (they may be rolled back), and must NOT commit `db`
# itself (that would prematurely persist the caller's unrelated pending state).
_usage_record_hook: Callable[[Any, Any, list, int], None] | None = None
# (db, user_id, delta_details, delta_actions) -> reason str if counting this
# in-flight run's live-so-far usage would push the team over a run-gated quota,
# else None. Polled per step (each LLM reply / tool call) during a run so a
# single long/expensive run is stopped mid-flight instead of only being metered
# at completion.
#
# CONTRACT: invoked SYNCHRONOUSLY on the event loop once per step. It MUST NOT
# block (no synchronous network/DB round-trips per call) — blocking work stalls
# the loop every step. Resolve/cache anything expensive out of band (the stock
# app layer caches the user->team lookup and checks quota off in-memory
# counters). Read-only: it must not write or commit `db` (same contract spirit
# as the metering hook).
_run_progress_gate_hook: Callable[[Any, Any, list, int], str | None] | None = None
# (db, user_id) -> reason str if the team is out of storage quota, else None
_storage_gate_hook: Callable[[Any, Any], str | None] | None = None
# (user_id) -> None; +1 billable action when a run is fired by a trigger
# (webhook / scheduled / API). Opens its own session on the application side.
_trigger_record_hook: Callable[[Any], None] | None = None


def set_run_gate_hook(hook: Callable[[Any, Any], str | None] | None) -> None:
    global _run_gate_hook
    _run_gate_hook = hook


def set_usage_record_hook(hook: Callable[[Any, Any, list, int], None] | None) -> None:
    global _usage_record_hook
    _usage_record_hook = hook


def set_run_progress_gate_hook(
    hook: Callable[[Any, Any, list, int], str | None] | None,
) -> None:
    global _run_progress_gate_hook
    _run_progress_gate_hook = hook


def set_storage_gate_hook(hook: Callable[[Any, Any], str | None] | None) -> None:
    global _storage_gate_hook
    _storage_gate_hook = hook


def set_trigger_record_hook(hook: Callable[[Any], None] | None) -> None:
    global _trigger_record_hook
    _trigger_record_hook = hook


def check_run_gate(db: Any, user_id: Any) -> str | Mapping[str, Any] | None:
    if _run_gate_hook is None or user_id is None:
        return None
    return _run_gate_hook(db, user_id)


def record_usage(
    db: Any, user_id: Any, delta_details: list, delta_actions: int
) -> None:
    if _usage_record_hook is None or user_id is None:
        return
    _usage_record_hook(db, user_id, delta_details, delta_actions)


def check_run_progress_gate(
    db: Any, user_id: Any, delta_details: list, delta_actions: int
) -> str | None:
    if _run_progress_gate_hook is None or user_id is None:
        return None
    return _run_progress_gate_hook(db, user_id, delta_details, delta_actions)


def check_storage_gate(db: Any, user_id: Any) -> str | None:
    if _storage_gate_hook is None or user_id is None:
        return None
    return _storage_gate_hook(db, user_id)


def record_trigger(user_id: Any) -> None:
    if _trigger_record_hook is None or user_id is None:
        return
    _trigger_record_hook(user_id)
