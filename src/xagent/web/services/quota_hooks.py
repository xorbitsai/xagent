"""Quota enforcement hook seams.

Core stays quota-agnostic. An application layer (e.g. xagent-cloud) registers
callbacks via the setters below; core calls the getters at run start, run
completion and KB ingest. When no hook is registered every gate is open and
recording is a no-op, so stock xagent is unaffected.

Follows the same setter/getter idiom as set_user_tool_overrides_hook etc.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Protocol


class UsageRecordHook(Protocol):
    """Post-run metering callback registered with :func:`set_usage_record_hook`.

    Called as ``hook(db, user_id, delta_details, delta_actions, task_id=...)``.
    ``task_id`` is the id of the task whose run is being metered, always
    passed as a keyword (``None`` only when the caller has no task), so the
    application layer can attribute the usage to that task (e.g. per-task
    spend in its own ledger). Implementations MUST accept the ``task_id``
    keyword; one that ignores it should still declare it, e.g.
    ``task_id: int | None = None``.
    """

    def __call__(
        self,
        db: Any,
        user_id: Any,
        delta_details: list,
        delta_actions: int,
        /,
        *,
        task_id: int | None = None,
    ) -> None: ...


# (db, user_id) -> reason if the team is out of quota (block the run), else None.
# The reason is either a plain message str or a structured mapping the app layer
# builds (e.g. {"code","metric","limit","plan","message"}). Core does not
# interpret the structure — it forwards it to the run result so the client can
# localise / branch on it; ``message`` is the human-readable fallback.
#
# EXECUTION CONTRACT: invoked synchronously on the asyncio event loop. Existing
# callbacks may rely on that affinity and must remain non-blocking.
_run_gate_hook: Callable[[Any, Any], str | Mapping[str, Any] | None] | None = None
# (db, user_id, delta_details, delta_actions, *, task_id) -> None; best-effort
# post-run metering (see UsageRecordHook). delta_details is this turn's
# per-model token breakdown (list of {"type","tokens","model"}) for cost-based
# credits; delta_actions counts tool calls (one billable action per tool
# invocation); task_id is the metered run's task, passed as a keyword.
#
# TRANSACTION CONTRACT: the hook is invoked from TaskTracker.complete_tracking
# only after the run/runner-fenced token-usage update commits. The hook owns
# separate durability — either persist through an independent
# session/transaction, or accumulate in a store that doesn't depend on the
# compatibility `db` session. It must NOT leave writes pending on `db`
# expecting the caller to commit them, and must NOT commit `db` itself.
#
# EXECUTION CONTRACT: invoked synchronously on the asyncio event loop. Existing
# callbacks may rely on that affinity and must remain non-blocking.
_usage_record_hook: UsageRecordHook | None = None
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
# (user_id, details, source) -> None; usage recorded outside an agent run.
# See ``set_standalone_usage_hook``.
_standalone_usage_hook: (
    Callable[[int | None, list[dict[str, Any]], str], None] | None
) = None
# (user_id, source) -> reason if non-run work may not spend for this user now,
# else None. See ``set_standalone_gate_hook``.
_standalone_gate_hook: Callable[[int, str], str | Mapping[str, Any] | None] | None = (
    None
)


def set_run_gate_hook(hook: Callable[[Any, Any], str | None] | None) -> None:
    global _run_gate_hook
    _run_gate_hook = hook


def set_usage_record_hook(hook: UsageRecordHook | None) -> None:
    """Register the post-run metering callback; it must accept ``task_id=``."""
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


def set_standalone_usage_hook(
    hook: Callable[[int | None, list[dict[str, Any]], str], None] | None,
) -> None:
    """Register the receiver of usage recorded outside an agent run.

    Called as ``hook(user_id, details, source)`` once per
    :func:`xagent.web.tracking.standalone_usage.usage_scope` that recorded
    anything: a knowledge-base search or ingestion, a memory API call, a model
    connection test, a speech transcription.

    - ``user_id``: the user the work was done for -- the requesting member, not
      a knowledge base's storage tenant. ``None`` when the work cannot be
      attributed; the usage is still delivered (and logged), never dropped and
      never assigned to a guessed owner.
    - ``details``: the ``TokenUsage.details`` rows recorded in the scope, as
      copies the hook may keep. Mostly ``type: "media"`` rows (embedding,
      rerank, ASR ...), but any LLM token rows recorded in the scope are
      included too, so nothing billable is lost.
    - ``source``: a short stable label for the entry point, e.g. ``kb_search``.

    This is not ``record_usage``: nothing here is an agent execution, and the
    ``delta_actions`` tool-call counter does not apply.

    EXECUTION CONTRACT: invoked synchronously, never on the asyncio event loop
    -- async entry points deliver from a worker thread. No database session is
    passed: the hook opens and commits its own. Exceptions are logged and
    swallowed; they never fail the work being metered.
    """
    global _standalone_usage_hook
    _standalone_usage_hook = hook


def get_standalone_usage_hook() -> (
    Callable[[int | None, list[dict[str, Any]], str], None] | None
):
    return _standalone_usage_hook


def set_standalone_gate_hook(
    hook: Callable[[int, str], str | Mapping[str, Any] | None] | None,
) -> None:
    """Register the gate checked before usage is incurred outside an agent run.

    Called as ``hook(user_id, source)`` when a
    :func:`xagent.web.tracking.standalone_usage.usage_scope` opens, before the
    work inside it reaches a provider, and by the entry points that would
    otherwise refuse only after writing state (KB ingest, TTS streaming).
    ``user_id`` and ``source`` mean what they mean for
    :func:`set_standalone_usage_hook`.

    Return ``None`` to allow the work, or a reason to refuse it: a message
    ``str`` or a structured mapping (e.g. ``{"code", "metric", "message"}``)
    that core forwards to the client without interpreting it. A refusal is
    surfaced as HTTP 402, a websocket close with code 1008, a failed
    background job, or a reply in the channel, depending on the entry point.

    Not consulted for work with no attributable user, nor for model
    connection tests, which run on credentials supplied in the request.

    EXECUTION CONTRACT: invoked synchronously, never on the asyncio event loop
    -- async entry points check from a worker thread. No database session is
    passed: the hook opens its own and must not write. Exceptions are logged
    and the work is allowed: a broken gate must not take every standalone
    feature down with it.
    """
    global _standalone_gate_hook
    _standalone_gate_hook = hook


def get_standalone_gate_hook() -> (
    Callable[[int, str], str | Mapping[str, Any] | None] | None
):
    return _standalone_gate_hook


def check_run_gate(db: Any, user_id: Any) -> str | Mapping[str, Any] | None:
    if _run_gate_hook is None or user_id is None:
        return None
    return _run_gate_hook(db, user_id)


def record_usage(
    db: Any,
    user_id: Any,
    delta_details: list,
    delta_actions: int,
    task_id: int | None = None,
) -> None:
    """Forward one finished run's usage, and the task it belongs to, to the hook."""
    if _usage_record_hook is None or user_id is None:
        return
    _usage_record_hook(db, user_id, delta_details, delta_actions, task_id=task_id)


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
