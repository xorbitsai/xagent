"""Per-run settlement notification for the hosting application.

The execution entry points build a ``TurnFacts`` for each run, the run fills
it in, and once the run's outcome is settled -- its own result committed, or
its lease settled by the entry point -- the entry point hands a frozen
``TurnSettlement`` to the hook registered with ``set_turn_settled_hook``. A
run whose result committed is reported even when its lease is then left to
lease recovery; a run deferred to lease recovery before its result was
written is not reported. Delivery is detached: the entry point never waits
for the hook, so a slow hook cannot delay the next turn of the same task.

A synchronous hook is called on the event loop and must not block; a hook
that does I/O should return an awaitable or move the work to a thread.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ...core.agent.result import normalize_tool_failure_code
from ...core.model.chat.exceptions import MODEL_PROVIDER_FAILURE_KINDS
from ...core.runtime_performance import increment_counter
from .client_error_messages import ClientErrorCode

logger = logging.getLogger(__name__)

HOOK_TIMEOUT_SECONDS = 10.0
_HOOK_METRIC = "xagent.turn_settlement.hook"
# The only error codes a settlement carries. Anything else a run reports is
# folded into ``task_execution_failed`` so free text never reaches the hook.
_ERROR_CODES = frozenset({code.value for code in ClientErrorCode} | {"quota_exceeded"})


@dataclass(frozen=True)
class TurnModelUsage:
    """One model's share of a run's tokens."""

    model_id: str
    model_name: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class TurnUsage:
    """What one run consumed, over the baseline its tracker was seeded with."""

    input_tokens: int
    output_tokens: int
    llm_calls: int
    tool_calls: int
    tool_errors: int
    by_model: tuple[TurnModelUsage, ...]


_UNSETTLED_ANSWER_STATUSES = frozenset({"paused", "waiting_for_user"})


@dataclass(frozen=True)
class SettledOutcome:
    """What one run's settlement left on the task row, as the hook reports it."""

    status: str
    error_code: str | None
    empty_visible_answer: bool


def settled_outcome(
    status: str,
    *,
    failure_code: Any = None,
    visible_text: str | None = None,
    output_missing: bool = False,
    ignored_result: bool = False,
) -> SettledOutcome:
    """Decide the reported outcome from what a settlement committed.

    ``status`` is the task status the settlement committed, or the terminal
    status a finalizer found and kept. ``failure_code`` is the code the
    committing write recorded for a failure. ``visible_text`` is the answer
    text this run wrote; ``None`` when it wrote none. ``output_missing`` is
    True when the run produced no output and a placeholder stands in for it
    in ``visible_text``. ``ignored_result`` is True when a finalizer found the
    task already terminal and discarded this run's result, so neither its
    code nor its answer describes the row.

    A paused run carries no error code: it is not an outcome yet, and its
    resumed run reports again. A failure always carries a code. Never raises.
    """
    if status == "paused":
        error_code: str | None = None
    elif ignored_result:
        error_code = (
            ClientErrorCode.TASK_EXECUTION_FAILED.value if status == "failed" else None
        )
    else:
        error_code = build_error_code(failure_code)
        if error_code is None and status == "failed":
            error_code = ClientErrorCode.TASK_EXECUTION_FAILED.value
    empty_visible_answer = (
        not ignored_result
        and visible_text is not None
        and status not in _UNSETTLED_ANSWER_STATUSES
        and (output_missing or not visible_text.strip())
    )
    return SettledOutcome(status, error_code, empty_visible_answer)


@dataclass
class TurnFacts:
    """Facts one run accumulates before its settlement is reported.

    Created by the execution entry point when the run starts; the run's
    usage, its committed outcome and its result metadata are recorded as
    they become known. Every field but ``started_at`` stays at its default
    when the run never got that far.
    """

    started_at: datetime
    usage: TurnUsage | None = None
    first_answer_delta_at: datetime | None = None
    # The outcome this run's own result finalizer settled: the status it
    # committed, or a terminal status it found and kept while discarding
    # this run's result. None when no finalizer ran to completion (the run
    # raised, or a cancellation landed before its finalizer was invoked).
    outcome: SettledOutcome | None = None
    completion_outcome: str | None = None
    llm_finish_reason: str | None = None
    empty_final_answer: bool = False
    # The run produced no output; the adapter substituted a placeholder.
    output_missing: bool = False
    model_error_kind: str | None = None
    tool_failure_code: str | None = None
    # Set once this run's settlement was reported; a run is reported at most
    # once even when two entry points both see its outcome settle.
    notified: bool = False

    def record_usage(self, usage: TurnUsage) -> None:
        self.usage = usage

    def absorb_result(self, result: Mapping[str, Any]) -> None:
        """Copy the allowlisted classification fields of a normalized run result.

        ``result`` is ``AgentExecutionAdapter`` output; only its top-level
        keys are read.
        """
        finish_reason = result.get("finish_reason")
        self.llm_finish_reason = (
            finish_reason if isinstance(finish_reason, str) else None
        )
        self.empty_final_answer = result.get("empty_final_answer") is True
        self.output_missing = result.get("output_missing") is True
        self.tool_failure_code = normalize_tool_failure_code(result.get("failure_code"))
        self.first_answer_delta_at = _parse_timestamp(
            result.get("first_answer_delta_at")
        )
        model_error = result.get("model_error")
        if isinstance(model_error, dict):
            kind = model_error.get("kind")
            self.model_error_kind = (
                kind
                if isinstance(kind, str) and kind in MODEL_PROVIDER_FAILURE_KINDS
                else None
            )


def _parse_timestamp(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class TurnSettlement:
    """How one run of a task ended, reported once its outcome was settled."""

    task_id: int
    run_id: str | None
    turn_id: str | None
    task_source: str | None
    status: str
    completion_outcome: str | None
    error_code: str | None
    llm_finish_reason: str | None
    empty_final_answer: bool
    empty_visible_answer: bool
    model_error_kind: str | None
    tool_failure_code: str | None
    usage: TurnUsage | None
    started_at: datetime
    ended_at: datetime
    first_answer_delta_at: datetime | None


TurnSettledHook = Callable[[TurnSettlement], Awaitable[None] | None]

_turn_settled_hook: TurnSettledHook | None = None
# Deliveries still running. Holding the handles keeps them from being
# collected mid-flight and lets shutdown wait for them.
_inflight: set[asyncio.Task[None]] = set()


def set_turn_settled_hook(hook: TurnSettledHook | None) -> None:
    """Register the callback that receives every settled run; None removes it."""
    global _turn_settled_hook
    _turn_settled_hook = hook


def build_error_code(raw: Any) -> str | None:
    """Return ``raw`` when it is a known error code, else a generic failure code.

    ``None`` stays ``None``. Any other value, including free text, becomes
    ``task_execution_failed``.
    """
    if raw is None:
        return None
    if isinstance(raw, str) and str(raw) in _ERROR_CODES:
        return str(raw)
    return ClientErrorCode.TASK_EXECUTION_FAILED.value


def build_turn_settlement(
    facts: TurnFacts,
    *,
    task_id: int,
    run_id: str | None,
    turn_id: str | None,
    task_source: str | None,
    outcome: SettledOutcome,
    ended_at: datetime,
) -> TurnSettlement:
    """Freeze ``facts`` into the settlement the hook receives.

    ``completion_outcome`` is only reported for a completed run.
    """
    return TurnSettlement(
        task_id=task_id,
        run_id=run_id,
        turn_id=turn_id,
        task_source=task_source,
        status=outcome.status,
        completion_outcome=(
            facts.completion_outcome if outcome.status == "completed" else None
        ),
        error_code=outcome.error_code,
        llm_finish_reason=facts.llm_finish_reason,
        empty_final_answer=facts.empty_final_answer,
        empty_visible_answer=outcome.empty_visible_answer,
        model_error_kind=facts.model_error_kind,
        tool_failure_code=facts.tool_failure_code,
        usage=facts.usage,
        started_at=facts.started_at,
        ended_at=ended_at,
        first_answer_delta_at=facts.first_answer_delta_at,
    )


def report_turn_settled(
    facts: TurnFacts,
    *,
    task_id: int,
    run_id: str | None,
    turn_id: str | None,
    task_source: str | None,
    outcome: SettledOutcome,
    ended_at: datetime,
) -> bool:
    """Report this run's settlement unless it was already reported.

    Returns whether this call reported it. A missing ``turn_id`` is counted
    and the settlement is still reported.
    """
    if facts.notified:
        return False
    facts.notified = True
    if turn_id is None:
        increment_counter(_HOOK_METRIC, attributes={"outcome": "missing_turn_id"})
    notify_turn_settled(
        build_turn_settlement(
            facts,
            task_id=task_id,
            run_id=run_id,
            turn_id=turn_id,
            task_source=task_source,
            outcome=outcome,
            ended_at=ended_at,
        )
    )
    return True


def notify_turn_settled(settlement: TurnSettlement) -> None:
    """Hand ``settlement`` to the registered hook without waiting for it.

    Must be called on the event loop. The hook runs as its own task, bounded
    by ``HOOK_TIMEOUT_SECONDS``; its failures are logged and counted, never
    raised to the caller.
    """
    hook = _turn_settled_hook
    if hook is None:
        increment_counter(_HOOK_METRIC, attributes={"outcome": "no_hook"})
        return
    task = asyncio.get_running_loop().create_task(_run_hook(hook, settlement))
    _inflight.add(task)
    task.add_done_callback(_inflight.discard)


async def _call(hook: TurnSettledHook, settlement: TurnSettlement) -> None:
    outcome = hook(settlement)
    if inspect.isawaitable(outcome):
        await outcome


async def _run_hook(hook: TurnSettledHook, settlement: TurnSettlement) -> None:
    try:
        await asyncio.wait_for(_call(hook, settlement), timeout=HOOK_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        raise
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning(
            "turn settled hook timed out for task %s run %s",
            settlement.task_id,
            settlement.run_id,
        )
        increment_counter(_HOOK_METRIC, attributes={"outcome": "timeout"})
    except Exception:
        logger.warning(
            "turn settled hook failed for task %s run %s",
            settlement.task_id,
            settlement.run_id,
            exc_info=True,
        )
        increment_counter(_HOOK_METRIC, attributes={"outcome": "failed"})
    else:
        increment_counter(_HOOK_METRIC, attributes={"outcome": "ok"})


async def drain_turn_settlement_hooks(timeout: float = 10.0) -> None:
    """Wait up to ``timeout`` seconds for in-flight deliveries, then cancel
    whatever is still running and wait for it to finish."""
    if not _inflight:
        return
    pending_count = len(_inflight)
    _done, pending = await asyncio.wait(set(_inflight), timeout=timeout)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    logger.info(
        "drained %s turn settlement deliveries (%s cancelled)",
        pending_count,
        len(pending),
    )
