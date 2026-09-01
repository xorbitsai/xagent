"""Transport-neutral execution seam for external-scope MESSAGE commands.

The durable task-command transport (``task_command_transport``) owns
admission for commands that must survive worker restarts and defer under
contention. External task input -- an end user's answer to a task parked
``WAITING_FOR_USER``, submitted from outside the first-party WebSocket --
is executed by the embedding application, which owns the external identity
fences, the delivery ledger, and the answer/turn identities. This module is
the seam the dispatcher routes those commands through: the embedding
application registers one executor at startup, and the dispatcher's command
adapter hands it every claimed ``MESSAGE`` command whose payload names the
external scope.

The executor reports outcomes exactly the way first-party command handlers
do: return a JSON-safe result dict on success, raise ``TaskCommandDeferred``
to retry without consuming the failure budget (for example while the parked
run still holds the resume lease), or raise ``TaskCommandRejected`` for a
terminal domain refusal. Any other exception consumes the failure budget.
An executor may attach a ``TerminalTaskEventDraft`` to the exceptions it
raises (``bind_terminal_event_draft``) to control the durable terminal
outcome's client-safe presentation.
"""

from __future__ import annotations

import threading
from typing import Any, Awaitable, Callable

from .task_command_transport import (
    ClaimedTaskCommand,
    TaskCommandDeferred,
    TaskCommandRejected,
)

# Broadcast, never stored -- the same single-consumer rule as the external
# cancel sentences in external_task_cancel.py. Neither sentence instructs
# retry or abandonment, because either instruction would sometimes be false:
# resubmitting the identical answer reopens a spent budget, while a revoked
# principal or a stale request makes any retry futile. They differ on the
# one axis the transport can actually prove -- whether the answer reached
# the downstream operation. The categorical sentence is reserved for
# outcomes that prove non-application; everything else gets the sentence
# that asserts only uncertainty, because a worker may have injected the
# answer before crashing and the reclaiming attempt cannot know.
EXTERNAL_INPUT_NOT_APPLIED_MESSAGE = "This answer was not applied to the task."
EXTERNAL_INPUT_UNCONFIRMED_MESSAGE = (
    "We could not confirm whether this answer was applied to the task."
)


def external_input_terminal_message(error: BaseException) -> str:
    """Wording for a terminal external MESSAGE outcome, by what is provable.

    A ``TaskCommandRejected`` is a durable domain refusal: the executor
    established that the answer was not applied. A ``TaskCommandDeferred``
    with ``resend_safe=True`` carries the same proof by its own contract --
    the flag is set only when the failed handoff shows the command never
    reached the downstream operation. Every other terminal -- an exhausted
    ``resend_safe=False`` deferral, a generic failure -- ends with the
    outcome unknown, and asserting non-application there could send the
    audience a false statement about a turn their answer already reached.
    """

    if isinstance(error, TaskCommandRejected):
        return EXTERNAL_INPUT_NOT_APPLIED_MESSAGE
    if isinstance(error, TaskCommandDeferred) and error.resend_safe:
        return EXTERNAL_INPUT_NOT_APPLIED_MESSAGE
    return EXTERNAL_INPUT_UNCONFIRMED_MESSAGE


ExternalTaskInputExecutor = Callable[
    [ClaimedTaskCommand], Awaitable[dict[str, Any] | None]
]

_executor_lock = threading.Lock()
_executor: ExternalTaskInputExecutor | None = None


def register_external_task_input_executor(
    executor: ExternalTaskInputExecutor,
) -> None:
    """Register the process-wide external input executor.

    Re-registering the same object is a no-op so embedding applications can
    run their startup wiring idempotently; a different object is refused
    rather than silently replaced, because two owners would race each
    other's claimed commands.
    """

    if not callable(executor):
        raise TypeError("external task input executor must be callable")
    global _executor
    with _executor_lock:
        if _executor is executor:
            return
        if _executor is not None:
            raise ValueError("An external task input executor is already registered")
        _executor = executor


def unregister_external_task_input_executor() -> ExternalTaskInputExecutor | None:
    """Remove and return the registered executor, if any."""

    global _executor
    with _executor_lock:
        executor, _executor = _executor, None
        return executor


def registered_external_task_input_executor() -> ExternalTaskInputExecutor | None:
    """The registered executor, or ``None`` when no embedding app owns one."""

    with _executor_lock:
        return _executor


async def execute_external_task_input_command(
    command: ClaimedTaskCommand,
) -> dict[str, Any] | None:
    """Execute one claimed external-scope input command via the seam.

    A deployment with no registered executor has no external execution core:
    the command is rejected terminally rather than deferred, because no
    amount of retrying will grow one, and an unbounded PENDING command would
    also block every later command for the same task behind the per-task
    FIFO.
    """

    executor = registered_external_task_input_executor()
    if executor is None:
        raise TaskCommandRejected(
            f"Message command {command.command_id} names the external task "
            "scope, but no external input execution core is registered",
            reason="unsupported_scope",
        )
    return await executor(command)
