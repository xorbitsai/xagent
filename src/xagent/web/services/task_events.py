"""Task event publication, with delivery supplied by the hosting process."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol

from ...core.runtime_performance import increment_counter

logger = logging.getLogger(__name__)


class DeliveryNotifier(Protocol):
    """Transport callback for acknowledging a deferred user message."""

    async def __call__(
        self,
        *,
        turn_id: str | None,
        accepted: bool,
        message: str | None = None,
        error_code: str | None = None,
        retry_with_new_id: bool = False,
        rejection_outcome: Literal["not_accepted", "outcome_unknown"] | None = None,
    ) -> None: ...


TaskEventSink = Callable[[dict[str, Any], int], Awaitable[None]]
_task_event_sink: TaskEventSink | None = None
_warned_missing_sink = False


def set_task_event_sink(sink: TaskEventSink | None) -> None:
    """Attach the host's live event delivery adapter; None disables delivery."""
    global _task_event_sink, _warned_missing_sink
    _task_event_sink = sink
    if sink is not None:
        _warned_missing_sink = False


async def publish_task_event(message: dict[str, Any], task_id: int) -> None:
    """Publish a live event without depending on connections or API handlers."""
    global _warned_missing_sink
    if _task_event_sink is None:
        increment_counter(
            "xagent.task_events.dropped", attributes={"outcome": "no_sink"}
        )
        if not _warned_missing_sink:
            _warned_missing_sink = True
            logger.warning(
                "Task events are not being delivered because no host event sink "
                "is registered; further warnings are suppressed until a sink "
                "is registered."
            )
        return
    await _task_event_sink(message, task_id)


CommandReply = Callable[[dict[str, Any]], Awaitable[None]]


class TaskCommandDelivery(Protocol):
    """Host-owned personal replies and their command-scoped lifetime.

    Replies must raise ConnectionError when delivery fails because the
    recipient disconnected. Hosts translate transport-specific exceptions.
    Commands without a local recipient intentionally use discard_command_reply,
    which neither delivers nor raises; command_reply also uses it without a host.
    """

    def reply_for(self, command_id: str, task_id: int) -> CommandReply: ...

    def discard_command(self, command_id: str, task_id: int) -> None: ...


_task_command_delivery: TaskCommandDelivery | None = None


def set_task_command_delivery(delivery: TaskCommandDelivery | None) -> None:
    """Attach a host without making command execution depend on connections."""
    global _task_command_delivery
    _task_command_delivery = delivery


async def discard_command_reply(_message: dict[str, Any]) -> None:
    """A recovered or remote command has no local personal reply recipient."""


def command_reply(command_id: str, task_id: int) -> CommandReply:
    if _task_command_delivery is None:
        return discard_command_reply
    return _task_command_delivery.reply_for(command_id, task_id)


def finish_task_command_delivery(command_id: str, task_id: int) -> None:
    if _task_command_delivery is not None:
        _task_command_delivery.discard_command(command_id, task_id)
