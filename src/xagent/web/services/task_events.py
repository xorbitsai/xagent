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
    """Attach the host's live event delivery adapter; None disables delivery.

    Installing a sink also clears any audience probe: the probe answers for
    one specific delivery target, so a new sink invalidates it. The host
    registers its own probe afterwards if it has one; until it does, the
    default answer is that an audience exists and no work is skipped.
    """
    global _task_event_sink, _warned_missing_sink, _task_audience_probe
    _task_event_sink = sink
    _task_audience_probe = None
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


TaskAudienceProbe = Callable[[int], bool]
_task_audience_probe: TaskAudienceProbe | None = None
_warned_failing_probe = False


def set_task_audience_probe(probe: TaskAudienceProbe | None) -> None:
    """Attach the host's audience predicate; None restores the default answer.

    The probe and the sink installed through ``set_task_event_sink`` must
    describe the same audience. A probe that reads one registry while the
    sink delivers to a different one answers "no audience" for every task
    the sink could still have reached, and the caller then skips the work
    silently: no counter, no warning, no frame. A host that replaces the
    sink must therefore replace this probe as well, or clear it with
    ``None`` to fall back to always doing the work.
    """
    global _task_audience_probe, _warned_failing_probe
    _task_audience_probe = probe
    if probe is not None:
        _warned_failing_probe = False


def task_has_audience(task_id: int) -> bool:
    """Answer whether anything is listening, so callers can skip wasted work.

    This is an optimization hint, not a delivery guarantee, and the two
    answers are not symmetric. A "yes" only says the work is worth
    doing: the sink above and whatever that sink hands the frame to
    still decide delivery, so a spurious "yes" costs work and nothing
    else. A "no" ends the caller's work, so a probe that wrongly answers
    "no" does suppress live frames.

    That asymmetry is why both failure modes here answer "yes": a host
    that registers no probe, and a probe that raises. A host that never
    opted into this optimization keeps exactly the behaviour it had, and
    a broken probe costs work instead of content.
    """
    global _warned_failing_probe
    if _task_audience_probe is None:
        return True
    try:
        return bool(_task_audience_probe(task_id))
    except Exception as exc:  # noqa: BLE001
        increment_counter(
            "xagent.task_events.audience_probe", attributes={"outcome": "failed"}
        )
        if not _warned_failing_probe:
            _warned_failing_probe = True
            logger.warning(
                "The registered task audience probe failed; assuming an "
                "audience is present so no work is skipped. Further "
                "warnings are suppressed until a probe is registered "
                "again. Cause: %s",
                exc,
            )
        return True


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
