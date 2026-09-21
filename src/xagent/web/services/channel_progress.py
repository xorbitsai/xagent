"""Progress and final replies share the durable channel delivery claim."""

from collections.abc import Awaitable, Callable
from typing import Any

from ...core.agent.trace import TraceEvent, TraceHandler
from .channel_delivery import ChannelDelivery, ChannelSender, deliver_channel_result


class DurableChannelProgress(TraceHandler):
    def __init__(
        self,
        command_id: int,
        progress_sender: Callable[
            [ChannelDelivery, TraceEvent | None], Awaitable[None]
        ],
        final_sender: ChannelSender,
    ) -> None:
        self.command_id = command_id
        self.progress_sender = progress_sender
        self.final_sender = final_sender

    async def handle_event(self, event: TraceEvent) -> None:
        await self.send(event)

    async def send(self, event: TraceEvent | None = None) -> None:
        async def sender(delivery: ChannelDelivery, result: dict[str, Any]) -> None:
            if result.get("status") == "accepted":
                await self.progress_sender(delivery, event)
            else:
                await self.final_sender(delivery, result)

        await deliver_channel_result(
            self.command_id, sender, pending_notice=True, progress=True
        )
