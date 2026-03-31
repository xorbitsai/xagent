import asyncio
import json
import logging
from typing import Any, Optional

from lark_oapi.api.im.v1 import (
    PatchMessageRequest,
    PatchMessageRequestBody,
)

from ....core.agent.trace import TraceAction, TraceCategory, TraceEvent, TraceHandler

logger = logging.getLogger(__name__)


class FeishuTraceHandler(TraceHandler):
    def __init__(
        self,
        task_id: int,
        api_client: Any,
        chat_id: str,
        message_id: Optional[str] = None,
    ):
        self.task_id = task_id
        self.api_client = api_client
        self.chat_id = chat_id
        self.message_id = (
            message_id  # The ID of the loading message we can potentially update
        )
        self.current_text = ""

    async def handle_event(self, event: TraceEvent) -> None:
        try:
            # We only care about message events and tool events for Feishu
            if (
                event.event_type.category == TraceCategory.MESSAGE
                and event.event_type.action == TraceAction.UPDATE
            ):
                data = event.data or {}
                role = data.get("role")
                content = data.get("content", "")

                if role == "assistant" and content:
                    await self._update_message(content)

            elif (
                event.event_type.category == TraceCategory.MESSAGE
                and event.event_type.action == TraceAction.END
            ):
                data = event.data or {}
                role = data.get("role")
                content = data.get("content", "")

                if role == "assistant" and content:
                    await self._update_message(content, final=True)

        except Exception as e:
            logger.warning(f"FeishuTraceHandler error for task {self.task_id}: {e}")

    async def _update_message(self, text: str, final: bool = False) -> None:
        if not text:
            return

        display_text = text if final else text + " ✍️"

        if self.current_text == display_text:
            return

        self.current_text = display_text

        if self.message_id:
            try:
                card_content = {
                    "config": {"wide_screen_mode": True},
                    "elements": [{"tag": "markdown", "content": display_text[:4000]}],
                }
                req = (
                    PatchMessageRequest.builder()
                    .message_id(self.message_id)
                    .request_body(
                        PatchMessageRequestBody.builder()
                        .content(json.dumps(card_content))
                        .build()
                    )
                    .build()
                )
                resp = await asyncio.get_event_loop().run_in_executor(
                    None, self.api_client.im.v1.message.patch, req
                )
                if not resp.success():
                    logger.error(
                        f"Failed to patch Feishu message: {resp.code}, {resp.msg}"
                    )
                    # If the message cannot be updated (e.g. 230001 NOT a card error)
                    # We fallback to creating a new message for final result
                    if resp.code == 230001 and final:
                        from lark_oapi.api.im.v1 import (
                            CreateMessageRequest,
                            CreateMessageRequestBody,
                        )

                        req_create = (
                            CreateMessageRequest.builder()
                            .receive_id_type("chat_id")
                            .request_body(
                                CreateMessageRequestBody.builder()
                                .receive_id(self.chat_id)
                                .msg_type("interactive")
                                .content(json.dumps(card_content))
                                .build()
                            )
                            .build()
                        )
                        await asyncio.get_event_loop().run_in_executor(
                            None, self.api_client.im.v1.message.create, req_create
                        )

            except Exception as e:
                logger.error(f"Error patching Feishu message: {e}")
