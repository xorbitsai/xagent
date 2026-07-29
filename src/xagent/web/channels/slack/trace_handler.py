from __future__ import annotations

import logging
import time

from slack_sdk.web.async_client import AsyncWebClient

from ....core.agent.trace import TraceAction, TraceCategory, TraceEvent, TraceHandler
from .utils import markdown_to_slack, strip_slack_file_refs

logger = logging.getLogger(__name__)


class SlackTraceHandler(TraceHandler):
    """Project coarse execution progress into one editable Slack message."""

    MIN_STATUS_UPDATE_INTERVAL_SECONDS = 3.0

    def __init__(
        self,
        task_id: int,
        client: AsyncWebClient,
        channel_id: str,
        message_ts: str,
    ) -> None:
        self.task_id = task_id
        self.client = client
        self.channel_id = channel_id
        self.message_ts = message_ts
        self.current_text = ""
        self._last_status_update_at = 0.0
        self._last_status_text = ""
        self._activity_items: list[str] = []

    async def handle_event(self, event: TraceEvent) -> None:
        try:
            if event.task_id is not None and str(event.task_id) != str(self.task_id):
                return

            if event.event_type.category == TraceCategory.MESSAGE:
                data = event.data if isinstance(event.data, dict) else {}
                content = str(data.get("content") or "")
                if data.get("role") == "assistant" and content:
                    await self._update_message(content)
                return

            if event.event_type.category != TraceCategory.TOOL:
                return

            data = event.data if isinstance(event.data, dict) else {}
            tool_name = str(data.get("tool_name") or "").strip()
            if not tool_name:
                return
            tool_label = tool_name.replace("_", " ")
            action = event.event_type.action
            if action == TraceAction.START:
                status = f"I'm checking with {tool_label}."
                activity = f"Started {tool_label}"
            elif action == TraceAction.END:
                status = f"I've finished {tool_label} and am preparing the answer."
                activity = f"Finished {tool_label}"
            elif action == TraceAction.ERROR:
                status = f"{tool_label} didn't work, so I'm trying another way."
                activity = f"{tool_label} did not work"
            else:
                return

            if not self._activity_items or self._activity_items[-1] != activity:
                self._activity_items.append(activity)
                self._activity_items = self._activity_items[-3:]
            await self._update_status(status)
        except Exception:
            logger.warning(
                "SlackTraceHandler failed for task %s",
                self.task_id,
                exc_info=True,
            )

    async def _update_status(self, status: str) -> None:
        if not status or status == self._last_status_text:
            return
        now = time.monotonic()
        if now - self._last_status_update_at < self.MIN_STATUS_UPDATE_INTERVAL_SECONDS:
            return
        self._last_status_text = status
        self._last_status_update_at = now
        activity = "\n".join(f"• {item}" for item in self._activity_items)
        text = f"I'm still working on this.\n\n{status}"
        if activity:
            text += f"\n\nRecent activity:\n{activity}"
        await self._update_message(text)

    async def _update_message(self, text: str) -> None:
        visible_text, _ = strip_slack_file_refs(text)
        display_text = markdown_to_slack(visible_text)[:3900]
        if not display_text or display_text == self.current_text:
            return
        self.current_text = display_text
        await self.client.chat_update(
            channel=self.channel_id,
            ts=self.message_ts,
            text=display_text,
            link_names=False,
        )
