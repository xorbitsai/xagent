import logging
import time
from typing import Optional

from aiogram import Bot
from aiogram.enums import ParseMode

from ....core.agent.trace import TraceAction, TraceCategory, TraceEvent, TraceHandler
from .utils import markdown_to_tg_html, strip_telegram_image_refs

logger = logging.getLogger(__name__)


class TelegramTraceHandler(TraceHandler):
    MIN_STATUS_UPDATE_INTERVAL_SECONDS = 2.0

    def __init__(
        self, task_id: int, bot: Bot, chat_id: int, message_id: Optional[int] = None
    ):
        self.task_id = task_id
        self.bot = bot
        self.chat_id = chat_id
        self.message_id = message_id
        self.current_text = ""
        self._last_status_update_at = 0.0
        self._last_status_text = ""
        self._activity_items: list[str] = []

    async def handle_event(self, event: TraceEvent) -> None:
        try:
            # We only care about assistant messages and coarse tool activity for Telegram.
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

            elif event.event_type.category == TraceCategory.TOOL:
                await self._handle_tool_event(event)

        except Exception as e:
            logger.warning(f"TelegramTraceHandler error for task {self.task_id}: {e}")

    async def _handle_tool_event(self, event: TraceEvent) -> None:
        tool_name = self._extract_tool_name(event)
        if not tool_name:
            return

        action = event.event_type.action
        tool_label = self._format_tool_name(tool_name)
        if action == TraceAction.START:
            status = (
                f"I'm checking with {tool_label} and will pull the results together."
            )
            activity = f"Started {tool_label}"
        elif action == TraceAction.END:
            status = f"I've finished {tool_label} and am putting the answer together."
            activity = f"Finished {tool_label}"
        elif action == TraceAction.ERROR:
            status = f"{tool_label} didn't work, so I'm trying another way."
            activity = f"{tool_label} did not work"
        else:
            return

        self._append_activity(activity)
        await self._update_status(status)

    def _extract_tool_name(self, event: TraceEvent) -> str:
        data = event.data if isinstance(event.data, dict) else {}
        raw_name = data.get("tool_name")
        if raw_name is None:
            return ""
        return str(raw_name).strip()

    def _format_tool_name(self, tool_name: str) -> str:
        return tool_name.replace("_", " ").strip() or "a tool"

    def _append_activity(self, activity: str) -> None:
        if not activity:
            return
        if self._activity_items and self._activity_items[-1] == activity:
            return
        self._activity_items.append(activity)
        self._activity_items = self._activity_items[-3:]

    async def _update_status(self, status: str) -> None:
        if not status or status == self._last_status_text:
            return

        now = time.monotonic()
        if now - self._last_status_update_at < self.MIN_STATUS_UPDATE_INTERVAL_SECONDS:
            return

        self._last_status_text = status
        self._last_status_update_at = now

        activity = "\n".join(f"• {item}" for item in self._activity_items)
        text = f"I'm still working on this and making progress.\n\n{status}"
        if activity:
            text += f"\n\nRecent activity:\n{activity}"
        await self._update_message(text)

    async def _update_message(self, text: str, final: bool = False) -> None:
        if not text:
            return

        text, image_refs = strip_telegram_image_refs(text)
        if not text and image_refs:
            text = self._image_placeholder_text(len(image_refs))

        # Add a typing indicator for non-final messages
        display_text = text if final else text + " ✍️"

        # Avoid updating if text hasn't changed much (to prevent rate limits)
        if self.current_text == display_text:
            return

        self.current_text = display_text

        try:
            html_text = markdown_to_tg_html(display_text[:4000])
            if self.message_id is None:
                try:
                    msg = await self.bot.send_message(
                        chat_id=self.chat_id, text=html_text, parse_mode=ParseMode.HTML
                    )
                except Exception:
                    # Fallback if HTML parsing fails
                    msg = await self.bot.send_message(
                        chat_id=self.chat_id, text=display_text[:4000]
                    )
                self.message_id = msg.message_id
            else:
                try:
                    await self.bot.edit_message_text(
                        chat_id=self.chat_id,
                        message_id=self.message_id,
                        text=html_text,
                        parse_mode=ParseMode.HTML,
                    )
                except Exception as e:
                    if "message is not modified" not in str(e).lower():
                        # Fallback if HTML parsing fails
                        await self.bot.edit_message_text(
                            chat_id=self.chat_id,
                            message_id=self.message_id,
                            text=display_text[:4000],
                        )
        except Exception as e:
            if "message is not modified" not in str(e).lower():
                logger.error(f"Error updating Telegram message: {e}")

    def _image_placeholder_text(self, image_count: int) -> str:
        return "Images generated." if image_count > 1 else "Image generated."
