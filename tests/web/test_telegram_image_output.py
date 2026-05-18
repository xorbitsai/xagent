import pytest

from xagent.core.agent.trace import (
    ACTION_END_TOOL,
    ACTION_START_TOOL,
    TraceAction,
    TraceCategory,
    TraceEvent,
    TraceEventType,
    TraceScope,
)
from xagent.web.channels.telegram import handler as telegram_handler
from xagent.web.channels.telegram.handler import TelegramTraceHandler
from xagent.web.channels.telegram.utils import strip_telegram_image_refs


def test_strip_telegram_image_refs_extracts_file_refs() -> None:
    text = (
        "Here is the result:\n\n![generated_image.jpg](file:abc-123)\nKeep this text."
    )

    cleaned, refs = strip_telegram_image_refs(text)

    assert cleaned == "Here is the result:\n\nKeep this text."
    assert [(ref.file_id, ref.alt_text) for ref in refs] == [
        ("abc-123", "generated_image.jpg")
    ]


def test_strip_telegram_image_refs_supports_file_urls_and_api_urls() -> None:
    cleaned, refs = strip_telegram_image_refs(
        "![a](file://abc%201) "
        "![b](/api/files/preview/def%202) "
        "![c](https://example.com/api/files/download/ghi%203?token=ignored)"
    )

    assert cleaned == ""
    assert [ref.file_id for ref in refs] == ["abc 1", "def 2", "ghi 3"]


def test_telegram_trace_handler_uses_plural_image_placeholder() -> None:
    handler = TelegramTraceHandler(
        task_id=421,
        bot=object(),
        chat_id=123,
        message_id=456,  # type: ignore[arg-type]
    )

    assert handler._image_placeholder_text(1) == "Image generated."
    assert handler._image_placeholder_text(2) == "Images generated."


@pytest.mark.asyncio
async def test_telegram_trace_handler_throttles_tool_status_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent_texts: list[str] = []

    class CapturingTelegramTraceHandler(TelegramTraceHandler):
        async def _update_message(self, text: str, final: bool = False) -> None:
            sent_texts.append(text)

    now = 10.0
    monkeypatch.setattr(telegram_handler.time, "monotonic", lambda: now)

    handler = CapturingTelegramTraceHandler(
        task_id=421,
        bot=object(),
        chat_id=123,
        message_id=456,  # type: ignore[arg-type]
    )

    await handler.handle_event(
        TraceEvent(
            ACTION_START_TOOL,
            task_id="421",
            step_id="step-1",
            data={"tool_name": "web_search"},
        )
    )
    await handler.handle_event(
        TraceEvent(
            ACTION_END_TOOL,
            task_id="421",
            step_id="step-1",
            data={"tool_name": "web_search"},
        )
    )

    assert len(sent_texts) == 1
    assert "I'm still working on this and making progress." in sent_texts[0]
    assert "I'm checking with web search" in sent_texts[0]


@pytest.mark.asyncio
async def test_telegram_trace_handler_updates_after_throttle_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent_texts: list[str] = []
    current_time = {"value": 10.0}

    class CapturingTelegramTraceHandler(TelegramTraceHandler):
        async def _update_message(self, text: str, final: bool = False) -> None:
            sent_texts.append(text)

    monkeypatch.setattr(
        telegram_handler.time, "monotonic", lambda: current_time["value"]
    )

    handler = CapturingTelegramTraceHandler(
        task_id=421,
        bot=object(),
        chat_id=123,
        message_id=456,  # type: ignore[arg-type]
    )

    await handler.handle_event(
        TraceEvent(
            ACTION_START_TOOL,
            task_id="421",
            step_id="step-1",
            data={"tool_name": "web_search"},
        )
    )
    current_time["value"] += handler.MIN_STATUS_UPDATE_INTERVAL_SECONDS + 0.1
    await handler.handle_event(
        TraceEvent(
            TraceEventType(TraceScope.ACTION, TraceAction.ERROR, TraceCategory.TOOL),
            task_id="421",
            step_id="step-1",
            data={"tool_name": "web_search"},
        )
    )

    assert len(sent_texts) == 2
    assert "web search didn't work" in sent_texts[1]
    assert "Started web search" in sent_texts[1]
    assert "web search did not work" in sent_texts[1]
