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
from xagent.web.channels.telegram.bot import TelegramBotInstance
from xagent.web.channels.telegram.handler import TelegramTraceHandler
from xagent.web.channels.telegram.utils import (
    TelegramFileRef,
    TelegramImageRef,
    markdown_to_tg_html,
    persist_telegram_assistant_turn,
    restore_telegram_task_context,
    strip_telegram_file_refs,
    strip_telegram_image_refs,
)


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


def test_strip_telegram_file_refs_extracts_local_file_links() -> None:
    text = (
        "Generated files:\n"
        "- [report.csv](file://file-1)\n"
        "- [poster](https://example.com/poster.html)\n"
        "![preview](file://image-1)"
    )

    cleaned, refs = strip_telegram_file_refs(text)

    assert cleaned == (
        "Generated files:\n"
        "- [poster](https://example.com/poster.html)\n"
        "![preview](file://image-1)"
    )
    assert [(ref.file_id, ref.label) for ref in refs] == [("file-1", "report.csv")]


def test_strip_telegram_file_refs_supports_api_download_urls() -> None:
    cleaned, refs = strip_telegram_file_refs(
        "[a](/api/files/download/abc%201) "
        "[b](https://example.com/api/files/preview/def%202?token=ignored)"
    )

    assert cleaned == ""
    assert [ref.file_id for ref in refs] == ["abc 1", "def 2"]


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
async def test_send_output_files_uploads_documents(tmp_path) -> None:
    output_path = tmp_path / "report.csv"
    output_path.write_text("a,b\n1,2\n")

    class FakeRecord:
        file_id = "file-1"
        storage_path = str(output_path)
        filename = "report.csv"

    class FakeQuery:
        def filter(self, *args):
            return self

        def all(self):
            return [FakeRecord()]

    class FakeDB:
        def query(self, *args):
            return FakeQuery()

    class FakeReply:
        def __init__(self) -> None:
            self.documents = []

        async def answer_document(self, document, caption=None):
            self.documents.append((document, caption))

    bot = object.__new__(TelegramBotInstance)
    reply = FakeReply()

    failed_refs = await bot._send_output_files(
        file_refs=[TelegramFileRef(file_id="file-1", label="report.csv")],
        user_id=7,
        task_id=423,
        db=FakeDB(),  # type: ignore[arg-type]
        reply_to=reply,  # type: ignore[arg-type]
    )

    assert failed_refs == []
    assert len(reply.documents) == 1
    assert reply.documents[0][1] == "report.csv"


def test_extract_telegram_output_refs_uses_final_answer_links_only() -> None:
    bot = object.__new__(TelegramBotInstance)

    assert bot._extract_telegram_output_refs(None) == ("", [], [])
    assert bot._extract_telegram_output_refs("") == ("", [], [])

    cleaned, image_refs, file_refs = bot._extract_telegram_output_refs(
        "Final files:\n"
        "- [report.csv](file:file-1)\n"
        "- [draft.csv](https://example.com/draft.csv)\n"
        "![plot.png](file:image-1)"
    )

    assert cleaned == "Final files:\n- [draft.csv](https://example.com/draft.csv)"
    assert [(ref.file_id, ref.alt_text) for ref in image_refs] == [
        ("image-1", "plot.png")
    ]
    assert [(ref.file_id, ref.label) for ref in file_refs] == [("file-1", "report.csv")]


def test_dedupe_telegram_output_refs_prefers_images_over_documents() -> None:
    bot = object.__new__(TelegramBotInstance)

    image_refs, file_refs = bot._dedupe_telegram_output_refs(
        [
            TelegramImageRef(file_id="image-1", alt_text="inline image"),
            TelegramImageRef(file_id="image-1", alt_text="structured image"),
        ],
        [
            TelegramFileRef(file_id="image-1", label="plot.png"),
            TelegramFileRef(file_id="file-1", label="report.csv"),
            TelegramFileRef(file_id="file-1", label="report duplicate.csv"),
        ],
    )

    assert [(ref.file_id, ref.alt_text) for ref in image_refs] == [
        ("image-1", "inline image")
    ]
    assert [(ref.file_id, ref.label) for ref in file_refs] == [("file-1", "report.csv")]


@pytest.mark.asyncio
async def test_telegram_trace_handler_ignores_events_for_other_tasks() -> None:
    sent_texts: list[str] = []

    class CapturingTelegramTraceHandler(TelegramTraceHandler):
        async def _update_message(self, text: str, final: bool = False) -> None:
            sent_texts.append(text)

    handler = CapturingTelegramTraceHandler(
        task_id=421,
        bot=object(),
        chat_id=123,
        message_id=456,  # type: ignore[arg-type]
    )

    await handler.handle_event(
        TraceEvent(
            ACTION_START_TOOL,
            task_id="422",
            step_id="step-1",
            data={"tool_name": "browser_navigate"},
        )
    )

    assert sent_texts == []


def test_markdown_to_tg_html_renders_tables_as_wrapped_lists() -> None:
    html = markdown_to_tg_html(
        "| 特性 | 说明 |\n"
        "|---|---|\n"
        "| 统一可观测引擎 | 指标、日志、链路、宽事件统一在一个数据库中。 |\n"
        "| 对象存储优先 | 成本最高降低 50 倍。 |\n"
    )

    assert "<pre>" not in html
    assert (
        "• <b>统一可观测引擎</b>: 指标、日志、链路、宽事件统一在一个数据库中。" in html
    )
    assert "• <b>对象存储优先</b>: 成本最高降低 50 倍。" in html


def test_markdown_to_tg_html_renders_multi_column_tables_as_field_lists() -> None:
    html = markdown_to_tg_html(
        "| Tool | Status | Notes |\n"
        "|---|---|---|\n"
        "| web_search | done | Found product page |\n"
    )

    assert "<pre>" not in html
    assert "• <b>web_search</b>" in html
    assert "  Status: done" in html
    assert "  Notes: Found product page" in html


def test_markdown_to_tg_html_accepts_single_hyphen_table_delimiters() -> None:
    html = markdown_to_tg_html(
        "| Feature | Detail |\n| - | - |\n| Storage | Object storage first |\n"
    )

    assert "<pre>" not in html
    assert "• <b>Storage</b>: Object storage first" in html


@pytest.mark.asyncio
async def test_restore_telegram_task_context_loads_transcript_and_recovery_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import chat_history_service, task_execution_context_service

    db = object()
    transcript = [{"role": "user", "content": "Generate a Spirited Away image"}]
    recovery_state = {
        "messages": [{"role": "tool", "content": "image generated"}],
        "skill_context": {"style": "ghibli"},
    }

    class FakeAgentService:
        def __init__(self) -> None:
            self.conversation_history = None
            self.execution_context_messages = None
            self.recovered_skill_context = None

        def set_conversation_history(self, messages):
            self.conversation_history = messages

        def set_execution_context_messages(self, messages):
            self.execution_context_messages = messages

        def set_recovered_skill_context(self, context):
            self.recovered_skill_context = context

    async def fake_load_recovery_state(received_db, received_task_id: int):
        assert received_db is db
        assert received_task_id == 423
        return recovery_state

    monkeypatch.setattr(
        chat_history_service,
        "load_task_transcript",
        lambda received_db, received_task_id: transcript,
    )
    monkeypatch.setattr(
        task_execution_context_service,
        "load_task_execution_recovery_state",
        fake_load_recovery_state,
    )

    agent_service = FakeAgentService()

    await restore_telegram_task_context(agent_service, db, 423)  # type: ignore[arg-type]

    assert agent_service.conversation_history == transcript
    assert agent_service.execution_context_messages == recovery_state["messages"]
    assert agent_service.recovered_skill_context == recovery_state["skill_context"]


def test_persist_telegram_assistant_turn_stores_failed_text_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import chat_history_service

    persisted = []

    def fake_persist_assistant_message(**kwargs):
        persisted.append(kwargs)

    monkeypatch.setattr(
        chat_history_service,
        "persist_assistant_message",
        fake_persist_assistant_message,
    )

    persist_telegram_assistant_turn(
        db=object(),  # type: ignore[arg-type]
        task_id=423,
        user_id=7,
        content="I could not generate that image.",
        interactions=[],
    )

    assert persisted == [
        {
            "db": persisted[0]["db"],
            "task_id": 423,
            "user_id": 7,
            "content": "I could not generate that image.",
            "interactions": [],
            "message_type": "assistant_message",
        }
    ]


def test_persist_telegram_assistant_turn_stores_interactions_as_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import chat_history_service

    persisted = []
    interactions = [{"type": "text_input", "label": "Destination"}]

    def fake_persist_assistant_message(**kwargs):
        persisted.append(kwargs)

    monkeypatch.setattr(
        chat_history_service,
        "persist_assistant_message",
        fake_persist_assistant_message,
    )

    persist_telegram_assistant_turn(
        db=object(),  # type: ignore[arg-type]
        task_id=423,
        user_id=7,
        content="",
        interactions=interactions,
        message_type="question",
    )

    assert persisted == [
        {
            "db": persisted[0]["db"],
            "task_id": 423,
            "user_id": 7,
            "content": "",
            "interactions": interactions,
            "message_type": "question",
        }
    ]


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
