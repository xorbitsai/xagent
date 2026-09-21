"""Both startup and CRUD synchronization obey the designated ingress setting."""

import importlib
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "platform,manager_name",
    [
        ("feishu", "FeishuChannelManager"),
        ("slack", "SlackChannelManager"),
        ("telegram", "TelegramChannelManager"),
    ],
)
async def test_other_web_replicas_do_not_load_or_connect_bots(
    monkeypatch, platform, manager_name
):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "web")
    monkeypatch.delenv("XAGENT_CHANNEL_INGRESS_ENABLED", raising=False)
    module = importlib.import_module(f"xagent.web.channels.{platform}.bot")
    load = AsyncMock(side_effect=AssertionError("Not the designated ingress"))
    monkeypatch.setattr(module, "load_active_channel_configs", load)
    manager = getattr(module, manager_name)()
    await manager.start()
    await manager._sync_bots_async()
    load.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["feishu", "telegram"])
async def test_channel_entry_submits_shared_turn_without_local_agent(
    monkeypatch, platform
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from xagent.web.services.channel_input_acceptance import AcceptedChannelInput
    from xagent.web.services.channel_runtime import SelectedChannelTask

    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "web")
    module = importlib.import_module(f"xagent.web.channels.{platform}.bot")
    accepted = AcceptedChannelInput(
        5,
        6,
        "command",
        "run",
        45,
        False,
        SelectedChannelTask(5, 45, True, 7, "sender", None, 0),
    )
    submit = Mock(return_value=accepted)
    monkeypatch.setattr(module, "accept_channel_input", submit)
    monkeypatch.setattr(
        module, "lookup_channel_inputs", lambda items: (45, items, (), ())
    )
    monkeypatch.setattr(module, "get_task_event_bridge", lambda: Mock(host_id="host"))
    local_agent = Mock(
        side_effect=AssertionError("Web ingress must not create an agent")
    )
    monkeypatch.setattr(module, "get_agent_manager", local_agent)
    persist = AsyncMock(
        side_effect=AssertionError("START acceptance owns the transcript")
    )
    monkeypatch.setattr(module, "persist_channel_user_message", persist)
    if platform == "feishu":
        from tests.web.test_feishu_message_queue import make_bot

        bot = make_bot()
        bot.channel_id = 7
        bot.channel_name = "test"
        bot.active_tasks = {}
        bot.api_client = object()
        bot._save_active_tasks = Mock()
        bot._observe_shared_input = AsyncMock()
        bot._send_text = AsyncMock(return_value="loading")
        bot._update_text = AsyncMock()
        message = SimpleNamespace(
            event=SimpleNamespace(
                message=SimpleNamespace(
                    chat_id="chat",
                    message_id="message",
                    message_type="text",
                    content='{"text": "hello"}',
                )
            )
        )
        await bot._process_messages_batch("sender", [message])
        assert submit.call_args.args[0].destination["chat_id"] == "chat"
    else:
        from tests.web.test_telegram_message_queue import make_bot

        bot = make_bot()
        bot.channel_id = 7
        bot.channel_name = "test"
        bot.active_tasks = {}
        bot.bot = object()
        bot._save_active_tasks = Mock()
        bot._observe_shared_input = AsyncMock()
        bot._extract_message_content = AsyncMock(return_value=("hello", []))
        loading = SimpleNamespace(
            message_id=77, edit_text=AsyncMock(), delete=AsyncMock()
        )
        message = SimpleNamespace(
            voice=None,
            message_id=76,
            message_thread_id=None,
            from_user=SimpleNamespace(id=123),
            chat=SimpleNamespace(id=456),
            answer=AsyncMock(return_value=loading),
        )
        await bot._process_user_messages_batch(123, [message])
        assert submit.call_args.args[0].destination["chat_id"] == 456
    assert submit.call_args.kwargs["payload"].transcript_message == "hello"
    bot._observe_shared_input.assert_awaited_once()
    local_agent.assert_not_called()
    persist.assert_not_awaited()
