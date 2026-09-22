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

    from xagent.web.services.channel_runtime import SelectedChannelTask
    from xagent.web.services.shared_channel_execution import SharedChannelTurn

    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "web")
    module = importlib.import_module(f"xagent.web.channels.{platform}.bot")
    turn = SharedChannelTurn(
        SelectedChannelTask(5, 45, True, 7, "sender", None, 0), SimpleNamespace()
    )
    turn.execute = AsyncMock(
        return_value={"success": True, "status": "completed", "output": "Worker answer"}
    )
    turn.close = AsyncMock()
    turn.deliver = AsyncMock(return_value=True)
    monkeypatch.setattr(
        module, "prepare_shared_channel_turn", AsyncMock(return_value=turn)
    )
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
        assert turn.delivery_destination["chat_id"] == "chat"
    else:
        from tests.web.test_telegram_message_queue import make_bot

        bot = make_bot()
        bot.channel_id = 7
        bot.channel_name = "test"
        bot.active_tasks = {}
        bot.bot = object()
        bot._save_active_tasks = Mock()
        bot._extract_message_content = AsyncMock(return_value=("hello", []))
        loading = SimpleNamespace(
            message_id=77, edit_text=AsyncMock(), delete=AsyncMock()
        )
        message = SimpleNamespace(
            message_id=76,
            message_thread_id=None,
            from_user=SimpleNamespace(id=123),
            chat=SimpleNamespace(id=456),
            answer=AsyncMock(return_value=loading),
        )
        await bot._process_user_messages_batch(123, [message])
        assert turn.delivery_destination["chat_id"] == 456
    assert turn.execute.await_args.args[0].transcript_message == "hello"
    turn.deliver.assert_awaited_once()
    assert callable(turn.deliver.await_args.args[0])
    assert turn.deliver.await_args.kwargs == {"pending_notice": False}
    turn.close.assert_awaited_once()
    local_agent.assert_not_called()
    persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_slack_entry_accepts_and_observes_without_local_agent(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from tests.web.test_slack_channel import make_bot
    from xagent.web.channels.slack import bot as module
    from xagent.web.services.channel_input_acceptance import AcceptedChannelInput
    from xagent.web.services.channel_runtime import SelectedChannelTask
    from xagent.web.services.shared_channel_execution import SharedChannelTurn

    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "web")
    accepted = AcceptedChannelInput(
        45,
        12,
        "command",
        "run",
        5,
        False,
        SelectedChannelTask(5, 45, True, 7, "sender", None, 0),
    )
    lookup = Mock(return_value=(5, None))
    accept = Mock(return_value=accepted)
    observe = AsyncMock(return_value={"success": True, "status": "completed"})
    close = AsyncMock()
    progress = AsyncMock()
    deliver = AsyncMock(return_value=True)
    local_agent = Mock(
        side_effect=AssertionError("Web ingress must not create an agent")
    )
    persist = AsyncMock(
        side_effect=AssertionError("START acceptance owns the transcript")
    )
    monkeypatch.setattr(module, "lookup_channel_input", lookup)
    monkeypatch.setattr(module, "accept_channel_input", accept)
    monkeypatch.setattr(
        module,
        "get_task_event_bridge",
        lambda: SimpleNamespace(require_ready=Mock(), host_id="host"),
    )
    monkeypatch.setattr(SharedChannelTurn, "observe", observe)
    monkeypatch.setattr(SharedChannelTurn, "close", close)
    monkeypatch.setattr(module.DurableChannelProgress, "send", progress)
    monkeypatch.setattr(module, "deliver_channel_result", deliver)
    monkeypatch.setattr(module, "get_agent_manager", local_agent)
    monkeypatch.setattr(module, "persist_channel_user_message", persist)
    bot = make_bot()
    bot.channel_id = 7
    bot._save_active_tasks = Mock()
    await bot._process_event(
        "conversation",
        {"team_id": "T1"},
        {
            "type": "message",
            "channel_type": "im",
            "channel": "D1",
            "user": "sender",
            "ts": "1.0",
            "text": "hello",
        },
    )
    lookup.assert_called_once()
    incoming = lookup.call_args.args[0]
    assert incoming.scope == ("T1", "D1")
    assert incoming.message_id == "1.0"
    assert incoming.destination["chat_id"] == "D1"
    accept.assert_called_once()
    assert accept.call_args.args == (incoming,)
    assert accept.call_args.kwargs["payload"].transcript_message == "hello"
    assert bot.active_tasks["conversation"] == 45
    progress.assert_awaited_once()
    observe.assert_awaited_once()
    assert isinstance(observe.await_args.args[0], module.SlackTraceHandler)
    deliver.assert_awaited_once_with(
        12, bot._deliver_shared_result, pending_notice=False
    )
    close.assert_awaited_once()
    local_agent.assert_not_called()
    persist.assert_not_awaited()
