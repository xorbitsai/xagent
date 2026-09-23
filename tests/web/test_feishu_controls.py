"""Feishu commands interrupt queued/preparing/running work outside batching."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from xagent.web.channels.feishu import bot as module
from xagent.web.services.channel_input_acceptance import AcceptedChannelInput
from xagent.web.services.channel_runtime import (
    ChannelAuthorizationError,
    SelectedChannelTask,
)
from xagent.web.services.shared_channel_execution import SharedChannelTurn


def message(text, identity="one"):
    return SimpleNamespace(
        event=SimpleNamespace(
            sender=SimpleNamespace(sender_id=SimpleNamespace(open_id="sender")),
            message=SimpleNamespace(
                message_type="text",
                content=json.dumps({"text": text}),
                chat_id="chat",
                message_id=identity,
            ),
        )
    )


@pytest.fixture
def bot(monkeypatch, tmp_path):
    instance = object.__new__(module.FeishuBotInstance)
    instance._initialize_batch_control()
    instance.user_active_trace_handlers = {}
    instance.control_tasks = set()
    instance.control_queues = {}
    instance.control_locks = {}
    instance._accepting = True
    instance.start_time = 0
    instance.channel_id = 7
    instance.channel_name = "test"
    instance.active_tasks = {"sender": "45"}
    instance.active_tasks_file = tmp_path / "active.json"
    instance.api_client = Mock()
    instance.queue_flush_delay_seconds = 0
    instance._send_text = AsyncMock(return_value="loading")
    instance._update_text = AsyncMock()
    monkeypatch.setattr(module, "authorize_channel_sender", AsyncMock())
    monkeypatch.setattr(module, "discard_channel_task_results", AsyncMock())
    monkeypatch.setattr(module, "get_shared_task_execution_enabled", lambda: True)
    return instance


async def control(bot, text):
    bot._handle_message_sync(message(text))
    await asyncio.gather(*list(bot.control_tasks))


@pytest.mark.asyncio
async def test_feishu_drains_inputs_arriving_during_execution(bot):
    entered, release = asyncio.Event(), asyncio.Event()
    batches = []

    async def process(user, messages):
        batches.append([json.loads(m.event.message.content)["text"] for m in messages])
        if len(batches) == 1:
            entered.set()
            await release.wait()

    bot._process_messages_batch = process
    bot._handle_message_sync(message("first"))
    await asyncio.wait_for(entered.wait(), 2)
    bot._handle_message_sync(message("second", "two"))
    task = bot.user_message_tasks["sender"]
    release.set()
    await asyncio.wait_for(task, 2)
    assert batches == [["first"], ["second"]]
    assert not bot.user_message_queues
    assert not bot.user_message_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/new", "/stop", "/pause"])
async def test_command_does_not_join_pending_text(bot, command):
    bot.queue_flush_delay_seconds = 60
    bot._process_messages_batch = AsyncMock()
    bot._handle_message_sync(message("queued"))
    await control(bot, command)
    assert not bot.user_message_queues
    bot._process_messages_batch.assert_not_awaited()
    assert bot.active_tasks["sender"] == ("-1" if command == "/new" else "45")
    await bot._drain_user_message_tasks()


@pytest.mark.asyncio
async def test_unauthorized_control_does_not_change_work(bot, monkeypatch):
    monkeypatch.setattr(
        module,
        "authorize_channel_sender",
        AsyncMock(side_effect=ChannelAuthorizationError()),
    )
    bot.user_message_queues["sender"] = [message("queued")]
    await control(bot, "/new")
    assert bot.active_tasks["sender"] == "45"
    assert len(bot.user_message_queues["sender"]) == 1


@pytest.mark.asyncio
async def test_new_save_failure_preserves_current_work(bot):
    bot._save_active_tasks = Mock(return_value=False)
    bot.user_message_queues["sender"] = [message("queued")]
    pause = Mock(return_value=True)
    bot.user_active_executions["sender"] = (
        45,
        SimpleNamespace(pause_execution_by_id=pause),
    )
    await control(bot, "/new")
    pause.assert_not_called()
    assert bot.active_tasks["sender"] == "45"
    assert bot.user_message_queues["sender"]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/new", "/stop"])
@pytest.mark.parametrize("retained", [False, True])
async def test_control_fences_late_preparation(bot, monkeypatch, command, retained):
    if retained:
        old = SharedChannelTurn(
            SelectedChannelTask(5, 45, False, 7, "sender", None, 0), None
        )
        old.request_stop = Mock(return_value=True)
        bot.user_active_executions["sender"] = (45, old)
    import threading

    entered, release = threading.Event(), threading.Event()
    accept = Mock(side_effect=AssertionError("Stopped preparation cannot accept"))

    def lookup(incoming):
        entered.set()
        assert release.wait(5)
        return 5, (), (), ()

    monkeypatch.setattr(module, "lookup_channel_inputs", lookup)
    monkeypatch.setattr(module, "accept_channel_input", accept)
    bot._handle_message_sync(message("request"))
    task = bot.user_message_tasks["sender"]
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        await control(bot, command)
    finally:
        release.set()
    await asyncio.wait_for(task, 2)
    accept.assert_not_called()
    assert bot.active_tasks["sender"] == ("-1" if command == "/new" else "45")


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/new", "/stop"])
async def test_control_reaches_running_shared_turn(bot, monkeypatch, command):
    entered, release = asyncio.Event(), asyncio.Event()
    turn = SharedChannelTurn(
        SelectedChannelTask(5, 45, False, 7, "sender", None, 0), None
    )

    async def execute(*args):
        turn.accepted = True
        entered.set()
        await release.wait()
        return {"status": "interrupted", "success": True, "output": "partial"}

    def stop():
        release.set()
        return True

    turn.observe = AsyncMock(side_effect=execute)
    turn.request_stop = Mock(side_effect=stop)
    turn.close = AsyncMock()
    turn.deliver = AsyncMock(return_value=True)
    turn.discard_delivery = AsyncMock()
    accepted = AcceptedChannelInput(45, 1, "command", "run", 5, False, turn.selection)
    monkeypatch.setattr(AcceptedChannelInput, "as_turn", lambda self: turn)
    monkeypatch.setattr(module.DurableChannelProgress, "send", AsyncMock())
    task = asyncio.create_task(bot._observe_shared_input("sender", accepted, 0))
    await asyncio.wait_for(entered.wait(), 2)
    handler = bot.user_active_trace_handlers["sender"]
    await control(bot, command)
    await asyncio.wait_for(task, 2)
    turn.request_stop.assert_called_once()
    assert handler.cancelled
    assert turn.discard_output == (command == "/new")
    if command == "/new":
        turn.deliver.assert_not_awaited()
        turn.discard_delivery.assert_awaited_once()
    else:
        turn.deliver.assert_awaited_once()
    assert not bot.user_active_executions


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["accepted", "completed"])
async def test_stop_after_shared_reply_timeout_retains_control(
    bot, monkeypatch, status
):
    turn = SharedChannelTurn(
        SelectedChannelTask(5, 45, False, 7, "sender", None, 0), None
    )
    turn.observe = AsyncMock(return_value={"status": status})
    turn.deliver = AsyncMock(return_value=False)
    turn.close = AsyncMock()
    turn.request_stop = Mock(return_value=True)
    accepted = AcceptedChannelInput(45, 1, "command", "run", 5, False, turn.selection)
    monkeypatch.setattr(AcceptedChannelInput, "as_turn", lambda self: turn)
    monkeypatch.setattr(module.DurableChannelProgress, "send", AsyncMock())
    await bot._observe_shared_input("sender", accepted, 0)
    await control(bot, "/stop")
    if status == "accepted":
        turn.request_stop.assert_called_once()
    else:
        turn.request_stop.assert_not_called()
        assert not bot.user_active_executions
        assert bot._send_text.await_args.args[1] == "No active run to stop."


@pytest.mark.asyncio
async def test_shutdown_drains_authorizing_command(bot, monkeypatch):
    entered = asyncio.Event()

    async def authorize(**kwargs):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(module, "authorize_channel_sender", authorize)
    bot._handle_message_sync(message("/new"))
    await asyncio.wait_for(entered.wait(), 2)
    bot._handle_message_sync(message("after command", "after"))
    assert bot.control_queues["sender"]
    bot._accepting = False
    await asyncio.wait_for(bot._drain_user_message_tasks(), 2)
    assert bot.active_tasks["sender"] == "45"
    assert not bot.control_tasks
    assert not bot.control_queues
    assert not bot.user_message_queues


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/new", "/stop"])
async def test_control_reaches_local_execution(bot, monkeypatch, command):
    from xagent.web.models.task import TaskStatus
    from xagent.web.services.task_execution_context_service import (
        TaskExecutionRecoverySnapshot,
    )

    monkeypatch.setattr(module, "get_shared_task_execution_enabled", lambda: False)
    entered, release = asyncio.Event(), asyncio.Event()
    lease = SimpleNamespace(
        lease=object(),
        heartbeat_task=None,
        close=AsyncMock(),
        finalize_result=AsyncMock(return_value=True),
    )
    service = Mock()

    def pause(*args, **kwargs):
        release.set()
        return True

    service.pause_execution_by_id.side_effect = pause

    async def execute(**kwargs):
        entered.set()
        await release.wait()
        return {"success": True, "status": "interrupted", "output": "partial"}

    manager = SimpleNamespace(
        get_agent_for_task=AsyncMock(return_value=service), execute_task=execute
    )
    monkeypatch.setattr(module, "get_agent_manager", lambda: manager)
    monkeypatch.setattr(
        module,
        "prepare_channel_task",
        AsyncMock(
            return_value=SimpleNamespace(
                task_id=45, user_id=5, is_new_task=False, managed_lease=lease
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "load_task_setup_snapshot_sync",
        lambda *args: SimpleNamespace(
            runtime_user=None,
            conversation_history=(),
            conversation_watermark=None,
            execution_recovery=TaskExecutionRecoverySnapshot(),
        ),
    )
    monkeypatch.setattr(module, "persist_channel_user_message", AsyncMock())
    bot._handle_message_sync(message("request"))
    await asyncio.wait_for(entered.wait(), 2)
    task = bot.user_message_tasks["sender"]
    await control(bot, command)
    await asyncio.wait_for(task, 2)
    service.pause_execution_by_id.assert_called_once()
    lease.close.assert_awaited_once()
    assert lease.finalize_result.await_args.kwargs["status"] == TaskStatus.PAUSED
    if command == "/new":
        bot._update_text.assert_not_awaited()
    else:
        bot._update_text.assert_awaited_once()
    assert not bot.user_active_executions


@pytest.mark.asyncio
async def test_failed_batch_does_not_strand_next_batch(bot):
    batches = []

    async def process(user, items):
        batches.append(items)
        if len(batches) == 1:
            bot._enqueue_user_message(user, "second")
            raise RuntimeError("batch failure")

    bot._process_messages_batch = process
    bot._enqueue_user_message("sender", "first")
    await asyncio.wait_for(bot.user_message_tasks["sender"], 2)
    assert batches == [["first"], ["second"]]
    assert not bot.user_message_tasks


def test_save_failure_keeps_previous_feishu_file(bot, monkeypatch):
    bot._save_active_tasks()
    bot.active_tasks["sender"] = "-1"
    monkeypatch.setattr(module.os, "replace", Mock(side_effect=OSError("disk error")))
    assert not bot._save_active_tasks()
    assert json.loads(bot.active_tasks_file.read_text()) == {"sender": "45"}


@pytest.mark.asyncio
async def test_new_conversation_suppresses_remaining_shared_reply_chunks(bot):
    current = True

    async def update(*args, **kwargs):
        nonlocal current
        current = False

    bot._update_text.side_effect = update
    delivery = SimpleNamespace(
        destination={"chat_id": "chat", "loading_message_id": "loading"}
    )
    with pytest.raises(module.ChannelDeliveryDiscarded):
        await bot._deliver_shared_result(
            delivery,
            {"success": True, "status": "completed", "output": "a" * 5000},
            is_current=lambda: current,
        )
    bot._update_text.assert_awaited_once()
    bot._send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_conversation_fences_recovered_reply_chunks(bot):
    async def update(*args, **kwargs):
        await control(bot, "/new")

    bot._update_text.side_effect = update
    delivery = SimpleNamespace(
        external_user_id="sender",
        task_id=45,
        destination={"chat_id": "chat", "loading_message_id": "loading"},
    )
    with pytest.raises(module.ChannelDeliveryDiscarded):
        await bot._deliver_shared_result(
            delivery, {"success": True, "status": "completed", "output": "a" * 5000}
        )
    bot._update_text.assert_awaited_once()
    assert bot._send_text.await_count == 1
    assert "a" * 1000 not in bot._send_text.await_args.args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("shared", [False, True])
async def test_new_conversation_blocks_final_update_fallback(bot, shared):
    entered, release = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()

    def patch(request):
        loop.call_soon_threadsafe(entered.set)
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result(timeout=2)
        return SimpleNamespace(
            success=lambda: False, code=230001, msg="not a card", error=None
        )

    bot.api_client.im.v1.message.patch.side_effect = patch
    bot._update_text = module.FeishuBotInstance._update_text.__get__(bot)
    generation = bot._conversation_generation("sender")
    if shared:
        pending = bot._deliver_shared_result(
            SimpleNamespace(
                external_user_id="sender",
                task_id=45,
                destination={"chat_id": "chat", "loading_message_id": "loading"},
            ),
            {"success": True, "status": "completed", "output": "old answer"},
        )
    else:
        pending = bot._update_text(
            "chat",
            "loading",
            "old answer",
            is_current=lambda: bot._conversation_generation("sender") == generation,
        )
    sending = asyncio.create_task(pending)
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await control(bot, "/new")
    finally:
        release.set()
    if shared:
        with pytest.raises(module.ChannelDeliveryDiscarded):
            await asyncio.wait_for(sending, 2)
    else:
        await asyncio.wait_for(sending, 2)
    assert bot._send_text.await_count == 1
    assert bot._send_text.await_args.args[1] != "old answer"


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/stop", "/new"])
async def test_control_pauses_when_local_setup_fails(bot, monkeypatch, command):
    from xagent.web.models.task import TaskStatus

    monkeypatch.setattr(module, "get_shared_task_execution_enabled", lambda: False)
    entered, release = asyncio.Event(), asyncio.Event()
    lease = SimpleNamespace(
        finalize_result=AsyncMock(return_value=True), close=AsyncMock()
    )

    async def setup(*args, **kwargs):
        entered.set()
        await release.wait()
        raise RuntimeError("setup failed")

    manager = SimpleNamespace(get_agent_for_task=setup, execute_task=AsyncMock())
    monkeypatch.setattr(module, "get_agent_manager", lambda: manager)
    monkeypatch.setattr(
        module,
        "prepare_channel_task",
        AsyncMock(
            return_value=SimpleNamespace(
                task_id=45, user_id=5, is_new_task=False, managed_lease=lease
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "load_task_setup_snapshot_sync",
        lambda *args: SimpleNamespace(runtime_user=None),
    )
    bot._handle_message_sync(message("request"))
    await asyncio.wait_for(entered.wait(), 2)
    task = bot.user_message_tasks["sender"]
    try:
        await control(bot, command)
    finally:
        release.set()
    await asyncio.wait_for(task, 2)
    lease.finalize_result.assert_awaited_once_with(status=TaskStatus.PAUSED)
    lease.close.assert_awaited_once()
    manager.execute_task.assert_not_awaited()
    bot._send_text.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("previous", [None, "-1"])
async def test_new_task_save_failure_prevents_execution(bot, monkeypatch, previous):
    from xagent.web.models.task import TaskStatus

    bot.active_tasks = {} if previous is None else {"sender": previous}
    assert bot._save_active_tasks()
    monkeypatch.setattr(module, "get_shared_task_execution_enabled", lambda: False)
    lease = SimpleNamespace(
        finalize_result=AsyncMock(return_value=True), close=AsyncMock()
    )
    monkeypatch.setattr(
        module,
        "prepare_channel_task",
        AsyncMock(
            return_value=SimpleNamespace(
                task_id=99, user_id=5, is_new_task=True, managed_lease=lease
            )
        ),
    )
    manager = Mock()
    monkeypatch.setattr(module, "get_agent_manager", lambda: manager)
    with monkeypatch.context() as failed_write:
        failed_write.setattr(
            module.os, "replace", Mock(side_effect=OSError("disk unavailable"))
        )
        bot._handle_message_sync(message("request"))
        await asyncio.wait_for(bot.user_message_tasks["sender"], 2)
    expected = {} if previous is None else {"sender": previous}
    assert bot.active_tasks == expected
    assert bot._load_active_tasks() == expected
    manager.get_agent_for_task.assert_not_called()
    lease.finalize_result.assert_awaited_once_with(status=TaskStatus.PAUSED)
    lease.close.assert_awaited_once()
    bot._send_text.assert_awaited_once()
    assert "try again" in bot._send_text.await_args.args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/new", "/stop"])
@pytest.mark.parametrize("authorized", [False, True])
async def test_message_after_authorizing_control_waits_and_is_preserved(
    bot, monkeypatch, command, authorized
):
    entered, release, processed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    observed = []

    async def authorize(**kwargs):
        entered.set()
        await release.wait()
        if not authorized:
            raise ChannelAuthorizationError()

    async def process(user, messages):
        observed.append(
            (
                bot.active_tasks[user],
                [json.loads(m.event.message.content)["text"] for m in messages],
            )
        )
        processed.set()

    monkeypatch.setattr(module, "authorize_channel_sender", authorize)
    bot._process_messages_batch = process
    bot._handle_message_sync(message(command))
    # Both callbacks may run before either scheduled task starts.
    bot._handle_message_sync(message("new request", "following"))
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.sleep(0)
    assert not processed.is_set()
    assert not bot.user_message_queues
    release.set()
    await asyncio.wait_for(processed.wait(), 2)
    assert observed == [
        ("-1" if authorized and command == "/new" else "45", ["new request"])
    ]
    await bot._drain_user_message_tasks()
    assert not bot.control_queues


@pytest.mark.asyncio
async def test_controls_keep_boundaries_between_multiple_commands(bot, monkeypatch):
    entered, release, second, finish = (asyncio.Event() for _ in range(4))
    calls = 0

    async def authorize(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
        else:
            second.set()
            await finish.wait()

    monkeypatch.setattr(module, "authorize_channel_sender", authorize)
    bot.queue_flush_delay_seconds = 60
    bot._handle_message_sync(message("/new"))
    await entered.wait()
    bot._handle_message_sync(message("between", "between"))
    bot._handle_message_sync(message("/stop", "stop"))
    bot._handle_message_sync(message("after", "after"))
    release.set()
    await second.wait()
    assert [
        json.loads(m.event.message.content)["text"]
        for m in bot.user_message_queues["sender"]
    ] == ["between"]
    finish.set()
    await asyncio.gather(*list(bot.control_tasks))
    assert [
        json.loads(m.event.message.content)["text"]
        for m in bot.user_message_queues["sender"]
    ] == ["after"]
    await bot._drain_user_message_tasks()


@pytest.mark.asyncio
async def test_new_cleanup_failure_reports_committed_selection(bot, monkeypatch):
    monkeypatch.setattr(
        module,
        "discard_channel_task_results",
        AsyncMock(side_effect=RuntimeError("db unavailable")),
    )
    await control(bot, "/new")
    assert bot._load_active_tasks() == {"sender": "-1"}
    assert "new conversation is selected" in bot._send_text.await_args.args[1]
    assert "couldn't finish cleaning" in bot._send_text.await_args.args[1]


@pytest.mark.asyncio
async def test_control_failure_reports_safe_error(bot, monkeypatch):
    monkeypatch.setattr(
        module,
        "authorize_channel_sender",
        AsyncMock(side_effect=RuntimeError("private database details")),
    )
    await control(bot, "/stop")
    bot._send_text.assert_awaited_once()
    assert "try again" in bot._send_text.await_args.args[1]
    assert "private" not in bot._send_text.await_args.args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("shared", [False, True])
async def test_startup_filter_preserves_controls_but_allows_shared_retries(
    bot, monkeypatch, shared
):
    monkeypatch.setattr(module, "get_shared_task_execution_enabled", lambda: shared)
    bot.start_time = 200
    bot._process_messages_batch = AsyncMock()
    for text in ("/new", "/stop", "retry"):
        data = message(text)
        data.event.message.create_time = "100"
        bot._handle_message_sync(data)
    tasks = list(bot.user_message_tasks.values())
    if tasks:
        await asyncio.gather(*tasks)
    assert not bot.control_tasks
    assert bot.active_tasks["sender"] == "45"
    assert bot._process_messages_batch.await_count == int(shared)
