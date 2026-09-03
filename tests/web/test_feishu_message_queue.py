import asyncio
from types import SimpleNamespace

import pytest

from xagent.web.channels.feishu.bot import FeishuBotInstance, FeishuChannelManager
from xagent.web.models.task import TaskStatus
from xagent.web.services.task_execution_context_service import (
    TaskExecutionRecoverySnapshot,
)
from xagent.web.services.task_lease_service import TaskLease

# Bound shutdown waits so a regression fails instead of hanging the CI job.
# Five seconds also leaves enough headroom for a loaded xdist worker; the
# production Feishu disconnect path deliberately yields for 100 ms first.
_TEST_TIMEOUT_SECONDS = 5.0


def make_bot() -> FeishuBotInstance:
    bot = object.__new__(FeishuBotInstance)
    bot._accepting = True
    bot._ingress_stopped = False
    bot._stop_lock = None
    bot._stop_loop = None
    bot.ws_client = None
    bot._ping_task = None
    bot.user_message_queues = {}
    bot.user_message_tasks = {}
    return bot


@pytest.mark.asyncio
async def test_error_after_prepare_settles_preclaimed_task_instead_of_orphaning_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = object.__new__(FeishuBotInstance)
    bot.channel_id = 1
    bot.channel_name = "Feishu prepare failure"
    bot.active_tasks = {}
    bot.api_client = object()
    bot._save_active_tasks = lambda: (_ for _ in ()).throw(
        RuntimeError("mapping persistence failed")
    )
    lease = TaskLease(task_id=45, runner_id="runner-a", run_id="run-a")
    finalized: list[TaskStatus] = []

    class FakeManagedLease:
        heartbeat_task = None

        def __init__(self) -> None:
            self.lease = lease
            self.closed = False

        async def finalize_result(self, *, status: TaskStatus, **_kwargs) -> bool:
            finalized.append(status)
            return True

        async def close(self) -> bool:
            self.closed = True
            return True

    managed = FakeManagedLease()

    async def prepare(**_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            user_id=5,
            task_id=45,
            is_new_task=True,
            managed_lease=managed,
        )

    sent_messages: list[str] = []

    async def send_text(_chat_id: str, text: str) -> str:
        sent_messages.append(text)
        return "message-id"

    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.prepare_channel_task",
        prepare,
    )
    bot._send_text = send_text
    message = SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                chat_id="chat-id",
                message_id="message-id",
                message_type="text",
                content='{"text": "hello"}',
            )
        )
    )

    await bot._process_messages_batch("open-id", [message])

    assert finalized == [TaskStatus.FAILED]
    assert managed.closed is True
    assert sent_messages == ["Sorry, an error occurred while processing your request."]


@pytest.mark.asyncio
async def test_channel_failure_suppresses_stale_error_after_exact_settlement_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = object.__new__(FeishuBotInstance)
    bot.channel_id = 1
    bot.channel_name = "Feishu exact settlement"
    bot.active_tasks = {}
    bot.api_client = object()
    bot._save_active_tasks = lambda: None

    lease = TaskLease(task_id=45, runner_id="runner-a", run_id="shared-run")

    class FakeManagedLease:
        heartbeat_task = None

        def __init__(self) -> None:
            self.lease = lease
            self.closed = False

        async def close(self) -> bool:
            self.closed = True
            return True

        async def finalize_result(
            self,
            *,
            status: TaskStatus,
            **_kwargs,
        ) -> bool:
            settlements.append((self.lease, status))
            return False

    managed = FakeManagedLease()
    settlements: list[tuple[TaskLease, TaskStatus]] = []

    class FakeTracer:
        def __init__(self) -> None:
            self.handlers: list[object] = []

        def add_handler(self, handler: object) -> None:
            self.handlers.append(handler)

        def remove_handler(self, handler: object) -> None:
            if handler in self.handlers:
                self.handlers.remove(handler)

    agent_service = SimpleNamespace(
        tracer=FakeTracer(),
        set_conversation_history=lambda _messages: None,
        set_execution_context_messages=lambda _messages: None,
        set_recovered_skill_context=lambda _context: None,
    )

    class FakeAgentManager:
        async def get_agent_for_task(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return agent_service

        async def execute_task(self, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("channel execution failed")

    sent_messages: list[str] = []

    async def send_text(_chat_id: str, text: str) -> str:
        sent_messages.append(text)
        return "loading-message-id"

    async def fake_prepare(**_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            user_id=5,
            task_id=45,
            is_new_task=True,
            managed_lease=managed,
        )

    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.prepare_channel_task",
        fake_prepare,
    )
    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.load_task_setup_snapshot_sync",
        lambda *_args: SimpleNamespace(
            runtime_user=None,
            conversation_history=(),
            execution_recovery=TaskExecutionRecoverySnapshot(),
        ),
    )

    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.get_agent_manager",
        lambda: FakeAgentManager(),
    )

    async def persist_message(**_kwargs) -> None:  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.persist_channel_user_message",
        persist_message,
    )
    bot._send_text = send_text

    message = SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                chat_id="chat-id",
                message_id="message-id",
                message_type="text",
                content='{"text": "hello"}',
            )
        )
    )

    await bot._process_messages_batch("open-id", [message])

    assert settlements == [(lease, TaskStatus.FAILED)]
    assert managed.closed is True
    assert sent_messages == [
        "⏳ **Task #45 is processing...**\n_Please wait for the result._"
    ]


@pytest.mark.parametrize(
    (
        "execution_result",
        "expected_status",
        "expected_message_type",
        "expected_content",
        "expected_error",
    ),
    [
        (
            {"success": True, "output": "completed reply"},
            TaskStatus.COMPLETED,
            "assistant_response",
            "completed reply",
            None,
        ),
        (
            {
                "success": False,
                "status": "waiting_for_user",
                "chat_response": {
                    "message": "Please confirm",
                    "interactions": [{"label": "Continue?", "options": ["Yes", "No"]}],
                },
            },
            TaskStatus.WAITING_FOR_USER,
            "question",
            "Please confirm",
            None,
        ),
        (
            {
                "success": False,
                "status": "error",
                "output": "provider token=secret",
                "error": "provider token=secret",
            },
            TaskStatus.FAILED,
            "assistant_response",
            "Task execution failed.",
            "provider token=secret",
        ),
    ],
)
@pytest.mark.asyncio
async def test_successful_channel_turn_persists_user_before_exact_assistant_settlement(
    monkeypatch: pytest.MonkeyPatch,
    execution_result: dict,
    expected_status: TaskStatus,
    expected_message_type: str,
    expected_content: str,
    expected_error: str | None,
) -> None:
    bot = object.__new__(FeishuBotInstance)
    bot.channel_id = 1
    bot.channel_name = "Feishu history"
    bot.active_tasks = {"open-id": "45"}
    bot.api_client = object()
    bot._save_active_tasks = lambda: None
    events: list[str] = []
    finalized: list[dict] = []

    lease = TaskLease(task_id=45, runner_id="runner-a", run_id="shared-run")

    class FakeManagedLease:
        heartbeat_task = None

        def __init__(self, claimed_lease: TaskLease) -> None:
            self.lease = claimed_lease

        async def close(self) -> bool:
            return True

        async def finalize_result(self, **kwargs) -> bool:  # type: ignore[no-untyped-def]
            events.append("assistant-settlement")
            finalized.append(kwargs)
            return True

    class FakeTracer:
        def __init__(self) -> None:
            self.handlers: list[object] = []

        def add_handler(self, handler: object) -> None:
            self.handlers.append(handler)

        def remove_handler(self, handler: object) -> None:
            if handler in self.handlers:
                self.handlers.remove(handler)

    agent_service = SimpleNamespace(
        tracer=FakeTracer(),
        set_conversation_history=lambda _messages: None,
        set_execution_context_messages=lambda _messages: None,
        set_recovered_skill_context=lambda _context: None,
    )

    class FakeAgentManager:
        async def get_agent_for_task(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return agent_service

        async def execute_task(self, **_kwargs):  # type: ignore[no-untyped-def]
            events.append("execute")
            return execution_result

    managed = FakeManagedLease(lease)

    async def fake_prepare(**_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            user_id=5,
            task_id=45,
            is_new_task=False,
            managed_lease=managed,
        )

    async def persist_message(**kwargs) -> None:  # type: ignore[no-untyped-def]
        assert kwargs["task_id"] == 45
        assert kwargs["user_id"] == 5
        assert kwargs["content"] == "hello"
        events.append("user-message")

    async def send_text(_chat_id: str, _text: str) -> str:
        return "loading-message-id"

    async def update_text(*_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.prepare_channel_task",
        fake_prepare,
    )
    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.load_task_setup_snapshot_sync",
        lambda *_args: SimpleNamespace(
            runtime_user=None,
            conversation_history=(),
            execution_recovery=TaskExecutionRecoverySnapshot(),
        ),
    )
    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.persist_channel_user_message",
        persist_message,
    )
    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.get_agent_manager",
        lambda: FakeAgentManager(),
    )
    bot._send_text = send_text
    bot._update_text = update_text

    message = SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                chat_id="chat-id",
                message_id="message-id",
                message_type="text",
                content='{"text": "hello"}',
            )
        )
    )
    await bot._process_messages_batch("open-id", [message])

    assert events == ["user-message", "execute", "assistant-settlement"]
    assert finalized == [
        {
            "status": expected_status,
            "assistant_content": expected_content,
            "interactions": (
                [{"label": "Continue?", "options": ["Yes", "No"]}]
                if expected_status == TaskStatus.WAITING_FOR_USER
                else []
            ),
            "message_type": expected_message_type,
            "error_message": expected_error,
            "execution_result": execution_result,
        }
    ]


class _FakeFeishuWebSocketClient:
    def __init__(
        self,
        *,
        disconnect_started: asyncio.Event | None = None,
        allow_disconnect: asyncio.Event | None = None,
        disconnect_error: Exception | None = None,
    ) -> None:
        self._auto_reconnect = False
        self._reconnect = None
        self.disconnect_started = disconnect_started
        self.allow_disconnect = allow_disconnect
        self.disconnect_error = disconnect_error
        self.disconnect_calls = 0

    async def _disconnect(self) -> None:
        self.disconnect_calls += 1
        if self.disconnect_started is not None:
            self.disconnect_started.set()
        if self.allow_disconnect is not None:
            await self.allow_disconnect.wait()
        if self.disconnect_error is not None:
            raise self.disconnect_error


def _feishu_message(open_id: str = "open-id") -> SimpleNamespace:
    return SimpleNamespace(
        event=SimpleNamespace(
            sender=SimpleNamespace(
                sender_id=SimpleNamespace(open_id=open_id),
            ),
            message=SimpleNamespace(create_time=None),
        )
    )


@pytest.mark.asyncio
async def test_stop_drains_each_feishu_turn_once_before_clearing_runtime_state() -> (
    None
):
    bot = make_bot()
    client = _FakeFeishuWebSocketClient()
    bot.ws_client = client
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def running_turn() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()
            cleanup_finished.set()

    turn_task = asyncio.create_task(running_turn())
    await asyncio.sleep(0)
    bot.user_message_tasks = {"one": turn_task, "two": turn_task}
    bot.user_message_queues = {"one": ["queued"]}

    stop_task = asyncio.create_task(bot.stop())
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=_TEST_TIMEOUT_SECONDS)
        await asyncio.sleep(0)

        assert turn_task.cancelling() == 1
        assert not stop_task.done()
        assert bot.user_message_tasks == {"one": turn_task, "two": turn_task}
        assert bot.user_message_queues == {"one": ["queued"]}

        allow_cleanup.set()
        await asyncio.wait_for(stop_task, timeout=_TEST_TIMEOUT_SECONDS)

        assert cleanup_finished.is_set()
        assert bot.user_message_tasks == {}
        assert bot.user_message_queues == {}
        assert client.disconnect_calls == 1
    finally:
        allow_cleanup.set()
        if not turn_task.done():
            turn_task.cancel()
        await asyncio.gather(turn_task, return_exceptions=True)
        if not stop_task.done():
            stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_drains_feishu_ping_cleanup() -> None:
    bot = make_bot()
    bot.ws_client = _FakeFeishuWebSocketClient()
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def ping_loop() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()
            cleanup_finished.set()

    ping_task = asyncio.create_task(ping_loop())
    await asyncio.sleep(0)
    bot._ping_task = ping_task

    stop_task = asyncio.create_task(bot.stop())
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=_TEST_TIMEOUT_SECONDS)
        await asyncio.sleep(0)

        assert not stop_task.done()
        assert not cleanup_finished.is_set()

        allow_cleanup.set()
        await asyncio.wait_for(stop_task, timeout=_TEST_TIMEOUT_SECONDS)

        assert cleanup_finished.is_set()
        assert ping_task.done()
        assert bot._ping_task is None
    finally:
        allow_cleanup.set()
        if not ping_task.done():
            ping_task.cancel()
        await asyncio.gather(ping_task, return_exceptions=True)
        if not stop_task.done():
            stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_drains_feishu_ping_cleanup_after_disconnect_failure() -> None:
    bot = make_bot()
    bot.ws_client = _FakeFeishuWebSocketClient(
        disconnect_error=RuntimeError("disconnect failed")
    )
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()

    async def ping_loop() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()

    ping_task = asyncio.create_task(ping_loop())
    await asyncio.sleep(0)
    bot._ping_task = ping_task

    stop_task = asyncio.create_task(bot.stop())
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=_TEST_TIMEOUT_SECONDS)
        assert not stop_task.done()

        allow_cleanup.set()
        with pytest.raises(RuntimeError, match="disconnect failed"):
            await asyncio.wait_for(stop_task, timeout=_TEST_TIMEOUT_SECONDS)

        assert ping_task.done()
        assert bot._ping_task is None
    finally:
        allow_cleanup.set()
        if not ping_task.done():
            ping_task.cancel()
        await asyncio.gather(ping_task, return_exceptions=True)
        if not stop_task.done():
            stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_feishu_stop_cancellation_waits_for_turn_cleanup_before_propagating() -> (
    None
):
    bot = make_bot()
    bot.ws_client = _FakeFeishuWebSocketClient()
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()

    async def running_turn() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()

    turn_task = asyncio.create_task(running_turn())
    await asyncio.sleep(0)
    bot.user_message_tasks = {"open-id": turn_task}
    bot.user_message_queues = {"open-id": ["queued"]}

    stop_task = asyncio.create_task(bot.stop())
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=_TEST_TIMEOUT_SECONDS)
        stop_task.cancel()
        await asyncio.sleep(0)

        assert not stop_task.done()
        assert bot.user_message_tasks == {"open-id": turn_task}
        assert bot.user_message_queues == {"open-id": ["queued"]}

        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(stop_task, timeout=_TEST_TIMEOUT_SECONDS)

        assert turn_task.done()
        assert bot.user_message_tasks == {}
        assert bot.user_message_queues == {}
    finally:
        allow_cleanup.set()
        if not turn_task.done():
            turn_task.cancel()
        await asyncio.gather(turn_task, return_exceptions=True)
        if not stop_task.done():
            stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_fence_rejects_late_feishu_ingress() -> None:
    bot = make_bot()
    bot.start_time = 0
    disconnect_started = asyncio.Event()
    allow_disconnect = asyncio.Event()
    bot.ws_client = _FakeFeishuWebSocketClient(
        disconnect_started=disconnect_started,
        allow_disconnect=allow_disconnect,
    )

    stop_task = asyncio.create_task(bot.stop())
    try:
        await asyncio.wait_for(disconnect_started.wait(), timeout=_TEST_TIMEOUT_SECONDS)
        bot._handle_message_sync(_feishu_message())

        assert bot.user_message_queues == {}
        assert bot.user_message_tasks == {}
    finally:
        allow_disconnect.set()
        await asyncio.gather(stop_task, return_exceptions=True)


def test_feishu_stop_is_concurrent_and_cross_loop_idempotent() -> None:
    bot = make_bot()
    client = _FakeFeishuWebSocketClient()
    bot.ws_client = client

    async def stop_concurrently() -> None:
        await asyncio.gather(bot.stop(), bot.stop())

    asyncio.run(stop_concurrently())
    asyncio.run(bot.stop())

    assert client.disconnect_calls == 1


@pytest.mark.asyncio
async def test_manager_stop_drains_feishu_polling_cleanup_before_removal() -> None:
    manager = FeishuChannelManager()
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def polling() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()
            cleanup_finished.set()

    polling_task = asyncio.create_task(polling())
    await asyncio.sleep(0)

    async def stop_bot() -> None:
        return None

    bot = SimpleNamespace(polling_task=polling_task, stop=stop_bot)
    manager.bots["app-id"] = bot

    stop_task = asyncio.create_task(manager._stop_bot_for_appid("app-id"))
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=_TEST_TIMEOUT_SECONDS)
        await asyncio.sleep(0)

        assert not stop_task.done()
        assert "app-id" in manager.bots
        assert not cleanup_finished.is_set()

        allow_cleanup.set()
        await asyncio.wait_for(stop_task, timeout=_TEST_TIMEOUT_SECONDS)

        assert cleanup_finished.is_set()
        assert polling_task.done()
        assert "app-id" not in manager.bots
    finally:
        allow_cleanup.set()
        if not polling_task.done():
            polling_task.cancel()
        await asyncio.gather(polling_task, return_exceptions=True)
        if not stop_task.done():
            stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_manager_stop_is_single_flight_per_feishu_bot() -> None:
    manager = FeishuChannelManager()
    stop_started = asyncio.Event()
    allow_stop = asyncio.Event()
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    stop_calls = 0

    async def polling() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()

    polling_task = asyncio.create_task(polling())
    await asyncio.sleep(0)

    async def stop_bot() -> None:
        nonlocal stop_calls
        stop_calls += 1
        stop_started.set()
        await allow_stop.wait()

    bot = SimpleNamespace(polling_task=polling_task, stop=stop_bot)
    manager.bots["app-id"] = bot

    stop_tasks = [
        asyncio.create_task(manager._stop_bot_for_appid("app-id")),
        asyncio.create_task(manager._stop_bot_for_appid("app-id")),
    ]
    try:
        await asyncio.wait_for(stop_started.wait(), timeout=_TEST_TIMEOUT_SECONDS)
        await asyncio.sleep(0)
        assert stop_calls == 1

        allow_stop.set()
        await asyncio.wait_for(cleanup_started.wait(), timeout=_TEST_TIMEOUT_SECONDS)
        assert polling_task.cancelling() == 1

        allow_cleanup.set()
        await asyncio.wait_for(
            asyncio.gather(*stop_tasks), timeout=_TEST_TIMEOUT_SECONDS
        )
        assert "app-id" not in manager.bots
    finally:
        allow_stop.set()
        allow_cleanup.set()
        if not polling_task.done():
            polling_task.cancel()
        await asyncio.gather(polling_task, return_exceptions=True)
        for stop_task in stop_tasks:
            if not stop_task.done():
                stop_task.cancel()
        await asyncio.gather(*stop_tasks, return_exceptions=True)
