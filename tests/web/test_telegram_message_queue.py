import asyncio

import pytest

from xagent.web.channels.telegram.bot import TelegramBotInstance


def make_bot() -> TelegramBotInstance:
    bot = object.__new__(TelegramBotInstance)
    bot.user_message_queues = {}
    bot.user_message_tasks = {}
    bot.user_active_executions = {}
    bot.user_preparing_executions = set()
    bot.user_stop_events = {}
    return bot


@pytest.mark.asyncio
async def test_process_user_queue_drains_messages_added_while_batch_runs() -> None:
    bot = make_bot()
    bot.queue_flush_delay_seconds = 0
    bot.user_message_queues = {123: ["first"]}

    processed_batches: list[list[str]] = []

    async def fake_process_batch(user_id: int, messages: list[str]) -> None:
        processed_batches.append(list(messages))
        if len(processed_batches) == 1:
            bot.user_message_queues.setdefault(user_id, []).append("second")

    bot._process_user_messages_batch = fake_process_batch

    queue_task = asyncio.create_task(bot._process_user_queue(123))
    bot.user_message_tasks[123] = queue_task

    await queue_task

    assert processed_batches == [["first"], ["second"]]
    assert bot.user_message_tasks == {}
    assert bot.user_message_queues == {}


@pytest.mark.asyncio
async def test_process_user_queue_drains_message_added_while_unregistering() -> None:
    bot = make_bot()
    bot.queue_flush_delay_seconds = 0
    bot.user_message_queues = {123: ["first"]}

    class RaceTaskDict(dict):
        def __init__(self, user_id: int) -> None:
            super().__init__()
            self.user_id = user_id
            self.injected = False

        def pop(self, key, default=None):  # type: ignore[no-untyped-def]
            value = super().pop(key, default)
            if key == self.user_id and not self.injected:
                self.injected = True
                bot.user_message_queues.setdefault(key, []).append("second")
            return value

    bot.user_message_tasks = RaceTaskDict(123)
    processed_batches: list[list[str]] = []

    async def fake_process_batch(user_id: int, messages: list[str]) -> None:
        processed_batches.append(list(messages))

    bot._process_user_messages_batch = fake_process_batch

    queue_task = asyncio.create_task(bot._process_user_queue(123))
    bot.user_message_tasks[123] = queue_task

    await queue_task

    assert processed_batches == [["first"], ["second"]]
    assert bot.user_message_tasks == {}
    assert bot.user_message_queues == {}


def test_start_new_conversation_clears_queue_and_pauses_active_execution() -> None:
    bot = make_bot()
    bot.user_message_queues = {123: ["old queued message"]}
    bot.active_tasks = {123: 456}
    bot.saved = False

    class FakeAgentService:
        def __init__(self) -> None:
            self.pause_calls: list[tuple[str, str | None]] = []

        def pause_execution_by_id(
            self, execution_id: str, reason: str | None = None
        ) -> bool:
            self.pause_calls.append((execution_id, reason))
            return True

    agent_service = FakeAgentService()
    bot.user_active_executions = {123: (456, agent_service)}

    def fake_save_active_tasks() -> None:
        bot.saved = True

    bot._save_active_tasks = fake_save_active_tasks

    assert bot._start_new_conversation(123) is True
    assert 123 not in bot.user_message_queues
    assert bot.active_tasks[123] == -1
    assert bot.saved is True
    assert agent_service.pause_calls == [("456", "new Telegram conversation requested")]


def test_stop_current_conversation_preserves_active_task() -> None:
    bot = make_bot()
    bot.user_message_queues = {123: ["old queued message"]}
    bot.active_tasks = {123: 456}
    bot.saved = False

    class FakeAgentService:
        def __init__(self) -> None:
            self.pause_calls: list[tuple[str, str | None]] = []

        def pause_execution_by_id(
            self, execution_id: str, reason: str | None = None
        ) -> bool:
            self.pause_calls.append((execution_id, reason))
            return True

    agent_service = FakeAgentService()
    bot.user_active_executions = {123: (456, agent_service)}

    def fake_save_active_tasks() -> None:
        bot.saved = True

    bot._save_active_tasks = fake_save_active_tasks

    assert bot._stop_current_conversation(123) is True
    assert 123 not in bot.user_message_queues
    assert bot.active_tasks[123] == 456
    assert bot.saved is False
    assert agent_service.pause_calls == [("456", "Telegram stop requested")]


def test_stop_current_conversation_clears_pending_queue_without_active_run() -> None:
    bot = make_bot()
    bot.user_message_queues = {123: ["queued before execution"]}
    bot.active_tasks = {123: 456}

    assert bot._stop_current_conversation(123) is True
    assert bot.user_message_queues == {}
    assert bot.active_tasks[123] == 456


def test_stop_current_conversation_records_stop_during_preparation() -> None:
    bot = make_bot()
    bot.active_tasks = {123: 456}
    bot.user_preparing_executions.add(123)

    assert bot._stop_current_conversation(123) is True
    assert bot.user_stop_events[123].is_set()
    assert bot.active_tasks[123] == 456


@pytest.mark.asyncio
async def test_await_execution_with_stop_monitor_pauses_pending_stop() -> None:
    bot = make_bot()

    class FakeAgentService:
        def __init__(self) -> None:
            self.pause_calls: list[tuple[str, str | None]] = []

        def pause_execution_by_id(
            self, execution_id: str, reason: str | None = None
        ) -> bool:
            self.pause_calls.append((execution_id, reason))
            return True

    agent_service = FakeAgentService()
    bot.user_active_executions = {123: (456, agent_service)}
    bot._request_user_stop(123)

    async def fake_execution() -> dict:
        await asyncio.sleep(0)
        return {"status": "interrupted"}

    result = await bot._await_execution_with_stop_monitor(
        123,
        fake_execution(),
        reason="Telegram stop requested",
    )

    assert result == {"status": "interrupted"}
    assert agent_service.pause_calls == [("456", "Telegram stop requested")]
    assert not bot.user_stop_events[123].is_set()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/stop", True),
        ("/stop@xagent_bot", True),
        ("/pause now", True),
        ("STOP", True),
        ("暂停", True),
        ("停止", True),
        ("请暂停一下", False),
        ("/new", False),
    ],
)
def test_stop_request_text_aliases(text: str, expected: bool) -> None:
    bot = make_bot()

    assert bot._is_stop_request_text(text) is expected
