"""Every channel turn binds the row's ``source`` AND its ``run_id``.

``task_source`` is the key the MCP approval gate selects a registration by,
and ``run_id`` is part of the execution identity that registration then
requires: ``ToolCallExecutionContext.is_complete()`` is false without it, and
a registered source presenting an incomplete identity is refused before
dispatch. Binding only the source would therefore turn a registration on that
source into a hard outage on these paths -- strictly worse than leaving them
unbound, where the call simply passes through ungated.

The three chat bots build their own ``context`` dict and call
``AgentService.execute_task`` directly, bypassing the WebSocket turn path that
binds the pair, so each binds it itself: the source from the task row it
already loaded, the run id from the lease it already holds -- never from the
inbound chat event. Their shared-turn path does not carry these dicts at all;
it rebuilds one in ``execute_channel_background``, covered at the bottom.

On ``source`` values: no production row carries "slack"/"telegram"/"feishu".
Bot-created tasks fall to the ``Task.source`` column default "internal", the
same value the web UI gets (``channel_runtime`` and ``task_command_execution``
both construct ``Task(...)`` without ``source=``). A host that wants to gate
only its own flow has to stamp a distinct source on the tasks it creates, so
these tests use "internal" plus a host-stamped value rather than a channel
name that does not exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from xagent.core.agent.context.execution import MODEL_CONTEXT_WATERMARK_METADATA_KEY
from xagent.core.agent.service import AgentService
from xagent.core.tools.adapters.vibe.base import AbstractBaseTool, ToolMetadata
from xagent.web.channels.feishu.bot import FeishuBotInstance
from xagent.web.channels.slack.bot import SlackBotInstance
from xagent.web.channels.telegram.bot import TelegramBotInstance
from xagent.web.services.task_execution_context_service import (
    TaskExecutionRecoverySnapshot,
)
from xagent.web.services.task_lease_service import TaskLease

_EVENT_WATERMARK = {"scope_id": "root", "event_id": "channel-history", "sequence": 9}


def _snapshot(source: str | None) -> Any:
    return SimpleNamespace(
        runtime_user=None,
        conversation_history=(),
        conversation_watermark=None,
        conversation_event_watermark=_EVENT_WATERMARK,
        execution_recovery=TaskExecutionRecoverySnapshot(),
        task=SimpleNamespace(source=source),
    )


class _FakeTracer:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def add_handler(self, handler: Any) -> None:
        self.handlers.append(handler)

    def remove_handler(self, handler: Any) -> None:
        if handler in self.handlers:
            self.handlers.remove(handler)


def _agent_service() -> Any:
    service = SimpleNamespace(
        workspace=None,
        tracer=_FakeTracer(),
        execution_metadata={},
        _execution_adapter=None,
        set_execution_context_messages=lambda _messages: None,
        set_recovered_skill_context=lambda _context: None,
    )

    service.set_conversation_history = AgentService.set_conversation_history.__get__(
        service, AgentService
    )
    return service


def _agent_manager(contexts: list[Any], service: Any) -> Any:
    class FakeAgentManager:
        async def get_agent_for_task(self, *_args: Any, **_kwargs: Any) -> Any:
            return service

        async def execute_task(self, **kwargs: Any) -> dict[str, Any]:
            assert (
                service.execution_metadata[MODEL_CONTEXT_WATERMARK_METADATA_KEY]
                == _EVENT_WATERMARK
            )
            contexts.append(kwargs["context"])
            return {"success": True, "output": "ok"}

    return FakeAgentManager()


# The two shapes a real row can present to this binding: the column default
# every bot-created task gets, and a source a host stamped itself (the only
# way to single out one tenant flow, since "internal" also covers the web UI).
# ``None`` is the legacy-row case, where the binding must still not fabricate
# a value.
_SOURCE_CASES = ["internal", "toby-slack", None]


class _FakeManagedLease:
    heartbeat_task = None

    def __init__(self, lease: TaskLease) -> None:
        self.lease = lease

    async def close(self) -> bool:
        return True

    async def finalize_result(self, **_kwargs: Any) -> bool:
        return True


async def _drive_slack_turn(
    monkeypatch: pytest.MonkeyPatch,
    source: str | None,
    *,
    run_id: str | None = "run-a",
) -> list[Any]:
    """Run one real Slack turn and return the context dicts it forwarded."""

    bot = object.__new__(SlackBotInstance)
    bot.channel_id = 7
    bot.channel_name = "Support Slack"
    bot.bot_user_id = "U_BOT"
    bot.web_client = object()
    bot.active_tasks = {}
    bot.event_queues = {}
    bot.event_tasks = {}
    bot._recent_event_ids = []
    bot._recent_event_id_set = set()
    bot._accepting = True
    bot._save_active_tasks = lambda: None

    managed = _FakeManagedLease(
        TaskLease(task_id=45, runner_id="runner-a", run_id=run_id)
    )
    contexts: list[Any] = []
    service = _agent_service()

    async def prepare(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            user_id=5, task_id=45, is_new_task=True, managed_lease=managed
        )

    async def persist(**_kwargs: Any) -> None:
        return None

    async def send_text(_channel_id: str, _text: str, *, thread_ts: Any) -> str:
        return "loading-ts"

    async def send_final_text(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("xagent.web.channels.slack.bot.prepare_channel_task", prepare)
    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.load_task_setup_snapshot_sync",
        lambda *_args: _snapshot(source),
    )
    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.get_agent_manager",
        lambda: _agent_manager(contexts, service),
    )
    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.persist_channel_user_message", persist
    )
    bot._send_text = send_text
    bot._send_final_text = send_final_text

    await bot._process_event(
        "T1:D1:U1:direct",
        {},
        {
            "type": "message",
            "channel_type": "im",
            "channel": "D1",
            "user": "U1",
            "ts": "1.0",
            "text": "hello",
        },
    )
    return contexts


@pytest.mark.parametrize("source", _SOURCE_CASES)
@pytest.mark.asyncio
async def test_slack_turn_binds_the_task_rows_source(
    monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    contexts = await _drive_slack_turn(monkeypatch, source)

    assert len(contexts) == 1
    assert contexts[0]["task_source"] == source
    assert contexts[0]["run_id"] == "run-a"


@pytest.mark.parametrize("source", _SOURCE_CASES)
@pytest.mark.asyncio
async def test_telegram_turn_binds_the_task_rows_source(
    monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    bot = object.__new__(TelegramBotInstance)
    bot.channel_id = 1
    bot.channel_name = "Telegram source binding"
    bot.active_tasks = {}
    bot.bot = object()
    bot._accepting = True
    bot._ingress_stopped = False
    bot._stop_lock = None
    bot._stop_loop = None
    # The shared initializer rather than a hand-listed set of fields, so a
    # future batch-control field does not silently break this harness.
    bot._initialize_batch_control()
    bot.user_active_trace_handlers = {}
    bot.user_switch_locks = {}
    bot.selected_agents = {}
    bot._save_selected_agents = lambda: True
    bot._save_active_tasks = lambda: True
    bot._consume_user_stop_request = lambda _user_id: False
    bot._clear_user_stop_request = lambda _user_id: None

    managed = _FakeManagedLease(
        TaskLease(task_id=48, runner_id="runner-a", run_id="run-a")
    )
    contexts: list[Any] = []
    service = _agent_service()

    async def prepare(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            user_id=5,
            task_id=48,
            is_new_task=True,
            managed_lease=managed,
            requested_agent_missing=False,
        )

    async def persist(**_kwargs: Any) -> None:
        return None

    async def extract_text(_message: Any) -> tuple[str, list[Any]]:
        return "hello", []

    async def await_execution(_user_id: Any, execution: Any, *, reason: Any) -> Any:
        return await execution

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.prepare_channel_task", prepare
    )
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.load_task_setup_snapshot_sync",
        lambda *_args: _snapshot(source),
    )
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.get_agent_manager",
        lambda: _agent_manager(contexts, service),
    )
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.persist_channel_user_message", persist
    )
    bot._extract_message_content = extract_text
    bot._await_execution_with_stop_monitor = await_execution

    class LoadingMessage:
        message_id = 77

        async def edit_text(self, _text: str, **_kwargs: Any) -> None:
            return None

        async def delete(self) -> None:
            return None

    class Message:
        from_user = SimpleNamespace(id=123)
        chat = SimpleNamespace(id=456)

        async def answer(self, _text: str, **_kwargs: Any) -> LoadingMessage:
            return LoadingMessage()

    await bot._process_user_messages_batch(123, [Message()])

    assert len(contexts) == 1
    assert contexts[0]["task_source"] == source
    assert contexts[0]["run_id"] == "run-a"


@pytest.mark.parametrize("source", _SOURCE_CASES)
@pytest.mark.asyncio
async def test_feishu_turn_binds_the_task_rows_source(
    monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    bot = object.__new__(FeishuBotInstance)
    bot.channel_id = 1
    bot.channel_name = "Feishu source binding"
    bot.active_tasks = {"open-id": "45"}
    bot.api_client = object()
    bot._save_active_tasks = lambda: None
    # The shared initializer rather than a hand-listed set of fields, so a
    # future batch-control field does not silently break this harness.
    bot._initialize_batch_control()
    bot.user_active_trace_handlers = {}

    managed = _FakeManagedLease(
        TaskLease(task_id=45, runner_id="runner-a", run_id="run-a")
    )
    contexts: list[Any] = []
    service = _agent_service()

    async def prepare(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            user_id=5, task_id=45, is_new_task=False, managed_lease=managed
        )

    async def persist(**_kwargs: Any) -> None:
        return None

    async def send_text(_chat_id: str, _text: str) -> str:
        return "loading-message-id"

    async def update_text(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("xagent.web.channels.feishu.bot.prepare_channel_task", prepare)
    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.load_task_setup_snapshot_sync",
        lambda *_args: _snapshot(source),
    )
    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.get_agent_manager",
        lambda: _agent_manager(contexts, service),
    )
    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.persist_channel_user_message", persist
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

    assert len(contexts) == 1
    assert contexts[0]["task_source"] == source
    assert contexts[0]["run_id"] == "run-a"


@pytest.mark.asyncio
async def test_shared_channel_turn_binds_the_snapshot_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared-turn executor is the fourth channel entry point.

    When shared task execution is enabled all three bots hand their turn to
    ``execute_channel_background``, which builds its own context dict instead
    of forwarding theirs. The binding has to exist there too, or the gate
    silently stops firing for every tenant on the shared path.
    """

    from xagent.web.services import shared_channel_execution

    contexts: list[Any] = []
    manager = _agent_manager(contexts, _agent_service())
    snapshot = SimpleNamespace(
        runtime_user=None,
        conversation_history=(),
        conversation_watermark=None,
        conversation_event_watermark=_EVENT_WATERMARK,
        execution_recovery=TaskExecutionRecoverySnapshot(),
        task=SimpleNamespace(source="internal", user_id=5),
    )

    async def _materialize(_recovery: Any) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(
        "xagent.web.services.agent_service_manager.get_agent_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        "xagent.web.services.task_execution_context_service."
        "materialize_task_execution_recovery_state",
        _materialize,
    )

    class _Forwarder:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def close(self, *, drain: bool = False) -> None:
            return None

    monkeypatch.setattr(
        shared_channel_execution, "ChannelProgressForwarder", _Forwarder
    )
    monkeypatch.setattr(
        shared_channel_execution, "get_task_event_bridge", lambda: MagicMock()
    )

    # Stop right after execute_task: everything past it (result projection,
    # lease finalization, command settlement) needs a live database and is
    # covered by test_shared_channel_execution.py. A sentinel is used rather
    # than a bare ``except Exception`` so an unrelated failure inside
    # execute_task is not swallowed.
    class _StopAfterExecute(Exception):
        pass

    def _explode(_result: Any) -> Any:
        raise _StopAfterExecute

    monkeypatch.setattr(
        "xagent.web.services.execution_result_projection."
        "project_execution_result_for_channel",
        _explode,
    )

    with pytest.raises(_StopAfterExecute):
        await shared_channel_execution.execute_channel_background(
            command=SimpleNamespace(task_id=45, command_id="cmd-1"),
            lease=TaskLease(task_id=45, runner_id="runner-a", run_id="run-a"),
            heartbeat_task=None,
            snapshot=snapshot,
            payload=SimpleNamespace(
                for_agent="hello",
                transcript_message="hello",
                attachments=[],
                file_ids=(),
            ),
        )

    assert len(contexts) == 1
    assert contexts[0]["task_source"] == "internal"
    assert contexts[0]["run_id"] == "run-a"


@pytest.mark.asyncio
async def test_a_lease_without_a_run_id_binds_neither_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both keys or neither -- never a half identity.

    A ``TaskLease`` carries ``run_id: str | None``. With a source bound but
    no run id, ``ToolCallExecutionContext.is_complete()`` is false and a
    registered source is refused before dispatch. Leaving both unbound keeps
    the documented safe default instead: the call passes through ungated.
    """

    contexts = await _drive_slack_turn(monkeypatch, "toby-slack", run_id=None)

    assert len(contexts) == 1
    assert "task_source" not in contexts[0]
    assert "run_id" not in contexts[0]


@pytest.mark.asyncio
async def test_a_bot_context_reaches_the_gate_hook_rather_than_failing_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reason the binding must carry both keys, end to end.

    Takes the context dict the real Slack turn above produced, registers a
    gate for that row's source, and runs it through
    ``AgentService.execute_task``. The hook must be reached with a complete
    execution identity -- not refused with ``_GATE_FAILURE`` before dispatch,
    which is what a source-only binding produces.
    """

    from xagent.core.agent.service import AgentService
    from xagent.core.tools.adapters.vibe.mcp_approval_gate import (
        GateDecision,
        gate_mcp_tools,
        register_mcp_approval_gate,
        unregister_mcp_approval_gate,
    )

    contexts = await _drive_slack_turn(monkeypatch, "toby-slack")
    bot_context = dict(contexts[0])
    assert bot_context["task_source"] == "toby-slack"

    seen: list[Any] = []

    async def gate(call: Any) -> Any:
        seen.append(call)
        return GateDecision.deny(message="Not approved.")

    async def resume(**_: Any) -> None:
        raise AssertionError("resume must not run for a denied call")

    target = _StubConnectorWrite()
    (gated,) = gate_mcp_tools([target], connection={"id": 41})
    handle = register_mcp_approval_gate(
        task_source="toby-slack", gate=gate, resume=resume
    )
    try:
        result = await AgentService(
            name="slack-agent",
            id="svc-slack-gate",
            tools=[gated],
            llm=_ScriptedLLM(),
            pattern="react",
        ).execute_task("Publish the post.", context=bot_context, task_id="task-248032")
    finally:
        unregister_mcp_approval_gate(handle)

    assert len(seen) == 1, (
        "the gate hook was never reached -- an incomplete execution identity "
        "was refused before dispatch"
    )
    execution = seen[0].execution_context
    assert execution.is_complete()
    assert execution.task_source == "toby-slack"
    assert execution.run_id == "run-a"
    assert target.calls == [], "a denied write must not be dispatched"
    assert "approval gate is unavailable" not in str(result)


class _StubConnectorWrite(AbstractBaseTool):
    """Minimal MCP-shaped write target for the end-to-end gate assertion."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._metadata = ToolMetadata(
            name="mcp_LinkedIn_create_post",
            description="Create a post.",
            concurrency_safe=False,
            read_only=False,
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    @property
    def name(self) -> str:
        return "mcp_LinkedIn_create_post"

    @property
    def description(self) -> str:
        return "Create a post."

    def args_type(self) -> type[BaseModel]:
        return _WriteArgs

    def return_type(self) -> type[BaseModel]:
        return BaseModel

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise AssertionError("the async path is the one under test")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        self.calls.append(dict(args))
        return {"success": True}


class _WriteArgs(BaseModel):
    text: str = ""


class _ScriptedLLM:
    model_name = "stub-model"

    def __init__(self) -> None:
        self.responses: list[Any] = [
            {
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "mcp_LinkedIn_create_post",
                            "arguments": '{"text":"publish"}',
                        },
                    }
                ]
            },
            {"content": "Done.", "done": True},
        ]

    async def chat(self, **_kwargs: Any) -> Any:
        return self.responses.pop(0)
