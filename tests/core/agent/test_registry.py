from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from xagent.core.agent import (
    Agent,
    ContextManager,
    ExecutionContext,
    ExecutionLifecycleStatus,
    PatternRuntime,
)
from xagent.core.agent import registry as registry_module
from xagent.core.agent.registry import ExecutionRegistry
from xagent.core.agent.runner import AgentRunner


@pytest.fixture(autouse=True)
def reset_context_manager() -> None:
    manager = ContextManager()
    manager._contexts.clear()  # type: ignore[attr-defined]
    yield
    manager._contexts.clear()  # type: ignore[attr-defined]


@dataclass
class FakeWorkspace:
    id: str
    workspace_dir: Path
    input_dir: Path
    output_dir: Path
    temp_dir: Path
    allowed_external_dirs: list[Path]


class FakeWorkspaceManager:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    def get_or_create_workspace(
        self,
        base_dir: str,
        task_id: str,
        allowed_external_dirs: list[str] | None = None,
    ) -> FakeWorkspace:
        del base_dir
        workspace_dir = self.tmp_path / task_id
        return FakeWorkspace(
            id=task_id,
            workspace_dir=workspace_dir,
            input_dir=workspace_dir / "input",
            output_dir=workspace_dir / "output",
            temp_dir=workspace_dir / "temp",
            allowed_external_dirs=[Path(path) for path in allowed_external_dirs or []],
        )


class TracerCheckpointStore:
    def __init__(self) -> None:
        self.by_execution_id: dict[str, dict[str, Any]] = {}

    async def checkpoint(self, **payload: Any) -> None:
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


class InterruptingPattern:
    def __init__(self, runner: AgentRunner, execution_id: str) -> None:
        self.runner = runner
        self.execution_id = execution_id

    async def run(
        self,
        *,
        context: ExecutionContext,
        runtime: PatternRuntime,
        **_: Any,
    ) -> dict[str, Any]:
        self.runner.pause(self.execution_id, reason="pause before step")
        if await runtime.should_interrupt():
            await runtime.checkpoint(
                "interrupted",
                context=context,
                pattern=self,
                status="interrupted",
                metadata={"safe_point": "during_pattern"},
            )
            return {
                "success": False,
                "status": "interrupted",
                "error": "InterruptingPattern interrupted.",
            }

        return {"success": True, "output": "continued"}


class SuccessfulPattern:
    async def run(self, **_: Any) -> dict[str, Any]:
        return {"success": True, "output": "done"}


class BlockingPattern:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, **_: Any) -> dict[str, Any]:
        self.started.set()
        await asyncio.Future()
        return {"success": True, "output": "unreachable"}


@pytest.mark.asyncio
async def test_registry_routes_live_execution_control_and_resume(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    execution_id = "exec-registry"
    registry = ExecutionRegistry()
    first_agent = Agent(name="writer", patterns=[])
    first_runner = AgentRunner(
        agent=first_agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    first_agent.patterns = [InterruptingPattern(first_runner, execution_id)]

    handle = registry.start(
        first_runner,
        execution_id=execution_id,
        task="Calculate 6*7",
    )
    assert handle.status == ExecutionLifecycleStatus.RUNNING
    assert handle.requested_task == "Calculate 6*7"
    interrupted = await handle.task

    assert interrupted["status"] == "interrupted"
    stored = registry.get(execution_id)
    assert stored is not None
    assert stored.status == ExecutionLifecycleStatus.INTERRUPTED
    assert stored.is_resumable is True
    assert stored.last_error == "InterruptingPattern interrupted."
    assert stored.to_dict()["status"] == "interrupted"
    assert registry.get_status(execution_id) == stored.to_dict()
    assert registry.list_statuses() == [stored.to_dict()]

    context = await registry.post_user_message(
        execution_id,
        "Reply with only the number.",
        request_interrupt=False,
    )
    assert context is not None
    assert any(
        message.role == "user" and message.content == "Reply with only the number."
        for message in context.messages
    )

    resumed_agent = Agent(
        name="writer",
        patterns=[SuccessfulPattern()],
    )
    resumed_runner = AgentRunner(
        agent=resumed_agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    registry.register(execution_id, resumed_runner)

    resumed = await registry.resume(execution_id)

    assert resumed is not None
    assert resumed["success"] is True
    assert resumed["output"] == "done"
    assert registry.get(execution_id) is None


@pytest.mark.asyncio
async def test_registry_emits_lifecycle_and_message_events(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    execution_id = "exec-events"
    registry = ExecutionRegistry()
    events: list[dict[str, Any]] = []
    subscription_id = registry.subscribe(events.append)
    first_agent = Agent(name="writer", patterns=[])
    first_runner = AgentRunner(
        agent=first_agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    first_agent.patterns = [InterruptingPattern(first_runner, execution_id)]

    handle = registry.start(
        first_runner,
        execution_id=execution_id,
        task="Calculate 6*7",
    )
    assert handle.task is not None
    await handle.task
    await registry.post_user_message(
        execution_id,
        "Reply with only the number.",
        request_interrupt=False,
    )

    event_types = [event["type"] for event in events]
    assert event_types == [
        "execution.started",
        "execution.interrupted",
        "execution.message_posted",
    ]
    assert all(event["execution_id"] == execution_id for event in events)
    assert events[1]["handle"]["status"] == "interrupted"
    assert events[2]["message"] == "Reply with only the number."
    assert events[2]["display_message"] == "Reply with only the number."
    assert events[2]["execution_message"] == "Reply with only the number."
    assert registry.unsubscribe(subscription_id) is True
    assert registry.unsubscribe(subscription_id) is False


@pytest.mark.asyncio
async def test_registry_logs_async_subscriber_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ExecutionRegistry()
    logged_messages: list[str] = []

    async def failing_subscriber(_: dict[str, Any]) -> None:
        raise RuntimeError("subscriber failed")

    def fake_exception(message: str) -> None:
        logged_messages.append(message)

    monkeypatch.setattr(registry_module.logger, "exception", fake_exception)
    registry.subscribe(failing_subscriber)

    registry.register(
        "exec-subscriber-error",
        AgentRunner(agent=Agent(name="writer", patterns=[SuccessfulPattern()])),
        requested_task="manual task",
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert logged_messages == ["Execution registry subscriber failed"]


def test_registry_logs_sync_subscriber_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ExecutionRegistry()
    logged_messages: list[str] = []
    events: list[dict[str, Any]] = []

    def failing_subscriber(_: dict[str, Any]) -> None:
        raise RuntimeError("subscriber failed")

    def fake_exception(message: str) -> None:
        logged_messages.append(message)

    monkeypatch.setattr(registry_module.logger, "exception", fake_exception)
    registry.subscribe(failing_subscriber)
    registry.subscribe(events.append)

    registry.register(
        "exec-sync-subscriber-error",
        AgentRunner(agent=Agent(name="writer", patterns=[SuccessfulPattern()])),
        requested_task="manual task",
    )

    assert logged_messages == ["Execution registry subscriber failed"]
    assert events[0]["type"] == "execution.registered"


@pytest.mark.asyncio
async def test_registry_cancel_running_execution_emits_cancelled_and_cleans_up(
    tmp_path: Path,
) -> None:
    registry = ExecutionRegistry()
    events: list[dict[str, Any]] = []
    registry.subscribe(events.append)
    pattern = BlockingPattern()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[pattern]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    handle = registry.start(
        runner,
        execution_id="exec-cancel",
        task="Block forever",
    )
    await pattern.started.wait()

    assert registry.cancel("exec-cancel", reason="user cancelled") is True
    assert handle.status == ExecutionLifecycleStatus.CANCELLED
    with pytest.raises(asyncio.CancelledError):
        assert handle.task is not None
        await handle.task

    assert registry.get("exec-cancel") is None
    event_types = [event["type"] for event in events]
    assert event_types == ["execution.started", "execution.cancelled"]
    assert events[-1]["handle"]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_registry_cancelled_execution_cannot_resume(tmp_path: Path) -> None:
    registry = ExecutionRegistry()
    pattern = BlockingPattern()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[pattern]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    handle = registry.start(
        runner,
        execution_id="exec-cancel-resume",
        task="Block forever",
    )
    await pattern.started.wait()
    assert registry.cancel("exec-cancel-resume") is True
    with pytest.raises(asyncio.CancelledError):
        assert handle.task is not None
        await handle.task

    assert await registry.resume("exec-cancel-resume") is None


@pytest.mark.asyncio
async def test_registry_unregisters_completed_execution_tasks(tmp_path: Path) -> None:
    registry = ExecutionRegistry()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[SuccessfulPattern()]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    handle = registry.start(
        runner,
        execution_id="exec-complete",
        task="Say done",
    )
    assert handle.task is not None
    result = await handle.task

    assert result["success"] is True
    assert registry.get("exec-complete") is None


@pytest.mark.asyncio
async def test_registry_returns_none_for_unknown_execution_on_message_post() -> None:
    registry = ExecutionRegistry()

    context = await registry.post_user_message("missing", "hello")

    assert context is None


@pytest.mark.asyncio
async def test_registry_pause_returns_false_for_unknown_execution() -> None:
    registry = ExecutionRegistry()

    assert registry.pause("missing") is False


@pytest.mark.asyncio
async def test_registry_cancel_returns_false_for_unknown_execution() -> None:
    registry = ExecutionRegistry()

    assert registry.cancel("missing") is False


@pytest.mark.asyncio
async def test_registry_resume_returns_none_for_unknown_execution() -> None:
    registry = ExecutionRegistry()

    result = await registry.resume("missing")

    assert result is None


@pytest.mark.asyncio
async def test_registry_registers_handle_metadata() -> None:
    runner = AgentRunner(agent=Agent(name="writer", patterns=[SuccessfulPattern()]))
    registry = ExecutionRegistry()

    handle = registry.register(
        "exec-meta",
        runner,
        requested_task="metadata task",
        metadata={"source": "test"},
    )

    assert handle.metadata == {"source": "test"}
    assert handle.requested_task == "metadata task"
    assert handle.status == ExecutionLifecycleStatus.REGISTERED
    assert registry.get("exec-meta") is handle
    assert registry.get_status("exec-meta") == handle.to_dict()
    assert registry.list_statuses() == [handle.to_dict()]
    assert handle.to_dict()["is_resumable"] is False


@pytest.mark.asyncio
async def test_registry_emits_registered_event_for_manual_handle_registration() -> None:
    runner = AgentRunner(agent=Agent(name="writer", patterns=[SuccessfulPattern()]))
    registry = ExecutionRegistry()
    events: list[dict[str, Any]] = []
    registry.subscribe(events.append)

    registry.register(
        "exec-manual",
        runner,
        requested_task="manual task",
        metadata={"source": "manual"},
    )

    assert [event["type"] for event in events] == ["execution.registered"]
    assert events[0]["handle"]["status"] == "registered"


@pytest.mark.asyncio
async def test_registry_cancel_non_running_handle_marks_terminal_and_unregisters() -> (
    None
):
    runner = AgentRunner(agent=Agent(name="writer", patterns=[SuccessfulPattern()]))
    registry = ExecutionRegistry()
    events: list[dict[str, Any]] = []
    registry.subscribe(events.append)

    registry.register(
        "exec-manual-cancel",
        runner,
        requested_task="manual task",
        metadata={"source": "manual"},
    )

    assert registry.cancel("exec-manual-cancel", reason="manual cancel") is True
    assert registry.get("exec-manual-cancel") is None
    assert [event["type"] for event in events] == [
        "execution.registered",
        "execution.cancelled",
    ]
    assert events[-1]["handle"]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_registry_status_queries_return_none_or_empty_for_missing_handles() -> (
    None
):
    registry = ExecutionRegistry()

    assert registry.get_status("missing") is None
    assert registry.list_statuses() == []
