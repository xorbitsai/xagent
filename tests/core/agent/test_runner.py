from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from xagent.core.agent import (
    Agent,
    ContextManager,
    ExecutionContext,
    PatternRuntime,
    TraceEventCallback,
)
from xagent.core.agent.runner import AgentRunner
from xagent.core.agent.runtime import LLMCallInterrupted


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
        self.calls: list[dict[str, Any]] = []

    def get_or_create_workspace(
        self,
        base_dir: str,
        task_id: str,
        allowed_external_dirs: list[str] | None = None,
        scope_segments: tuple[str, ...] = (),
    ) -> FakeWorkspace:
        self.calls.append(
            {
                "base_dir": base_dir,
                "task_id": task_id,
                "allowed_external_dirs": allowed_external_dirs,
            }
        )
        workspace_dir = self.tmp_path / task_id
        return FakeWorkspace(
            id=task_id,
            workspace_dir=workspace_dir,
            input_dir=workspace_dir / "input",
            output_dir=workspace_dir / "output",
            temp_dir=workspace_dir / "temp",
            allowed_external_dirs=[Path(path) for path in allowed_external_dirs or []],
        )


class FakeMemoryManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_or_create_session(
        self,
        *,
        execution_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "execution_id": execution_id,
                "user_id": user_id,
                "session_id": session_id,
            }
        )
        return {
            "session_id": session_id or f"memory-{execution_id}",
            "snapshot": {"summary": f"resume {execution_id}"},
        }


class AsyncMemoryManager(FakeMemoryManager):
    async def get_or_create_session(
        self,
        *,
        execution_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        return super().get_or_create_session(
            execution_id=execution_id,
            user_id=user_id,
            session_id=session_id,
        )


class FakePattern:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return dict(self.result)


class FailingPattern:
    def __init__(self, error: str) -> None:
        self.error = error

    async def run(self, **_: Any) -> dict[str, Any]:
        return {"success": False, "error": self.error}


class LLMInterruptedPattern:
    async def run(self, **_: Any) -> dict[str, Any]:
        raise LLMCallInterrupted("paused during LLM call")


class StatefulPattern:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []

    def load_state(self, state: dict[str, Any]) -> None:
        self.state = state

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "success": True,
            "output": self.state["output"],
            "message_count": len(kwargs["context"].messages),
        }


class InjectingPattern:
    def __init__(self, runner: AgentRunner, execution_id: str) -> None:
        self.runner = runner
        self.execution_id = execution_id

    async def run(self, *, context: ExecutionContext, **_: Any) -> dict[str, Any]:
        injected = await self.runner.inject_user_message(
            self.execution_id,
            "Injected while resumed.",
            request_interrupt=False,
        )
        return {
            "success": True,
            "same_context": injected is context,
            "messages": [message.content for message in context.messages],
        }


class TrackingCallback:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def on_run_start(self, **payload: Any) -> None:
        context = payload["context"]
        self.events.append(("start", context.execution_id))

    async def on_run_end(self, **payload: Any) -> None:
        context = payload["context"]
        self.events.append(("end", context.execution_id))


class FailingUserMessageCallback:
    async def on_user_message_posted(self, **_: Any) -> None:
        raise RuntimeError("trace callback failed")


class RecordingTraceEventTracer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        step_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> str:
        self.events.append(
            {
                "event_type": getattr(event_type, "value", str(event_type)),
                "task_id": task_id,
                "step_id": step_id,
                "data": data or {},
            }
        )
        return str(len(self.events))


class InterruptingPattern:
    def __init__(
        self,
        runner: AgentRunner,
        execution_id: str,
        *,
        before_interrupt_check: Any | None = None,
    ) -> None:
        self.runner = runner
        self.execution_id = execution_id
        self.before_interrupt_check = before_interrupt_check

    async def run(
        self,
        *,
        context: ExecutionContext,
        runtime: PatternRuntime,
        **_: Any,
    ) -> dict[str, Any]:
        if callable(self.before_interrupt_check):
            maybe_result = self.before_interrupt_check()
            if maybe_result is not None:
                await maybe_result
        else:
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
                "error": runtime.interrupt_reason or "interrupted",
            }

        return {"success": True, "output": "continued"}


class TracerCheckpointStore:
    def __init__(self) -> None:
        self.by_execution_id: dict[str, dict[str, Any]] = {}

    async def checkpoint(self, **payload: Any) -> None:
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


@pytest.mark.asyncio
async def test_runner_builds_context_and_invokes_pattern(tmp_path: Path) -> None:
    workspace_manager = FakeWorkspaceManager(tmp_path)
    memory_manager = FakeMemoryManager()
    callback = TrackingCallback()
    pattern = FakePattern({"success": True, "output": "done"})
    agent = Agent(
        name="writer",
        patterns=[pattern],
        tools=["local-tool"],
        llm="fake-llm",
        system_prompt="System prompt",
    )
    runner = AgentRunner(
        agent=agent,
        workspace_manager=workspace_manager,
        memory_manager=memory_manager,
        callbacks=[callback],
        workspace_base_dir=str(tmp_path / "workspaces"),
    )

    result = await runner.run(
        task="Write a summary",
        execution_id="exec-1",
        user_id="user-1",
        session_id="session-1",
        allowed_external_dirs=[str(tmp_path / "kb")],
        extra_tools=["extra-tool"],
        metadata={"source": "test"},
    )

    assert result["success"] is True
    assert result["execution_id"] == "exec-1"
    context = result["context"]
    assert isinstance(context, ExecutionContext)
    assert context.system_prompt == "System prompt"
    assert context.user_id == "user-1"
    assert context.session_id == "session-1"
    assert context.workspace_id == "exec-1"
    assert context.memory_session_id == "session-1"
    assert context.memory_snapshot == {"summary": "resume exec-1"}
    assert context.metadata["task"] == "Write a summary"
    assert context.metadata["source"] == "test"
    assert [message.role for message in context.messages] == ["user", "assistant"]
    assert context.messages[0].content == "Write a summary"
    assert context.messages[1].content == "done"
    assert ContextManager().get_context("exec-1") is context

    pattern_call = pattern.calls[0]
    assert pattern_call["task"] == "Write a summary"
    assert pattern_call["context"] is context
    assert pattern_call["tools"] == ["local-tool", "extra-tool"]
    assert pattern_call["llm"] == "fake-llm"
    assert isinstance(pattern_call["runtime"], PatternRuntime)
    assert workspace_manager.calls[0]["task_id"] == "exec-1"
    assert callback.events == [("start", "exec-1"), ("end", "exec-1")]


@pytest.mark.asyncio
async def test_runner_awaits_async_memory_manager(tmp_path: Path) -> None:
    memory_manager = AsyncMemoryManager()
    pattern = FakePattern({"success": True, "output": "done"})
    agent = Agent(name="writer", patterns=[pattern])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
        memory_manager=memory_manager,
    )

    result = await runner.run(
        task="Write a summary",
        execution_id="exec-async-memory",
        session_id="session-async",
    )

    assert result["success"] is True
    assert result["context"].memory_session_id == "session-async"
    assert result["context"].memory_snapshot == {"summary": "resume exec-async-memory"}


@pytest.mark.asyncio
async def test_runner_tries_multiple_patterns_and_collects_failures(
    tmp_path: Path,
) -> None:
    first = FailingPattern("first failed")
    second = FakePattern({"success": True, "message": "second worked"})
    agent = Agent(name="writer", patterns=[first, second])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(task="Recover", execution_id="exec-2")

    assert result["success"] is True
    assert result["pattern"] == "FakePattern"
    context = result["context"]
    assert [message.content for message in context.messages] == [
        "Recover",
        "second worked",
    ]


@pytest.mark.asyncio
async def test_runner_returns_aggregate_error_when_all_patterns_fail(
    tmp_path: Path,
) -> None:
    agent = Agent(
        name="writer",
        patterns=[FailingPattern("first failed"), FailingPattern("second failed")],
    )
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(task="Impossible", execution_id="exec-3")

    assert result["success"] is False
    assert result["patterns_attempted"] == 2
    assert len(result["pattern_errors"]) == 2
    assert result["context"].messages[0].content == "Impossible"


@pytest.mark.asyncio
async def test_runner_returns_single_pattern_failure_result(tmp_path: Path) -> None:
    agent = Agent(
        name="writer",
        patterns=[
            FakePattern(
                {
                    "success": False,
                    "status": "failed",
                    "failure_reason": "structured_failure",
                    "error": "failed with details",
                }
            )
        ],
    )
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(task="Impossible", execution_id="exec-single-fail")

    assert result["success"] is False
    assert result["status"] == "failed"
    assert result["failure_reason"] == "structured_failure"
    assert result["error"] == "failed with details"
    assert "pattern_errors" not in result


@pytest.mark.asyncio
async def test_runner_does_not_add_empty_user_message_for_missing_task(
    tmp_path: Path,
) -> None:
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(task=None, execution_id="exec-empty-task")

    assert result["success"] is True
    assert result["context"].messages == []


@pytest.mark.asyncio
async def test_runner_stops_on_llm_call_interrupt(tmp_path: Path) -> None:
    fallback = FakePattern({"success": True, "output": "should not run"})
    agent = Agent(name="writer", patterns=[LLMInterruptedPattern(), fallback])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(task="Pause me", execution_id="exec-llm-interrupt")

    assert result["success"] is False
    assert result["status"] == "interrupted"
    assert result["error"] == "paused during LLM call"
    assert result["pattern"] == "LLMInterruptedPattern"
    assert fallback.calls == []


@pytest.mark.asyncio
async def test_runner_restores_context_and_pattern_from_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_context = ExecutionContext(execution_id="exec-resume")
    checkpoint_context.add_user_message("Original task")
    checkpoint = {
        "context": checkpoint_context.to_dict(),
        "pattern": "StatefulPattern",
        "pattern_state": {"output": "restored"},
    }
    pattern = StatefulPattern()
    agent = Agent(name="writer", patterns=[pattern])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(
        task="Should not be appended",
        execution_id="exec-resume",
        checkpoint=checkpoint,
    )

    assert result["success"] is True
    assert result["output"] == "restored"
    assert result["message_count"] == 1
    assert pattern.state == {"output": "restored"}
    assert [message.content for message in result["context"].messages] == [
        "Original task",
        "restored",
    ]


@pytest.mark.asyncio
async def test_runner_registers_restored_context_for_live_message_injection(
    tmp_path: Path,
) -> None:
    checkpoint_context = ExecutionContext(execution_id="exec-restore-inject")
    checkpoint_context.add_user_message("Original task")
    checkpoint = {
        "context": checkpoint_context.to_dict(),
        "pattern": "InjectingPattern",
        "pattern_state": {},
    }
    agent = Agent(name="writer", patterns=[])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    agent.patterns = [InjectingPattern(runner, "exec-restore-inject")]

    result = await runner.run(
        task=None,
        execution_id="exec-restore-inject",
        checkpoint=checkpoint,
    )

    assert result["success"] is True
    assert result["same_context"] is True
    assert result["messages"] == ["Original task", "Injected while resumed."]


@pytest.mark.asyncio
async def test_runner_pause_requests_interrupt_for_active_execution(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    agent = Agent(name="writer", patterns=[])
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    agent.patterns = [InterruptingPattern(runner, "exec-pause")]

    result = await runner.run(task="Calculate 6*7", execution_id="exec-pause")

    assert result["success"] is False
    assert result["status"] == "interrupted"
    assert tracer.by_execution_id["exec-pause"]["label"] == "interrupted"
    assert (
        tracer.by_execution_id["exec-pause"]["metadata"]["safe_point"]
        == "during_pattern"
    )


@pytest.mark.asyncio
async def test_runner_inject_user_message_updates_live_context_and_requests_interrupt(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    agent = Agent(name="writer", patterns=[])
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    agent.patterns = [
        InterruptingPattern(
            runner,
            "exec-inject",
            before_interrupt_check=lambda: runner.inject_user_message(
                "exec-inject",
                "Use metric units.",
                reason="new user message",
            ),
        )
    ]

    result = await runner.run(task="Calculate 6*7", execution_id="exec-inject")
    context = result["context"]

    assert result["success"] is False
    assert result["status"] == "interrupted"
    user_messages = [msg.content for msg in context.messages if msg.role == "user"]
    assert user_messages == ["Calculate 6*7", "Use metric units."]
    checkpoint_messages = tracer.by_execution_id["exec-inject"]["context"]["messages"]
    assert any(
        message["role"] == "user" and message["content"] == "Use metric units."
        for message in checkpoint_messages
    )


@pytest.mark.asyncio
async def test_runner_resume_restores_from_latest_checkpoint_after_restart(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    execution_id = "exec-restart"
    first_agent = Agent(name="writer", patterns=[])
    first_runner = AgentRunner(
        agent=first_agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    first_agent.patterns = [InterruptingPattern(first_runner, execution_id)]

    interrupted = await first_runner.run(
        task="Calculate 6*7",
        execution_id=execution_id,
    )

    assert interrupted["status"] == "interrupted"
    await first_runner.inject_user_message(
        execution_id,
        "Reply with only the number.",
        request_interrupt=False,
    )

    agent = Agent(
        name="writer",
        patterns=[FakePattern({"success": True, "response": "42"})],
    )
    resumed_runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    resumed = await resumed_runner.resume(execution_id)

    assert resumed["success"] is True
    assert resumed["response"] == "42"
    resumed_contents = [message.content for message in resumed["context"].messages]
    assert "Reply with only the number." in resumed_contents
    assert resumed_contents.index(
        "Reply with only the number."
    ) < resumed_contents.index("42")


@pytest.mark.asyncio
async def test_runner_inject_user_message_with_files_dispatches_trace_callback(
    tmp_path: Path,
) -> None:
    """End-to-end coverage of the continuation chip path: a websocket-style
    ``post_user_message`` call with attachments must (a) attach the files to
    the new Message so they survive checkpoints and (b) fire the trace
    callback so the chip is broadcast live (instead of only appearing after
    a page reload via historical replay)."""
    tracer = RecordingTraceEventTracer()
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        callbacks=[TraceEventCallback()],
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    await runner.run(task="Original task", execution_id="exec-cont-files")

    files = [
        {
            "file_id": "fid-cont",
            "name": "follow-up.pdf",
            "size": 2048,
            "type": "application/pdf",
        }
    ]
    context = await runner.post_user_message(
        "exec-cont-files",
        "Use the attached PDF.",
        request_interrupt=False,
        files=files,
    )

    assert context is not None
    new_user_message = next(
        msg for msg in reversed(context.messages) if msg.role == "user"
    )
    assert new_user_message.metadata.get("files") == files
    turn_id = new_user_message.metadata.get("turn_id")
    assert isinstance(turn_id, str) and turn_id

    user_message_events = [
        event
        for event in tracer.events
        if event["event_type"] == "task_start_message"
        and event["data"].get("message") == "Use the attached PDF."
    ]
    assert len(user_message_events) == 1
    assert user_message_events[0]["data"]["turn_id"] == turn_id
    assert user_message_events[0]["data"]["files"] == files
    assert user_message_events[0]["data"]["attachments"] == files


@pytest.mark.asyncio
async def test_runner_post_user_message_alias_matches_inject_behavior(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    checkpoint_context = ExecutionContext(execution_id="exec-alias")
    checkpoint_context.add_user_message("Original task")
    await tracer.checkpoint(
        type="checkpoint",
        execution_id="exec-alias",
        pattern="FakePattern",
        label="before_llm",
        status="interrupted",
        context=checkpoint_context.to_dict(),
        pattern_state={},
        metadata={},
    )

    context = await runner.post_user_message(
        "exec-alias",
        "Follow-up from user.",
        request_interrupt=False,
    )

    assert context is not None
    user_messages = [
        message.content for message in context.messages if message.role == "user"
    ]
    assert user_messages == ["Original task", "Follow-up from user."]


@pytest.mark.asyncio
async def test_runner_post_user_message_deduplicates_explicit_turn_id_after_failure(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    execution_id = "exec-idempotent-message"
    checkpoint_context = ExecutionContext(execution_id=execution_id)
    checkpoint_context.add_user_message("Original task")
    await tracer.checkpoint(
        type="checkpoint",
        execution_id=execution_id,
        pattern="FakePattern",
        label="waiting_for_user",
        status="waiting_for_user",
        context=checkpoint_context.to_dict(),
        pattern_state={},
        metadata={},
    )
    failing_runner = AgentRunner(
        agent=Agent(name="writer", patterns=[FakePattern({"success": True})]),
        tracer=tracer,
        callbacks=[FailingUserMessageCallback()],
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    with pytest.raises(RuntimeError, match="trace callback failed"):
        await failing_runner.post_user_message(
            execution_id,
            "Choose B",
            turn_id="a2a:42:msg-1",
            request_interrupt=False,
        )

    failing_runner.context_manager.remove_context(execution_id)
    retry_runner = AgentRunner(
        agent=Agent(name="writer", patterns=[FakePattern({"success": True})]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = await retry_runner.post_user_message(
        execution_id,
        "Choose B",
        turn_id="a2a:42:msg-1",
        request_interrupt=False,
    )

    assert context is not None
    retried_messages = [
        message
        for message in context.messages
        if message.role == "user" and message.metadata.get("turn_id") == "a2a:42:msg-1"
    ]
    assert len(retried_messages) == 1
    assert retried_messages[0].content == "Choose B"


@pytest.mark.asyncio
async def test_runner_rejects_reused_turn_id_with_different_content(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    execution_id = "exec-conflicting-message"
    checkpoint_context = ExecutionContext(execution_id=execution_id)
    checkpoint_context.add_user_message(
        "Choose A",
        metadata={"turn_id": "a2a:42:msg-1"},
    )
    await tracer.checkpoint(
        type="checkpoint",
        execution_id=execution_id,
        pattern="FakePattern",
        label="waiting_for_user",
        status="waiting_for_user",
        context=checkpoint_context.to_dict(),
        pattern_state={},
        metadata={},
    )
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[FakePattern({"success": True})]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    with pytest.raises(ValueError, match="different user message"):
        await runner.post_user_message(
            execution_id,
            "Choose B",
            turn_id="a2a:42:msg-1",
            request_interrupt=False,
        )


@pytest.mark.asyncio
async def test_runner_post_user_message_preserves_display_and_execution_contract(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    checkpoint_context = ExecutionContext(execution_id="exec-display-contract")
    checkpoint_context.add_user_message("Original task")
    await tracer.checkpoint(
        type="checkpoint",
        execution_id="exec-display-contract",
        pattern="FakePattern",
        label="before_llm",
        status="interrupted",
        context=checkpoint_context.to_dict(),
        pattern_state={},
        metadata={},
    )

    execution_message = "Read file\n\n## UPLOADED FILES\nfile_id=file-123"
    files = [{"file_id": "file-123", "name": "notes.txt"}]
    context = await runner.post_user_message(
        "exec-display-contract",
        execution_message=execution_message,
        display_message="Read file",
        files=files,
        turn_id="client-turn-123",
        request_interrupt=False,
    )

    assert context is not None
    latest_user = [message for message in context.messages if message.role == "user"][
        -1
    ]
    assert latest_user.content == execution_message
    assert latest_user.metadata["display_message"] == "Read file"
    assert latest_user.metadata["files"] == files
    turn_id = latest_user.metadata.get("turn_id")
    assert turn_id == "client-turn-123"

    checkpoint_messages = tracer.by_execution_id["exec-display-contract"]["context"][
        "messages"
    ]
    latest_checkpoint_user = [
        message for message in checkpoint_messages if message["role"] == "user"
    ][-1]
    assert latest_checkpoint_user["content"] == execution_message
    assert latest_checkpoint_user["metadata"]["display_message"] == "Read file"
    assert latest_checkpoint_user["metadata"]["files"] == files
    assert latest_checkpoint_user["metadata"]["turn_id"] == turn_id


@pytest.mark.asyncio
async def test_runner_initial_user_message_preserves_display_metadata(
    tmp_path: Path,
) -> None:
    tracer = RecordingTraceEventTracer()
    agent = Agent(
        name="writer",
        patterns=[FakePattern({"success": True, "response": "Done"})],
    )
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        callbacks=[TraceEventCallback()],
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    execution_message = "Read file\n\n## UPLOADED FILES\nfile_id=file-123"
    files = [{"file_id": "file-123", "name": "notes.txt"}]
    result = await runner.run(
        task=execution_message,
        execution_id="exec-initial-display",
        metadata={"request_context": {"display_message": "Read file", "files": files}},
    )

    first_user = next(
        message for message in result["context"].messages if message.role == "user"
    )
    assert first_user.content == execution_message
    assert first_user.metadata["display_message"] == "Read file"
    assert first_user.metadata["files"] == files
    turn_id = first_user.metadata.get("turn_id")
    assert isinstance(turn_id, str) and turn_id
    user_event = next(
        event for event in tracer.events if event["event_type"] == "task_start_message"
    )
    assert user_event["data"]["message"] == "Read file"
    assert user_event["data"]["turn_id"] == turn_id


@pytest.mark.asyncio
async def test_runner_post_user_message_rejects_execution_without_display(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    checkpoint_context = ExecutionContext(execution_id="exec-display-required")
    checkpoint_context.add_user_message("Original task")
    await tracer.checkpoint(
        type="checkpoint",
        execution_id="exec-display-required",
        pattern="FakePattern",
        label="before_llm",
        status="interrupted",
        context=checkpoint_context.to_dict(),
        pattern_state={},
        metadata={},
    )

    with pytest.raises(ValueError, match="requires display_message"):
        await runner.post_user_message(
            "exec-display-required",
            execution_message="Read file\n\n## UPLOADED FILES\nfile_id=file-123",
            request_interrupt=False,
        )


@pytest.mark.asyncio
async def test_trace_callback_does_not_emit_completion_for_interrupted_run(
    tmp_path: Path,
) -> None:
    tracer = RecordingTraceEventTracer()
    agent = Agent(
        name="paused",
        patterns=[
            FakePattern(
                {
                    "success": False,
                    "status": "interrupted",
                    "error": "Paused by user.",
                }
            )
        ],
    )
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        callbacks=[TraceEventCallback()],
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(task="Pause this", execution_id="exec-paused")

    assert result["status"] == "interrupted"
    event_types = [event["event_type"] for event in tracer.events]
    assert event_types == ["task_start_message"]
    assert "task_end_general" not in event_types


@pytest.mark.asyncio
async def test_trace_callback_unwraps_final_answer_and_omits_success_context(
    tmp_path: Path,
) -> None:
    tracer = RecordingTraceEventTracer()
    agent = Agent(
        name="writer",
        patterns=[
            FakePattern(
                {
                    "success": True,
                    "output": (
                        '```json\n{"action":"final_answer",'
                        '"action_input":"Done cleanly."}\n```'
                    ),
                    "message": (
                        '```json\n{"action":"final_answer",'
                        '"action_input":"Done cleanly."}\n```'
                    ),
                }
            )
        ],
    )
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        callbacks=[TraceEventCallback()],
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(task="Finish", execution_id="exec-success")

    assert result["output"] == "Done cleanly."
    assert result["message"] == "Done cleanly."
    ai_event = next(
        event for event in tracer.events if event["event_type"] == "task_end_message"
    )
    assert ai_event["data"]["content"] == "Done cleanly."
    assert "context" not in ai_event["data"]


class _FakeLLM:
    def __init__(self, context_window: Any) -> None:
        self.context_window = context_window


def _threshold_runner(context_window: Any) -> AgentRunner:
    agent = Agent(name="t", patterns=[FakePattern({})], llm=_FakeLLM(context_window))
    return AgentRunner(agent=agent)


def test_resolve_compact_threshold_uses_window_ratio(monkeypatch) -> None:
    monkeypatch.delenv("XAGENT_COMPACT_THRESHOLD_RATIO", raising=False)
    # 128000 * 0.75
    assert _threshold_runner(128000)._resolve_compact_threshold() == 96000


def test_resolve_compact_threshold_respects_ratio_env(monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_COMPACT_THRESHOLD_RATIO", "0.8")
    assert _threshold_runner(200000)._resolve_compact_threshold() == 160000


@pytest.mark.parametrize("window", [None, 0, -1, "128000"])
def test_resolve_compact_threshold_falls_back_to_default(monkeypatch, window) -> None:
    monkeypatch.delenv("XAGENT_COMPACT_THRESHOLD_DEFAULT", raising=False)
    # None / non-positive / non-int all fall back to the global default.
    assert _threshold_runner(window)._resolve_compact_threshold() == 32000


def test_resolve_compact_threshold_default_env_override(monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_COMPACT_THRESHOLD_DEFAULT", "50000")
    assert _threshold_runner(None)._resolve_compact_threshold() == 50000


def test_resolve_compact_threshold_missing_llm() -> None:
    agent = Agent(name="t", patterns=[FakePattern({})], llm=None)
    assert AgentRunner(agent=agent)._resolve_compact_threshold() == 32000
