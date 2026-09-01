from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from xagent.core.agent import (
    Agent,
    ContextManager,
    ExecutionContext,
    PatternRuntime,
    TraceEventCallback,
)
from xagent.core.agent.attachments import build_image_context_references
from xagent.core.agent.checkpoint import (
    CheckpointAccessRefusedError,
    CheckpointCorruptError,
    CheckpointUnavailableError,
)
from xagent.core.agent.language import (
    OUTPUT_LANGUAGE_METADATA_KEY,
    OUTPUT_LANGUAGE_SOURCE_METADATA_KEY,
    reset_output_language_to_request_context,
)
from xagent.core.agent.runner import AgentRunner
from xagent.core.agent.runtime import LLMCallInterrupted
from xagent.core.task_runtime import PREFERRED_INPUT_MODALITIES_METADATA_KEY


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


class StatusAwareTeardownTool:
    def __init__(self) -> None:
        self.teardown_calls: list[tuple[str | None, str | None]] = []

    async def setup(self, task_id: str | None = None) -> None:
        return None

    async def teardown(
        self,
        task_id: str | None = None,
        execution_status: str | None = None,
    ) -> None:
        self.teardown_calls.append((task_id, execution_status))


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


class EmptyCanonicalCheckpointStore:
    def __init__(self) -> None:
        self.legacy_reads = 0

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        del execution_id
        return None

    def get_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        del execution_id
        self.legacy_reads += 1
        raise AssertionError("canonical empty result must end checkpoint lookup")


@pytest.mark.asyncio
async def test_runner_treats_canonical_empty_checkpoint_as_authoritative() -> None:
    checkpoint_store = EmptyCanonicalCheckpointStore()
    runner = AgentRunner(
        agent=Agent(name="checkpoint-reader", patterns=[], llm=None),
        tracer=checkpoint_store,
    )

    context = await runner.inject_user_message(
        "missing-execution",
        "Continue",
        request_interrupt=False,
    )

    assert context is None
    assert checkpoint_store.legacy_reads == 0


class ContextlessCheckpointStore:
    """Returns a recognized checkpoint payload that carries no context.

    Every production checkpoint writer persists a ``context`` dict; a
    stored payload without one is malformed. The runner must classify it
    as corrupt rather than silently building fresh state on resume."""

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any]:
        return {"type": "checkpoint", "execution_id": execution_id}


class UnavailableCheckpointStore:
    """Every read fails -- distinct from ``EmptyCanonicalCheckpointStore``,
    whose ``None`` is an authoritative "no checkpoint" the runner may act
    on by building fresh state."""

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any]:
        del execution_id
        raise CheckpointUnavailableError("checkpoint store unavailable")


class FlakyOnceCheckpointStore:
    """Fails the first read, then behaves like a normal checkpoint store.

    Models a transient failure during ``inject_user_message``'s baseline
    read: the retry after the failure must actually persist the message,
    not find a "ghost" confirmation left behind by the rejected attempt.
    """

    def __init__(self) -> None:
        self.by_execution_id: dict[str, dict[str, Any]] = {}
        self.load_calls = 0

    async def checkpoint(self, **payload: Any) -> None:
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        self.load_calls += 1
        if self.load_calls == 1:
            raise CheckpointUnavailableError("transient")
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


@pytest.mark.asyncio
async def test_run_resume_does_not_build_fresh_context_on_unavailable() -> None:
    runner = AgentRunner(
        agent=Agent(name="checkpoint-reader", patterns=[], llm=None),
        tracer=UnavailableCheckpointStore(),
    )
    build_context_calls: list[Any] = []
    original_build_context = runner._build_context

    async def spy_build_context(*args: Any, **kwargs: Any) -> Any:
        build_context_calls.append((args, kwargs))
        return await original_build_context(*args, **kwargs)

    runner._build_context = spy_build_context  # type: ignore[method-assign]

    with pytest.raises(CheckpointUnavailableError):
        await runner.run(
            task=None,
            execution_id="exec-resume-unavailable",
            resume=True,
        )

    assert build_context_calls == []
    assert runner.context_manager.get_context("exec-resume-unavailable") is None


@pytest.mark.asyncio
async def test_inject_user_message_propagates_unavailable() -> None:
    """Distinct from the canonical-empty-checkpoint case above: a read
    failure must not be swallowed into the same ``None`` "no checkpoint"
    result -- the caller cannot tell a real failure from genuine absence."""
    runner = AgentRunner(
        agent=Agent(name="checkpoint-reader", patterns=[], llm=None),
        tracer=UnavailableCheckpointStore(),
    )

    with pytest.raises(CheckpointUnavailableError):
        await runner.inject_user_message(
            "missing-execution-unavailable",
            "Continue",
            request_interrupt=False,
        )


class RunProvenanceUnavailableCheckpointStore:
    """Every read refuses with ``run_provenance_unavailable``.

    Models the merge-baseline read inside ``inject_user_message`` hitting a
    checkpoint pointer row this reader cannot verify. Unlike
    ``UnavailableCheckpointStore``, this refusal is downgraded to a ``None``
    baseline at that one call site instead of propagating: the context is
    already live by the time this read runs, so injection does not depend
    on it succeeding.
    """

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any]:
        del execution_id
        raise CheckpointAccessRefusedError(
            "checkpoint pointer row is missing its run-partition field",
            reason="run_provenance_unavailable",
        )


@pytest.mark.asyncio
async def test_inject_user_message_baseline_refusal_still_injects() -> None:
    """A live context with no cached runtime checkpoint (the common
    already-paused, process-still-warm case) takes the merge-baseline read
    at the bottom of ``inject_user_message``, not the cold-start read at the
    top. A ``run_provenance_unavailable`` refusal there must not turn an
    otherwise-successful injection into a rejected one -- only the
    cold-start read (see test_inject_user_message_propagates_unavailable)
    is allowed to fail the whole call."""
    runner = AgentRunner(
        agent=Agent(name="checkpoint-reader", patterns=[], llm=None),
        tracer=RunProvenanceUnavailableCheckpointStore(),
    )
    context = ExecutionContext(execution_id="exec-live-baseline-refused")
    runner.context_manager.set_context(context)

    result = await runner.inject_user_message(
        "exec-live-baseline-refused",
        "Continue",
        request_interrupt=False,
    )

    assert result is context
    assert len(context.messages) == 1
    assert context.messages[0].content == "Continue"


@pytest.mark.asyncio
async def test_inject_rejection_leaves_no_dedupe_residue() -> None:
    tracer = FlakyOnceCheckpointStore()
    runner = AgentRunner(
        agent=Agent(name="checkpoint-reader", patterns=[], llm=None),
        tracer=tracer,
    )
    context = ExecutionContext(execution_id="exec-residue")
    runner.context_manager.set_context(context)

    with pytest.raises(CheckpointUnavailableError):
        await runner.inject_user_message(
            "exec-residue",
            "Continue",
            turn_id="turn-1",
            request_interrupt=False,
        )

    # The rejected attempt must not have mutated the in-memory context.
    assert context.messages == []

    # A retry must actually persist the message -- it must not find a
    # ghost confirmation left behind by the failed attempt above.
    result = await runner.inject_user_message(
        "exec-residue",
        "Continue",
        turn_id="turn-1",
        request_interrupt=False,
    )

    assert result is context
    assert len(context.messages) == 1
    assert tracer.by_execution_id["exec-residue"]["context"]["messages"]


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
async def test_runner_inserts_synthetic_user_turn_before_a_leading_assistant_initial_message(
    tmp_path: Path,
) -> None:
    # The marketplace Hire flow seeds a persona greeting as a task's very
    # first persisted message (see seed_assistant_message in
    # src/xagent/web/api/chat.py) - initial_messages then starts with role
    # "assistant" and no prior user turn. Anthropic's Messages API (and
    # every claude_compatible provider routed through it) rejects a request
    # whose first message isn't role "user", so the runner must correct
    # this before it's ever replayed into context.
    workspace_manager = FakeWorkspaceManager(tmp_path)
    memory_manager = FakeMemoryManager()
    pattern = FakePattern({"success": True, "output": "done"})
    agent = Agent(name="writer", patterns=[pattern], tools=[], llm="fake-llm")
    runner = AgentRunner(
        agent=agent,
        workspace_manager=workspace_manager,
        memory_manager=memory_manager,
        workspace_base_dir=str(tmp_path / "workspaces"),
    )

    result = await runner.run(
        task="Let's get started",
        execution_id="exec-seed",
        user_id="user-1",
        initial_messages=[
            {
                "role": "assistant",
                "content": "Hi - I'm Maya, your Social Media Content Manager.",
            }
        ],
    )

    context = result["context"]
    assert [message.role for message in context.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert context.messages[0].content == "(conversation start)"
    assert context.messages[0].metadata.get("_xagent_synthetic") == "leading_user_turn"
    assert (
        context.messages[1].content
        == "Hi - I'm Maya, your Social Media Content Manager."
    )
    assert context.messages[2].content == "Let's get started"


@pytest.mark.asyncio
async def test_runner_does_not_insert_synthetic_turn_for_user_first_initial_messages(
    tmp_path: Path,
) -> None:
    workspace_manager = FakeWorkspaceManager(tmp_path)
    memory_manager = FakeMemoryManager()
    pattern = FakePattern({"success": True, "output": "done"})
    agent = Agent(name="writer", patterns=[pattern], tools=[], llm="fake-llm")
    runner = AgentRunner(
        agent=agent,
        workspace_manager=workspace_manager,
        memory_manager=memory_manager,
        workspace_base_dir=str(tmp_path / "workspaces"),
    )

    result = await runner.run(
        task="Follow up",
        execution_id="exec-normal",
        user_id="user-1",
        initial_messages=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello, how can I help?"},
        ],
    )

    context = result["context"]
    assert [message.role for message in context.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert context.messages[0].content == "Hi"


@pytest.mark.asyncio
async def test_runner_passes_waiting_status_to_tool_teardown(tmp_path: Path) -> None:
    tool = StatusAwareTeardownTool()
    agent = Agent(
        name="interactive",
        patterns=[
            FakePattern(
                {
                    "success": False,
                    "status": "waiting_for_user",
                    "message": "Provide the missing value.",
                }
            )
        ],
    )
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(
        task="Run an interactive tool",
        execution_id="interaction-task",
        extra_tools=[tool],
    )

    assert result["status"] == "waiting_for_user"
    assert tool.teardown_calls == [("interaction-task", "waiting_for_user")]


class LiveStepTasksPattern:
    """A pattern that reports a waiting_for_user exit while its
    has_live_step_tasks() predicate is under test control, to pin the
    runner-side guard independently of any real pattern implementation."""

    def __init__(self, *, live_step_tasks: bool) -> None:
        self._live_step_tasks = live_step_tasks

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "success": False,
            "status": "waiting_for_user",
            "message": "Pick one.",
            "clarification_draft": {"source": "test"},
        }

    def has_live_step_tasks(self) -> bool:
        return self._live_step_tasks


@pytest.mark.asyncio
async def test_runner_raises_when_waiting_exit_still_has_live_step_tasks(
    tmp_path: Path,
) -> None:
    agent = Agent(
        name="interactive",
        patterns=[LiveStepTasksPattern(live_step_tasks=True)],
    )
    runner = AgentRunner(agent=agent, workspace_manager=FakeWorkspaceManager(tmp_path))

    with pytest.raises(AssertionError, match="live step tasks"):
        await runner.run(task="Run an interactive tool", execution_id="live-tasks-task")


@pytest.mark.asyncio
async def test_runner_allows_waiting_exit_once_step_tasks_are_clear(
    tmp_path: Path,
) -> None:
    agent = Agent(
        name="interactive",
        patterns=[LiveStepTasksPattern(live_step_tasks=False)],
    )
    runner = AgentRunner(agent=agent, workspace_manager=FakeWorkspaceManager(tmp_path))

    result = await runner.run(
        task="Run an interactive tool", execution_id="no-live-tasks-task"
    )

    assert result["status"] == "waiting_for_user"


@pytest.mark.asyncio
async def test_runner_passes_failed_status_when_tool_setup_raises(
    tmp_path: Path,
) -> None:
    class SetupFailingTool(StatusAwareTeardownTool):
        async def setup(self, task_id: str | None = None) -> None:
            raise RuntimeError("setup failed")

    tool = SetupFailingTool()
    runner = AgentRunner(
        agent=Agent(
            name="setup-failure",
            patterns=[FakePattern({"success": True, "output": "unused"})],
        ),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    with pytest.raises(RuntimeError, match="setup failed"):
        await runner.run(
            task="Initialize tools",
            execution_id="setup-failure-task",
            extra_tools=[tool],
        )

    assert tool.teardown_calls == [("setup-failure-task", "failed")]


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
async def test_initial_messages_replay_tool_pairs(tmp_path: Path) -> None:
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    initial_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_name": "read_file",
            "tool_call_id": "call-1",
            "raw_result": {"output": "file contents"},
        },
    ]

    result = await runner.run(
        task=None,
        execution_id="exec-replay",
        initial_messages=initial_messages,
    )

    assert result["success"] is True
    messages = result["context"].messages
    # initial_messages[0] is role "assistant", so AgentRunner.run prepends a
    # synthetic leading user turn (Anthropic's Messages API rejects a request
    # whose first message isn't role "user") before replaying the
    # assistant/tool pair.
    assert [message.role for message in messages] == ["user", "assistant", "tool"]

    synthetic_message = messages[0]
    assert synthetic_message.content == "(conversation start)"
    assert synthetic_message.metadata["_xagent_synthetic"] == "leading_user_turn"

    assistant_message = messages[1]
    assert assistant_message.content == ""
    assert assistant_message.tool_calls == initial_messages[0]["tool_calls"]

    tool_message = messages[2]
    assert tool_message.content == "Tool read_file returned: file contents"
    assert tool_message.tool_call_id == "call-1"
    assert tool_message.metadata["raw_result"] == {"output": "file contents"}
    assert tool_message.metadata["tool_name"] == "read_file"
    # The synthetic turn is prepended, not interleaved, so tool-call pairing
    # is undisturbed: every "tool" message is still immediately preceded by
    # the assistant message declaring its tool_call_id.
    for index, message in enumerate(messages):
        if message.role == "tool":
            assert messages[index - 1].role == "assistant"
            assert message.tool_call_id in {
                call["id"] for call in (messages[index - 1].tool_calls or [])
            }


@pytest.mark.asyncio
async def test_initial_assistant_with_empty_content_and_tool_calls_survives(
    tmp_path: Path,
) -> None:
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    initial_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": "{}"},
                }
            ],
        },
    ]

    result = await runner.run(
        task=None,
        execution_id="exec-empty-content-tool-calls",
        initial_messages=initial_messages,
    )

    assert result["success"] is True
    messages = result["context"].messages
    # The original defect this test guards: an assistant message with
    # content="" and non-empty tool_calls must not be dropped by the replay
    # loop's "is there anything to keep" check. The leading synthetic user
    # turn (added because initial_messages[0] is role "assistant") must not
    # mask that — the assistant message must still be present right after it.
    assert [message.role for message in messages] == ["user", "assistant"]

    synthetic_message = messages[0]
    assert synthetic_message.content == "(conversation start)"
    assert synthetic_message.metadata["_xagent_synthetic"] == "leading_user_turn"

    assistant_message = messages[1]
    assert assistant_message.content == ""
    assert assistant_message.tool_calls == initial_messages[0]["tool_calls"]


@pytest.mark.asyncio
async def test_initial_messages_starting_with_user_no_synthetic_turn(
    tmp_path: Path,
) -> None:
    """A realistically-shaped reconstruction that already starts with a user
    transcript message, followed by an assistant/tool pair, must NOT trigger
    the synthetic leading-user-turn correction: it is only needed when the
    replay would otherwise start with role "assistant".
    """
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    initial_messages = [
        {"role": "user", "content": "Please read the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-3",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_name": "read_file",
            "tool_call_id": "call-3",
            "raw_result": {"output": "file contents"},
        },
    ]

    result = await runner.run(
        task=None,
        execution_id="exec-replay-user-first",
        initial_messages=initial_messages,
    )

    assert result["success"] is True
    messages = result["context"].messages
    assert [message.role for message in messages] == ["user", "assistant", "tool"]
    assert messages[0].content == "Please read the file"
    assert not any(
        message.metadata.get("_xagent_synthetic") == "leading_user_turn"
        for message in messages
    )


@pytest.mark.asyncio
async def test_initial_messages_plain_roles_unchanged(tmp_path: Path) -> None:
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    initial_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello there"},
        {"role": "user", "content": ""},  # dropped: no content/context_refs
    ]

    result = await runner.run(
        task=None,
        execution_id="exec-plain-initial",
        initial_messages=initial_messages,
    )

    assert result["success"] is True
    messages = result["context"].messages
    assert [(message.role, message.content) for message in messages] == [
        ("system", "You are helpful."),
        ("user", "Hello there"),
    ]


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
    checkpoint_context.metadata[PREFERRED_INPUT_MODALITIES_METADATA_KEY] = ["text"]
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
        metadata={PREFERRED_INPUT_MODALITIES_METADATA_KEY: ["image"]},
    )

    assert result["success"] is True
    assert result["output"] == "restored"
    assert result["message_count"] == 1
    assert pattern.state == {"output": "restored"}
    assert result["context"].metadata[PREFERRED_INPUT_MODALITIES_METADATA_KEY] == [
        "image"
    ]
    assert [message.content for message in result["context"].messages] == [
        "Original task",
        "restored",
    ]


@pytest.mark.asyncio
async def test_runner_clears_checkpointed_modality_preference(
    tmp_path: Path,
) -> None:
    checkpoint_context = ExecutionContext(execution_id="exec-clear-modality")
    checkpoint_context.add_user_message("Original task")
    checkpoint_context.metadata[PREFERRED_INPUT_MODALITIES_METADATA_KEY] = ["image"]
    checkpoint = {
        "context": checkpoint_context.to_dict(),
        "pattern": "StatefulPattern",
        "pattern_state": {"output": "restored"},
    }
    pattern = StatefulPattern()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[pattern]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(
        task="Should not be appended",
        execution_id="exec-clear-modality",
        checkpoint=checkpoint,
        metadata={PREFERRED_INPUT_MODALITIES_METADATA_KEY: []},
    )

    assert PREFERRED_INPUT_MODALITIES_METADATA_KEY not in result["context"].metadata


def test_merge_context_metadata_restored_clears_absent_modality_key(
    tmp_path: Path,
) -> None:
    """Restored merges clear the modality key just like fresh-context merges."""

    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[StatefulPattern()]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id="exec-merge-absent")
    context.metadata[PREFERRED_INPUT_MODALITIES_METADATA_KEY] = ["image"]
    context.metadata["execution_type"] = "checkpointed"

    runner._merge_context_metadata(context, {}, restored=True)

    assert PREFERRED_INPUT_MODALITIES_METADATA_KEY not in context.metadata
    assert context.metadata["execution_type"] == "checkpointed"


@pytest.mark.asyncio
async def test_runner_empty_resume_metadata_preserves_non_modality_metadata(
    tmp_path: Path,
) -> None:
    """Resume metadata is authoritative for the modality key only.

    Every other checkpointed metadata entry survives an empty resume metadata
    mapping; the modality preference is cleared because the current run did not
    declare one.
    """

    checkpoint_context = ExecutionContext(execution_id="exec-preserve-metadata")
    checkpoint_context.add_user_message("Original task")
    checkpoint_context.metadata.update(
        {
            PREFERRED_INPUT_MODALITIES_METADATA_KEY: ["image"],
            "execution_type": "checkpointed",
        }
    )
    checkpoint = {
        "context": checkpoint_context.to_dict(),
        "pattern": "StatefulPattern",
        "pattern_state": {"output": "restored"},
    }
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[StatefulPattern()]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(
        task="Should not be appended",
        execution_id="exec-preserve-metadata",
        checkpoint=checkpoint,
        metadata={},
    )

    assert PREFERRED_INPUT_MODALITIES_METADATA_KEY not in result["context"].metadata
    assert result["context"].metadata["execution_type"] == "checkpointed"


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
    failing_runner.pause = MagicMock(return_value=True)

    accepted = await failing_runner.post_user_message(
        execution_id,
        "Choose B",
        turn_id="a2a:42:msg-1",
        request_interrupt=True,
    )

    assert accepted is not None
    failing_runner.pause.assert_called_once_with(
        execution_id,
        reason="new user message",
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
async def test_runner_attaches_uploaded_image_refs_to_initial_user_message(
    tmp_path: Path,
) -> None:
    agent = Agent(
        name="vision",
        patterns=[FakePattern({"success": True, "response": "Done"})],
    )
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    references = build_image_context_references(
        [{"file_id": "image-123", "name": "diagram.png", "type": "image/png"}]
    )

    result = await runner.run(
        task="What is shown?",
        execution_id="exec-initial-image",
        task_context_refs=references,
    )

    first_user = next(
        message for message in result["context"].messages if message.role == "user"
    )
    assert first_user.context_refs == references


@pytest.mark.asyncio
async def test_runner_attaches_uploaded_image_refs_to_injected_user_message(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    agent = Agent(name="vision", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    await runner.run(task="Start", execution_id="exec-injected-image")

    context = await runner.inject_user_message(
        "exec-injected-image",
        "Inspect the new image",
        files=[{"file_id": "image-456", "name": "screen.jpg", "type": "image/jpeg"}],
        request_interrupt=False,
    )

    assert context is not None
    assert context.messages[-1].context_refs[0].file_id == "image-456"


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


@pytest.mark.asyncio
async def test_run_resume_raises_corrupt_on_contextless_checkpoint() -> None:
    runner = AgentRunner(
        agent=Agent(name="checkpoint-reader", patterns=[], llm=None),
        tracer=ContextlessCheckpointStore(),
    )
    build_context_calls: list[Any] = []
    original_build_context = runner._build_context

    async def spy_build_context(*args: Any, **kwargs: Any) -> Any:
        build_context_calls.append((args, kwargs))
        return await original_build_context(*args, **kwargs)

    runner._build_context = spy_build_context  # type: ignore[method-assign]

    with pytest.raises(CheckpointCorruptError):
        await runner.run(
            task=None,
            execution_id="exec-resume-contextless",
            resume=True,
        )

    assert build_context_calls == []
    assert runner.context_manager.get_context("exec-resume-contextless") is None


@pytest.mark.asyncio
async def test_inject_user_message_raises_corrupt_on_contextless_checkpoint() -> None:
    """A found checkpoint without a context dict is malformed, not absent:
    returning ``None`` here would be indistinguishable from "no checkpoint"
    and the caller would defer forever against a row that can never resume."""
    runner = AgentRunner(
        agent=Agent(name="checkpoint-reader", patterns=[], llm=None),
        tracer=ContextlessCheckpointStore(),
    )

    with pytest.raises(CheckpointCorruptError):
        await runner.inject_user_message(
            "exec-inject-contextless",
            message="hello",
        )


@pytest.mark.asyncio
async def test_resume_drops_legacy_router_output_language(tmp_path: Path) -> None:
    checkpoint_context = ExecutionContext(execution_id="exec-legacy-router-language")
    checkpoint_context.metadata["pattern"] = "auto"
    checkpoint_context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    checkpoint_context.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "auto_router"
    checkpoint_context.add_user_message("Summarize the release notes in one paragraph.")
    child_context = ExecutionContext(execution_id="exec-legacy-router-language_child")
    child_context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    child_context.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "auto_router"
    checkpoint = {
        "context": checkpoint_context.to_dict(),
        "pattern": "StatefulPattern",
        "pattern_state": {
            "output": "restored",
            "active_step_contexts": {"step_1": child_context.to_dict()},
        },
    }
    pattern = StatefulPattern()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[pattern]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(
        task=None,
        execution_id="exec-legacy-router-language",
        checkpoint=checkpoint,
    )

    assert result["success"] is True
    metadata = result["context"].metadata
    assert OUTPUT_LANGUAGE_METADATA_KEY not in metadata
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in metadata
    restored_child = pattern.state["active_step_contexts"]["step_1"]["metadata"]
    assert OUTPUT_LANGUAGE_METADATA_KEY not in restored_child
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in restored_child
    system_content = result["context"].get_messages_for_llm()[0]["content"]
    assert "Output language: Simplified Chinese" not in system_content
    assert "Summarize the release notes in one paragraph." in system_content


@pytest.mark.asyncio
async def test_resume_drops_legacy_plan_output_language(tmp_path: Path) -> None:
    checkpoint_context = ExecutionContext(execution_id="exec-legacy-plan-language")
    checkpoint_context.metadata["pattern"] = "dag_plan_execute"
    checkpoint_context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    checkpoint_context.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "dag_plan"
    checkpoint_context.add_user_message("Summarize the release notes in one paragraph.")
    child_context = ExecutionContext(execution_id="exec-legacy-plan-language_child")
    child_context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    child_context.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "dag_plan"
    checkpoint = {
        "context": checkpoint_context.to_dict(),
        "pattern": "StatefulPattern",
        "pattern_state": {
            "output": "restored",
            "active_step_contexts": {"step_1": child_context.to_dict()},
        },
    }
    pattern = StatefulPattern()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[pattern]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(
        task=None,
        execution_id="exec-legacy-plan-language",
        checkpoint=checkpoint,
    )

    assert result["success"] is True
    metadata = result["context"].metadata
    assert OUTPUT_LANGUAGE_METADATA_KEY not in metadata
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in metadata
    restored_child = pattern.state["active_step_contexts"]["step_1"]["metadata"]
    assert OUTPUT_LANGUAGE_METADATA_KEY not in restored_child
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in restored_child
    system_content = result["context"].get_messages_for_llm()[0]["content"]
    assert "Output language: Simplified Chinese" not in system_content
    assert "Summarize the release notes in one paragraph." in system_content


@pytest.mark.asyncio
async def test_resume_keeps_caller_supplied_output_language(tmp_path: Path) -> None:
    checkpoint_context = ExecutionContext(execution_id="exec-caller-language")
    checkpoint_context.metadata["request_context"] = {
        OUTPUT_LANGUAGE_METADATA_KEY: "French"
    }
    checkpoint_context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "French"
    checkpoint_context.add_user_message("Summarize the release notes.")
    checkpoint = {
        "context": checkpoint_context.to_dict(),
        "pattern": "StatefulPattern",
        "pattern_state": {"output": "restored"},
    }
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[StatefulPattern()]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(
        task=None,
        execution_id="exec-caller-language",
        checkpoint=checkpoint,
    )

    assert result["success"] is True
    assert result["context"].metadata[OUTPUT_LANGUAGE_METADATA_KEY] == "French"
    system_content = result["context"].get_messages_for_llm()[0]["content"]
    assert "Output language: French" in system_content


class _StoredContextCheckpointStore:
    def __init__(self, context: ExecutionContext) -> None:
        self.payload = {"type": "checkpoint", "context": context.to_dict()}

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any]:
        del execution_id
        return self.payload


def _cold_start_runner(context: ExecutionContext) -> AgentRunner:
    return AgentRunner(
        agent=Agent(name="writer", patterns=[], llm=None),
        tracer=_StoredContextCheckpointStore(context),
    )


@pytest.mark.asyncio
async def test_inject_user_message_cold_start_drops_legacy_output_language() -> None:
    stored = ExecutionContext(execution_id="exec-inject-legacy-language")
    stored.metadata["pattern"] = "auto"
    stored.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    stored.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "auto_router"
    stored.add_user_message("Summarize the release notes.")

    context = await _cold_start_runner(stored).inject_user_message(
        "exec-inject-legacy-language",
        message="continue",
    )

    assert context is not None
    assert OUTPUT_LANGUAGE_METADATA_KEY not in context.metadata
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in context.metadata


@pytest.mark.asyncio
async def test_inject_user_message_cold_start_keeps_caller_output_language() -> None:
    stored = ExecutionContext(execution_id="exec-inject-caller-language")
    stored.metadata["request_context"] = {OUTPUT_LANGUAGE_METADATA_KEY: "French"}
    stored.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "French"
    stored.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "auto_router"
    stored.add_user_message("Summarize the release notes.")

    context = await _cold_start_runner(stored).inject_user_message(
        "exec-inject-caller-language",
        message="continue",
    )

    assert context is not None
    assert context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] == "French"
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in context.metadata


def test_resume_migration_only_touches_execution_context_nodes() -> None:
    """The migration owns ExecutionContext metadata and nothing else: a
    ``metadata`` dict inside a message, a tool argument, or a step result is
    someone else's payload, and a cold start would persist a silent edit."""
    root = ExecutionContext(execution_id="exec-migration-ownership")
    root.metadata["pattern"] = "dag_plan_execute"
    root.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    root.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "dag_plan"
    root.add_user_message(
        "Translate the attached note.",
        metadata={OUTPUT_LANGUAGE_METADATA_KEY: "French"},
    )
    child = ExecutionContext(execution_id="exec-migration-ownership_child")
    child.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    child.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "dag_plan"

    checkpoint = {
        "context": root.to_dict(),
        "pattern": "DAGPattern",
        "metadata": {OUTPUT_LANGUAGE_METADATA_KEY: "French"},
        "pattern_state": {
            "active_step_contexts": {"step_1": child.to_dict()},
            "step_results": {
                "step_1": {"metadata": {OUTPUT_LANGUAGE_METADATA_KEY: "French"}}
            },
            "active_step_pattern_states": {
                "step_1": {
                    "last_response": {
                        "tool_calls": [
                            {
                                "arguments": {
                                    "metadata": {OUTPUT_LANGUAGE_METADATA_KEY: "French"}
                                }
                            }
                        ]
                    }
                }
            },
        },
    }

    reset_output_language_to_request_context(checkpoint)

    assert OUTPUT_LANGUAGE_METADATA_KEY not in checkpoint["context"]["metadata"]
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in checkpoint["context"]["metadata"]
    restored_child = checkpoint["pattern_state"]["active_step_contexts"]["step_1"]
    assert OUTPUT_LANGUAGE_METADATA_KEY not in restored_child["metadata"]
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in restored_child["metadata"]

    message_metadata = checkpoint["context"]["messages"][0]["metadata"]
    assert message_metadata[OUTPUT_LANGUAGE_METADATA_KEY] == "French"
    assert checkpoint["metadata"][OUTPUT_LANGUAGE_METADATA_KEY] == "French"
    step_result = checkpoint["pattern_state"]["step_results"]["step_1"]
    assert step_result["metadata"][OUTPUT_LANGUAGE_METADATA_KEY] == "French"
    tool_arguments = checkpoint["pattern_state"]["active_step_pattern_states"][
        "step_1"
    ]["last_response"]["tool_calls"][0]["arguments"]
    assert tool_arguments["metadata"][OUTPUT_LANGUAGE_METADATA_KEY] == "French"


def test_resume_migration_reaches_a_nested_auto_pattern_child_context() -> None:
    child = ExecutionContext(execution_id="exec-migration-nested_child")
    child.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    child.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "auto_router"
    checkpoint = {
        "context": ExecutionContext(execution_id="exec-migration-nested").to_dict(),
        "pattern_state": {
            "dag_state": {"active_step_contexts": {"step_1": child.to_dict()}}
        },
    }

    reset_output_language_to_request_context(checkpoint)

    nested = checkpoint["pattern_state"]["dag_state"]["active_step_contexts"]["step_1"]
    assert OUTPUT_LANGUAGE_METADATA_KEY not in nested["metadata"]
