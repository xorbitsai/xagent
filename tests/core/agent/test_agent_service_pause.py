"""Pause semantics for ``AgentService``.

These tests drive the real ``AgentService`` -> ``AgentExecutionAdapter`` ->
``ExecutionRegistry`` -> ``AgentRunner`` -> ``PatternRuntime`` chain. Only the
pattern is a test double, which is the same seam ``test_registry.py`` uses.
``pause_execution`` is never mocked: the regression these tests protect is
precisely that a mocked ``pause_execution`` hides.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from xagent.core.agent import Agent, ContextManager, ExecutionContext, PatternRuntime
from xagent.core.agent.execution_adapter import (
    AgentExecutionAdapter,
    AgentExecutionConfig,
)
from xagent.core.agent.registry import ExecutionRegistry
from xagent.core.agent.runner import AgentRunner
from xagent.core.agent.service import AgentService


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
        scope_segments: tuple[str, ...] = (),
    ) -> FakeWorkspace:
        del base_dir, scope_segments
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


class PollingPattern:
    """Runs until the runtime reports an interrupt, like a real pattern loop."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.observed_interrupt = False

    async def run(
        self,
        *,
        context: ExecutionContext,
        runtime: PatternRuntime,
        **_: Any,
    ) -> dict[str, Any]:
        self.started.set()
        while not await runtime.should_interrupt():
            await asyncio.sleep(0)
        self.observed_interrupt = True
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
            "error": "PollingPattern interrupted.",
        }


class StubLLM:
    """Only ever inspected for logging / "is an LLM configured" checks."""

    model_name = "stub-model"


def _build_service(tmp_path: Path) -> tuple[AgentService, ExecutionRegistry, Any]:
    """Real service wired to a real registry, sharing one tracer."""

    tracer = TracerCheckpointStore()
    registry = ExecutionRegistry()
    service = AgentService(name="stopper", id="svc-1", tools=[], llm=StubLLM())
    service._current_task_id = "task-1"
    service._execution_adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="stopper",
            pattern="react",
            llm=StubLLM(),
            tracer=tracer,
            current_task_id="task-1",
            service_id=service.id,
            registry=registry,
        )
    )
    return service, registry, tracer


async def _start_live_run(
    registry: ExecutionRegistry,
    tracer: Any,
    tmp_path: Path,
    *,
    execution_id: str,
    task: str,
) -> tuple[PollingPattern, asyncio.Task]:
    """Start one real run and wait until its pattern is actually executing."""

    pattern = PollingPattern()
    runner = AgentRunner(
        agent=Agent(name="stopper", patterns=[pattern]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    handle = registry.start(runner, execution_id=execution_id, task=task)
    await asyncio.wait_for(pattern.started.wait(), timeout=5)
    # Hold the asyncio task itself: the registry clears ``handle.task`` from
    # its done-callback, so it is None by the time a settled run is inspected.
    assert handle.task is not None
    return pattern, handle.task


async def _settle(run_task: asyncio.Task, *, label: str) -> dict[str, Any]:
    """Await a run that a pause should have ended, failing loudly if it didn't."""

    done, _pending = await asyncio.wait({run_task}, timeout=5)
    if not done:
        run_task.cancel()
        pytest.fail(f"{label} kept running: the pause never reached the runtime")
    return run_task.result()


@pytest.mark.asyncio
async def test_second_pause_after_new_turn_interrupts_the_new_run(
    tmp_path: Path,
) -> None:
    """A stop on turn 2 must reach the runtime, even after turn 1 was stopped.

    Regression for the stale ``_is_paused`` flag: turn 1's pause left a
    service-level flag set, and because a user message on a PAUSED task starts
    an APPEND turn (not a resume), nothing ever cleared it. The second
    ``pause_execution`` then short-circuited and never requested an interrupt.
    """

    service, registry, tracer = _build_service(tmp_path)

    first_pattern, first_run = await _start_live_run(
        registry, tracer, tmp_path, execution_id="task-1", task="first turn"
    )
    assert await service.pause_execution() is True
    first_result = await _settle(first_run, label="turn 1")
    assert first_result["status"] == "interrupted"
    assert first_pattern.observed_interrupt is True

    # The APPEND turn: a brand new run under the same task id, exactly what
    # ``agent_manager.execute_task`` does for a user message on a PAUSED task.
    second_pattern, second_run = await _start_live_run(
        registry, tracer, tmp_path, execution_id="task-1", task="second turn"
    )

    assert await service.pause_execution() is True

    second_result = await _settle(second_run, label="turn 2")
    assert second_result["status"] == "interrupted"
    assert second_pattern.observed_interrupt is True


@pytest.mark.asyncio
async def test_pause_reports_false_when_no_run_is_live(tmp_path: Path) -> None:
    """With no live execution, pause must report failure rather than success."""

    service, registry, tracer = _build_service(tmp_path)

    _pattern, run_task = await _start_live_run(
        registry, tracer, tmp_path, execution_id="task-1", task="first turn"
    )
    assert await service.pause_execution() is True
    await _settle(run_task, label="turn 1")

    registry.unregister("task-1")

    assert await service.pause_execution() is False
