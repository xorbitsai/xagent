"""``AgentService.post_user_message``'s fresh-vs-replay report.

These tests drive the real ``AgentService`` -> ``AgentExecutionAdapter`` ->
``ExecutionRegistry`` -> ``AgentRunner`` chain, the same seam
``test_agent_service_pause.py`` uses. ``post_user_message`` is never
mocked: the outcome this test pins is produced by
``AgentRunner.inject_user_message`` and must reach this boundary without
being folded into a bare bool anywhere along the way.
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
from xagent.core.agent.runner import AgentRunner, UserMessageInjectionOutcome
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


def _build_service(execution_id: str) -> tuple[AgentService, ExecutionRegistry, Any]:
    """Real service wired to a real registry, sharing one tracer.

    ``current_task_id`` is set to ``execution_id`` so ``pause_execution``
    (which resolves ``self._current_task_id or self.id`` internally,
    taking no argument) targets the same run this test starts.
    """

    tracer = TracerCheckpointStore()
    registry = ExecutionRegistry()
    service = AgentService(
        name="fresh-replay", id="svc-fresh-replay", tools=[], llm=StubLLM()
    )
    service._current_task_id = execution_id
    service._execution_adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="fresh-replay",
            pattern="react",
            llm=StubLLM(),
            tracer=tracer,
            current_task_id=execution_id,
            service_id=service.id,
            registry=registry,
        )
    )
    return service, registry, tracer


async def _start_interrupted_run(
    service: AgentService,
    registry: ExecutionRegistry,
    tracer: Any,
    tmp_path: Path,
    *,
    execution_id: str,
) -> None:
    """Start one real run, pause it, and wait for it to land in an
    interrupted (and therefore resumable) state, so its handle stays in
    the registry for ``post_user_message`` to find."""

    pattern = PollingPattern()
    runner = AgentRunner(
        agent=Agent(name="fresh-replay", patterns=[pattern]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    handle = registry.start(runner, execution_id=execution_id, task="Wait")
    await asyncio.wait_for(pattern.started.wait(), timeout=5)
    assert handle.task is not None
    assert await service.pause_execution() is True
    await handle.task


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["fresh", "replay", "conflicting_content"])
async def test_service_post_user_message_reports_fresh_vs_replay(
    tmp_path: Path, scenario: str
) -> None:
    """The outermost layer forwards the execution adapter's report
    unmodified, no ``bool(...)`` fold: a first write is POSTED_FRESH, a
    repeat of the same turn id with the same content short-circuits as
    POSTED_REPLAY, and a repeat with different content still raises --
    the pre-existing conflict behavior, untouched by this contract."""
    execution_id = "exec-service-fresh-replay"
    service, registry, tracer = _build_service(execution_id)
    await _start_interrupted_run(
        service, registry, tracer, tmp_path, execution_id=execution_id
    )

    first = await service.post_user_message(
        execution_id,
        "Choose B",
        turn_id="turn-service-fresh-replay",
        request_interrupt=False,
    )
    assert first is UserMessageInjectionOutcome.POSTED_FRESH

    if scenario == "replay":
        second = await service.post_user_message(
            execution_id,
            "Choose B",
            turn_id="turn-service-fresh-replay",
            request_interrupt=False,
        )
        assert second is UserMessageInjectionOutcome.POSTED_REPLAY
    elif scenario == "conflicting_content":
        with pytest.raises(ValueError, match="different user message"):
            await service.post_user_message(
                execution_id,
                "Choose C",
                turn_id="turn-service-fresh-replay",
                request_interrupt=False,
            )
