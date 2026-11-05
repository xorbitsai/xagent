"""Structured output through ``AgentService``.

These tests drive the real ``AgentService`` -> ``AgentExecutionAdapter`` ->
``ExecutionRegistry`` -> ``AgentRunner`` -> ``PatternRuntime`` chain; only the
pattern and the models are test doubles, as in ``test_agent_service_pause.py``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from tests.core.agent.test_agent_service_pause import (
    FakeWorkspaceManager,
    TracerCheckpointStore,
)
from xagent.core.agent import Agent, ExecutionContext, PatternRuntime
from xagent.core.agent.execution_adapter import (
    AgentExecutionAdapter,
    AgentExecutionConfig,
)
from xagent.core.agent.registry import ExecutionRegistry
from xagent.core.agent.runner import AgentRunner
from xagent.core.agent.service import AgentService
from xagent.core.agent.structured_output import (
    STRUCTURED_OUTPUT_RESULT_KEY,
    StructuredOutputFormat,
)

SHAPE = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}
FMT = StructuredOutputFormat("shape", None, SHAPE)
READY = {"status": "ready", "output": {"name": "Bo"}, "reason": None}
DONE = {"success": True, "status": "completed", "output": "Bo is here."}
INTERRUPTED = {"success": False, "status": "interrupted", "error": "stop"}


class ScriptedModel:
    """Answers the extraction with the right arguments and records its calls."""

    def __init__(self, name: str = "model") -> None:
        self.model_name = name
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(copy.deepcopy(kwargs))
        return {
            "type": "tool_call",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "submit_result",
                        "arguments": json.dumps({"name": "Bo"}),
                    },
                }
            ],
        }


class ScriptedPattern:
    """A pattern that returns its results in turn and keeps what it was given."""

    def __init__(self, *results: dict[str, Any]) -> None:
        self.results = list(results)
        self.metadata: list[dict[str, Any]] = []

    async def run(
        self, *, context: ExecutionContext, runtime: PatternRuntime, **_: Any
    ) -> dict[str, Any]:
        self.metadata.append(copy.deepcopy(dict(context.metadata)))
        result = dict(self.results.pop(0))
        if result["status"] == "interrupted":
            await runtime.checkpoint(
                "interrupted",
                context=context,
                pattern=self,
                status="interrupted",
                metadata={"safe_point": "test"},
            )
        return result


class RecordingTracer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(self, event_type: Any, task_id: Any = None, **kwargs: Any):
        self.events.append({"task_id": task_id, **kwargs})


class Rig:
    def __init__(self, tmp_path: Path, pattern: ScriptedPattern) -> None:
        self.tmp_path = tmp_path
        self.pattern = pattern
        self.store = TracerCheckpointStore()
        self.registry = ExecutionRegistry()
        self.service_model = ScriptedModel("service")
        self.config_model = ScriptedModel("config")
        self.tracer = RecordingTracer()
        self.service = AgentService(
            name="s", id="svc-1", tools=[], llm=self.service_model
        )
        self.service._current_task_id = "task-1"
        self.service.tracer = self.tracer  # type: ignore[assignment]
        self.adapter = AgentExecutionAdapter(
            AgentExecutionConfig(
                name="s",
                pattern="react",
                llm=self.config_model,
                tracer=self.store,
                current_task_id="task-1",
                service_id="svc-1",
                registry=self.registry,
            )
        )
        self.service._execution_adapter = self.adapter
        self.adapter._build_runner = lambda: (  # type: ignore[method-assign]
            self.runner(self.adapter.config.llm),
            "agent_react",
        )

    def runner(self, llm: Any) -> AgentRunner:
        return AgentRunner(
            agent=Agent(name="s", patterns=[self.pattern], llm=llm),
            tracer=self.store,
            workspace_manager=FakeWorkspaceManager(self.tmp_path),
        )

    async def leave_resumable(self, llm: Any, *, keep_handle: bool) -> None:
        handle = self.registry.start(
            self.runner(llm), execution_id="task-1", task="first"
        )
        assert handle.task is not None
        assert (await handle.task)["status"] == "interrupted"
        if not keep_handle:
            self.registry.unregister("task-1")


async def test_a_new_run_with_a_format_gets_its_structured_result(tmp_path):
    rig = Rig(tmp_path, ScriptedPattern(DONE))

    result = await rig.service.execute_task(
        "do it", {"run_id": "run-9"}, "task-1", structured_output_format=FMT
    )

    assert result[STRUCTURED_OUTPUT_RESULT_KEY] == READY
    assert len(rig.service_model.calls) == 1 and rig.config_model.calls == []
    (event,) = rig.tracer.events
    assert event["task_id"] == "task-1"
    assert event["data"]["status"] == "ready"


async def test_a_resumed_run_uses_the_model_of_the_live_handle(tmp_path):
    rig = Rig(tmp_path, ScriptedPattern(INTERRUPTED, DONE))
    handle_model = ScriptedModel("handle")
    await rig.leave_resumable(handle_model, keep_handle=True)

    result = await rig.service.resume_execution_by_id(
        "task-1", structured_output_format=FMT, metadata={"run_id": "run-9"}
    )

    assert result is not None and result[STRUCTURED_OUTPUT_RESULT_KEY] == READY
    assert len(handle_model.calls) == 1 and rig.config_model.calls == []
    # The run finished and the handle is gone, but the model was taken before.
    assert rig.registry.get("task-1") is None
    assert rig.tracer.events[0]["task_id"] == "task-1"


async def test_a_resumed_run_without_a_handle_uses_the_configured_model(tmp_path):
    rig = Rig(tmp_path, ScriptedPattern(INTERRUPTED, DONE))
    await rig.leave_resumable(ScriptedModel("gone"), keep_handle=False)

    result = await rig.service.resume_execution_by_id(
        "task-1", structured_output_format=FMT
    )

    assert result is not None and result[STRUCTURED_OUTPUT_RESULT_KEY] == READY
    assert len(rig.config_model.calls) == 1


async def _registry_reads_of_a_resume(tmp_path: Path, **kwargs: Any) -> int:
    rig = Rig(tmp_path, ScriptedPattern(INTERRUPTED, DONE))
    await rig.leave_resumable(ScriptedModel("handle"), keep_handle=True)
    reads: list[str] = []
    real_get = rig.registry.get

    def get(execution_id: str) -> Any:
        reads.append(execution_id)
        return real_get(execution_id)

    rig.registry.get = get  # type: ignore[method-assign]
    await rig.service.resume_execution_by_id("task-1", **kwargs)
    return len(reads)


async def test_the_registry_is_read_for_a_model_only_with_a_format(tmp_path):
    plain = await _registry_reads_of_a_resume(tmp_path / "a")
    with_format = await _registry_reads_of_a_resume(
        tmp_path / "b", structured_output_format=FMT
    )

    assert with_format == plain + 1


async def test_a_resume_that_has_nothing_to_resume_returns_none(tmp_path):
    rig = Rig(tmp_path, ScriptedPattern())
    rig.adapter.resume = AsyncMock(return_value=None)  # type: ignore[method-assign]

    result = await rig.service.resume_execution_by_id(
        "task-1", structured_output_format=FMT
    )

    assert result is None
    assert rig.tracer.events == []


async def test_a_run_without_a_format_is_left_alone(tmp_path):
    rig = Rig(tmp_path, ScriptedPattern())
    new_result: dict[str, Any] = dict(DONE)
    resumed_result: dict[str, Any] = dict(DONE)
    rig.service._execute_agent_task = AsyncMock(return_value=new_result)  # type: ignore[method-assign]
    rig.adapter.resume = AsyncMock(return_value=resumed_result)  # type: ignore[method-assign]

    assert await rig.service.execute_task("do it", None, "task-1") is new_result
    assert await rig.service.resume_execution_by_id("task-1") is resumed_result

    for result in (new_result, resumed_result):
        assert STRUCTURED_OUTPUT_RESULT_KEY not in result
        assert result["output"] == "Bo is here."
    assert rig.service_model.calls == [] and rig.config_model.calls == []
    assert rig.tracer.events == []


def _holds(value: Any, needle: str) -> bool:
    return needle in repr(value)


async def test_the_format_reaches_neither_the_context_nor_the_checkpoint(tmp_path):
    rig = Rig(tmp_path, ScriptedPattern(INTERRUPTED, DONE))
    seen: dict[str, list[dict[str, Any]]] = {"start": [], "resume": []}
    real_start, real_resume = rig.registry.start, rig.registry.resume

    def start(*args: Any, **kwargs: Any) -> Any:
        seen["start"].append(kwargs)
        return real_start(*args, **kwargs)

    async def resume(*args: Any, **kwargs: Any) -> Any:
        seen["resume"].append(kwargs)
        return await real_resume(*args, **kwargs)

    rig.registry.start = start  # type: ignore[method-assign]
    rig.registry.resume = resume  # type: ignore[method-assign]

    first = await rig.service.execute_task(
        "do it", None, "task-1", structured_output_format=FMT
    )
    assert first["status"] == "interrupted"
    assert STRUCTURED_OUTPUT_RESULT_KEY not in first
    await rig.service.resume_execution_by_id("task-1", structured_output_format=FMT)

    for needle in ("structured_output_format", "StructuredOutputFormat"):
        assert not _holds(seen, needle)
        assert not _holds(rig.pattern.metadata, needle)
        assert not _holds(rig.store.by_execution_id, needle)
