from __future__ import annotations

from typing import Any, cast

import pytest

from xagent.core.agent.builtin import (
    BuiltinAgentCapabilityError,
    BuiltinAgentExecutor,
    BuiltinAgentModelUnavailableError,
    BuiltinAgentRegistry,
    BuiltinAgentSpec,
)
from xagent.core.tools.adapters.vibe.base import ToolCategory, ToolMetadata


class FakeLLM:
    model_name = "fake-model"

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class RecordingService:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.execute_kwargs: dict[str, Any] | None = None

    async def execute_task(self, task: str, **kwargs: Any) -> dict[str, Any]:
        self.execute_kwargs = {"task": task, **kwargs}
        return {
            "status": "completed",
            "success": True,
            "output": "done",
            "metadata": {"pattern": self.init_kwargs["pattern"]},
        }


class RecordingServiceFactory:
    def __init__(self) -> None:
        self.service: RecordingService | None = None

    def __call__(self, **kwargs: Any) -> RecordingService:
        self.service = RecordingService(**kwargs)
        return self.service


class AgentCategoryTool:
    metadata = ToolMetadata(
        name="delegate",
        description="Delegate work.",
        category=ToolCategory.AGENT,
    )


class OtherCategoryTool:
    metadata = ToolMetadata(
        name="uncategorized",
        description="Tool without an assignable category.",
    )


class BasicCategoryTool:
    metadata = ToolMetadata(
        name="basic",
        description="A normal built-in tool.",
        category=ToolCategory.BASIC,
    )


def _registry(**overrides: Any) -> BuiltinAgentRegistry:
    values: dict[str, Any] = {
        "name": "internal_worker",
        "version": "1",
        "system_prompt": "Perform one internal task.",
    }
    values.update(overrides)
    return BuiltinAgentRegistry([BuiltinAgentSpec(**values)])


@pytest.mark.asyncio
async def test_executor_builds_a_least_privilege_agent_and_stamps_metadata() -> None:
    model = FakeLLM()
    resolver_calls: list[tuple[str, str]] = []
    service_factory = RecordingServiceFactory()

    async def resolve_model(role: str, context: Any) -> FakeLLM:
        resolver_calls.append((role, context.execution_id))
        return model

    executor = BuiltinAgentExecutor(
        registry=_registry(),
        model_resolver=cast(Any, resolve_model),
        service_factory=cast(Any, service_factory),
    )

    result = await executor.execute(
        "internal_worker",
        task="Perform internal work",
        execution_id="run-1",
        request_context={"work_item_count": 3},
    )

    assert resolver_calls == [("general", "run-1")]
    assert service_factory.service is not None
    init = service_factory.service.init_kwargs
    assert init["name"] == "builtin:internal_worker"
    assert init["id"] == "builtin--internal_worker--run-1"
    assert init["pattern"] == "single_call"
    assert init["llm"] is model
    assert init["tools"] == []
    assert init["tools_initialized"] is True
    assert init["tool_config"] is None
    assert init["enable_default_tools"] is False
    assert init["enable_workspace"] is False
    assert init["memory_enabled"] is False
    assert init["skills_enabled"] is False
    assert init["user_interaction_enabled"] is False
    assert "agent_type" not in init

    execute = service_factory.service.execute_kwargs
    assert execute is not None
    assert execute["task"] == "Perform internal work"
    assert execute["task_id"] == "run-1"
    assert execute["context"]["work_item_count"] == 3
    assert execute["context"]["builtin_agent"] == {
        "agent_type": "builtin",
        "builtin_agent_name": "internal_worker",
        "builtin_agent_version": "1",
        "builtin_model_role": "general",
    }
    assert result["metadata"]["builtin_agent_name"] == "internal_worker"
    assert result["metadata"]["builtin_agent_version"] == "1"


@pytest.mark.asyncio
async def test_executor_supports_execution_scoped_async_tool_builders() -> None:
    tool = BasicCategoryTool()
    builder_execution_ids: list[str] = []

    async def build_tools(context: Any) -> list[Any]:
        builder_execution_ids.append(context.execution_id)
        context.artifacts["built"] = True
        return [tool]

    service_factory = RecordingServiceFactory()
    executor = BuiltinAgentExecutor(
        registry=_registry(build_tools=build_tools),
        model_resolver=cast(Any, lambda _role, _context: FakeLLM()),
        service_factory=cast(Any, service_factory),
    )

    result = await executor.execute(
        "internal_worker",
        task="Perform internal work",
        execution_id="run-tools",
    )

    assert builder_execution_ids == ["run-tools"]
    assert service_factory.service is not None
    assert service_factory.service.init_kwargs["tools"] == [tool]
    assert result["builtin_artifacts"] == {"built": True}


@pytest.mark.asyncio
async def test_executor_fails_before_service_creation_when_model_is_unavailable() -> (
    None
):
    service_factory = RecordingServiceFactory()
    executor = BuiltinAgentExecutor(
        registry=_registry(model_role="fast"),
        model_resolver=lambda _role, _context: None,
        service_factory=cast(Any, service_factory),
    )

    with pytest.raises(BuiltinAgentModelUnavailableError, match="role 'fast'"):
        await executor.execute(
            "internal_worker",
            task="Perform internal work",
            execution_id="run-no-model",
        )

    assert service_factory.service is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool",
    [AgentCategoryTool(), OtherCategoryTool(), object()],
)
async def test_executor_rejects_tools_without_an_assignable_category(
    tool: Any,
) -> None:
    executor = BuiltinAgentExecutor(
        registry=_registry(build_tools=lambda _context: [tool]),
        model_resolver=cast(Any, lambda _role, _context: FakeLLM()),
        service_factory=cast(Any, RecordingServiceFactory()),
    )

    with pytest.raises(BuiltinAgentCapabilityError, match="assignable category"):
        await executor.execute(
            "internal_worker",
            task="Perform internal work",
            execution_id="run-delegation",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_root", [None, "", " "])
async def test_executor_requires_an_explicit_root_for_workspace_access(
    tmp_path: Any,
    missing_root: str | None,
) -> None:
    service_factory = RecordingServiceFactory()
    executor = BuiltinAgentExecutor(
        registry=_registry(workspace_enabled=True),
        model_resolver=cast(Any, lambda _role, _context: FakeLLM()),
        service_factory=cast(Any, service_factory),
    )

    with pytest.raises(BuiltinAgentCapabilityError, match="workspace_base_dir"):
        await executor.execute(
            "internal_worker",
            task="Perform internal work",
            execution_id="run-workspace",
            workspace_base_dir=missing_root,
        )
    assert service_factory.service is None

    await executor.execute(
        "internal_worker",
        task="Perform internal work",
        execution_id="run-workspace",
        workspace_base_dir=str(tmp_path),
    )

    assert service_factory.service is not None
    assert service_factory.service.init_kwargs["enable_workspace"] is True
    assert service_factory.service.init_kwargs["workspace_base_dir"] == str(tmp_path)
    assert (
        service_factory.service.init_kwargs["id"]
        == "builtin--internal_worker--run-workspace"
    )


@pytest.mark.asyncio
async def test_executor_runs_through_real_agent_service_without_default_tools(
    tmp_path: Any,
) -> None:
    model = FakeLLM(["work complete"])
    executor = BuiltinAgentExecutor(
        registry=_registry(),
        model_resolver=cast(Any, lambda _role, _context: model),
    )

    result = await executor.execute(
        "internal_worker",
        task="Perform internal work",
        execution_id="run-real-service",
        workspace_base_dir=str(tmp_path),
    )

    assert result["success"] is True
    assert result["output"] == "work complete"
    assert result["metadata"]["execution_type"] == "agent_single_call"
    assert result["metadata"]["builtin_agent_name"] == "internal_worker"
    tool_names = {
        tool["function"]["name"] for tool in model.calls[0].get("tools") or []
    }
    assert tool_names == {"final_answer"}
    assert not any(path.name.startswith("builtin:") for path in tmp_path.iterdir())
